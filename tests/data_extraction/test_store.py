"""The extraction file round-trips, and its missing values survive the trip.

The file is the deliverable — the dashboard and every later analysis read it
and nothing else. Two things it must never do: lose the distinction between
"not re-detected" and "score zero", and lose mask geometry to the compression
or the chunking.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from data_extraction.orient_offline import SKIP_NO_MASK, SKIP_NONE, OrientationResult
from data_extraction.sam3_offline import SOURCE_FORWARD, SOURCE_REVERSE, SlotFrame, Tracks
from data_extraction.store import SCHEMA, write_extraction

h5py = pytest.importorskip("h5py")

H, W, F = 5, 7, 4
SLOTS = ("obj0", "obj1")


class FakeEpisode:
    path = "/tmp/episode_0.hdf5"
    name = "run/episode_0"
    eye = "left"
    height, width = H, W
    K = np.eye(3)
    timestamps_ns = np.arange(F, dtype=np.int64) * 33_000_000
    task_instruction = "put the red block in the holder"
    anchor_object = "block"


def blob(i: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[1:3, i:i + 2] = True
    return m


def make_tracks() -> Tracks:
    frames = {}
    for s_i, slot in enumerate(SLOTS):
        rows = []
        for i in range(F):
            if i == 0:                      # nothing found at all
                rows.append(SlotFrame())
                continue
            m = blob(i)
            rows.append(SlotFrame(
                mask_bits=np.packbits(m),
                box_xyxy=np.array([i, 1, i + 2, 3], dtype=np.float32),
                # frame 1 is memory propagation: det_score STAYS None
                det_score=None if i == 1 else 0.5 + 0.1 * i,
                tracker_score=0.7,
                mask_area_px=int(m.sum()),
                occluded=False,
                obj_id=10 + s_i,
                source=SOURCE_REVERSE if i == 1 else SOURCE_FORWARD,
                mask_usable=True,
                crop_usable=i != 1,
                gate_reason="" if i != 1 else "not re-detected (memory propagation)",
            ))
        frames[slot] = rows
    return Tracks(slot_ids=SLOTS, n_frames=F, height=H, width=W,
                  frames=frames, stats={"passes": ["forward", "reverse"]})


def make_orientation() -> OrientationResult:
    az = {s: np.arange(F, dtype=np.float32) * 10 for s in SLOTS}
    el = {s: np.zeros(F, dtype=np.float32) for s in SLOTS}
    ro = {s: np.zeros(F, dtype=np.float32) for s in SLOTS}
    R = {s: np.tile(np.eye(3, dtype=np.float32), (F, 1, 1)) for s in SLOTS}
    skip = {s: np.full(F, SKIP_NONE, dtype=np.uint8) for s in SLOTS}
    for s in SLOTS:
        az[s][0] = np.nan
        R[s][0] = np.nan
        skip[s][0] = SKIP_NO_MASK
    return OrientationResult(azimuth_deg=az, elevation_deg=el, roll_deg=ro,
                             R_cam=R, skip=skip, stats={"crops": 6})


@pytest.fixture
def written(tmp_path):
    out = write_extraction(
        tmp_path / "episode_0.h5", episode=FakeEpisode(), tracks=make_tracks(),
        orientation=make_orientation(),
        meta={"prompt_to_slot": {"red block": "obj0", "holder": "obj1"},
              "sam3": {"passes": ["forward", "reverse"]}})
    return out


def test_the_file_has_one_group_per_roster_slot(written):
    with h5py.File(written, "r") as f:
        assert f.attrs["schema"] == SCHEMA
        assert [s.decode() if isinstance(s, bytes) else s
                for s in f["objects"][:]] == list(SLOTS)
        assert set(f["obj"].keys()) == set(SLOTS)


def test_masks_survive_compression_and_chunking_bit_for_bit(written):
    with h5py.File(written, "r") as f:
        m = f["obj/obj0/mask"]
        assert m.chunks == (1, H, W)         # one frame per chunk: the read
        assert m.compression == "gzip"       # pattern a dashboard uses
        for i in range(1, F):
            np.testing.assert_array_equal(m[i].astype(bool), blob(i))
        assert not m[0].any()


def test_not_re_detected_reads_back_as_nan_not_zero(written):
    """The single most important missing value in the file.

    `det_score` NaN means the DETECTOR did not re-find the object — the mask
    is memory propagation and the crop is a guess (S1). Zero would mean it
    looked and scored it zero. Collapsing the two erases the distinction every
    downstream gate is built on.
    """
    with h5py.File(written, "r") as f:
        det = f["obj/obj0/det_score"][:]
    assert np.isnan(det[1])
    assert not np.isnan(det[2])
    assert det[2] == pytest.approx(0.7)


def test_a_frame_with_no_mask_has_no_box_and_no_rotation(written):
    with h5py.File(written, "r") as f:
        assert np.isnan(f["obj/obj0/box_xyxy"][0]).all()
        assert np.isnan(f["obj/obj0/R_cam"][0]).all()
        assert f["obj/obj0/orient_skip"][0] == SKIP_NO_MASK


def test_provenance_survives_so_the_reverse_pass_is_countable(written):
    with h5py.File(written, "r") as f:
        src = f["obj/obj0/source"][:]
    assert src[1] == SOURCE_REVERSE
    assert src[2] == SOURCE_FORWARD


def test_the_gates_are_recorded_not_applied(written):
    """A crop the online gate would reject is still in the file, WITH its
    rotation. That is the experiment: you cannot ask whether the gate was
    right about a frame it threw away unless the frame is there."""
    with h5py.File(written, "r") as f:
        assert f["obj/obj0/crop_usable"][1] == False        # noqa: E712
        assert f["obj/obj0/mask"][1].any()                  # kept anyway
        reason = f["obj/obj0/gate_reason"][1]
    assert "memory propagation" in (
        reason.decode() if isinstance(reason, bytes) else reason)


def test_the_sidecar_is_readable_json_not_double_encoded(written):
    meta = json.loads(written.with_suffix(".h5.meta.json").read_text())
    assert meta["schema"] == SCHEMA
    # dict, not a JSON string that happens to look like one
    assert meta["prompt_to_slot"]["red block"] == "obj0"
    assert meta["per_object"]["obj0"]["coverage"] == pytest.approx(0.75)


def test_per_object_coverage_matches_the_masks(written):
    with h5py.File(written, "r") as f:
        g = f["obj/obj0"]
        assert g.attrs["coverage"] == pytest.approx(3 / 4)
        assert g.attrs["detection_rate"] == pytest.approx(2 / 4)
        assert json.loads(g.attrs["source_counts"])["reverse"] == 1
