"""The offline-only logic: mask packing, hotstart retraction, pass merging.

None of this needs SAM 3, a GPU or weights — and all of it is exactly the kind
of code that fails silently. A merge that quietly prefers the wrong pass, or a
retraction that misses the frames before the removal fired, produces a file
that looks completely normal and is wrong everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_extraction.sam3_offline import (
    SOURCE_BOTH_FORWARD, SOURCE_BOTH_REVERSE, SOURCE_FORWARD, SOURCE_NONE,
    SOURCE_REVERSE, OfflineSam3, SlotFrame,
)
from ego2g1.deploy.perception.v2.sam3_source import SlotObservation

H, W = 6, 8


def mask(area: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m.flat[:area] = True
    return m


def obs(*, area=10, det=0.9, trk=0.8, obj_id=1, occluded=False):
    m = mask(area) if area else None
    return SlotObservation(
        instance_id="obj0", mask=m, box_xyxy=np.array([0.0, 0, 3, 3]),
        det_score=det, tracker_score=trk,
        mask_area_px=0 if m is None else int(m.sum()),
        occluded=occluded, obj_id=obj_id)


def frame(*, area=10, det=0.9, trk=0.8, obj_id=1, source=SOURCE_FORWARD):
    return SlotFrame.from_observation(
        obs(area=area, det=det, trk=trk, obj_id=obj_id), source)


# -- packing ----------------------------------------------------------------

def test_a_packed_mask_round_trips_exactly():
    sf = frame(area=13)
    np.testing.assert_array_equal(sf.mask(H, W), mask(13))


def test_packing_is_eight_times_smaller():
    sf = frame(area=48)
    assert sf.mask_bits.nbytes * 8 >= H * W          # no data lost
    assert sf.mask_bits.nbytes <= (H * W) // 8 + 1   # and it is actually bits


def test_a_mask_whose_size_is_not_a_multiple_of_eight_is_not_truncated():
    # H*W = 48 here, so force the awkward case explicitly: unpackbits pads to
    # a byte boundary and `count` is the only thing that trims it back.
    m = np.zeros((3, 5), dtype=bool)      # 15 px -> 2 bytes -> 1 pad bit
    m[-1, -1] = True
    sf = SlotFrame(mask_bits=np.packbits(m), mask_area_px=1)
    np.testing.assert_array_equal(sf.mask(3, 5), m)


def test_no_mask_stays_none():
    sf = SlotFrame()
    assert sf.mask(H, W) is None
    assert not sf.has_mask


# -- retraction -------------------------------------------------------------

def test_hotstart_retraction_wipes_frames_before_the_removal_fired():
    """The offline-only half of hotstart.

    `postprocess_outputs` filters against the removed set AS IT STANDS when it
    is called, and the iterator only buffers 15 frames — so a tracklet
    retracted at the end of the video survives at the start of the pass. This
    is the pass that fixes that, and if it regresses the file is silently full
    of the spurious tracklets hotstart exists to delete.
    """
    per_frame = [{"obj0": frame(obj_id=7)} for _ in range(50)]
    wiped = OfflineSam3._retract(per_frame, {7})
    assert wiped == 50
    assert all(f["obj0"].source == SOURCE_NONE for f in per_frame)
    assert all(not f["obj0"].has_mask for f in per_frame)


def test_retraction_leaves_other_tracklets_alone():
    per_frame = [{"a": frame(obj_id=7), "b": frame(obj_id=8)} for _ in range(4)]
    OfflineSam3._retract(per_frame, {7})
    assert all(not f["a"].has_mask for f in per_frame)
    assert all(f["b"].has_mask for f in per_frame)


def test_retraction_with_nothing_removed_is_a_no_op():
    per_frame = [{"a": frame(obj_id=7)}]
    assert OfflineSam3._retract(per_frame, set()) == 0
    assert per_frame[0]["a"].has_mask


# -- merge ------------------------------------------------------------------

def test_a_detected_frame_beats_a_propagated_one_regardless_of_score():
    """S1's governing distinction, carried into the merge.

    A propagated mask can carry a very high TRACKER score while being pure
    memory — that is exactly the occluded case. Ranking on score alone would
    pick it over a genuine re-detection with a modest detection score.
    """
    detected = frame(det=0.55, trk=0.10)
    propagated = frame(det=None, trk=0.99, source=SOURCE_REVERSE)
    assert OfflineSam3._pick(detected, propagated).source == SOURCE_BOTH_FORWARD
    assert OfflineSam3._pick(propagated, detected).source == SOURCE_BOTH_REVERSE


def test_between_two_detections_the_higher_score_wins():
    assert OfflineSam3._pick(frame(det=0.6), frame(det=0.9)).source == \
        SOURCE_BOTH_REVERSE
    assert OfflineSam3._pick(frame(det=0.9), frame(det=0.6)).source == \
        SOURCE_BOTH_FORWARD


def test_between_two_propagations_the_tracker_score_breaks_the_tie():
    a = frame(det=None, trk=0.2)
    b = frame(det=None, trk=0.7)
    assert OfflineSam3._pick(a, b).source == SOURCE_BOTH_REVERSE


def test_the_reverse_pass_fills_a_frame_the_forward_pass_missed():
    """The whole reason for a second pass.

    A causal tracker has nothing before the detector first fires, and in an
    egocentric recording that is routinely the first seconds of the episode.
    """
    picked = OfflineSam3._pick(None, frame(source=SOURCE_REVERSE))
    assert picked.source == SOURCE_REVERSE
    assert picked.has_mask


def test_an_empty_mask_does_not_count_as_a_mask():
    empty = frame(area=0)
    assert not empty.has_mask
    assert OfflineSam3._pick(empty, None).source == SOURCE_NONE


def test_neither_pass_finding_anything_is_an_empty_frame_not_a_crash():
    picked = OfflineSam3._pick(None, None)
    assert picked.source == SOURCE_NONE
    assert picked.det_score is None
    assert not picked.has_mask


def test_the_merge_never_unions_the_two_masks():
    """Deliberate: the passes seed independent tracklets, so a union would
    manufacture a silhouette neither model produced — and orientation reads
    global shape, so a fabricated silhouette is worse than a smaller true
    one."""
    a = frame(area=8)
    b = frame(area=40, source=SOURCE_REVERSE)
    picked = OfflineSam3._pick(a, b)
    assert picked.mask_area_px in (8, 40)


# -- construction guards ----------------------------------------------------

def test_an_unknown_pass_name_is_refused():
    with pytest.raises(ValueError, match="unknown pass"):
        OfflineSam3(object(), passes=("forward", "sideways"))


def test_no_passes_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one pass"):
        OfflineSam3(object(), passes=())
