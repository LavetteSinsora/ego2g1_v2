"""Orient Anything V2 over every mask in an episode.

Two differences from the deploy stage (`orientation_v2.OrientAnythingV2
.estimate`), both of them the point of the exercise:

GATE
    Deploy runs orientation only where `crop_usable` (S1), because a sliver
    crop returns a confident wrong answer that poisons the symmetry-snap
    reference for every later frame. That is the correct ONLINE rule — online
    there is no way to find out afterwards whether the answer was wrong.

    Here we run it on EVERY frame that has a mask, and record `crop_usable`
    alongside. The gate's threshold set is the thing under test: with the
    rejected frames' orientations in the output, "would the gate have thrown
    away a good answer?" and "did an answer it admitted turn out to be
    garbage?" both become measurable. Answering them is how the numbers in
    `VisibilityConfig` stop being bring-up defaults (plan Q10).

    The one thing still skipped is a mask too small to crop at all —
    `crop_from_mask` returns None below `min_side_px`, and there is no image
    to run. That is the user-stated boundary: no mask, no orientation.

BATCH
    Deploy batches the three roster slots of one frame; the measured cost is
    ~24 ms fixed + ~30 ms per crop, so at N=3 nearly a third of the time is
    the fixed part. Offline there is no frame boundary to respect, so crops
    batch across FRAMES — a batch of 24 amortises that fixed cost eight times
    further. The forward and the decode are `OrientAnythingV2
    .angles_for_crops`, unchanged, so the numbers stay comparable.

NOT DONE HERE, deliberately
    `compose_relational_rotation` — training gives only the ANCHOR object the
    model's raw rotation and CONSTRUCTS every other object's from the
    direction to the anchor. That construction needs camera-frame 3D
    translations, which need depth, which this monocular pass does not
    produce. So what is written out is the RAW model rotation for every slot,
    which is the honest raw material: the relational form is a pure function
    of (raw rotation, own position, anchor position) and can be applied later
    without re-running the model. The anchor id is recorded in the metadata so
    it is not lost.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from data_extraction.symmetry import SymmetryFitter, summarise
from ego2g1.deploy.perception.v2.orientation_v2 import (
    angles_to_matrix, crop_from_mask,
)

logger = logging.getLogger(__name__)

__all__ = ["OrientationResult", "estimate_over_episode"]

# Why a (frame, slot) has no rotation. Written out as a uint8 so the dashboard
# can separate "the tracker lost it" from "the crop was unusable" — those have
# completely different fixes.
SKIP_NONE = 0        # has a rotation
SKIP_NO_MASK = 1     # SAM 3 produced nothing (both passes)
SKIP_TINY = 2        # mask present but too small to cut a crop from
SKIP_NAMES = {SKIP_NONE: "ok", SKIP_NO_MASK: "no mask", SKIP_TINY: "crop too small"}


@dataclasses.dataclass
class OrientationResult:
    """Per slot, per frame. NaN rows where `skip != SKIP_NONE`."""

    azimuth_deg: dict[str, np.ndarray]      # (F,) float32
    elevation_deg: dict[str, np.ndarray]
    roll_deg: dict[str, np.ndarray]
    R_cam: dict[str, np.ndarray]            # (F, 3, 3) float32
    skip: dict[str, np.ndarray]             # (F,) uint8
    stats: dict
    # Rotational symmetry order per frame: {0, 1, 2, 4}, or -1 for "not
    # measured". See symmetry.py — 0 is a real answer ("no confident call"),
    # which is why "not measured" needs its own value.
    alpha: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def rate(self, slot: str) -> float:
        s = self.skip[slot]
        return float((s == SKIP_NONE).mean()) if s.size else 0.0


def estimate_over_episode(orient, episode, tracks, *, batch_size: int = 24,
                          fit_symmetry: bool = True,
                          progress: bool = True) -> OrientationResult:
    """Run Orient Anything V2 on every masked (frame, slot) in `tracks`.

    `orient` is a constructed `OrientAnythingV2`; `tracks` is the merged
    output of `OfflineSam3.run`. Frames are decoded one at a time and their
    crops accumulate into a cross-frame batch, so peak memory is one frame of
    RGB plus `batch_size` small crops however long the episode is.
    """
    F = tracks.n_frames
    slots = tracks.slot_ids
    H, W = tracks.height, tracks.width

    az = {s: np.full(F, np.nan, dtype=np.float32) for s in slots}
    el = {s: np.full(F, np.nan, dtype=np.float32) for s in slots}
    ro = {s: np.full(F, np.nan, dtype=np.float32) for s in slots}
    R = {s: np.full((F, 3, 3), np.nan, dtype=np.float32) for s in slots}
    skip = {s: np.full(F, SKIP_NO_MASK, dtype=np.uint8) for s in slots}
    alpha = {s: np.full(F, -1, dtype=np.int8) for s in slots}

    fitter = SymmetryFitter(enabled=fit_symmetry)
    if fit_symmetry and not fitter.available:
        print(f"  [orient] symmetry fit unavailable ({fitter.reason}); "
              f"alpha will be -1")

    pending_crops: list = []
    pending_keys: list[tuple[str, int]] = []
    n_forward = 0
    t_model = t_alpha = 0.0
    t0 = time.perf_counter()

    def flush() -> None:
        nonlocal pending_crops, pending_keys, n_forward, t_model, t_alpha
        if not pending_crops:
            return
        t = time.perf_counter()
        # The azimuth DISTRIBUTION, not just its argmax — the symmetry order
        # is read off the shape of that curve, so the argmax alone cannot
        # produce it and a second forward would be pure waste.
        out = orient.angles_for_crops(
            pending_crops, return_azimuth_distribution=fitter.available)
        a, e, r = out[:3]
        t_model += time.perf_counter() - t
        n_forward += 1

        if fitter.available:
            t = time.perf_counter()
            fitted = fitter(out[3])
            t_alpha += time.perf_counter() - t
            for i, (slot, frame_idx) in enumerate(pending_keys):
                if i < len(fitted):
                    alpha[slot][frame_idx] = fitted[i]
        # `angles_to_matrix` returns (N, 3, 3) for array input and (3, 3) for
        # scalars. Reshape rather than `atleast_3d`, which would turn a single
        # (3, 3) into (3, 3, 1) and silently transpose the batch axis.
        mats = np.asarray(angles_to_matrix(a, e, r, convention=orient.convention),
                          dtype=np.float64).reshape(-1, 3, 3)
        for i, (slot, frame_idx) in enumerate(pending_keys):
            az[slot][frame_idx] = a[i]
            el[slot][frame_idx] = e[i]
            ro[slot][frame_idx] = r[i]
            R[slot][frame_idx] = mats[i]
            skip[slot][frame_idx] = SKIP_NONE
        pending_crops, pending_keys = [], []

    for frame_idx in range(F):
        wanted = [s for s in slots if tracks.frames[s][frame_idx].has_mask]
        if not wanted:
            continue
        rgb = episode.frame(frame_idx)
        for slot in wanted:
            sf = tracks.frames[slot][frame_idx]
            crop = crop_from_mask(rgb, sf.mask(H, W), pad=orient.crop_pad,
                                  background=orient.background)
            if crop is None:
                skip[slot][frame_idx] = SKIP_TINY
                continue
            pending_crops.append(crop)
            pending_keys.append((slot, frame_idx))
            if len(pending_crops) >= batch_size:
                flush()
        if progress and (frame_idx + 1) % 100 == 0:
            print(f"  [orient] {frame_idx + 1}/{F} frames", flush=True)
    flush()

    wall = time.perf_counter() - t0
    n_crops = int(sum(int((skip[s] == SKIP_NONE).sum()) for s in slots))
    stats = {
        "crops": n_crops,
        "batches": n_forward,
        "batch_size": batch_size,
        "wall_s": round(wall, 2),
        "model_s": round(t_model, 2),
        "ms_per_crop": round(1000 * t_model / max(1, n_crops), 2),
        "input_size": orient.size,
        "cast_weights": orient.cast_weights,
        "background": orient.background,
        "crop_pad": orient.crop_pad,
        "symmetry_available": fitter.available,
        "symmetry_s": round(t_alpha, 2),
        "symmetry": {s: summarise(alpha[s]) for s in slots},
    }
    if progress:
        print(f"  [orient] {n_crops} crops in {n_forward} batches, "
              f"{t_model:.1f} s model time ({stats['ms_per_crop']:.1f} ms/crop)")
        if fitter.available:
            print(f"  [orient] symmetry fit {t_alpha:.1f} s; " + ", ".join(
                f"{s}=alpha {stats['symmetry'][s]['mode']} "
                f"({stats['symmetry'][s]['agreement']:.0%} of frames)"
                for s in slots))
    return OrientationResult(azimuth_deg=az, elevation_deg=el, roll_deg=ro,
                             R_cam=R, skip=skip, stats=stats, alpha=alpha)


def empty_result(tracks) -> OrientationResult:
    """What to write when orientation is disabled or unavailable.

    Every row NaN and every skip reason honest, rather than an absent group —
    a consumer that reads the file should not have to branch on whether the
    orientation stage ran.
    """
    F = tracks.n_frames
    slots = tracks.slot_ids
    return OrientationResult(
        azimuth_deg={s: np.full(F, np.nan, dtype=np.float32) for s in slots},
        elevation_deg={s: np.full(F, np.nan, dtype=np.float32) for s in slots},
        roll_deg={s: np.full(F, np.nan, dtype=np.float32) for s in slots},
        R_cam={s: np.full((F, 3, 3), np.nan, dtype=np.float32) for s in slots},
        skip={s: np.full(F, SKIP_NO_MASK, dtype=np.uint8) for s in slots},
        stats={"crops": 0, "disabled": True},
        alpha={s: np.full(F, -1, dtype=np.int8) for s in slots},
    )
