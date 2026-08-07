"""The CPU-testable half of the SAM 3 stage: prompts, visibility, join.

`Sam3Source` itself needs weights and a GPU. What is exercised here is
everything AROUND the model call — which is where the silent failures live:
a prompt that never matches, a gate that stops discriminating, a depth sample
taken from the gripper.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from ego2g1.deploy.perception.v2.sam3_source import (
    SlotObservation, VisibilityConfig, VisibilityGate, build_prompt_map,
    join_to_camera, normalize_prompt,
)

from .conftest import Spec

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def square(side, *, shape=(24, 24), origin=(2, 2)):
    """A `side`x`side` True block. Sized above the 64-px default floors, so a
    test that means to exercise a THRESHOLD is not silently rejected by the
    minimum-area guard first."""
    mask = np.zeros(shape, dtype=bool)
    r, c = origin
    mask[r:r + side, c:c + side] = True
    return mask


def obs(instance_id="obj0", *, mask=None, det=0.9, tracker=0.9, area=None,
        occluded=False):
    if mask is None:
        mask = square(12)                                # 144 px
    return SlotObservation(
        instance_id=instance_id, mask=mask, box_xyxy=None, det_score=det,
        tracker_score=tracker,
        mask_area_px=int(mask.sum()) if area is None else area,
        occluded=occluded)


# --- prompts ----------------------------------------------------------------

def test_grounding_dino_separator_is_stripped(caplog):
    """A live suspect for the plan's §2.4 defect: the v1 example config
    documents prompts ending in ' .', which is GroundingDINO's phrase
    separator and pure noise to SAM 3's text encoder."""
    with caplog.at_level(logging.WARNING):
        assert normalize_prompt("a red cube .") == "a red cube"
    assert "GroundingDINO" in caplog.text


def test_trailing_period_is_stripped():
    assert normalize_prompt("a red cube.") == "a red cube"
    assert normalize_prompt("  a red cube  ") == "a red cube"


def test_an_empty_prompt_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        normalize_prompt(" . ")


def test_duplicate_prompts_are_refused():
    """SAM 3 keys detections by prompt, so two slots on one prompt makes the
    assignment ambiguous — and guessing wrong feeds the model one object's
    geometry under another's slot."""
    with pytest.raises(ValueError, match="share the SAM 3 prompt"):
        build_prompt_map([Spec("a", "a red cube"), Spec("b", "a red cube .")])


def test_prompt_map_normalises_both_sides():
    assert build_prompt_map([Spec("a", "a red cube .")]) == {"a red cube": "a"}


# --- visibility (S1) --------------------------------------------------------

def test_a_re_detected_full_mask_is_usable_for_everything():
    v = VisibilityGate().update(obs())
    assert v.mask_usable and v.crop_usable


def test_memory_propagation_keeps_position_but_not_orientation():
    """The governing asymmetry: position survives occlusion and orientation
    does not. A gate that stopped both would freeze the object in place during
    every reach."""
    gate = VisibilityGate()
    gate.update(obs())
    v = gate.update(obs(det=None))
    assert v.mask_usable is True
    assert v.crop_usable is False
    assert "not re-detected" in v.reason


def test_area_collapse_rejects_the_crop_but_not_the_position():
    gate = VisibilityGate(VisibilityConfig(min_area_fraction=0.5))
    gate.update(obs())                                   # 144 px, sets the max
    v = gate.update(obs(mask=square(9)))                 # 81 px = 56%... of 144
    assert v.crop_usable, "56% is above the 50% floor"
    v = gate.update(obs(mask=square(8)))                 # 64 px = 44%
    assert v.mask_usable and not v.crop_usable
    assert "running max" in v.reason


def test_a_tiny_mask_is_unusable_for_position_too():
    v = VisibilityGate().update(obs(mask=square(3)))     # 9 px
    assert not v.mask_usable and not v.crop_usable


def test_low_scores_gate_the_crop():
    cfg = VisibilityConfig(min_det_score=0.8, min_tracker_score=0.8)
    assert not VisibilityGate(cfg).update(obs(det=0.5)).crop_usable
    assert not VisibilityGate(cfg).update(obs(tracker=0.5)).crop_usable


def test_the_area_maximum_decays_so_scale_changes_are_forgiven():
    """Without decay the maximum is a permanent high-water mark: an object
    once close to the camera and since placed further away would be marked
    unusable for the rest of the episode."""
    gate = VisibilityGate(VisibilityConfig(min_area_fraction=0.5,
                                           area_max_decay=0.9))
    gate.update(obs(mask=square(20)))                    # 400 px, close up
    far = square(13)                                     # 169 px, under 50%
    assert not gate.update(obs(mask=far)).crop_usable
    for _ in range(20):                                  # the max decays away
        result = gate.update(obs(mask=far))
    assert result.crop_usable


def test_a_propagated_mask_cannot_raise_the_maximum():
    """A memory-propagated mask can drift larger while the object is hidden;
    letting it set the bar would make every later real detection look like a
    collapse."""
    gate = VisibilityGate(VisibilityConfig(min_area_fraction=0.5,
                                           area_max_decay=1.0))
    small = square(9)                                    # 81 px
    gate.update(obs(mask=small))
    gate.update(obs(mask=square(20), det=None))          # 400 px, propagated
    assert gate.update(obs(mask=small)).crop_usable, (
        "the real detection must not be judged against a propagated mask")


def test_occlusion_flag_gates_everything():
    v = VisibilityGate().update(obs(occluded=True))
    assert not v.mask_usable and not v.crop_usable


# --- join -------------------------------------------------------------------

def _depth(value=0.8, shape=(24, 24)):
    return np.full(shape, value, dtype=np.float64)


def test_back_projection_is_the_pinhole_inverse():
    mask = np.zeros((480, 640), dtype=bool)
    mask[236:246, 316:326] = True                        # centred at (320.5, 240.5)
    out = join_to_camera({"o": obs(mask=mask)}, np.full((480, 640), 2.0), K)
    point, z = out["o"]
    assert z == pytest.approx(2.0)
    np.testing.assert_allclose(point, [(320.5 - 320) * 2 / 600,
                                       (240.5 - 240) * 2 / 600, 2.0])


def test_depth_is_the_median_over_the_mask():
    """Median, not mean: a mask boundary that bleeds onto the background
    produces outliers a mean would swallow."""
    depth = _depth()
    depth[3, 3] = 99.0                                   # one bad pixel
    (_, z) = join_to_camera({"o": obs()}, depth, K)["o"]
    assert z == pytest.approx(0.8)


def test_holes_in_the_depth_map_yield_no_measurement():
    """None means "no measurement", which is the right answer for a
    textureless object and is NOT the same as a bad measurement."""
    depth = _depth()
    depth[:] = 0.0                                       # depth.py's invalid
    assert join_to_camera({"o": obs()}, depth, K)["o"] is None


def test_a_small_mask_yields_no_measurement():
    assert join_to_camera({"o": obs(mask=square(3))}, _depth(), K)["o"] is None


def test_join_is_gated_on_mask_usable_not_crop_usable():
    """Position survives occlusion. Gating depth on the strict crop gate would
    freeze the object in place during every reach, contradicting S1's own
    stated asymmetry."""
    from ego2g1.deploy.perception.v2.sam3_source import Visibility
    args = ({"o": obs()}, _depth(), K)
    assert join_to_camera(*args,
                          visibility={"o": Visibility(True, False)})["o"] is not None
    assert join_to_camera(*args,
                          visibility={"o": Visibility(False, False)})["o"] is None


def test_shape_mismatch_is_survivable():
    assert join_to_camera({"o": obs()}, _depth(shape=(48, 48)), K)["o"] is None
