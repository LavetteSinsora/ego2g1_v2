"""The pure half of the orientation stage: crop geometry and angle decoding.

⚠ These tests pin PROPERTIES (orthonormality, handedness, composition,
continuity), never absolute correctness against Orient Anything V2's canonical
frame — because that frame has not been measured, and a test written from the
same assumption as the code would only pin the assumption. The convention is a
calibration to be validated on hardware; see docs/perception_v2_notes.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from ego2g1.deploy.perception.v2.orientation_v2 import (
    CAMERA_OPENCV, TRAINING_ANGLES_TO_MATRIX, OrientationConvention,
    angles_to_matrix, compose_relational_rotation, crop_from_mask,
)


# --- angles -> rotation -----------------------------------------------------

def test_zero_angles_are_the_identity():
    np.testing.assert_allclose(angles_to_matrix(0.0, 0.0, 0.0), np.eye(3),
                               atol=1e-12)


def test_output_is_always_a_proper_rotation():
    """A non-orthonormal or left-handed "rotation" would flow straight into
    the state vector as a valid-looking pose. Nothing downstream checks."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        az, el, ro = rng.uniform([0, -90, -180], [360, 90, 180])
        R = angles_to_matrix(az, el, ro)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_vectorised_and_scalar_agree():
    az = np.array([10.0, 200.0, 350.0])
    el = np.array([-30.0, 0.0, 45.0])
    ro = np.array([0.0, -120.0, 90.0])
    batch = angles_to_matrix(az, el, ro)
    assert batch.shape == (3, 3, 3)
    for i in range(3):
        np.testing.assert_allclose(batch[i],
                                   angles_to_matrix(az[i], el[i], ro[i]))


def test_the_composition_order_is_roll_elevation_azimuth():
    """Azimuth innermost (the object's own turn), elevation next (tilting that
    turned object toward the camera), roll outermost (an in-image rotation
    applied last). Order is not a detail — the three do not commute."""
    a, e, r = 30.0, 20.0, 40.0
    R_a = angles_to_matrix(a, 0, 0)
    R_e = angles_to_matrix(0, e, 0)
    R_r = angles_to_matrix(0, 0, r)
    np.testing.assert_allclose(angles_to_matrix(a, e, r), R_r @ R_e @ R_a,
                               atol=1e-12)
    assert not np.allclose(R_r @ R_e @ R_a, R_a @ R_e @ R_r)


def test_each_angle_turns_about_its_configured_camera_axis():
    axis_of = {0: np.array([1.0, 0, 0]), 1: np.array([0, 1.0, 0]),
               2: np.array([0, 0, 1.0])}
    c = CAMERA_OPENCV
    for angles, axis in (((37.0, 0, 0), c.azimuth_axis),
                         ((0, 37.0, 0), c.elevation_axis),
                         ((0, 0, 37.0), c.roll_axis)):
        R = angles_to_matrix(*angles)
        np.testing.assert_allclose(R @ axis_of[axis], axis_of[axis], atol=1e-12)


def test_flipping_a_sign_is_a_config_change_not_a_code_change():
    """The whole reason `OrientationConvention` exists: fixing the frame on
    hardware must not require editing the decode. A wrong sign mirrors the
    object about a plane and still looks entirely plausible downstream."""
    flipped = OrientationConvention(azimuth_sign=-1.0)
    # Flipping the sign is exactly negating the angle, and nothing else.
    np.testing.assert_allclose(angles_to_matrix(45.0, 0, 0, convention=flipped),
                               angles_to_matrix(-45.0, 0, 0), atol=1e-12)
    # ...and it is a real difference, not a no-op.
    assert not np.allclose(angles_to_matrix(45.0, 0, 0, convention=flipped),
                           angles_to_matrix(45.0, 0, 0))


def test_the_decode_is_continuous_across_the_azimuth_wrap():
    """Bin 359 and bin 0 are one degree apart physically; a decode that jumped
    there would show up as the object spinning once per revolution."""
    a = angles_to_matrix(359.0, 0, 0)
    b = angles_to_matrix(0.0, 0, 0)
    assert np.degrees(np.arccos((np.trace(a.T @ b) - 1) / 2)) < 1.5


def test_mismatched_angle_shapes_are_rejected():
    with pytest.raises(ValueError, match="shapes disagree"):
        angles_to_matrix(np.zeros(3), np.zeros(2), np.zeros(3))


# --- crops ------------------------------------------------------------------

def _rgb(h=100, w=120):
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


def test_crop_is_square_so_the_later_resize_cannot_shear_orientation():
    """Preprocessing resizes the longest side and pads the rest, so a
    non-square crop arrives at a different aspect ratio than training used —
    which shears apparent orientation, the one thing this stage measures."""
    mask = np.zeros((100, 120), dtype=bool)
    mask[30:70, 40:50] = True                  # tall and thin
    crop = crop_from_mask(_rgb(), mask, pad=0.0)
    assert crop.size[0] == crop.size[1]


def test_crop_adds_context_around_the_silhouette():
    mask = np.zeros((100, 120), dtype=bool)
    mask[40:60, 40:60] = True                  # 20x20
    tight = crop_from_mask(_rgb(), mask, pad=0.0)
    padded = crop_from_mask(_rgb(), mask, pad=0.5)
    assert padded.size[0] > tight.size[0]


def test_crop_is_clipped_to_the_frame():
    mask = np.zeros((100, 120), dtype=bool)
    mask[0:10, 0:10] = True                    # hard against the corner
    crop = crop_from_mask(_rgb(), mask, pad=1.0)
    assert crop.size[0] <= 120 and crop.size[1] <= 100


def test_an_empty_or_sliver_mask_yields_no_crop():
    assert crop_from_mask(_rgb(), np.zeros((100, 120), bool)) is None
    sliver = np.zeros((100, 120), dtype=bool)
    sliver[50, 50:53] = True
    assert crop_from_mask(_rgb(), sliver, pad=0.0) is None


# --- matching the training pipeline -----------------------------------------

def test_the_decode_matches_the_training_pipeline_exactly():
    """THE test that matters for orientation.

    Training labels came from the same model through
    `data_extraction_zh/.../OrientAnything.py::angles_to_rot_matrix`. The
    absolute frame is irrelevant; agreeing with THAT function is everything.
    A mismatch is a fixed remapping of every rotation the policy sees — still
    a valid rotation, so nothing downstream detects it.

    This caught a real bug: the first draft used Ry(-az).
    """
    rng = np.random.default_rng(1)
    for _ in range(500):
        az, el, ro = rng.uniform([0, -90, -180], [360, 90, 180])
        np.testing.assert_allclose(angles_to_matrix(az, el, ro),
                                   TRAINING_ANGLES_TO_MATRIX(az, el, ro),
                                   atol=1e-12)


def test_the_default_convention_is_the_training_convention():
    assert CAMERA_OPENCV.azimuth_sign == 1.0
    np.testing.assert_allclose(angles_to_matrix(90.0, 0, 0),
                               TRAINING_ANGLES_TO_MATRIX(90.0, 0, 0),
                               atol=1e-12)


# --- anchor / context -------------------------------------------------------

def test_a_context_rotation_points_its_x_axis_at_the_anchor():
    """Training constructs non-anchor rotations rather than trusting the
    model's (OrientAnything.py, is_anchor=False). Feeding the raw model
    rotation instead would be structurally different from training for every
    slot but one."""
    R_model = angles_to_matrix(37.0, 12.0, -5.0)
    t = np.array([0.0, 0.0, 1.0])
    anchor = np.array([0.3, 0.0, 1.0])
    R = compose_relational_rotation(R_model, t, anchor)

    np.testing.assert_allclose(R[:, 1], R_model[:, 1], atol=1e-12)  # y kept
    to_anchor = anchor - t
    assert np.dot(R[:, 0], to_anchor) > 0                            # x toward
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_a_context_x_axis_is_perpendicular_to_the_kept_y_axis():
    R_model = angles_to_matrix(200.0, -40.0, 90.0)
    R = compose_relational_rotation(R_model, np.array([0.1, 0.2, 0.9]),
                                    np.array([-0.2, 0.5, 1.2]))
    assert abs(float(np.dot(R[:, 0], R[:, 1]))) < 1e-9


def test_a_stacked_object_falls_back_to_the_models_x_axis():
    """Object directly above the anchor: the projection vanishes and there is
    no relational direction to be had. Training calls this
    `vlm_context_stacked_fallback`."""
    R_model = angles_to_matrix(10.0, 0.0, 0.0)     # y-axis is camera Y
    t = np.array([0.0, 0.0, 1.0])
    anchor = t + R_model[:, 1] * 0.2               # displaced purely along y
    R = compose_relational_rotation(R_model, t, anchor)
    np.testing.assert_allclose(R[:, 0], R_model[:, 0], atol=1e-12)


# --- background -------------------------------------------------------------

def test_the_background_is_removed_to_match_training():
    """Training runs rembg (`do_rm_bkg=True`), so the checkpoint's labels came
    from a model that saw a matted foreground. A raw crop at deploy is a
    domain shift on the model's input."""
    rgb = np.full((40, 40, 3), 7, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    crop = np.asarray(crop_from_mask(rgb, mask, pad=0.5, background="white"))
    assert (crop == 255).any(), "outside the mask must be filled"
    assert (crop == 7).any(), "inside the mask must survive"
    assert set(np.unique(crop)) == {7, 255}


def test_background_none_leaves_the_raw_crop():
    rgb = np.full((40, 40, 3), 7, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    crop = np.asarray(crop_from_mask(rgb, mask, pad=0.5, background="none"))
    assert set(np.unique(crop)) == {7}


def test_an_unknown_background_is_rejected():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    with pytest.raises(ValueError, match="background must be"):
        crop_from_mask(np.zeros((40, 40, 3), np.uint8), mask, background="puce")
