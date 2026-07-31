"""StereoSGBMDepthSource on a synthetic stereo pair with a KNOWN pixel
disparity (two fronto-parallel textured planes at two different depths).
Stereo matching needs texture, so both planes are random noise, not blank.

Camera model: parallel, undistorted, unit-baseline-along-x rig
(R = I, T = [-baseline, 0, 0], the standard "right camera baseline metres
to the +x side of the left camera" convention -- see StereoCalibration's
docstring). With R = I and zero distortion, cv2.stereoRectify's rectified
frame is (up to floating point) the input frame itself, so the synthetic
images can be built directly in pixel space without needing to also model
rectification.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ego2g1.deploy.perception.depth import StereoCalibration, StereoSGBMDepthSource


W, H = 320, 240
FX = FY = 300.0
CX, CY = W / 2.0, H / 2.0
BASELINE_M = 0.06  # 60 mm, matches the datasheet nominal (§6.1)


def _calib() -> StereoCalibration:
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    return StereoCalibration(
        K_left=K.copy(),
        K_right=K.copy(),
        dist_left=np.zeros(5),
        dist_right=np.zeros(5),
        R=np.eye(3),
        T=np.array([-BASELINE_M, 0.0, 0.0]),
        image_size=(W, H),
    )


def _disparity_for_depth(z_m: float) -> float:
    return FX * BASELINE_M / z_m


def _make_stereo_pair(rng):
    """Background plane at Z_FAR everywhere, a smaller textured patch at
    Z_NEAR composited on top -- both integer-pixel disparities so the
    ground truth has no sub-pixel ambiguity."""
    z_far, z_near = 2.0, 1.0
    d_far = _disparity_for_depth(z_far)
    d_near = _disparity_for_depth(z_near)
    assert d_far == int(d_far) and d_near == int(d_near)
    d_far, d_near = int(d_far), int(d_near)

    # background: one wide noise texture, sliced twice (0 shift disparity
    # for "left", d_far shift for "right") so right[y, x] == left[y, x+d_far]
    # holds exactly for every interior pixel.
    wide_bg = rng.integers(0, 256, size=(H, W + d_far), dtype=np.uint8)
    left_bg = wide_bg[:, :W]
    right_bg = wide_bg[:, d_far:d_far + W]

    # near patch: fully interior, far enough from the left edge that its
    # right-image placement (shifted left by d_near) stays in-bounds.
    ph, pw = 80, 80
    r0, c0 = 80, 160
    assert c0 - d_near >= 0
    patch = rng.integers(0, 256, size=(ph, pw), dtype=np.uint8)

    left = left_bg.copy()
    left[r0:r0 + ph, c0:c0 + pw] = patch
    right = right_bg.copy()
    right[r0:r0 + ph, c0 - d_near:c0 - d_near + pw] = patch

    left_rgb = np.stack([left] * 3, axis=-1)
    right_rgb = np.stack([right] * 3, axis=-1)
    return left_rgb, right_rgb, z_far, z_near, (r0, c0, ph, pw)


def test_recovers_known_depth_on_textured_planes():
    rng = np.random.default_rng(0)
    left, right, z_far, z_near, (r0, c0, ph, pw) = _make_stereo_pair(rng)

    # A small numDisparities (our synthetic disparities are 9/18 px) so the
    # SGBM-mandated invalid band at the left edge (it needs numDisparities
    # columns of search context, so columns [0, numDisparities) are always
    # invalid) stays narrow -- the default 128 would swallow both the
    # background-check region and part of the patch.
    source = StereoSGBMDepthSource(_calib(), num_disparities=32)
    depth = source.estimate(left, right)

    assert depth.shape == (H, W)
    assert depth.dtype == np.float32

    # background region: away from the patch, the left-edge invalid band
    # (columns < numDisparities=32), and image borders.
    bg_region = depth[20:60, 50:150]
    assert np.mean(bg_region > 0) > 0.8, "expected most background pixels to match"
    valid_bg = bg_region[bg_region > 0]
    assert np.median(valid_bg) == pytest.approx(z_far, rel=0.1)

    # patch region: central sub-window, away from the block-matcher's own
    # edge-blur zone.
    margin = 15
    patch_region = depth[r0 + margin:r0 + ph - margin, c0 + margin:c0 + pw - margin]
    assert np.mean(patch_region > 0) > 0.8, "expected most patch pixels to match"
    valid_patch = patch_region[patch_region > 0]
    assert np.median(valid_patch) == pytest.approx(z_near, rel=0.1)


def test_invalid_pixels_are_zero_not_nan():
    rng = np.random.default_rng(1)
    # a blank (textureless) pair: no matches possible anywhere.
    left = np.full((H, W, 3), 128, dtype=np.uint8)
    right = np.full((H, W, 3), 128, dtype=np.uint8)

    source = StereoSGBMDepthSource(_calib())
    depth = source.estimate(left, right)

    assert not np.isnan(depth).any(), "0.0 is the documented invalid sentinel, not NaN"
    assert np.all(depth == 0.0)


def test_estimate_rejects_wrong_shape():
    source = StereoSGBMDepthSource(_calib())
    with pytest.raises(ValueError):
        source.estimate(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10, 3), np.uint8))


def test_stereo_calibration_roundtrip(tmp_path):
    calib = _calib()
    path = tmp_path / "calib.npz"
    calib.save(path)
    loaded = StereoCalibration.load(path)

    np.testing.assert_allclose(loaded.K_left, calib.K_left)
    np.testing.assert_allclose(loaded.K_right, calib.K_right)
    np.testing.assert_allclose(loaded.R, calib.R)
    np.testing.assert_allclose(loaded.T, calib.T)
    assert loaded.image_size == calib.image_size
    assert loaded.baseline_m() == pytest.approx(BASELINE_M)


def test_num_disparities_must_be_multiple_of_16():
    with pytest.raises(ValueError):
        StereoSGBMDepthSource(_calib(), num_disparities=100)


def test_block_size_must_be_odd():
    with pytest.raises(ValueError):
        StereoSGBMDepthSource(_calib(), block_size=4)
