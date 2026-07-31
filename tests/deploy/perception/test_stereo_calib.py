"""calibrate_from_image_pairs on FULLY SYNTHETIC checkerboard image pairs,
rendered with a known camera K/R/T via cv2.projectPoints + a perspective
warp (exact for a planar target, so the only error source is corner
localization/solver noise, same as a real calibration).

Rendering approach: a checkerboard PATTERN is drawn once as a flat template
image; for each of several board poses, the template's 4 outer corners are
projected into each eye with the ground-truth camera model, and
cv2.getPerspectiveTransform + warpPerspective produces the camera view --
this is exact because a homography is the true image-to-image map for a
planar surface under pinhole projection (no 3D mesh/renderer needed).
"""

import cv2
import numpy as np
import pytest

from ego2g1.deploy.perception.stereo_calib import calibrate_from_image_pairs


COLS, ROWS = 7, 6          # inner corners (must be non-square, see stereo_calib.py)
SQUARE_SIZE_M = 0.025
SQ_PX = 50                 # template pixels per square
MARGIN_SQ = 2              # quiet-zone margin, in squares

FX_TRUE = FY_TRUE = 500.0
IMG_W, IMG_H = 640, 480
CX_TRUE, CY_TRUE = IMG_W / 2.0, IMG_H / 2.0
K_TRUE = np.array([[FX_TRUE, 0, CX_TRUE], [0, FY_TRUE, CY_TRUE], [0, 0, 1]], dtype=np.float64)

R_STEREO_TRUE = np.eye(3)
T_STEREO_TRUE = np.array([0.06, 0.0, 0.0])  # 60 mm baseline, left->right


def _make_board_template() -> np.ndarray:
    """(rows+1)x(cols+1)-square checkerboard pattern, quiet-zone margin,
    flat grayscale image."""
    sx, sy = COLS + 1, ROWS + 1
    w = (sx + 2 * MARGIN_SQ) * SQ_PX
    h = (sy + 2 * MARGIN_SQ) * SQ_PX
    img = np.full((h, w), 255, dtype=np.uint8)
    for j in range(sy):
        for i in range(sx):
            if (i + j) % 2 == 0:
                r0, c0 = (MARGIN_SQ + j) * SQ_PX, (MARGIN_SQ + i) * SQ_PX
                img[r0:r0 + SQ_PX, c0:c0 + SQ_PX] = 0
    return img


def _template_px_to_board_squares(uv: np.ndarray) -> np.ndarray:
    """Inverse of the "inner corner (c, r) -> template pixel" map used to
    build the template (corner (0, 0) sits at pixel
    ((MARGIN_SQ+1)*SQ_PX, (MARGIN_SQ+1)*SQ_PX))."""
    x = uv[:, 0] / SQ_PX - MARGIN_SQ - 1
    y = uv[:, 1] / SQ_PX - MARGIN_SQ - 1
    return np.stack([x, y], axis=-1)


def _render_view(template: np.ndarray, R_board_cam: np.ndarray, t_board_cam: np.ndarray,
                  image_size: tuple[int, int]) -> np.ndarray:
    """Warp `template` into a camera view given the board's pose in that
    camera's frame (`R_board_cam`, `t_board_cam`) and the TRUE camera K
    (zero distortion) -- exact for a planar target."""
    h_t, w_t = template.shape
    template_corners = np.array(
        [[0, 0], [w_t - 1, 0], [0, h_t - 1], [w_t - 1, h_t - 1]], dtype=np.float32
    )
    board_squares = _template_px_to_board_squares(template_corners)
    board_m = np.zeros((4, 3), dtype=np.float64)
    board_m[:, :2] = board_squares * SQUARE_SIZE_M

    rvec, _ = cv2.Rodrigues(R_board_cam)
    proj, _ = cv2.projectPoints(board_m, rvec, t_board_cam, K_TRUE, np.zeros(5))
    proj = proj.reshape(-1, 2).astype(np.float32)

    homography = cv2.getPerspectiveTransform(template_corners, proj)
    return cv2.warpPerspective(template, homography, image_size, borderValue=255)


def _synthetic_pairs(n_poses=10, seed=42):
    rng = np.random.default_rng(seed)
    template = _make_board_template()
    pairs = []
    for _ in range(n_poses):
        rvec = rng.uniform(-0.25, 0.25, size=3)
        rvec[2] += rng.uniform(-0.3, 0.3)
        tvec = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.06, 0.06), rng.uniform(0.5, 0.9)])
        R_bl, _ = cv2.Rodrigues(rvec)
        t_bl = tvec

        left = _render_view(template, R_bl, t_bl, (IMG_W, IMG_H))
        R_br = R_STEREO_TRUE @ R_bl
        t_br = R_STEREO_TRUE @ t_bl + T_STEREO_TRUE
        right = _render_view(template, R_br, t_br, (IMG_W, IMG_H))

        pairs.append((np.stack([left] * 3, axis=-1), np.stack([right] * 3, axis=-1)))
    return pairs


def test_recovers_known_camera_model():
    pairs = _synthetic_pairs()
    calib = calibrate_from_image_pairs(pairs, (COLS, ROWS), SQUARE_SIZE_M)

    assert calib.image_size == (IMG_W, IMG_H)

    # intrinsics: focal length within 5%, principal point within 6 px
    for K in (calib.K_left, calib.K_right):
        assert K[0, 0] == pytest.approx(FX_TRUE, rel=0.05)
        assert K[1, 1] == pytest.approx(FY_TRUE, rel=0.05)
        assert K[0, 2] == pytest.approx(CX_TRUE, abs=6.0)
        assert K[1, 2] == pytest.approx(CY_TRUE, abs=6.0)

    # extrinsics: near-identity rotation (angle from true R under 2 degrees)
    R_err = calib.R @ R_STEREO_TRUE.T
    angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    assert np.degrees(angle_err) < 2.0

    # translation: baseline (mostly along X) within 15% and each component
    # within 8 mm of ground truth -- Z (along the optical axis) is the
    # least-constrained direction in planar stereo calibration, so it gets
    # the same absolute (not relative) bound as X/Y.
    np.testing.assert_allclose(calib.T, T_STEREO_TRUE, atol=0.008)
    assert calib.baseline_m() == pytest.approx(np.linalg.norm(T_STEREO_TRUE), rel=0.15)


def test_rejects_mismatched_left_right_size_within_a_pair():
    left = np.zeros((100, 100, 3), dtype=np.uint8)
    right = np.zeros((100, 120, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="differ"):
        calibrate_from_image_pairs([(left, right)], (COLS, ROWS), SQUARE_SIZE_M)


def test_rejects_no_checkerboard_found():
    blank_left = np.full((IMG_H, IMG_W, 3), 200, dtype=np.uint8)
    blank_right = np.full((IMG_H, IMG_W, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError, match="not found"):
        calibrate_from_image_pairs([(blank_left, blank_right)], (COLS, ROWS), SQUARE_SIZE_M)


def test_rejects_square_board_size():
    pairs = _synthetic_pairs(n_poses=1)
    with pytest.raises(ValueError, match="cols must differ"):
        calibrate_from_image_pairs(pairs, (7, 7), SQUARE_SIZE_M)


def test_rejects_empty_pairs():
    with pytest.raises(ValueError, match="at least one"):
        calibrate_from_image_pairs([], (COLS, ROWS), SQUARE_SIZE_M)


def test_skips_pairs_missing_a_board_but_still_calibrates():
    pairs = _synthetic_pairs(n_poses=6)
    blank = np.full((IMG_H, IMG_W, 3), 200, dtype=np.uint8)
    pairs_with_a_miss = [pairs[0], (blank, blank), *pairs[1:]]

    calib = calibrate_from_image_pairs(pairs_with_a_miss, (COLS, ROWS), SQUARE_SIZE_M)
    assert calib.K_left[0, 0] == pytest.approx(FX_TRUE, rel=0.1)
