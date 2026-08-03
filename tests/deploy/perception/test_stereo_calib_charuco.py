"""calibrate_from_charuco_image_pairs + detect_aruco_dictionary, on FULLY
SYNTHETIC ChArUco board image pairs -- same rendering approach as
test_stereo_calib.py (homography warp of a flat template, exact for a
planar target under pinhole projection), except the "flat pixel <-> board
metric position" mapping is derived EMPIRICALLY by running the real
CharucoDetector against the undistorted flat template once, rather than
assumed by hand -- this way the test can't silently disagree with
whatever corner-origin/axis convention cv2.aruco.CharucoBoard actually
uses internally.
"""

import cv2
import numpy as np
import pytest

from ego2g1.deploy.perception.stereo_calib import (
    MIN_RECOMMENDED_VIEWS,
    calibrate_from_charuco_image_pairs,
    detect_aruco_dictionary,
)

# 8x7 squares (42 corners) and a wide tilt range, matching what
# test_stereo_calib.py's plain-checkerboard test uses -- a SMALLER board
# (6x5 squares/20 corners) with a narrow tilt range (+-0.3 rad) is a
# genuinely under-constrained calibration problem even with perfectly exact,
# noise-free correspondences (a real, known effect in planar camera
# calibration, confirmed by hand: fx/fy recovery was off by 7-22% with the
# smaller/narrower setup and converges to ~1% with this one) -- this was a
# test-design bug, not a `calibrate_from_charuco_image_pairs` bug.
SQUARES_X, SQUARES_Y = 8, 7   # squares per side (NOT inner corners -- see module docstring)
SQUARE_SIZE_M = 0.020
MARKER_SIZE_M = 0.015
SQ_PX = 80                    # template pixels per square (generous, for reliable detection)
ARUCO_DICT_NAME = "DICT_5X5_100"

FX_TRUE = FY_TRUE = 600.0
IMG_W, IMG_H = 640, 480
CX_TRUE, CY_TRUE = IMG_W / 2.0, IMG_H / 2.0
K_TRUE = np.array([[FX_TRUE, 0, CX_TRUE], [0, FY_TRUE, CY_TRUE], [0, 0, 1]], dtype=np.float64)

R_STEREO_TRUE = np.eye(3)
T_STEREO_TRUE = np.array([0.06, 0.0, 0.0])


def _make_board():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    return cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M, dictionary)


def _flat_template_and_reference(board):
    """Flat, unwarped board image + the empirically-detected (flat pixel,
    board 3D position) correspondence for every corner the detector finds
    on it -- the ground truth this test's synthetic rendering is built from,
    so it can never silently disagree with the library's own convention."""
    w, h = SQUARES_X * SQ_PX, SQUARES_Y * SQ_PX
    template = board.generateImage((w, h), marginSize=0)

    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _marker_corners, _marker_ids = detector.detectBoard(template)
    if ids is None or len(ids) < 8:
        raise RuntimeError("flat template itself wasn't detectable -- test setup is broken")
    all_corners_3d = board.getChessboardCorners()
    flat_px = corners.reshape(-1, 2).astype(np.float64)
    board_xy = all_corners_3d[ids.reshape(-1)][:, :2].astype(np.float64)  # z is always 0
    return template, flat_px, board_xy


def _render_view(template, flat_px, board_xy, R_board_cam, t_board_cam, image_size):
    """Warp `template` into a camera view given the board's pose in that
    camera's frame -- exact for a planar target. The homography is fit from
    (flat-template pixel) -> (this view's projected pixel) using the SAME
    empirically-grounded correspondence set for every corner, not just 4
    arbitrarily-chosen ones, which is both more robust and avoids picking
    corners near the image edge (least reliably detected on the flat
    reference in the first place)."""
    board_xyz = np.concatenate([board_xy, np.zeros((len(board_xy), 1))], axis=1)
    rvec, _ = cv2.Rodrigues(R_board_cam)
    proj, _ = cv2.projectPoints(board_xyz, rvec, t_board_cam, K_TRUE, np.zeros(5))
    proj = proj.reshape(-1, 2).astype(np.float64)

    homography, _mask = cv2.findHomography(flat_px, proj, method=0)
    return cv2.warpPerspective(
        template, homography, image_size, borderValue=255, flags=cv2.INTER_LINEAR
    )


def _synthetic_pairs(n_poses=16, seed=11):
    """Poses span a WIDE tilt range (up to ~30 deg/axis) -- narrow tilts are
    a classic ill-conditioned case for planar calibration (see module-level
    comment on SQUARES_X/SQUARES_Y). Poses that leave either eye with too
    few detected corners for cv2.calibrateCamera's own minimum (a real
    possibility at the wider end of this tilt range, from steep foreshortening)
    are silently retried with a fresh sample rather than included -- this
    mirrors what a human capturing real images would naturally do (glance at
    a shot, retake it if it's clearly unusable), not a special case for the
    solver."""
    board = _make_board()
    template, flat_px, board_xy = _flat_template_and_reference(board)
    rng = np.random.default_rng(seed)
    pairs = []
    attempts = 0
    while len(pairs) < n_poses and attempts < 20 * n_poses:
        attempts += 1
        rvec = rng.uniform(-0.5, 0.5, size=3)
        rvec[2] += rng.uniform(-0.4, 0.4)
        tvec = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.06, 0.06), rng.uniform(0.35, 0.75)])
        R_bl, _ = cv2.Rodrigues(rvec)
        t_bl = tvec

        left = _render_view(template, flat_px, board_xy, R_bl, t_bl, (IMG_W, IMG_H))
        R_br = R_STEREO_TRUE @ R_bl
        t_br = R_STEREO_TRUE @ t_bl + T_STEREO_TRUE
        right = _render_view(template, flat_px, board_xy, R_br, t_br, (IMG_W, IMG_H))

        detector = cv2.aruco.CharucoDetector(board)
        _, ids_l, _, _ = detector.detectBoard(left)
        _, ids_r, _, _ = detector.detectBoard(right)
        n_l = 0 if ids_l is None else len(ids_l)
        n_r = 0 if ids_r is None else len(ids_r)
        if n_l < 6 or n_r < 6:  # cv2.calibrateCamera's own per-view minimum
            continue

        pairs.append((np.stack([left] * 3, axis=-1), np.stack([right] * 3, axis=-1)))
    if len(pairs) < n_poses:
        raise RuntimeError(f"only got {len(pairs)}/{n_poses} usable synthetic poses in {attempts} attempts")
    return pairs


def test_detects_the_correct_aruco_dictionary():
    pairs = _synthetic_pairs(n_poses=2)
    detected = detect_aruco_dictionary([pairs[0][0], pairs[1][0]])
    assert detected == ARUCO_DICT_NAME


def test_recovers_known_camera_model():
    pairs = _synthetic_pairs()
    calib, report = calibrate_from_charuco_image_pairs(
        pairs, (SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M,
        aruco_dict_name=ARUCO_DICT_NAME,
    )

    assert report.n_found == report.n_given == len(pairs)
    assert calib.image_size == (IMG_W, IMG_H)

    for K in (calib.K_left, calib.K_right):
        assert K[0, 0] == pytest.approx(FX_TRUE, rel=0.05)
        assert K[1, 1] == pytest.approx(FY_TRUE, rel=0.05)
        assert K[0, 2] == pytest.approx(CX_TRUE, abs=15.0)
        assert K[1, 2] == pytest.approx(CY_TRUE, abs=15.0)

    R_err = calib.R @ R_STEREO_TRUE.T
    angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
    assert np.degrees(angle_err) < 2.0

    np.testing.assert_allclose(calib.T, T_STEREO_TRUE, atol=0.008)
    assert calib.baseline_m() == pytest.approx(np.linalg.norm(T_STEREO_TRUE), rel=0.15)


def test_auto_detects_dictionary_when_not_given():
    pairs = _synthetic_pairs(n_poses=6)
    calib, report = calibrate_from_charuco_image_pairs(
        pairs, (SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M,
        aruco_dict_name=None,  # must auto-detect ARUCO_DICT_NAME correctly
    )
    assert report.n_found == len(pairs)
    assert calib.K_left[0, 0] == pytest.approx(FX_TRUE, rel=0.1)


def test_rejects_empty_pairs():
    with pytest.raises(ValueError, match="at least one"):
        calibrate_from_charuco_image_pairs([], (SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M)


def test_pair_with_no_board_is_excluded_and_reported():
    pairs = _synthetic_pairs(n_poses=5)
    blank = np.full((IMG_H, IMG_W, 3), 200, dtype=np.uint8)
    pairs_with_a_miss = [pairs[0], (blank, blank), *pairs[1:]]

    calib, report = calibrate_from_charuco_image_pairs(
        pairs_with_a_miss, (SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M,
        aruco_dict_name=ARUCO_DICT_NAME,
    )
    assert report.n_given == 6
    assert report.n_found == 5
    assert report.failed_indices == (1,)
    assert calib.K_left[0, 0] == pytest.approx(FX_TRUE, rel=0.1)


def test_warns_on_too_few_views(capsys):
    pairs = _synthetic_pairs(n_poses=1)
    calib, report = calibrate_from_charuco_image_pairs(
        pairs, (SQUARES_X, SQUARES_Y), SQUARE_SIZE_M, MARKER_SIZE_M,
        aruco_dict_name=ARUCO_DICT_NAME,
    )
    assert report.n_found < MIN_RECOMMENDED_VIEWS
    out = capsys.readouterr().out
    assert "WARNING" in out
