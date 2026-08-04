"""handeye_calib's AX=XB solve + consistency check, on FULLY SYNTHETIC data
(no camera, no robot): construct a ground-truth `T_pelvis_camera` and a
ground-truth grip offset `T_flange_marker`, generate several random flange
poses with real rotational diversity, and derive each sample's
`T_camera_marker` DIRECTLY from equation (*) in the module docstring --
`T_camera_marker_i = T_pelvis_camera^-1 @ T_base_flange_i @ T_flange_marker`
-- so a correct `solve_eye_to_hand` MUST recover the ground-truth
`T_pelvis_camera` to float64 precision, independent of ever knowing
`T_flange_marker` (which is deliberately never passed to the solver).

`detect_tag_pose` is tested separately below, on a synthetically rendered
marker image (homography warp of `cv2.aruco.generateImageMarker`, same
"exact for a planar target" technique `test_stereo_calib_charuco.py` uses).
"""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ego2g1.core.se3 import se3_inv
from ego2g1.deploy.perception.handeye_calib import (
    CONSISTENCY_WARN_ROTATION_DEG,
    CONSISTENCY_WARN_TRANSLATION_M,
    HandEyeSample,
    detect_tag_pose,
    mount_consistency_report,
    rotation_spread_deg,
    solve_eye_to_hand,
)


def _random_se3(rng, *, translation_scale=0.5) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    T[:3, 3] = rng.uniform(-translation_scale, translation_scale, size=3)
    return T


def _synthetic_samples(n=14, seed=3, T_pelvis_camera=None, T_flange_marker=None):
    rng = np.random.default_rng(seed)
    if T_pelvis_camera is None:
        T_pelvis_camera = _random_se3(rng)
    if T_flange_marker is None:
        T_flange_marker = _random_se3(rng, translation_scale=0.05)

    T_camera_pelvis = se3_inv(T_pelvis_camera)
    samples = []
    for _ in range(n):
        T_base_flange = _random_se3(rng)
        T_camera_marker = T_camera_pelvis @ T_base_flange @ T_flange_marker
        samples.append(HandEyeSample(T_base_flange=T_base_flange, T_camera_marker=T_camera_marker))
    return samples, T_pelvis_camera, T_flange_marker


class TestSolveEyeToHand:
    def test_recovers_ground_truth_without_knowing_the_grip_offset(self):
        samples, T_true, _T_flange_marker = _synthetic_samples()

        T_solved, report = solve_eye_to_hand(samples)

        np.testing.assert_allclose(T_solved[:3, :3], T_true[:3, :3], atol=1e-6)
        np.testing.assert_allclose(T_solved[:3, 3], T_true[:3, 3], atol=1e-6)
        assert report["n_samples"] == len(samples)

    def test_consistency_report_is_near_zero_for_the_correct_solve(self):
        samples, T_true, _ = _synthetic_samples()
        T_solved, report = solve_eye_to_hand(samples)

        assert report["translation_std_m"] < 1e-5
        assert report["rotation_spread_deg_estimate"] < 1e-3

    def test_consistency_report_is_large_for_a_perturbed_extrinsic(self):
        """A wrong T_pelvis_camera makes each sample "explain" a DIFFERENT
        apparent grip offset -- the whole point of this residual check."""
        samples, T_true, _ = _synthetic_samples()

        perturbed = T_true.copy()
        perturbed[:3, :3] = T_true[:3, :3] @ Rotation.from_euler("x", 5, degrees=True).as_matrix()

        report = mount_consistency_report(samples, perturbed)
        assert (
            report["translation_std_m"] > CONSISTENCY_WARN_TRANSLATION_M
            or report["rotation_spread_deg_estimate"] > CONSISTENCY_WARN_ROTATION_DEG
        )

    def test_raises_below_three_samples(self):
        samples, _, _ = _synthetic_samples(n=2)
        with pytest.raises(ValueError, match=">= 3"):
            solve_eye_to_hand(samples)

    def test_warns_on_low_rotation_spread(self, capsys):
        """All flange poses at nearly the SAME orientation (only translation
        varies) is the textbook ill-conditioned case for AX=XB -- rotation
        alone disambiguates X's rotation part, so near-identical rotations
        must trigger the warning regardless of sample count."""
        rng = np.random.default_rng(5)
        T_true = _random_se3(rng)
        T_flange_marker = _random_se3(rng, translation_scale=0.05)
        T_camera_pelvis = se3_inv(T_true)
        fixed_R = Rotation.random(random_state=rng).as_matrix()

        samples = []
        for _ in range(12):
            T_base_flange = np.eye(4)
            T_base_flange[:3, :3] = fixed_R
            T_base_flange[:3, 3] = rng.uniform(-0.3, 0.3, size=3)
            T_camera_marker = T_camera_pelvis @ T_base_flange @ T_flange_marker
            samples.append(HandEyeSample(T_base_flange=T_base_flange, T_camera_marker=T_camera_marker))

        solve_eye_to_hand(samples)
        assert "rotation spread" in capsys.readouterr().out.lower()

    def test_rotation_spread_deg_matches_a_known_pair(self):
        rng = np.random.default_rng(1)
        T0 = _random_se3(rng)
        T1 = T0.copy()
        T1[:3, :3] = T0[:3, :3] @ Rotation.from_euler("z", 30, degrees=True).as_matrix()
        samples = [
            HandEyeSample(T_base_flange=T0, T_camera_marker=np.eye(4)),
            HandEyeSample(T_base_flange=T1, T_camera_marker=np.eye(4)),
        ]
        assert rotation_spread_deg(samples) == pytest.approx(30.0, abs=1e-6)


# --------------------------------------------------------------------------
# detect_tag_pose: synthetic rendering (planar homography warp, exact),
# same technique as test_stereo_calib_charuco.py's board rendering.
# --------------------------------------------------------------------------

TAG_SIZE_M = 0.05
FX = FY = 700.0
IMG_W, IMG_H = 640, 480
CX, CY = IMG_W / 2.0, IMG_H / 2.0
K_TRUE = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
DIST_TRUE = np.zeros(5)
ARUCO_DICT_NAME = "DICT_4X4_50"
TAG_ID = 7


def _flat_template_and_reference():
    """Flat marker template (WITH a white quiet-zone margin -- a bare
    `generateImageMarker` output has none, and `ArucoDetector` cannot find a
    marker with no light-colored margin around its black border at all, a
    real thing confirmed by hand: detection returned zero markers against
    the un-padded template) + the EMPIRICALLY detected flat-pixel corners,
    same "don't assume the library's own convention, measure it" approach
    `test_stereo_calib_charuco.py` uses for the ChArUco board."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
    margin = 40
    raw = cv2.aruco.generateImageMarker(dictionary, TAG_ID, 200)
    template = cv2.copyMakeBorder(raw, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)

    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(template)
    if ids is None or int(ids[0][0]) != TAG_ID:
        raise RuntimeError("flat template itself wasn't detectable -- test setup is broken")
    flat_px = corners[0].reshape(4, 2).astype(np.float64)
    return template, flat_px


# `detect_tag_pose`'s object points are OpenCV's own documented SOLVEPNP_
# IPPE_SQUARE/ArUco pairing ((-s/2,s/2,0) <-> corner 0, etc). Confirmed BY
# HAND (not assumed) that rendering with R=identity against THAT convention
# does NOT reproduce a real, straight-on marker view -- it comes out
# top/bottom-flipped and fails to decode at all (OpenCV's image-Y-DOWN pixel
# convention vs. the object frame's own +Y-is-"up" labeling disagree for the
# canonical/flat pose). Tried the 4 obvious candidate fixes (identity,
# 180 deg about X, Y, Z) against a real render-then-detect roundtrip;
# 180-about-X (and, equivalently, about Y) is what actually decodes. This
# constant is that empirically-verified correction, applied ONCE here so
# every test's "R_true" can mean "the pose to test", not "the pose relative
# to some flip only this rendering path needs to know about".
_CANONICAL_BASE_R, _ = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))


def _render_tag_view(R_marker_cam, t_marker_cam) -> np.ndarray:
    """Render the tag at pose `R_marker_cam @ _CANONICAL_BASE_R`, `t_marker_cam`
    IN THE CAMERA FRAME, as a full IMG_W x IMG_H grayscale-then-RGB frame --
    exact for a planar target under pinhole projection (a homography warp of
    the flat marker template). Callers compare `detect_tag_pose`'s result
    against this SAME composed rotation, not bare `R_marker_cam` (see
    `_CANONICAL_BASE_R`'s comment for why)."""
    template, flat_px = _flat_template_and_reference()

    half = TAG_SIZE_M / 2.0
    # Object points in the SAME corner order ArucoDetector/detect_tag_pose
    # use (top-left, top-right, bottom-right, bottom-left), Z=0.
    obj_xyz = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float64,
    )

    R_render = np.asarray(R_marker_cam) @ _CANONICAL_BASE_R
    rvec, _ = cv2.Rodrigues(R_render)
    proj, _ = cv2.projectPoints(obj_xyz, rvec, t_marker_cam, K_TRUE, DIST_TRUE)
    proj = proj.reshape(-1, 2).astype(np.float64)

    homography, _mask = cv2.findHomography(flat_px, proj, method=0)
    warped = cv2.warpPerspective(
        template, homography, (IMG_W, IMG_H), borderValue=255, flags=cv2.INTER_LINEAR
    )
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)


class TestDetectTagPose:
    def test_recovers_a_known_pose(self):
        R_true = Rotation.from_euler("xyz", [15, -10, 5], degrees=True).as_matrix()
        t_true = np.array([0.03, -0.02, 0.4])
        image = _render_tag_view(R_true, t_true)

        result = detect_tag_pose(image, TAG_SIZE_M, K_TRUE, DIST_TRUE, dictionary_name=ARUCO_DICT_NAME)

        assert result is not None
        T_camera_marker, marker_id = result
        assert marker_id == TAG_ID
        # atol here is pixel-level (homography warp + real corner-detection
        # interpolation noise on a ~90px-wide rendered marker), not symbolic
        # exactness -- 0.02 on a rotation-matrix element is sub-degree,
        # 5mm translation at 0.4m range is sub-1.5%, both consistent with a
        # real (if small) raster round-trip rather than an exact projection.
        np.testing.assert_allclose(T_camera_marker[:3, :3], R_true @ _CANONICAL_BASE_R, atol=0.02)
        np.testing.assert_allclose(T_camera_marker[:3, 3], t_true, atol=0.005)

    def test_auto_detects_dictionary(self):
        R_true = np.eye(3)
        t_true = np.array([0.0, 0.0, 0.35])
        image = _render_tag_view(R_true, t_true)

        result = detect_tag_pose(image, TAG_SIZE_M, K_TRUE, DIST_TRUE, dictionary_name=None)

        assert result is not None
        np.testing.assert_allclose(result[0][:3, 3], t_true, atol=1e-3)

    def test_returns_none_when_no_marker_present(self):
        blank = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
        assert detect_tag_pose(blank, TAG_SIZE_M, K_TRUE, DIST_TRUE, dictionary_name=ARUCO_DICT_NAME) is None
