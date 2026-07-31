"""Depth from the head camera's stereo RGB pair -- there is no other source.

docs/relation_deploy_plan.md §5.2 (researched directly, not assumed): the
G1-D head module is a passive stereo-RGB "HD Binocular Camera" (125deg FOV,
60mm baseline nominal per datasheet); Unitree's own `image_server` never
wires a depth channel to any client regardless of hardware, and a RealSense
would need a hardware swap plus non-trivial server patches. `StereoSGBM` on
the existing `cam_left_high`/`cam_right_high` pair is therefore the ONLY
depth implementation this plan builds, not a placeholder-pending-something-
better -- see the module docstring in the plan for the full research trail.

This module does not read the camera itself (that stays in `camera.py`,
untouched, per this pass's scope) -- `estimate()` takes raw left/right RGB
arrays as plain arguments, so it works the same whether they came from
`HeadCamera` (which today only exposes one eye; wiring up the second is a
separate, already-flagged concern), a recorded pair, or a synthetic test.

Depth convention (pick one, documented once, never re-litigated per call
site): `estimate()` returns metres, `float32`, same (H, W) as the input
images, with **0.0** marking invalid/unmatched pixels -- NOT NaN. NaN
propagates silently through arithmetic (one NaN pixel can NaN out a masked
mean or a whole downstream reduction that forgot to filter first); 0.0 is a
valid sentinel a caller must explicitly test for (`depth > 0`), which is the
same "0 means missing, filter before using" discipline this codebase already
uses elsewhere (e.g. `core/hand/retarget.py`'s open=0 motor convention is a
different fact, but the same "zero is a real, checkable sentinel" spirit).
"""

import dataclasses

import numpy as np


@dataclasses.dataclass
class StereoCalibration:
    """Stereo intrinsics + extrinsics for one calibrated camera pair.

    Conventions are `cv2.stereoCalibrate`'s own, kept verbatim so this
    container is a drop-in for anything already speaking OpenCV's stereo API
    (in particular, `stereo_calib.py`'s `calibrate_from_image_pairs` is the
    thing that PRODUCES one of these; `StereoSGBMDepthSource` CONSUMES one):

      K_left, K_right   (3, 3) float64 pixel-space intrinsic matrices.
      dist_left/right   (5,) (or (8,) with the rational-model extras) float64
                         OpenCV distortion coeffs [k1, k2, p1, p2, k3, ...].
      R                 (3, 3) rotation FROM the left camera frame TO the
                         right camera frame: a point in left-camera
                         coordinates `p_l` has right-camera coordinates
                         `p_r = R @ p_l + T`.
      T                 (3,) translation, metres, same convention as R.
      image_size        (width, height) in pixels -- both eyes share this;
                         `estimate()` requires input images to match it.
    """

    K_left: np.ndarray
    K_right: np.ndarray
    dist_left: np.ndarray
    dist_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: tuple[int, int]  # (W, H)

    def baseline_m(self) -> float:
        """|T|, metres -- the measured stereo baseline (compare against the
        60mm datasheet nominal as a sanity bound, per §6.1's bootstrap-only
        stance on spec-sheet numbers)."""
        return float(np.linalg.norm(np.asarray(self.T, dtype=np.float64)))

    def save(self, path) -> None:
        """Persist as a flat `.npz` -- mirrors `b_calib.npz`'s convention
        for this project's other measured-calibration asset."""
        np.savez(
            path,
            K_left=np.asarray(self.K_left, dtype=np.float64),
            K_right=np.asarray(self.K_right, dtype=np.float64),
            dist_left=np.asarray(self.dist_left, dtype=np.float64),
            dist_right=np.asarray(self.dist_right, dtype=np.float64),
            R=np.asarray(self.R, dtype=np.float64),
            T=np.asarray(self.T, dtype=np.float64),
            image_size=np.asarray(self.image_size, dtype=np.int64),
        )

    @classmethod
    def load(cls, path) -> "StereoCalibration":
        data = np.load(path)
        return cls(
            K_left=data["K_left"],
            K_right=data["K_right"],
            dist_left=data["dist_left"],
            dist_right=data["dist_right"],
            R=data["R"],
            T=data["T"],
            image_size=tuple(int(x) for x in data["image_size"]),
        )


class DepthSource:
    """Duck-typed depth interface -- matches this repo's own convention for
    swappable implementations (`camera.py`'s `HeadCamera`/`StaticCamera`,
    `executor.py`'s `UnitreeExecutor`/`MockExecutor`: plain classes with a
    matching method signature, no `abc.ABC`/`typing.Protocol` machinery).
    Subclassing this is a convenience (shared docstring, an explicit place
    to hang shared helpers later) rather than a requirement -- anything
    with a matching `estimate(rgb_left, rgb_right)` method works.
    """

    def estimate(self, rgb_left: np.ndarray, rgb_right: np.ndarray) -> np.ndarray:
        """rgb_left, rgb_right: (H, W, 3) uint8, same shape.

        Returns (H, W) float32 metres; 0.0 marks invalid pixels (see module
        docstring for why 0.0 and not NaN).
        """
        raise NotImplementedError


# Default StereoSGBM parameters.
#
# docs/relation_deploy_plan.md §5.2/§9(task 7) directs reusing the exact
# parameter choices `data_extraction_zh/src/ego_relation/s1_pico_mode2/
# stereo_depth.py` found to work on the Pico's own stereo pair, rather than
# guessing fresh ones. FLAGGED DISCREPANCY: that path does not exist on this
# machine/checkout -- `data_extraction_zh` is a separate sibling uv project
# and is not reachable from this worktree (confirmed: not under the repo,
# not a submodule, not found by search). The values below are therefore
# standard, widely-documented OpenCV SGBM defaults for an LR-consistency-
# checked matcher (non-negative `disp12MaxDiff` turns the check on), NOT the
# literal numbers from that file. Re-read the real file and update these
# before trusting them on hardware.
DEFAULT_MIN_DISPARITY = 0
DEFAULT_NUM_DISPARITIES = 128  # must be a positive multiple of 16
DEFAULT_BLOCK_SIZE = 5
DEFAULT_UNIQUENESS_RATIO = 10
DEFAULT_SPECKLE_WINDOW_SIZE = 100
DEFAULT_SPECKLE_RANGE = 32
DEFAULT_DISP12_MAX_DIFF = 1  # >=0 enables the LR-consistency check


class StereoSGBMDepthSource(DepthSource):
    """`cv2.StereoSGBM` on a rectified stereo pair, lifted to metric depth
    via the calibration's own `Q` reprojection matrix.

    Calibration (K/dist/R/T) is passed in at construction, never hardcoded
    (there is no stereo calibration data for this camera anywhere in the
    repo yet -- see `stereo_calib.py`, which produces the `StereoCalibration`
    this class consumes). Rectification maps are precomputed once here, not
    per call.
    """

    def __init__(
        self,
        calib: StereoCalibration,
        *,
        min_disparity: int = DEFAULT_MIN_DISPARITY,
        num_disparities: int = DEFAULT_NUM_DISPARITIES,
        block_size: int = DEFAULT_BLOCK_SIZE,
        uniqueness_ratio: int = DEFAULT_UNIQUENESS_RATIO,
        speckle_window_size: int = DEFAULT_SPECKLE_WINDOW_SIZE,
        speckle_range: int = DEFAULT_SPECKLE_RANGE,
        disp12_max_diff: int = DEFAULT_DISP12_MAX_DIFF,
        rectify_alpha: float = 0.0,
    ):
        import cv2  # lazy: perception/__init__ discipline, see module docstring

        if num_disparities <= 0 or num_disparities % 16 != 0:
            raise ValueError(f"num_disparities must be a positive multiple of 16, got {num_disparities}")
        if block_size < 1 or block_size % 2 == 0:
            raise ValueError(f"block_size must be a positive odd integer, got {block_size}")

        self.calib = calib
        self._cv2 = cv2

        K_l = np.asarray(calib.K_left, dtype=np.float64)
        K_r = np.asarray(calib.K_right, dtype=np.float64)
        d_l = np.asarray(calib.dist_left, dtype=np.float64)
        d_r = np.asarray(calib.dist_right, dtype=np.float64)
        R = np.asarray(calib.R, dtype=np.float64)
        T = np.asarray(calib.T, dtype=np.float64)
        size = tuple(int(x) for x in calib.image_size)  # (W, H)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_l, d_l, K_r, d_r, size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=rectify_alpha,
        )
        self._map1x, self._map1y = cv2.initUndistortRectifyMap(K_l, d_l, R1, P1, size, cv2.CV_32FC1)
        self._map2x, self._map2y = cv2.initUndistortRectifyMap(K_r, d_r, R2, P2, size, cv2.CV_32FC1)
        self._Q = Q
        self._image_size = size  # (W, H)

        p1 = 8 * 3 * block_size**2
        p2 = 32 * 3 * block_size**2
        self._matcher = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=disp12_max_diff,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self._min_disparity = min_disparity

    def estimate(self, rgb_left: np.ndarray, rgb_right: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        rgb_left = np.asarray(rgb_left)
        rgb_right = np.asarray(rgb_right)
        expect_hw = (self._image_size[1], self._image_size[0])
        if rgb_left.shape[:2] != expect_hw or rgb_right.shape[:2] != expect_hw:
            raise ValueError(
                f"expected (H, W) = {expect_hw} (from calib.image_size), "
                f"got left {rgb_left.shape[:2]} right {rgb_right.shape[:2]}"
            )

        left_r = cv2.remap(rgb_left, self._map1x, self._map1y, cv2.INTER_LINEAR)
        right_r = cv2.remap(rgb_right, self._map2x, self._map2y, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(left_r, cv2.COLOR_RGB2GRAY) if left_r.ndim == 3 else left_r
        gray_r = cv2.cvtColor(right_r, cv2.COLOR_RGB2GRAY) if right_r.ndim == 3 else right_r

        disp_fixed = self._matcher.compute(gray_l, gray_r)
        disparity = disp_fixed.astype(np.float32) / 16.0

        points_3d = cv2.reprojectImageTo3D(disparity, self._Q)
        depth = points_3d[..., 2].astype(np.float32)

        valid = (disparity > self._min_disparity) & np.isfinite(depth) & (depth > 0)
        return np.where(valid, depth, 0.0).astype(np.float32)
