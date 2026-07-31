"""Checkerboard stereo calibration -- solves the `StereoCalibration` that
`depth.py`'s `StereoSGBMDepthSource` needs to produce metric depth.

docs/relation_deploy_plan.md §5.2/§9(task 6b): there is no stereo
calibration for this camera anywhere in the repo (the "60mm baseline" is a
datasheet nominal, not a per-unit measurement) -- `StereoSGBM` needs a real
calibration to turn disparity into metres at all. This is "ordinary,
well-understood tooling (`cv2.stereoCalibrate`)", per the plan, not a new
research problem: capture N image pairs of a checkerboard from both eyes,
find the board corners in each, solve.

Split into a pure function (`calibrate_from_image_pairs`, the actual math,
unit-testable with no camera or physical board) and a thin CLI wrapper that
owns image I/O -- capture the pairs however is convenient (this repo's
`ego2g1.deploy.camera.HeadCamera` run twice, one per eye, or
`python -m ego2g1.deploy.check camera`, saved to disk), this tool only
consumes already-saved images.
"""

import dataclasses
import glob
import os

import numpy as np

from .depth import StereoCalibration

# Below this many successfully-detected views, cv2.calibrateCamera is known
# to silently return physically nonsensical intrinsics (severely
# underconstrained focal length / principal point) rather than raising --
# OpenCV's own calibration tutorials recommend 10-20 well-varied views;
# this is a conservative floor for "don't even provisionally trust it",
# not a claim that this many guarantees a good result.
MIN_RECOMMENDED_VIEWS = 8


@dataclasses.dataclass
class CalibrationReport:
    """What actually happened during a `calibrate_from_image_pairs` solve --
    surfaced because the previous version of this tool printed `len(pairs)`
    (images GIVEN) where it meant `n_found` (images the board was actually
    detected in), silently hiding a garbage-in-garbage-out failure mode."""

    n_given: int
    n_found: int
    failed_indices: tuple[int, ...]
    rms_left_px: float
    rms_right_px: float
    rms_stereo_px: float


def _object_points(board_size: tuple[int, int], square_size_m: float) -> np.ndarray:
    """(cols, rows) INNER corners -> (cols*rows, 3) planar object points
    (board frame, Z=0), in the same corner order `cv2.findChessboardCorners`
    returns (row-major raster scan)."""
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)  # raster order, row-major
    objp[:, :2] = grid * square_size_m
    return objp


def calibrate_from_image_pairs(
    pairs,
    board_size: tuple[int, int],
    square_size_m: float,
    *,
    criteria=None,
) -> StereoCalibration:
    """Solve a `StereoCalibration` from N checkerboard image pairs.

    pairs:          sequence of (img_left, img_right), each (H, W) or
                    (H, W, 3) uint8, ALL sharing one image size. Pairing is
                    positional (`pairs[i] = (left_i, right_i)`), so there is
                    no separate "count mismatch" to check across two lists
                    -- a mismatched left/right SIZE within one pair (a
                    genuinely bad capture) is checked and raises instead.
    board_size:     (cols, rows) INNER corners, as `cv2.findChessboardCorners`
                    counts them (one less than squares per side). Must be
                    non-square (cols != rows) -- a square board has a 90deg
                    corner-labeling ambiguity `cv2` cannot resolve on its own.
    square_size_m:  physical edge length of one checker square, metres.

    Returns `(StereoCalibration, CalibrationReport)` -- the report carries
    exactly how many of the GIVEN pairs actually had the board detected in
    BOTH eyes (`n_found`/`n_given`), which frames failed and why (mask not
    found vs. size mismatch), and the mean per-view reprojection error
    (pixels) each stage's own solve reports -- all of it load-bearing for
    catching a garbage calibration BEFORE it gets trusted downstream. A
    calibration solved from too few views is a well-known OpenCV failure
    mode: with only a handful of poorly-varied detections, focal length and
    principal point are severely underconstrained and `cv2.calibrateCamera`
    will still return SOME numbers -- often wildly asymmetric fx/fy, huge
    reprojection error, or a nonsensical baseline -- with no exception
    raised. Silence here is exactly how that goes undiagnosed.

    Pipeline (the standard two-stage pattern, e.g. OpenCV's own
    stereo_calib.cpp sample): per-eye `cv2.calibrateCamera` for an intrinsic
    initial guess, then `cv2.stereoCalibrate(..., flags=CALIB_FIX_INTRINSIC)`
    to solve just the extrinsics (R, T) against those fixed intrinsics.
    """
    import cv2  # lazy, see package __init__ docstring

    if len(pairs) == 0:
        raise ValueError("need at least one image pair")
    cols, rows = board_size
    if cols == rows:
        raise ValueError(
            f"board_size {board_size}: cols must differ from rows (a square "
            "board's corner order is ambiguous to cv2.findChessboardCorners)"
        )
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    objp = _object_points(board_size, square_size_m).astype(np.float32)

    obj_points, img_points_l, img_points_r = [], [], []
    image_size = None
    failed_indices: list[int] = []

    for i, (left, right) in enumerate(pairs):
        left = np.asarray(left)
        right = np.asarray(right)
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(
                f"pair {i}: left/right image sizes differ, "
                f"{left.shape[:2]} vs {right.shape[:2]}"
            )
        size = (int(left.shape[1]), int(left.shape[0]))  # (W, H)
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(
                f"pair {i}: image size {size} does not match earlier pair(s) {image_size}"
            )

        gray_l = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY) if left.ndim == 3 else left
        gray_r = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY) if right.ndim == 3 else right
        gray_l = np.ascontiguousarray(gray_l, dtype=np.uint8)
        gray_r = np.ascontiguousarray(gray_r, dtype=np.uint8)

        ok_l, corners_l = cv2.findChessboardCorners(gray_l, (cols, rows))
        ok_r, corners_r = cv2.findChessboardCorners(gray_r, (cols, rows))
        if not (ok_l and ok_r):
            failed_indices.append(i)
            continue

        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points_l.append(corners_l)
        img_points_r.append(corners_r)

    n_found = len(obj_points)
    if n_found == 0:
        raise ValueError(
            f"checkerboard ({cols}x{rows} inner corners) not found in both eyes "
            f"of any of the {len(pairs)} pair(s) given"
        )
    if n_found < MIN_RECOMMENDED_VIEWS:
        print(
            f"WARNING: the board was only detected in {n_found}/{len(pairs)} pair(s) "
            f"(failed: {failed_indices}) -- {MIN_RECOMMENDED_VIEWS}+ well-varied views "
            "are recommended. With this few, cv2.calibrateCamera is known to return "
            "physically nonsensical numbers (wildly asymmetric fx/fy, a nonsensical "
            "baseline) WITHOUT raising an error -- treat this calibration as "
            "provisional and check the reprojection error below before trusting it."
        )

    rms_l, K_l, dist_l, _, _ = cv2.calibrateCamera(obj_points, img_points_l, image_size, None, None)
    rms_r, K_r, dist_r, _, _ = cv2.calibrateCamera(obj_points, img_points_r, image_size, None, None)

    rms_stereo, K_l, dist_l, K_r, dist_r, R, T, _, _ = cv2.stereoCalibrate(
        obj_points, img_points_l, img_points_r,
        K_l, dist_l, K_r, dist_r, image_size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )

    calib = StereoCalibration(
        K_left=K_l,
        K_right=K_r,
        dist_left=dist_l.flatten(),
        dist_right=dist_r.flatten(),
        R=R,
        T=T.flatten(),
        image_size=image_size,
    )
    report = CalibrationReport(
        n_given=len(pairs),
        n_found=n_found,
        failed_indices=tuple(failed_indices),
        rms_left_px=float(rms_l),
        rms_right_px=float(rms_r),
        rms_stereo_px=float(rms_stereo),
    )
    return calib, report


def _cli_calibrate(
    image_dir: str,
    cols: int,
    rows: int,
    square_size_m: float,
    out_npz: str = "stereo_calib.npz",
    left_glob: str = "left_*.png",
    right_glob: str = "right_*.png",
) -> None:
    """CLI: calibrate from left/right image pairs already saved on disk.

    Capture pairs beforehand with whatever gets frames off the two eyes
    (e.g. `ego2g1.deploy.camera.HeadCamera` run once per `eye`, or
    `python -m ego2g1.deploy.check camera`, saved as PNGs) -- this tool only
    does the math on already-captured images; `left_glob`/`right_glob` must
    sort into the same pairing order.
    """
    import cv2  # lazy

    left_paths = sorted(glob.glob(os.path.join(image_dir, left_glob)))
    right_paths = sorted(glob.glob(os.path.join(image_dir, right_glob)))
    if len(left_paths) != len(right_paths):
        raise ValueError(
            f"found {len(left_paths)} left image(s) ({left_glob!r}) but "
            f"{len(right_paths)} right image(s) ({right_glob!r}) in "
            f"{image_dir!r} -- capture is paired, counts must match"
        )
    if not left_paths:
        raise ValueError(f"no images matching {left_glob!r}/{right_glob!r} in {image_dir!r}")

    pairs = []
    for lp, rp in zip(left_paths, right_paths):
        left = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        pairs.append((left, right))

    calib, report = calibrate_from_image_pairs(pairs, (cols, rows), square_size_m)
    calib.save(out_npz)

    print(f"board found in {report.n_found}/{report.n_given} pair(s)")
    if report.failed_indices:
        failed_names = [os.path.basename(left_paths[i]) for i in report.failed_indices]
        print(f"  NOT found in: {failed_names}")
    print(f"reprojection error (px, lower is better -- <0.5 good, >1.0 suspect): "
          f"left={report.rms_left_px:.3f} right={report.rms_right_px:.3f} "
          f"stereo={report.rms_stereo_px:.3f}")
    print(f"baseline: {calib.baseline_m() * 1000:.1f} mm (datasheet nominal: 60 mm)")
    print(f"K_left:\n{calib.K_left}")
    print(f"K_right:\n{calib.K_right}")
    print(f"R (left->right):\n{calib.R}")
    print(f"T (left->right), mm: {calib.T * 1000}")

    fx_l, fy_l = calib.K_left[0, 0], calib.K_left[1, 1]
    suspect = (
        report.n_found < MIN_RECOMMENDED_VIEWS
        or report.rms_stereo_px > 1.0
        or abs(fx_l / fy_l - 1.0) > 0.2
    )
    if suspect:
        print(
            "\nSUSPECT CALIBRATION -- do not trust this before investigating. "
            f"fx/fy ratio (left) = {fx_l / fy_l:.2f} (should be close to 1.0 for "
            "a real lens); a real camera's focal length is the same in both "
            "directions, so a ratio far from 1.0, together with few detected "
            f"views ({report.n_found}) and/or high reprojection error, means the "
            "solve was underconstrained garbage, not a real measurement. See the "
            "failed-pair list above and re-capture with more, more varied views."
        )
    print(f"saved -> {out_npz}  (SUSPECT, see above)" if suspect else f"saved -> {out_npz}")


if __name__ == "__main__":
    import tyro

    tyro.cli(_cli_calibrate)
