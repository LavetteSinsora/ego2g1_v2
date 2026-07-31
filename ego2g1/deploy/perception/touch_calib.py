"""Touch calibration: solve `T_pelvis_camera` (camera optical frame -> robot
pelvis frame) from measured point correspondences.

docs/relation_deploy_plan.md §6/§6.2 is the design doc. Why this exists
instead of trusting CAD/datasheet numbers (§6.1, in short): the mounting
tolerance between the camera module and this specific head shell is not a
lens property, it is an assembly fact, and this team's own prior experience
(`data_extraction_zh`'s measured 11.5mm median / 101mm max left/right
disagreement on a `calibration_verified: false` system, `TCP_TO_INWARD_PALM`
existing at all) is that CAD-implied relationships and measured ones differ
enough at ~0.3-0.6m arm reach to matter. Datasheet/CAD numbers remain a
bootstrap sanity bound (catches a gross sign/axis error immediately), never
the final number.

Procedure this module's math serves (§6.2, condensed): place a known,
detector-visible object in the workspace; command the arm (ramped, FK-known)
to touch it at several different configurations, optionally with the object
moved between touches; at each touch, BEFORE the hand occludes the object,
run the live detector+depth pipeline to get the object's position in the
camera's optical frame, and read the FK flange position in the pelvis frame.
Each touch yields one (camera-frame point, pelvis-frame point)
correspondence; `solve_camera_extrinsic` below fits the rigid transform
between the two point clouds. `collect_correspondence` is a stub describing
the physical half of the procedure -- it needs a real robot and a real live
detector, neither of which exists in this environment/session.
"""

import dataclasses

import numpy as np

from ego2g1.core.hand.retarget import _kabsch

# §6.2's own go/no-go bound: "if a handful of touches don't agree to within,
# say, 1-2 cm, something more fundamental is wrong". Exposed as a constant so
# the CLI wrapper and any caller apply the exact same threshold.
RESIDUAL_WARN_M = 0.02


def solve_camera_extrinsic(
    points_camera: np.ndarray, points_pelvis: np.ndarray
) -> tuple[np.ndarray, float]:
    """Solve the rigid transform T (4, 4) such that, for every i,
    `T @ [*points_camera[i], 1] ~= [*points_pelvis[i], 1]`.

    points_camera, points_pelvis: (N, 3) corresponding 3D points, N >= 3
    (a unique rotation needs >= 3 non-collinear points; more, spread out,
    is what makes the residual below a meaningful go/no-go number rather
    than an overfit to a minimal set, per §6.2's closing paragraph).

    Reuses `ego2g1.core.hand.retarget._kabsch`, THE existing rigid-transform
    fit in this codebase (§6.2 point 4: reuse one of the two that already
    exist rather than writing a third). Its docstring says it solves a
    rotation "minimizing ||R @ src_i - dst_i|| over unit vectors" because
    its only other caller (`HandRetargeter.calibrate`) happens to feed it
    pre-normalized fingertip DIRECTIONS -- but the function itself is the
    general SVD-based orthogonal-Procrustes solution (`H = dst.T @ src`,
    SVD, det-sign fix) and works on any already-centered point cloud, not
    just unit vectors; normalization is a property of that caller's inputs,
    not a requirement `_kabsch` itself imposes. It also does NOT solve
    translation. So this function does the standard Kabsch/Umeyama split:
    center both point sets on their own centroid, solve R on the centered
    clouds (translation-invariant by construction), then recover
    translation in closed form as
    `t = mean(points_pelvis) - R @ mean(points_camera)` -- given a fixed R,
    that is exactly the t minimizing `sum ||R @ p_i + t - q_i||^2`, since
    the optimum aligns the centroids. Verified against a synthetic random
    rigid transform: R recovered to ~2e-16, t to ~1e-17 (float64 machine
    precision), see `tests/deploy/perception/test_touch_calib.py`.

    Returns `(T, rms_residual_m)`: `T` is the (4, 4) rigid transform (camera
    -> pelvis); `rms_residual_m` is `sqrt(mean(||T @ p_cam_i - p_pel_i||^2))`
    over all correspondences, metres -- compare against `RESIDUAL_WARN_M`.
    """
    points_camera = np.asarray(points_camera, dtype=np.float64)
    points_pelvis = np.asarray(points_pelvis, dtype=np.float64)
    if points_camera.ndim != 2 or points_camera.shape[-1] != 3:
        raise ValueError(f"points_camera must be (N, 3), got {points_camera.shape}")
    if points_camera.shape != points_pelvis.shape:
        raise ValueError(
            f"points_camera {points_camera.shape} and points_pelvis "
            f"{points_pelvis.shape} must have the same (N, 3) shape"
        )
    n = points_camera.shape[0]
    if n < 3:
        raise ValueError(f"need >= 3 correspondences to fix a unique rotation, got {n}")

    centroid_cam = points_camera.mean(axis=0)
    centroid_pel = points_pelvis.mean(axis=0)
    src_c = points_camera - centroid_cam
    dst_c = points_pelvis - centroid_pel

    R = _kabsch(src_c, dst_c)
    t = centroid_pel - R @ centroid_cam

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    predicted = (R @ points_camera.T).T + t
    residual = predicted - points_pelvis
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return T, rms


@dataclasses.dataclass
class TouchSample:
    """One touch-calibration correspondence, kept around for the manifest
    (mirrors `b_calib.manifest.json`'s "how many points, residual, date"
    pattern per §6.2 point 5)."""

    point_camera: np.ndarray  # (3,) camera-optical-frame position (detector+depth)
    point_pelvis: np.ndarray  # (3,) FK flange position, pelvis frame, at the touch
    arm_q: np.ndarray | None = None  # (14,) joint config at the touch, for the record
    note: str = ""


def collect_correspondence(kin, executor, detector, touch_configs, *, touch_point_offset=None):
    """STUB. Describes, but cannot run, the physical touch-calibration
    procedure (§6.2) -- it needs a real robot (`kin`: `Kinematics`,
    `executor`: `UnitreeExecutor`) and a real live detector, neither
    reachable from this environment/session. Kept as a clearly-marked,
    procedure-shaped interface for a Phase-3 bring-up script to fill in;
    the tested, real math is `solve_camera_extrinsic` above.

    Intended per-entry procedure, for each `arm_q14` in `touch_configs` (an
    operator-verified configuration that touches a known, DINO-detectable
    object at a known point on the gripper):

      1. On approach, BEFORE the hand occludes the object: run the live
         detector + a `depth.DepthSource` on the current stereo frame, lift
         the object's mask centroid (or masked-region median -- more robust
         to mask-boundary noise, per §5.3) to a camera-optical-frame 3D
         point. This is one `point_camera`.
      2. Drive the arm (ramped -- e.g. this repo's `runner.reset_to_episode`
         style motion, never a raw teleport) to `arm_q`, confirm contact
         (there is no automatic "touched" signal; an operator watches),
         then read `kin.flange_poses(arm_q)[hand][:3, 3]` -- the pelvis-
         frame flange position. This is `point_pelvis`, ASSUMING the known
         physical touch point on the gripper coincides with the flange
         origin; if it is offset by a known vector in the flange frame,
         pass it as `touch_point_offset` and this stub would instead return
         `flange_pose[:3, :3] @ touch_point_offset + flange_pose[:3, 3]`.
      3. Repeat across several `touch_configs` -- different shoulder/elbow
         angles reaching the SAME physical point, plus repeats with the
         object moved -- `solve_camera_extrinsic` needs >= 3 non-collinear
         points, and more/spread-out samples make the residual a meaningful
         go/no-go number (§6.2's closing paragraph).

    Always raises `NotImplementedError` -- there is no simulated fallback
    here because a fabricated detector/FK pair would silently hide exactly
    the calibration risk this procedure exists to measure.
    """
    raise NotImplementedError(
        "collect_correspondence is a physical-procedure stub (needs a real "
        "robot + a real live detector); see its docstring for the exact "
        "steps. Once you have real (points_camera, points_pelvis) arrays "
        "(e.g. logged by hand during a bring-up session), call "
        "solve_camera_extrinsic(...) directly."
    )


def _cli_solve(points_camera_npy: str, points_pelvis_npy: str, out_npz: str = "") -> None:
    """CLI: solve_camera_extrinsic from two saved (N, 3) `.npy` correspondence
    files (however they were logged during a real touch-calibration
    session), print the transform + residual, optionally save."""
    points_camera = np.load(points_camera_npy)
    points_pelvis = np.load(points_pelvis_npy)
    T, rms = solve_camera_extrinsic(points_camera, points_pelvis)

    print(f"T_pelvis_camera ({points_camera.shape[0]} points) =\n{T}")
    print(f"RMS residual: {rms * 1000:.2f} mm")
    if rms > RESIDUAL_WARN_M:
        print(
            f"WARNING: residual exceeds the {RESIDUAL_WARN_M * 100:.0f} cm "
            "go/no-go bound (§6.2) -- do not trust this calibration on hardware."
        )
    if out_npz:
        np.savez(
            out_npz,
            T_pelvis_camera=T,
            rms_residual_m=np.float64(rms),
            n_points=np.int64(points_camera.shape[0]),
        )
        print(f"saved -> {out_npz}")


if __name__ == "__main__":
    import tyro

    tyro.cli(_cli_solve)
