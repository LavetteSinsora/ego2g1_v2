"""Stereo depth per frame, and the depth of each object's centre.

The episode already carries everything a calibration needs — per-eye intrinsics
and per-eye SDK extrinsics — so nothing external has to be measured. What is
derived here is the LEFT->RIGHT rigid transform OpenCV's stereo API wants, and
the derivation is checked against physics rather than trusted.

THE RECTIFICATION TRAP
    `StereoSGBMDepthSource.estimate` rectifies internally and returns depth in
    the RECTIFIED left frame. The SAM 3 masks are cut on the RAW left image.
    Those are different pixel grids, and sampling one with the other is a
    silent, plausible-looking error — the depth comes back as a real number
    from the wrong place.

    (`PerceptionRound` hands raw-frame masks and `calib.K_left` straight to
    `join_to_camera` against exactly this depth map. It has not bitten anyone
    because the only calibration it has ever run with is the e2e bench's
    placeholder, where `R = I` and `dist = 0` make rectification the identity.
    On this dataset it is nearly the identity too — but "nearly" is measured
    below, not assumed.)

    Handled by warping the MASK through the same maps SGBM used, sampling
    depth there, and rotating the back-projected point by `R1.T` to land back
    in the raw left camera frame. Everything this module returns is therefore
    in the raw left frame, the same frame the masks, crops and dashboard live
    in.

THE STATISTIC
    Median depth over the mask, not the depth at the centroid pixel. Same
    choice `join_to_camera` documents: a mask boundary that bleeds onto the
    background produces depth outliers, and a single pixel read at the
    centroid can land in an SGBM hole or on the gripper. The centroid supplies
    the DIRECTION (u, v) and the mask supplies the RANGE.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["StereoRig", "rig_from_episode", "DepthResult", "depth_over_episode"]


def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Scalar-LAST quaternion -> rotation matrix.

    Scalar-last because that is what the recording documents for every other
    pose field it writes (`body_pose_format`: "[x, y, z, qx, qy, qz, qw]").
    `rig_from_episode` does not take this on faith — it checks the resulting
    relative rotation against what two eyes of one rigid headset must satisfy.
    """
    x, y, z, w = (float(v) for v in np.asarray(q, dtype=np.float64)[:4])
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


@dataclasses.dataclass
class StereoRig:
    """A `StereoCalibration` plus the rectification geometry that goes with it."""

    calib: object                # perception.depth.StereoCalibration
    R1: np.ndarray               # raw-left -> rectified-left rotation
    P1: np.ndarray               # (3, 4) rectified-left projection
    map1x: np.ndarray            # rectified -> raw sampling maps for the left eye
    map1y: np.ndarray
    baseline_m: float
    rel_rot_deg: float           # |relative rotation| of the two eyes
    rectify_shift_px: float      # how far rectification actually moves a pixel

    @property
    def K_rect(self) -> np.ndarray:
        return np.asarray(self.P1, dtype=np.float64)[:3, :3]


def rig_from_episode(episode, *, swap: bool = False) -> StereoRig | None:
    """Build the rig from the episode's own recorded calibration.

    Returns None (with a reason logged) when the episode cannot support
    stereo. Depth is the only thing lost — every other stage still runs.
    """
    import cv2

    from ego2g1.deploy.perception.depth import StereoCalibration

    if not episode.has_stereo:
        logger.warning("%s: no second eye / no extrinsics — depth unavailable",
                       episode.name)
        return None

    e_l, e_r = episode.extrinsics["left"], episode.extrinsics["right"]
    R_l, R_r = _quat_xyzw_to_R(e_l[3:7]), _quat_xyzw_to_R(e_r[3:7])
    t_l, t_r = e_l[:3], e_r[:3]

    # p_head = R_l p_l + t_l = R_r p_r + t_r  =>  p_r = (R_r^T R_l) p_l + R_r^T (t_l - t_r)
    R = R_r.T @ R_l
    T = R_r.T @ (t_l - t_r)

    K_this = np.asarray(episode.K, dtype=np.float64)
    K_other = np.asarray(episode.K_other, dtype=np.float64)
    # `eye` names which image this extraction is built on. OpenCV's R and T go
    # LEFT to RIGHT, so extracting on the right eye means the pair is reversed
    # and the transform has to be inverted along with the intrinsics.
    if episode.eye == "right" or swap:
        R, T = R.T, -R.T @ T
        K_left, K_right = K_this, K_other
    else:
        K_left, K_right = K_this, K_other

    # No distortion coefficients are recorded, and the intrinsics are exact
    # pinholes (principal point at the geometric centre), so the SDK is
    # handing back already-undistorted images. Zeros are the honest value —
    # inventing coefficients would warp a correct image.
    calib = StereoCalibration(
        K_left=K_left, K_right=K_right,
        dist_left=np.zeros(5), dist_right=np.zeros(5),
        R=R, T=T, image_size=(episode.width, episode.height))

    rel = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    baseline = calib.baseline_m()

    # Two physical facts about one rigid headset. If either fails, the
    # quaternion convention or the extrinsics semantics are not what is
    # assumed here, and every depth downstream would be quietly wrong.
    if not (0.02 < baseline < 0.20):
        logger.error("%s: derived stereo baseline %.4f m is not plausible for "
                     "a headset — refusing to produce depth. Check the "
                     "extrinsics convention.", episode.name, baseline)
        return None
    if rel > 5.0:
        logger.error("%s: the two eyes differ by %.1f deg of rotation. Two "
                     "eyes of one rigid headset are near-parallel; this says "
                     "the quaternion layout is not scalar-last. Refusing to "
                     "produce depth.", episode.name, rel)
        return None

    size = (episode.width, episode.height)
    R1, _R2, P1, _P2, _Q, _, _ = cv2.stereoRectify(
        calib.K_left, calib.dist_left, calib.K_right, calib.dist_right,
        size, calib.R, calib.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib.K_left, calib.dist_left, R1, P1, size, cv2.CV_32FC1)

    # How much rectification actually moves things. Near zero means the raw
    # and rectified grids coincide and the mask warp is a formality; large
    # means it was load-bearing. Either way it is measured and recorded.
    gx, gy = np.meshgrid(np.arange(size[0], dtype=np.float32),
                         np.arange(size[1], dtype=np.float32))
    shift = float(np.nanmax(np.hypot(map1x - gx, map1y - gy)))

    logger.info("%s: baseline %.1f mm, eyes %.2f deg apart, rectification "
                "moves a pixel by at most %.2f px",
                episode.name, 1000 * baseline, rel, shift)
    return StereoRig(calib=calib, R1=np.asarray(R1), P1=np.asarray(P1),
                     map1x=map1x, map1y=map1y, baseline_m=baseline,
                     rel_rot_deg=rel, rectify_shift_px=shift)


@dataclasses.dataclass
class DepthResult:
    """Per slot, per frame. NaN where there is no usable depth."""

    depth_m: dict[str, np.ndarray]        # (F,) median depth over the mask
    point_cam: dict[str, np.ndarray]      # (F, 3) raw-left camera frame
    depth_px: dict[str, np.ndarray]       # (F,) valid depth pixels sampled
    depth_map: np.ndarray | None          # (F, H, W) uint16 mm, or None
    stats: dict

    def rate(self, slot: str) -> float:
        d = self.depth_m[slot]
        return float(np.isfinite(d).mean()) if d.size else 0.0


def depth_over_episode(rig: StereoRig, episode, tracks, *,
                       sgbm: dict | None = None, min_depth_px: int = 16,
                       save_map: bool = False, progress: bool = True
                       ) -> DepthResult:
    """SGBM per frame, then a median depth per object over its own mask."""
    import cv2

    from ego2g1.deploy.perception.depth import StereoSGBMDepthSource

    F, H, W = tracks.n_frames, tracks.height, tracks.width
    slots = tracks.slot_ids
    source = StereoSGBMDepthSource(rig.calib, **(sgbm or {}))
    valid_frame = rig_valid = episode.extrinsics.get("stereo_valid")

    depth_m = {s: np.full(F, np.nan, dtype=np.float32) for s in slots}
    point = {s: np.full((F, 3), np.nan, dtype=np.float32) for s in slots}
    npx = {s: np.zeros(F, dtype=np.int32) for s in slots}
    dmap = (np.zeros((F, H, W), dtype=np.uint16) if save_map else None)

    K = rig.K_rect
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R1T = np.asarray(rig.R1, dtype=np.float64).T

    n_ok = 0
    for i in range(F):
        if rig_valid is not None and i < len(rig_valid) and not rig_valid[i]:
            continue                       # unpaired frames: no honest depth
        wanted = [s for s in slots if tracks.frames[s][i].has_mask]
        if not wanted and not save_map:
            continue
        depth = source.estimate(episode.frame(i), episode.frame_other(i))
        good = np.isfinite(depth) & (depth > 0)
        if save_map:
            dmap[i] = np.clip(np.where(good, depth, 0) * 1000.0,
                              0, 65535).astype(np.uint16)
        n_ok += 1

        for s in wanted:
            m = tracks.frames[s][i].mask(H, W)
            # The mask is cut on the RAW image; the depth lives on the
            # RECTIFIED grid. Push the mask through the same maps rather than
            # sampling one grid with the other.
            m_rect = cv2.remap(m.astype(np.uint8), rig.map1x, rig.map1y,
                               cv2.INTER_NEAREST) > 0
            sel = m_rect & good
            n = int(sel.sum())
            npx[s][i] = n
            if n < min_depth_px:
                continue                   # SGBM hole: textureless or occluded
            z = float(np.median(depth[sel]))
            ys, xs = np.nonzero(m_rect)
            u, v = float(xs.mean()), float(ys.mean())
            p_rect = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
            depth_m[s][i] = z
            point[s][i] = R1T @ p_rect      # back into the raw left frame
        if progress and (i + 1) % 100 == 0:
            print(f"  [depth] {i + 1}/{F}", flush=True)

    stats = {
        "frames_with_depth": n_ok,
        "baseline_mm": round(1000 * rig.baseline_m, 2),
        "eye_rel_rotation_deg": round(rig.rel_rot_deg, 3),
        "rectify_shift_px": round(rig.rectify_shift_px, 3),
        "min_depth_px": min_depth_px,
        "sgbm": dict(sgbm or {}),
        "saved_map": bool(save_map),
        "per_object_rate": {},
    }
    result = DepthResult(depth_m=depth_m, point_cam=point, depth_px=npx,
                         depth_map=dmap, stats=stats)
    stats["per_object_rate"] = {s: round(result.rate(s), 4) for s in slots}
    if progress:
        med = {s: (float(np.nanmedian(depth_m[s]))
                   if np.isfinite(depth_m[s]).any() else float("nan"))
               for s in slots}
        print(f"  [depth] {n_ok}/{F} frames; median object depth (m): "
              + ", ".join(f"{s}={v:.3f}" for s, v in med.items()))
    return result


def empty_depth(tracks) -> DepthResult:
    """What to write when stereo is unavailable — NaN everywhere, honestly."""
    F, slots = tracks.n_frames, tracks.slot_ids
    return DepthResult(
        depth_m={s: np.full(F, np.nan, dtype=np.float32) for s in slots},
        point_cam={s: np.full((F, 3), np.nan, dtype=np.float32) for s in slots},
        depth_px={s: np.zeros(F, dtype=np.int32) for s in slots},
        depth_map=None,
        stats={"frames_with_depth": 0, "disabled": True})
