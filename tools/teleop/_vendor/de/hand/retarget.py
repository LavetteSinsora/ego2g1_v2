"""Human hand keypoints (Pico 26-joint OpenXR) -> BrainCo Revo2 motor commands.

Mechanism (per hand, per frame):
  1. Express the five fingertip positions in the wrist's local frame
     (v = R_wrist^T (p_tip - p_wrist)), discarding fingertip orientation --
     the Revo2's tip orientation is mechanically slaved to flexion anyway.
  2. Rotate into the Revo2 base frame with a fixed alignment R (Kabsch fit on
     the most-open calibration frame) and scale per finger by
     s_f = |robot open tip| / |human open tip| to absorb morphology.
  3. Index/middle/ring/pinky are 1-DOF: project each scaled target onto that
     finger's precomputed fingertip arc (nearest sample = command).
  4. Thumb (flex, rot) minimizes over its precomputed 2-D grid:
     pinch terms (thumb<->index, thumb<->middle pad distances) weighted above
     a wrist->thumb position regularizer. When the *human* pinch distance
     drops below snap_dist, the target distance snaps to slightly negative to
     commit to contact instead of hovering (DexPilot-style).
  5. Commands are rate-limited with the URDF joint velocity limits.

Output: (T, 6) float32 in [0,1], motor order [thumb_flex, thumb_rot, index,
middle, ring, pinky] -- exactly what the Unitree DDS bridge consumes.
"""

import numpy as np

from .constants import CMD_RATE_LIMIT, FINGERS, MOTOR_ORDER, XR_TIPS, XR_WRIST
from .fk_tables import Revo2FK, load_tables, palm_frame_from_points

# OpenXR joint indices of the palm-frame landmarks (proximal knuckles)
XR_INDEX_K, XR_MIDDLE_K, XR_PINKY_K = 7, 12, 22

# thumb objective weights
W_POS = 1.0
W_PINCH = 3.0
W_PINCH_SNAPPED = 10.0
SNAP_DIST = 0.025   # human thumb<->finger pad distance (m) that triggers snap
SNAP_TARGET = -0.003  # "slightly negative" target distance to enforce contact


def _quat_to_rot(q):
    """(...,4) [qx,qy,qz,qw] -> (...,3,3) rotation matrices."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.maximum(np.sqrt(x * x + y * y + z * z + w * w), 1e-12)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def wrist_frame_tips(hand_pose):
    """(T,26,7) world-frame joint poses -> (T,5,3) tip positions in the wrist
    frame, finger order [thumb, index, middle, ring, pinky]."""
    wrist_p = hand_pose[:, XR_WRIST, :3]
    wrist_R = _quat_to_rot(hand_pose[:, XR_WRIST, 3:7])
    tip_idx = [XR_TIPS[f] for f in ["thumb"] + FINGERS]
    rel = hand_pose[:, tip_idx, :3] - wrist_p[:, None, :]
    return np.einsum("tji,tfj->tfi", wrist_R, rel)  # R^T @ rel


def _kabsch(src, dst):
    """Rotation R minimizing ||R @ src_i - dst_i|| over unit vectors."""
    H = dst.T @ src
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


class HandRetargeter:
    def __init__(self, hand, align="geometric"):
        """align: "geometric" (palm frame from wrist+knuckle landmarks; no
        fitted bias — measured better than kabsch on all test episodes,
        thumb/index residuals roughly halved) or "kabsch" (fit tip directions
        at the most-open frame; kept for comparison)."""
        self.hand = hand
        self.align_mode = align
        t = load_tables(hand)
        self.arc_cmds = t["arc_cmds"]
        self.arcs = np.stack([t[f"arc_{f}"] for f in FINGERS])  # (4, N, 3)
        self.grid_cmds = t["grid_cmds"]
        self.thumb_grid = t["thumb_grid"]  # (Nf, Nr, 3)
        self.robot_palm = t["robot_palm"]  # (3,3) geometric palm frame
        self.R_align = None
        self.scales = None  # (5,) [thumb, index, middle, ring, pinky]

    # ---------------- calibration ----------------

    def calibrate(self, hand_pose, valid=None):
        """Fit alignment rotation + per-finger scales from the most-open frame
        of an episode (largest summed tip-to-wrist distance)."""
        tips = wrist_frame_tips(hand_pose)  # (T,5,3)
        spread = np.linalg.norm(tips, axis=2).sum(axis=1)
        if valid is not None:
            spread = np.where(valid, spread, -np.inf)
        open_t = int(np.argmax(spread))
        human_open = tips[open_t]  # (5,3)

        robot_open = np.vstack([self.thumb_grid[0, 0][None], self.arcs[:, 0]])  # (5,3)
        self.scales = np.linalg.norm(robot_open, axis=1) / np.linalg.norm(human_open, axis=1)
        if self.align_mode == "geometric":
            # palm frame from landmarks, expressed in the wrist frame, mapped
            # onto the robot's geometric palm frame: R = G_robot . G_human^T
            p = hand_pose[open_t, :, :3]
            wrist_R = _quat_to_rot(hand_pose[open_t, XR_WRIST, 3:7])
            G_h = wrist_R.T @ palm_frame_from_points(
                p[XR_WRIST], p[XR_INDEX_K], p[XR_MIDDLE_K], p[XR_PINKY_K], self.hand
            )
            self.R_align = self.robot_palm @ G_h.T
        else:
            self.R_align = _kabsch(
                human_open / np.linalg.norm(human_open, axis=1, keepdims=True),
                robot_open / np.linalg.norm(robot_open, axis=1, keepdims=True),
            )
        return open_t

    # ---------------- per-frame solve ----------------

    def _solve_fingers(self, targets):
        """targets (4,3) scaled base-frame goals -> cmds (4,), robot tips (4,3)."""
        d2 = ((self.arcs - targets[:, None, :]) ** 2).sum(axis=2)  # (4, N)
        idx = d2.argmin(axis=1)
        return self.arc_cmds[idx], self.arcs[np.arange(4), idx]

    def _solve_thumb(self, thumb_target, robot_index, robot_middle, t_ti, t_tm, snapped):
        g = self.thumb_grid
        w_pinch = W_PINCH_SNAPPED if snapped else W_PINCH
        cost = W_POS * ((g - thumb_target) ** 2).sum(axis=2)
        cost += w_pinch * (np.linalg.norm(g - robot_index, axis=2) - t_ti) ** 2
        cost += w_pinch * (np.linalg.norm(g - robot_middle, axis=2) - t_tm) ** 2
        i, j = np.unravel_index(cost.argmin(), cost.shape)
        return self.grid_cmds[i], self.grid_cmds[j], g[i, j]

    def step(self, tips_t):
        """One frame: wrist-frame tip positions (5,3) -> cmds (6,), residual (5,), snaps (2,).

        The whole per-frame solve, with no reference to any other frame. `retarget`
        is this in a loop -- teleop calls it directly, at whatever rate the tracker
        runs. Requires `calibrate` to have been called (R_align, scales).

        `tips_t` is in the RAW wrist frame (the output of `wrist_frame_tips`), not
        the aligned/scaled one: the pinch distances below are measured on the human
        hand, and scaling them first would change what "pads are touching" means.
        """
        if self.R_align is None or self.scales is None:
            raise RuntimeError("call calibrate() before step()")

        scaled = (self.R_align @ tips_t.T).T * self.scales[:, None]

        d_ti = float(np.linalg.norm(tips_t[0] - tips_t[1]))
        d_tm = float(np.linalg.norm(tips_t[0] - tips_t[2]))
        s_ti = 0.5 * (self.scales[0] + self.scales[1])
        s_tm = 0.5 * (self.scales[0] + self.scales[2])

        f_cmds, f_tips = self._solve_fingers(scaled[1:])
        snap_i, snap_m = d_ti < SNAP_DIST, d_tm < SNAP_DIST
        t_ti = SNAP_TARGET if snap_i else s_ti * d_ti
        t_tm = SNAP_TARGET if snap_m else s_tm * d_tm
        flex, rot, thumb_tip = self._solve_thumb(
            scaled[0], f_tips[0], f_tips[1], t_ti, t_tm, snap_i or snap_m
        )

        cmds = np.array([flex, rot, *f_cmds], dtype=np.float32)
        residual = np.empty(5, dtype=np.float32)
        residual[0] = np.linalg.norm(thumb_tip - scaled[0])
        residual[1:] = np.linalg.norm(f_tips - scaled[1:], axis=1)
        return cmds, residual, (snap_i, snap_m)

    def retarget(self, hand_pose, timestamps_ns=None, active=None, recalibrate=True):
        """(T,26,7) -> dict with cmds (T,6) in MOTOR_ORDER + diagnostics.

        Frames with lost tracking (active=False and/or all-zero rows, as in
        the Pico recordings) are skipped and hold the last valid command.
        """
        T = hand_pose.shape[0]
        valid = np.abs(hand_pose).sum(axis=(1, 2)) > 0
        valid &= np.isfinite(hand_pose).all(axis=(1, 2))
        if active is not None:
            valid &= np.asarray(active, dtype=bool)
        if not valid.any():
            raise ValueError("no valid tracking frames in episode")
        if recalibrate or self.R_align is None:
            open_t = self.calibrate(hand_pose, valid=valid)
        else:
            open_t = -1

        tips = wrist_frame_tips(hand_pose)  # (T,5,3) [thumb,index,middle,ring,pinky]

        cmds = np.zeros((T, 6), dtype=np.float32)
        residual = np.zeros((T, 5), dtype=np.float32)  # tip-target miss (m)
        snap_flags = np.zeros((T, 2), dtype=bool)

        # One frame at a time, through the SAME `step` the teleop loop calls. Keeping
        # a second vectorized copy of this solve here is how the two silently drift.
        for t in range(T):
            if not valid[t]:
                continue
            cmds[t], residual[t], snap_flags[t] = self.step(tips[t])

        # invalid frames hold the last valid command (leading ones take the
        # first valid command) so the output track is gap-free
        valid_idx = np.flatnonzero(valid)
        fill_from = valid_idx[np.clip(
            np.searchsorted(valid_idx, np.arange(T), side="right") - 1, 0, None
        )]
        cmds = cmds[fill_from]

        raw = cmds.copy()
        cmds = self._rate_limit(cmds, timestamps_ns)
        return {
            "cmds": cmds,
            "valid": valid,
            "cmds_raw": raw,
            "residual_m": residual,
            "snap_flags": snap_flags,
            "calib_frame": open_t,
            "scales": self.scales.astype(np.float32),
            "R_align": self.R_align.astype(np.float32),
        }

    @staticmethod
    def _rate_limit(cmds, timestamps_ns):
        if timestamps_ns is None:
            return cmds
        rates = np.array([CMD_RATE_LIMIT[m] for m in MOTOR_ORDER], dtype=np.float32)
        out = cmds.copy()
        dt = np.diff(np.asarray(timestamps_ns, dtype=np.float64)) * 1e-9
        for t in range(1, len(out)):
            step = rates * max(dt[t - 1], 1e-4)
            out[t] = np.clip(out[t], out[t - 1] - step, out[t - 1] + step)
        return out
