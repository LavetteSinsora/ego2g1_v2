"""Episode loading + 30 Hz control-grid resampling for wrist_replay.

Reads the Pico episode HDF5 format (same files as pico2usable/dashboard.py,
same field semantics) and produces per-tick wrist poses in the MuJoCo world
frame, ready for retargeting:

- wrist = joint index 1 of left/right_hand_pose (N, 26, 7)
- raw rows are [x y z qx qy qz qw] (scalar-LAST) in the Pico body-tracking
  world frame (OpenXR-style: x right, y up, z backward)
- control grid = uniform ticks at `hz` anchored at the first camera frame
  (the dashboard's convention); each tick is interpolated from its bracketing
  tracking samples (lerp position, sign-corrected slerp quaternion) and is
  invalid when the bracket gap exceeds `max_gap_ms` or a bracketing sample
  has the hand inactive.
"""

import os
from dataclasses import dataclass, field

import h5py
import numpy as np

from . import frames

# Pico body-tracking (OpenXR-style) -> MuJoCo world: mj x = forward (-xr z),
# mj y = left (-xr x), mj z = up (xr y). Proper rotation (det = +1).
R_XR_TO_MJ = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


@dataclass
class Episode:
    path: str
    name: str
    attrs: dict
    # tracking stream (device clock, ns)
    track_ns: np.ndarray          # (N,)
    lw7: np.ndarray               # (N, 7) raw [xyz xyzw], Pico frame
    rw7: np.ndarray               # (N, 7)
    l_active: np.ndarray          # (N,) bool
    r_active: np.ndarray          # (N,) bool
    # camera stream
    cam_ns: np.ndarray            # (M,)
    jpegs: list = field(repr=False, default=None)   # M jpeg byte strings


@dataclass
class ControlGrid:
    ticks_ns: np.ndarray          # (T,) device-clock tick times
    hz: float
    # per side, MuJoCo world frame, wxyz quaternions
    l_pos: np.ndarray             # (T, 3)
    l_quat: np.ndarray            # (T, 4)
    l_valid: np.ndarray           # (T,) bool
    r_pos: np.ndarray
    r_quat: np.ndarray
    r_valid: np.ndarray
    cam_match: np.ndarray         # (T,) nearest camera frame index per tick

    def wrist_se3(self, side, k):
        p = self.l_pos if side == "left" else self.r_pos
        q = self.l_quat if side == "left" else self.r_quat
        return frames.se3(p[k], q[k])


def load_episode(path):
    with h5py.File(path, "r") as f:
        lh = f["left_hand_pose"][:]
        rh = f["right_hand_pose"][:]
        return Episode(
            path=path,
            name=os.path.join(
                os.path.basename(os.path.dirname(os.path.abspath(path))),
                os.path.splitext(os.path.basename(path))[0]),
            attrs={k: (v.item() if isinstance(v, np.generic) else v)
                   for k, v in f.attrs.items()},
            track_ns=f["timestamps_ns"][:].astype(np.int64),
            lw7=lh[:, 1, :].astype(np.float64),   # wrist = joint 1
            rw7=rh[:, 1, :].astype(np.float64),
            l_active=f["left_hand_active"][:].astype(bool),
            r_active=f["right_hand_active"][:].astype(bool),
            cam_ns=f["camera/timestamps_ns"][:].astype(np.int64),
            jpegs=[bytes(b) for b in f["camera/images_left_jpeg"][:]],
        )


def verify_up_axis(ep):
    """Hard-fail if the data does not look like the expected OpenXR frame
    (y up: wrists of a standing/seated person should sit around 0.5-1.3 m on
    the y axis and have the largest mean magnitude there)."""
    pos = np.concatenate([ep.lw7[:, :3], ep.rw7[:, :3]])
    mean_abs = np.abs(pos).mean(axis=0)
    up = int(np.argmax(mean_abs))
    print(f"  up-axis check: per-axis mean|pos| = {np.round(mean_abs, 3)} "
          f"(x,y,z) -> dominant axis {'xyz'[up]}")
    if up != 1:
        raise SystemExit(
            f"up-axis check FAILED: expected data-y to dominate (OpenXR y-up), "
            f"got axis {'xyz'[up]}. The Pico frame convention changed - "
            f"re-derive R_XR_TO_MJ before trusting any output.")
    if not (0.3 < mean_abs[1] < 1.6):
        raise SystemExit(
            f"up-axis check FAILED: wrist height {mean_abs[1]:.2f} m is not a "
            f"plausible human wrist height; frame convention is suspect.")


def spike_mask(track_ns, pose7, active, max_speed_m_s=2.0, min_step_m=0.03):
    """Flag raw samples adjacent to physically implausible position steps.

    The Pico hand tracker occasionally re-solves the wrist pose in a jump
    (~83% radial to the headset: a depth re-estimate) while still reporting
    active=1, so validity masks never fire. Both endpoints of any
    active-active step whose implied speed exceeds `max_speed_m_s` AND whose
    displacement exceeds `min_step_m` are flagged (the step floor keeps tiny
    dt from amplifying millimeter jitter into fake speed). Flagged samples
    count as inactive downstream: the bracketing-validity check invalidates
    the surrounding ticks, strict splitting cuts the episode there, and a
    short glitch island between two cuts falls under min_subepisode_ticks
    and is dropped entirely."""
    n = len(track_ns)
    bad = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not (active[i] and active[i - 1]):
            continue
        dt = (track_ns[i] - track_ns[i - 1]) / 1e9
        if dt <= 0:
            continue
        step = float(np.linalg.norm(pose7[i, :3] - pose7[i - 1, :3]))
        if step > min_step_m and step / dt > max_speed_m_s:
            bad[i] = bad[i - 1] = True
    return bad


def make_control_grid_ticks(ep, hz):
    """Uniform ticks anchored at the first camera frame, covering the span
    where tracking data exists."""
    t0 = int(ep.cam_ns[0])
    end = int(ep.track_ns[-1])
    n = max(0, int(np.floor((end - t0) * hz / 1e9))) + 1
    return (t0 + np.round(np.arange(n) * 1e9 / hz)).astype(np.int64)


def _resample_side(track_ns, pose7, active, ticks_ns, max_gap_ms):
    T = len(ticks_ns)
    pos = np.zeros((T, 3))
    quat = np.zeros((T, 4))
    valid = np.zeros(T, dtype=bool)
    quat_xyzw = pose7[:, 3:]
    max_gap_ns = max_gap_ms * 1e6

    hi = np.searchsorted(track_ns, ticks_ns)          # first sample >= tick
    for k in range(T):
        h = hi[k]
        lo = h - 1
        if h < len(track_ns) and track_ns[h] == ticks_ns[k]:
            lo = h                                     # exact hit
        if lo < 0 or h >= len(track_ns):
            quat[k] = np.array([1.0, 0, 0, 0])
            continue
        qa = frames.quat_wxyz_from_xyzw(quat_xyzw[lo])
        if lo == h:
            pos[k] = pose7[h, :3]
            quat[k] = frames.quat_normalize(qa)
            valid[k] = bool(active[h])
            continue
        gap = track_ns[h] - track_ns[lo]
        u = (ticks_ns[k] - track_ns[lo]) / gap
        qb = frames.quat_wxyz_from_xyzw(quat_xyzw[h])
        pos[k] = (1 - u) * pose7[lo, :3] + u * pose7[h, :3]
        quat[k] = frames.quat_slerp(qa, qb, u)
        valid[k] = bool(active[lo]) and bool(active[h]) and gap <= max_gap_ns

    # invalid ticks (e.g. before the first tracking sample) keep placeholder
    # poses that would poison anchors and deltas - fill each from the nearest
    # valid tick; the valid mask still marks them as held for the runtime.
    if not valid.any():
        raise SystemExit("no valid control ticks for one hand - cannot replay")
    ok = np.flatnonzero(valid)
    for k in np.flatnonzero(~valid):
        j = ok[np.argmin(np.abs(ok - k))]
        pos[k], quat[k] = pos[j], quat[j]
    return pos, quat, valid


def _to_mj_world(pos, quat):
    """Apply the fixed Pico->MuJoCo axis remap: p' = R p, R' = R R_xr."""
    q_r = frames.quat_from_mat(R_XR_TO_MJ)
    pos_mj = pos @ R_XR_TO_MJ.T
    quat_mj = np.stack([frames.quat_mul(q_r, q) for q in quat])
    return pos_mj, quat_mj


def build_control_grid(ep, hz=30.0, max_gap_ms=70.0,
                       spike_speed_m_s=2.0, spike_step_cm=3.0):
    ticks = make_control_grid_ticks(ep, hz)
    l_active, r_active = ep.l_active, ep.r_active
    if spike_speed_m_s > 0:
        l_active = l_active & ~spike_mask(ep.track_ns, ep.lw7, ep.l_active,
                                          spike_speed_m_s, spike_step_cm / 100.0)
        r_active = r_active & ~spike_mask(ep.track_ns, ep.rw7, ep.r_active,
                                          spike_speed_m_s, spike_step_cm / 100.0)
    l_pos, l_quat, l_valid = _resample_side(
        ep.track_ns, ep.lw7, l_active, ticks, max_gap_ms)
    r_pos, r_quat, r_valid = _resample_side(
        ep.track_ns, ep.rw7, r_active, ticks, max_gap_ms)
    l_pos, l_quat = _to_mj_world(l_pos, l_quat)
    r_pos, r_quat = _to_mj_world(r_pos, r_quat)

    # nearest camera frame per tick (for the report's human-view column)
    idx = np.clip(np.searchsorted(ep.cam_ns, ticks), 0, len(ep.cam_ns) - 1)
    prev = np.clip(idx - 1, 0, len(ep.cam_ns) - 1)
    take_prev = np.abs(ep.cam_ns[prev] - ticks) < np.abs(ep.cam_ns[idx] - ticks)
    cam_match = np.where(take_prev, prev, idx)

    return ControlGrid(ticks_ns=ticks, hz=hz,
                       l_pos=l_pos, l_quat=l_quat, l_valid=l_valid,
                       r_pos=r_pos, r_quat=r_quat, r_valid=r_valid,
                       cam_match=cam_match)


def apply_world_transform(grid, S):
    """Left-multiply every wrist pose by the rigid transform S (4x4), i.e.
    re-express the human data in a world where the robot base sits at its
    XML pose. Mutates and returns the grid."""
    R, t = S[:3, :3], S[:3, 3]
    qS = frames.quat_from_mat(R)
    for pre in ("l", "r"):
        pos = getattr(grid, f"{pre}_pos")
        quat = getattr(grid, f"{pre}_quat")
        setattr(grid, f"{pre}_pos", pos @ R.T + t)
        setattr(grid, f"{pre}_quat", np.stack([frames.quat_mul(qS, q) for q in quat]))
    return grid
