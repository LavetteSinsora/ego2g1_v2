"""s004b_smooth: denoise the ACTION labels, within each s004 good run.

Runs AFTER s004 so the real jitters (tracker teleports, IK divergence, hand-contact) are
already flagged and cut -- we smooth only what survived, and only INSIDE a sub-episode span,
so a low-pass never bleeds across a split or over a dropped run. This is the "filter, THEN
smooth" order the pipeline was asked for: s004 removes the outliers a smoother would
otherwise smear into their neighbours (an 18 cm tracker jump must be cut, not averaged).

What is smoothed (the labels the policy imitates):
- finger commands (hand_cmds, 6-vec in [0,1]) -- Savitzky-Golay per motor, then re-clamped
  to [0,1] and to the URDF velocity limit (reusing HandRetargeter._rate_limit);
- EEF pose (9-vec = 3 translation + 6d rotation) -- SavGol on translation, and a zero-phase
  windowed quaternion average on rotation (6d is not a vector space, so smoothing it
  component-wise would distort the rotation).

What is NOT smoothed: proprioception (s003 state_eef / arm_qpos). It is the deployment-
faithful FK of the ORIGINAL IK targets; smoothing it would misrepresent what the robot
measures and would need a post-s004 IK re-solve (a DAG inversion). The hand portion of the
written `state` DOES move -- but only because it is literally this same hand_cmds array,
which s005 reuses for both state and action.
"""

import numpy as np

from ...core import frames
from .. import io
from ...core.rot6d import mat_to_6d, rot6d_to_mat


def _savgol(x, window, poly):
    """Savitzky-Golay along axis 0. Window is shrunk to fit a short span (kept odd and
    strictly greater than the polynomial order); too-short spans pass through unchanged."""
    from scipy.signal import savgol_filter

    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    w = min(int(window), T if T % 2 == 1 else T - 1)
    if w < 3 or w <= poly:
        return x.copy()
    return savgol_filter(x, w, poly, axis=0, mode="interp")


def _smooth_rot6d(pose6d, window):
    """(T,6) 6d rotations -> smoothed (T,6). Convert to quaternions, hemisphere-align for a
    well-posed linear average, average over a centred window (= chordal mean for a short
    window, as b_alignment's dataset_mean does), renormalize, back to 6d. Edges auto-clamp,
    so window >= span length degrades gracefully; window <= 1 is a no-op."""
    pose6d = np.asarray(pose6d, dtype=np.float64)
    T = len(pose6d)
    R = rot6d_to_mat(pose6d)                                       # (T,3,3)
    q = np.stack([frames.quat_from_mat(R[t]) for t in range(T)])  # (T,4) wxyz
    for t in range(1, T):                                          # hemisphere continuity
        if np.dot(q[t], q[t - 1]) < 0.0:
            q[t] = -q[t]
    half = int(window) // 2
    out = np.empty_like(pose6d)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        qm = frames.quat_normalize(q[lo:hi].mean(axis=0))
        out[t] = mat_to_6d(frames.mat_from_quat(qm))
    return out


def _repair_span(pose, cmds, bad):
    """Fill bridged bad ticks (anchor_bad) from good neighbours so they cannot leak into the
    smoother. Only reached when cfg.bridge_max_ticks > 0; strict-split leaves spans clean.
    `pose`/`cmds` are views into the full arrays -- mutated in place."""
    good = np.flatnonzero(~bad)
    if len(good) == 0 or not bad.any():
        return
    idx = np.arange(len(bad))
    for j in range(cmds.shape[1]):
        cmds[bad, j] = np.interp(idx[bad], good, cmds[good, j])
    for j in range(3):                                            # translation
        pose[bad, j] = np.interp(idx[bad], good, pose[good, j])
    for k in np.flatnonzero(bad):                                 # rotation: nearest good
        nn = good[np.argmin(np.abs(good - k))]
        pose[k, 3:9] = pose[nn, 3:9]


def run_episode(cfg, ep_path):
    from ...core.hand.retarget import HandRetargeter

    stem = ep_path.stem
    s001, _ = io.load_stage(cfg, stem, "s001")
    s002_01, _ = io.load_stage(cfg, stem, "s002_01")
    s002_02, _ = io.load_stage(cfg, stem, "s002_02")
    s004, _ = io.load_stage(cfg, stem, "s004")

    ticks_ns = s001["ticks_ns"]
    T = len(ticks_ns)
    spans = [(int(a), int(b)) for a, b in zip(s004["subep_start"], s004["subep_end"])]
    anchor_bad = s004["anchor_bad"].astype(bool) if "anchor_bad" in s004 \
        else np.zeros(T, dtype=bool)

    out, meta = {}, {"n_spans": len(spans), "window": int(cfg.smooth_window)}
    for side, pre in (("left", "l"), ("right", "r")):
        pose = s002_01[f"pose_{pre}"].astype(np.float64).copy()       # (T,9)
        cmds = s002_02[f"hand_cmds_{pre}"].astype(np.float64).copy()  # (T,6)
        raw_cmds = cmds.copy()
        for a, b in spans:
            sl = slice(a, b)
            bad = anchor_bad[sl]
            if bad.any():
                _repair_span(pose[sl], cmds[sl], bad)
            if cfg.smooth_eef:
                pose[sl, :3] = _savgol(pose[sl, :3], cfg.smooth_window, cfg.smooth_polyorder)
                pose[sl, 3:9] = _smooth_rot6d(pose[sl, 3:9], cfg.smooth_window)
            if cfg.smooth_hand:
                sm = np.clip(_savgol(cmds[sl], cfg.smooth_window, cfg.smooth_polyorder),
                             0.0, 1.0).astype(np.float32)
                cmds[sl] = HandRetargeter._rate_limit(sm, ticks_ns[sl]).astype(np.float64)
        out[f"pose_{pre}"] = pose.astype(np.float32)
        out[f"hand_cmds_{pre}"] = cmds.astype(np.float32)
        if spans:
            kept = np.concatenate([np.arange(a, b) for a, b in spans])
            meta[f"finger_rms_removed_{pre}"] = float(
                np.sqrt(np.mean((cmds[kept] - raw_cmds[kept]) ** 2)))
    return out, meta
