"""s004c_resolve: final proprioception by re-solving IK on the SMOOTHED targets.

Why this stage exists (docs/jitter_root_cause.md): s003_state IKs the RAW
targets, and the mink QP is fully converged at 5 iterations — so every bit of
Pico tracking noise (1197 deg/s² angular accel RMS on episode_1's right hand)
lands in its `arm_qpos` as ~26 rad/s² joint zig-zag. Replaying those joints
judders on the real robot through ANY executor. s004b smooths the action
labels but deliberately not the proprioception; this stage closes that gap by
re-running the same IK (same collision limits, same iteration budget) on the
s004b-smoothed poses, producing the `state_eef_*`/`arm_qpos` that s005 writes.

Proprioception thus becomes FK of smoothed-target tracking — which is also
what the deployed robot's measured state will look like, since deploy smooths
its EEF targets the same way before IK (ego2g1.kin.filters). s003's raw-target
IK stays untouched as the s004 filter-signal source (its error signals must
see the raw glitches, not a smoothed cover-up).

Solved per sub-episode span: converge statically onto the span's first pose
(a span start is a deployment episode start), then track tick by tick.
Ticks outside every span are never written to the dataset; they carry the
nearest span value so the arrays stay finite. Meta records the smoothness
improvement (joint accel RMS vs s003) and the EEF cost of tracking smoothed
targets, so a regression is visible in the stage log rather than on the robot.
"""

import numpy as np

from ...core import frames
from ...core.rot6d import se3_to_vec9, vec9_to_se3
from .. import io


def _joint_accel_rms(qpos, spans, hz):
    """Worst-joint accel RMS (rad/s^2) over in-span ticks; the jitter metric."""
    accs = []
    for a, b in spans:
        q = qpos[a:b]
        if len(q) < 3:
            continue
        accs.append(np.diff(q, n=2, axis=0) * hz * hz)
    if not accs:
        return 0.0
    a = np.concatenate(accs, axis=0)
    return float(np.sqrt((a ** 2).mean(axis=0)).max())


def run_episode(cfg, ep_path):
    from ...kin.g1 import ARM_JOINTS, DualArmIK, G1Backend

    stem = ep_path.stem
    smooth, _ = io.load_stage(cfg, stem, "s004b_smooth")
    s004, _ = io.load_stage(cfg, stem, "s004")
    s003, _ = io.load_stage(cfg, stem, "s003_state")

    T = len(smooth["pose_l"])
    spans = [(int(a), int(b)) for a, b in zip(s004["subep_start"], s004["subep_end"])]

    state = {s: np.zeros((T, 9), dtype=np.float32) for s in ("left", "right")}
    qpos = np.zeros((T, 14), dtype=np.float32)
    err_pos = {s: np.zeros(T, dtype=np.float32) for s in ("left", "right")}
    err_ori = {s: np.zeros(T, dtype=np.float32) for s in ("left", "right")}

    if not cfg.resolve_proprio:
        # A/B escape hatch: pass s003's raw-target proprioception through
        arrays = {"state_eef_l": s003["state_eef_l"], "state_eef_r": s003["state_eef_r"],
                  "arm_qpos": s003["arm_qpos"],
                  "resolve_pos_cm_l": err_pos["left"], "resolve_pos_cm_r": err_pos["right"],
                  "resolve_ori_deg_l": err_ori["left"], "resolve_ori_deg_r": err_ori["right"]}
        return arrays, {"resolve_proprio": False}

    backend = G1Backend()
    T_base = backend.base_pose()
    T_base_inv = frames.se3_inv(T_base)
    tgt = {"left": [T_base @ vec9_to_se3(smooth["pose_l"][k]) for k in range(T)],
           "right": [T_base @ vec9_to_se3(smooth["pose_r"][k]) for k in range(T)]}
    arm_adr = np.concatenate([backend.arm_qpos_adr["left"],
                              backend.arm_qpos_adr["right"]])

    smooth_cost = float(cfg.resolve_smooth_cost)
    for a, b in spans:
        ik = DualArmIK(backend,
                       posture_cost=(smooth_cost if smooth_cost > 0 else 1e-3),
                       collision_min_dist=(
                           cfg.ik_collision_min_m if cfg.ik_collision else None))
        # Seed from s003's solution at the span start, NOT the nominal pose:
        # the raw-target solve already found a branch whose errors passed the
        # s004 filters, and a fresh nominal-seeded solve can converge into a
        # worse null-space basin (seen on episode_2: elbow pinned at its joint
        # limit for a stretch, wrist flailing to compensate, 214 rad/s² spike).
        backend.reset_nominal()
        full = backend.data.qpos.copy()
        full[arm_adr] = s003["arm_qpos"][a]
        backend.set_qpos(full)
        ik.config.update(backend.data.qpos.copy())
        ik.solve_static(tgt["left"][a], tgt["right"][a])
        for k in range(a, b):
            if smooth_cost > 0:
                # posture tracks the previous solution -> ||q - q_last||^2 term
                ik.posture.set_target_from_configuration(ik.config)
            ik.solve_tick(tgt["left"][k], tgt["right"][k], iters=cfg.ik_iters)
            qpos[k] = backend.data.qpos[arm_adr]
            for s in ("left", "right"):
                ach = backend.flange_pose(s)
                state[s][k] = se3_to_vec9(T_base_inv @ ach)
                err_pos[s][k] = np.linalg.norm(ach[:3, 3] - tgt[s][k][:3, 3]) * 100.0
                err_ori[s][k] = frames.rot_geodesic_deg(ach[:3, :3], tgt[s][k][:3, :3])

    # out-of-span ticks: hold the nearest computed value (never consumed by s005)
    if spans:
        covered = np.zeros(T, dtype=bool)
        for a, b in spans:
            covered[a:b] = True
        idx = np.where(covered)[0]
        nearest = idx[np.abs(np.arange(T)[:, None] - idx[None, :]).argmin(axis=1)]
        for arr in (qpos, state["left"], state["right"]):
            arr[~covered] = arr[nearest[~covered]]

    hz = cfg.control_hz
    meta = {"resolve_proprio": True, "n_spans": len(spans),
            "accel_rms_raw": _joint_accel_rms(s003["arm_qpos"], spans, hz),
            "accel_rms_resolved": _joint_accel_rms(qpos, spans, hz)}
    for s, p in (("left", "l"), ("right", "r")):
        if spans:
            kept = np.concatenate([np.arange(a, b) for a, b in spans])
            meta[s] = {"eef_pos_cm_mean": float(err_pos[s][kept].mean()),
                       "eef_pos_cm_max": float(err_pos[s][kept].max()),
                       "eef_ori_deg_mean": float(err_ori[s][kept].mean()),
                       "eef_ori_deg_max": float(err_ori[s][kept].max())}

    arrays = {"state_eef_l": state["left"], "state_eef_r": state["right"],
              "arm_qpos": qpos,
              "resolve_pos_cm_l": err_pos["left"], "resolve_pos_cm_r": err_pos["right"],
              "resolve_ori_deg_l": err_ori["left"], "resolve_ori_deg_r": err_ori["right"]}
    return arrays, meta
