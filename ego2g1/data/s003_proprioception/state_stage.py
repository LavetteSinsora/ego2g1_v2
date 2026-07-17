"""s003_state: proprioception + IK feasibility signals.

Runs the whole recording through per-tick IK against the ground-truth flange
targets G(t) = S·T_wrist(t)·B and reads the state the way a deployed robot
would: forward kinematics of the achieved joint configuration, expressed in
the pelvis (base) frame. The per-tick tracking error doubles as an s004
filter signal. proprio_source="direct" skips IK (state = target pose) for
fast passes; errors are then zero and the IK filter is inert.
"""

import numpy as np

from ...core import frames
from ...core.rot6d import se3_to_vec9


def run_episode(cfg, ep_path):
    from ...kin.g1 import (ARM_JOINTS, G1Backend, DualArmIK,
                          collision_geom_groups, self_clearance)
    from .grid_util import episode_B, load_grid

    grid, _, S = load_grid(cfg, ep_path.stem, with_S=True)
    backend = G1Backend()
    B = episode_B(cfg, ep_path.stem, grid, backend)
    T = len(grid.ticks_ns)
    T_base_inv = frames.se3_inv(backend.base_pose())

    targets = {s: [grid.wrist_se3(s, k) @ B[s] for k in range(T)]
               for s in ("left", "right")}
    valid = {"left": grid.l_valid, "right": grid.r_valid}

    state = {s: np.zeros((T, 9), dtype=np.float32) for s in ("left", "right")}
    qpos = np.zeros((T, 14), dtype=np.float32)
    err_pos = {s: np.zeros(T, dtype=np.float32) for s in ("left", "right")}
    err_ori = {s: np.zeros(T, dtype=np.float32) for s in ("left", "right")}
    # min self-clearance of the achieved configuration (m, negative =
    # penetration); with the IK collision limit on this should never go
    # negative - stored so s004 can enforce that as a hard filter
    clear = {s: np.full(T, 0.2, dtype=np.float32) for s in ("left", "right")}

    if cfg.proprio_source == "direct":
        for s in ("left", "right"):
            for k in range(T):
                state[s][k] = se3_to_vec9(T_base_inv @ targets[s][k])
    elif cfg.proprio_source == "ik_fk":
        ik = DualArmIK(backend, collision_min_dist=(
            cfg.ik_collision_min_m if cfg.ik_collision else None))
        groups = collision_geom_groups(backend.model)
        backend.reset_nominal()
        ik.config.update(backend.data.qpos.copy())
        # converge onto the t0 targets from the ready pose (t0 residual is the
        # placement gate; the runtime loop then tracks tick by tick)
        init_err = ik.solve_static(targets["left"][0], targets["right"][0])
        last_cmd = {s: targets[s][0] for s in ("left", "right")}
        arm_adr = np.concatenate([backend.arm_qpos_adr["left"],
                                  backend.arm_qpos_adr["right"]])
        for k in range(T):
            cmd = {}
            for s in ("left", "right"):
                cmd[s] = targets[s][k] if valid[s][k] else last_cmd[s]
                last_cmd[s] = cmd[s]
            ik.solve_tick(cmd["left"], cmd["right"], iters=cfg.ik_iters)
            qpos[k] = backend.data.qpos[arm_adr]
            for s in ("left", "right"):
                ach = backend.flange_pose(s)
                state[s][k] = se3_to_vec9(T_base_inv @ ach)
                err_pos[s][k] = np.linalg.norm(ach[:3, 3] - cmd[s][:3, 3]) * 100.0
                err_ori[s][k] = frames.rot_geodesic_deg(ach[:3, :3], cmd[s][:3, :3])
                clear[s][k] = self_clearance(backend, groups, s)
    else:
        raise SystemExit(f"unknown proprio_source: {cfg.proprio_source}")

    arrays = {
        "state_eef_l": state["left"], "state_eef_r": state["right"],
        "arm_qpos": qpos,
        "ik_pos_cm_l": err_pos["left"], "ik_pos_cm_r": err_pos["right"],
        "ik_ori_deg_l": err_ori["left"], "ik_ori_deg_r": err_ori["right"],
        "self_clear_m_l": clear["left"], "self_clear_m_r": clear["right"],
    }
    meta = {"proprio_source": cfg.proprio_source,
            "ik_collision": bool(cfg.ik_collision)}
    for s, pre in (("left", "l"), ("right", "r")):
        live = valid[s]
        if cfg.proprio_source == "ik_fk" and live.any():
            meta[s] = {"ik_pos_cm_mean": float(err_pos[s][live].mean()),
                       "ik_pos_cm_max": float(err_pos[s][live].max()),
                       "ik_ori_deg_mean": float(err_ori[s][live].mean()),
                       "ik_ori_deg_max": float(err_ori[s][live].max()),
                       "self_clear_m_min": float(clear[s][live].min())}
    if cfg.proprio_source == "ik_fk":
        meta["init_residual_m"] = [float(e) for e in init_err]
    return arrays, meta
