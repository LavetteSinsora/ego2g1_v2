"""s002_01: canonical per-tick flange poses - the action-label source.

Stores absolute poses, never action chunks: the loader gathers H+1 poses and
differences them (Δ_k = pose_0^-1 · pose_k), so the chunk length is a
training-time choice. With cfg.pose_frame == "flange" the stored pose is
G(t) = pelvis^-1 · S · T_wrist(t) · B: the deltas the loader computes are
then flange-frame and compose at deployment as T_anchor_FK · Δ_k with no B
anywhere on the robot side.

Gates (hard-fail):
- selftest_identity: chunk anchor/delta composition reproduces the direct
  retarget map exactly (catches every frame-convention bug at once);
- continuity: no single-tick rotation jump above cfg.max_tick_rotation_deg
  between consecutive valid ticks (catches quaternion/interp bugs).
"""

import numpy as np

from ...core import frames
from ...core.rot6d import se3_to_vec9


def run_episode(cfg, ep_path):
    from ...core.chunks import selftest_identity
    from ...kin.g1 import G1Backend
    from ..s003_proprioception.grid_util import episode_B, load_grid

    grid, _, S = load_grid(cfg, ep_path.stem, with_S=True)
    backend = G1Backend()
    B = episode_B(cfg, ep_path.stem, grid, backend)
    T_base_inv = frames.se3_inv(backend.base_pose())
    T = len(grid.ticks_ns)

    arrays, meta = {}, {}
    for side, pre in (("left", "l"), ("right", "r")):
        T_h = [grid.wrist_se3(side, k) for k in range(T)]
        worst = selftest_identity(T_h, B[side], cfg.action_horizon)

        if cfg.pose_frame == "flange":
            G = [t @ B[side] for t in T_h]
        elif cfg.pose_frame == "wrist":
            G = T_h
        else:
            raise SystemExit(f"unknown pose_frame: {cfg.pose_frame}")

        valid = getattr(grid, f"{pre}_valid")
        jumps = [frames.rot_geodesic_deg(G[k - 1][:3, :3], G[k][:3, :3])
                 for k in range(1, T) if valid[k - 1] and valid[k]]
        max_jump = float(max(jumps)) if jumps else 0.0
        if max_jump > cfg.max_tick_rotation_deg:
            raise SystemExit(
                f"{ep_path.stem} [{side}]: single-tick rotation jump "
                f"{max_jump:.1f} deg > {cfg.max_tick_rotation_deg} deg - "
                f"orientation track is suspect, refusing to label")

        arrays[f"pose_{pre}"] = np.stack(
            [se3_to_vec9(T_base_inv @ g) for g in G]).astype(np.float32)
        meta[side] = {"selftest_max_err": float(worst),
                      "max_tick_rot_deg": max_jump}
    return arrays, meta
