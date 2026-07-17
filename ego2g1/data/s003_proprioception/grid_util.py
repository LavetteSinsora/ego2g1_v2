"""Rebuild a ControlGrid-shaped view from s001 stage arrays.

The placement / alignment / IK code (migrated from wrist_replay) operates on
the grid attributes (l_pos, l_quat, l_valid, ...); stages reload them from
the s001 npz instead of recomputing from the HDF5.
"""

import numpy as np

from ...core import frames
from .. import io
from ...core.episode import apply_world_transform


class GridView:
    def __init__(self, arrays):
        self.ticks_ns = arrays["ticks_ns"]
        for pre in ("l", "r"):
            setattr(self, f"{pre}_pos", arrays[f"{pre}_pos"].astype(np.float64).copy())
            setattr(self, f"{pre}_quat", arrays[f"{pre}_quat"].astype(np.float64).copy())
            setattr(self, f"{pre}_valid", arrays[f"{pre}_valid"].astype(bool))

    def wrist_se3(self, side, k):
        pre = side[0]
        return frames.se3(getattr(self, f"{pre}_pos")[k],
                          getattr(self, f"{pre}_quat")[k])


def load_grid(cfg, ep_name, with_S=False):
    """-> (GridView, s001 meta[, S]) ; with_S applies the s003 placement."""
    arrays, meta = io.load_stage(cfg, ep_name, "s001")
    grid = GridView(arrays)
    if not with_S:
        return grid, meta
    p_arrays, _ = io.load_stage(cfg, ep_name, "s003_placement")
    S = p_arrays["S"]
    apply_world_transform(grid, S)
    return grid, meta, S


def episode_B(cfg, ep_name, grid_with_S, backend):
    """The alignment B to use for this episode: the global b_calib output,
    or a per-episode t0 calibration when cfg.b_alignment == 'per_episode'
    (diagnostic mode - injects per-episode label inconsistency)."""
    from ...kin.placement import calibrate_alignment, first_valid_tick
    if cfg.b_alignment == "per_episode":
        return calibrate_alignment(grid_with_S, first_valid_tick(grid_with_S), backend)
    arrays, _ = io.load_stage(cfg, None, "b_calib")
    return {"left": arrays["B_left"], "right": arrays["B_right"]}
