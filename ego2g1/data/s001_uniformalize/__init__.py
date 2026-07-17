"""s001: lay the irregular recording streams onto the uniform control grid.

Reuses wrist_replay.episode_data for the wrist tracks (Pico->MuJoCo remap,
lerp/slerp, validity from bracketing-gap + hand-active) and generalizes the
same resampling to the full 26-joint hand arrays needed by the BrainCo
retargeter (positions lerped for all joints; only the wrist quaternion is
slerped - the retargeter consumes tip/knuckle positions + wrist orientation).

Frame conventions in the output:
- wrist tracks: MuJoCo world, quaternions wxyz (arm/EEF pipeline).
- hand arrays: RAW Pico frame, wrist quat xyzw (hand retargeting happens in
  the wrist's local frame, so the world frame cancels; keep the format the
  retargeter was validated on).
Images are NOT copied here - `cam_match` indexes into the source HDF5.
"""

import numpy as np

from ...core import episode as episode_data
from ...core import frames


def _lerp_tracks(track_ns, values, ticks_ns):
    """Linear interpolation of (N, ...) values onto ticks; returns
    (T, ...) plus lo/hi/u bracket arrays for reuse."""
    hi = np.searchsorted(track_ns, ticks_ns)
    exact = (hi < len(track_ns)) & (np.take(track_ns, np.clip(hi, 0, len(track_ns) - 1)) == ticks_ns)
    lo = np.where(exact, hi, hi - 1)
    in_range = (lo >= 0) & (hi < len(track_ns))
    lo_c = np.clip(lo, 0, len(track_ns) - 1)
    hi_c = np.clip(hi, 0, len(track_ns) - 1)
    gap = track_ns[hi_c] - track_ns[lo_c]
    u = np.where(gap > 0, (ticks_ns - track_ns[lo_c]) / np.maximum(gap, 1), 0.0)
    shape_pad = (slice(None),) + (None,) * (values.ndim - 1)
    out = (1 - u)[shape_pad] * values[lo_c] + u[shape_pad] * values[hi_c]
    return out, lo_c, hi_c, u, in_range


def _slerp_track(track_quat_xyzw, lo, hi, u):
    """Sign-corrected slerp of raw xyzw quaternions at precomputed brackets;
    output stays xyzw (raw Pico convention)."""
    out = np.zeros((len(lo), 4))
    for k in range(len(lo)):
        qa = frames.quat_wxyz_from_xyzw(track_quat_xyzw[lo[k]])
        if lo[k] == hi[k]:
            q = frames.quat_normalize(qa)
        else:
            qb = frames.quat_wxyz_from_xyzw(track_quat_xyzw[hi[k]])
            q = frames.quat_slerp(qa, qb, float(u[k]))
        out[k] = [q[1], q[2], q[3], q[0]]   # back to xyzw
    return out


def _fill_invalid(arr, valid):
    """Replace rows at invalid ticks with the nearest valid tick's row."""
    ok = np.flatnonzero(valid)
    if len(ok) == 0:
        raise SystemExit("no valid control ticks - cannot process episode")
    for k in np.flatnonzero(~valid):
        arr[k] = arr[ok[np.argmin(np.abs(ok - k))]]
    return arr


def run_episode(cfg, ep_path):
    assert cfg.grid_anchor == "first_camera_frame", cfg.grid_anchor
    ep = episode_data.load_episode(str(ep_path))
    episode_data.verify_up_axis(ep)
    grid = episode_data.build_control_grid(
        ep, cfg.control_hz, cfg.max_gap_ms,
        spike_speed_m_s=cfg.spike_speed_m_s, spike_step_cm=cfg.spike_step_cm)
    spikes = {side: int(episode_data.spike_mask(
                  ep.track_ns, getattr(ep, f"{side[0]}w7"),
                  getattr(ep, f"{side[0]}_active"),
                  cfg.spike_speed_m_s, cfg.spike_step_cm / 100.0).sum())
              for side in ("left", "right")} if cfg.spike_speed_m_s > 0 \
        else {"left": 0, "right": 0}
    if any(spikes.values()):
        print(f"  [{ep.name}] tracker spike samples flagged: {spikes}")
    ticks = grid.ticks_ns
    T = len(ticks)

    import h5py
    arrays = {
        "ticks_ns": ticks,
        "cam_match": grid.cam_match.astype(np.int32),
        "cam_gap_ms": np.abs(ep.cam_ns[grid.cam_match] - ticks) / 1e6,
    }
    with h5py.File(ep_path, "r") as f:
        for side, pre in (("left", "l"), ("right", "r")):
            arrays[f"{pre}_pos"] = getattr(grid, f"{pre}_pos")
            arrays[f"{pre}_quat"] = getattr(grid, f"{pre}_quat")
            arrays[f"{pre}_valid"] = getattr(grid, f"{pre}_valid")
            hand = f[f"{side}_hand_pose"][:].astype(np.float64)   # (N,26,7)
            valid = arrays[f"{pre}_valid"]
            pos, lo, hi, u, _ = _lerp_tracks(ep.track_ns, hand[:, :, :3], ticks)
            wrist_q = _slerp_track(hand[:, 1, 3:7], lo, hi, u)
            arrays[f"{pre}_hand_pos"] = _fill_invalid(pos, valid).astype(np.float32)
            arrays[f"{pre}_hand_wrist_quat"] = _fill_invalid(wrist_q, valid).astype(np.float32)

    meta = {
        "source": str(ep_path),
        "episode": ep.name,
        "n_ticks": T,
        "hz": cfg.control_hz,
        "valid_frac": {"left": float(arrays["l_valid"].mean()),
                       "right": float(arrays["r_valid"].mean())},
        "spike_samples": spikes,
        "cam_gap_ms_max": float(arrays["cam_gap_ms"].max()),
        "n_camera_frames": int(len(ep.cam_ns)),
    }
    return arrays, meta
