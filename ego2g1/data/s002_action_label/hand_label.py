"""s002_02: BrainCo Revo2 commands per control tick.

Runs the migrated fingertip retargeter (wrist-local, so the world frame and
placement S never enter) on the s001 grid arrays. Rate limiting then operates
at the control period, which is what the deployed hand will experience.

Calibration scope:
- per_episode: most-open frame of this episode (fragile if the episode never
  opens the hand);
- shared_recording: calibrate once from cfg.hand_calib_recording (a dedicated
  flat-open-hand recording), shared by every episode of the subject.
"""

import numpy as np

from .. import io


def _grid_hand_pose(arrays, pre):
    """(T,26,7) in the raw layout the retargeter was validated on: lerped
    positions for all joints, slerped quat only at the wrist (index 1) - the
    solver reads tip/knuckle positions + wrist orientation, nothing else."""
    pos = arrays[f"{pre}_hand_pos"].astype(np.float64)
    T = len(pos)
    pose = np.zeros((T, 26, 7))
    pose[:, :, :3] = pos
    pose[:, :, 6] = 1.0                                   # identity xyzw
    pose[:, 1, 3:7] = arrays[f"{pre}_hand_wrist_quat"].astype(np.float64)
    return pose


def run_episode(cfg, ep_path):
    from ...core.hand.retarget import HandRetargeter

    if cfg.fingertip_source != "pico":
        raise NotImplementedError(
            f"fingertip_source={cfg.fingertip_source} (HaMeR is a stub)")

    arrays_in, _ = io.load_stage(cfg, ep_path.stem, "s001")
    ticks_ns = arrays_in["ticks_ns"]

    out, meta = {}, {}
    for side, pre in (("left", "l"), ("right", "r")):
        r = HandRetargeter(side, align=cfg.hand_align)
        recalibrate = True
        if cfg.hand_calib == "shared_recording":
            if not cfg.hand_calib_recording:
                raise SystemExit("hand_calib=shared_recording needs hand_calib_recording")
            import h5py
            with h5py.File(cfg.hand_calib_recording, "r") as f:
                r.calibrate(f[f"{side}_hand_pose"][:].astype(np.float64),
                            valid=f[f"{side}_hand_active"][:].astype(bool))
            recalibrate = False
        elif cfg.hand_calib != "per_episode":
            raise SystemExit(f"unknown hand_calib: {cfg.hand_calib}")

        res = r.retarget(_grid_hand_pose(arrays_in, pre),
                         timestamps_ns=ticks_ns if cfg.hand_rate_limit else None,
                         active=arrays_in[f"{pre}_valid"],
                         recalibrate=recalibrate)
        out[f"hand_cmds_{pre}"] = res["cmds"]
        out[f"hand_cmds_raw_{pre}"] = res["cmds_raw"]
        out[f"hand_residual_{pre}"] = res["residual_m"]
        out[f"hand_snap_{pre}"] = res["snap_flags"]
        out[f"hand_valid_{pre}"] = res["valid"]

        v = res["valid"]
        meta[side] = {
            "calib": cfg.hand_calib,
            "calib_frame": int(res["calib_frame"]),
            "scales": [float(x) for x in res["scales"]],
            "valid_frac": float(v.mean()),
            "residual_mm_mean": float(res["residual_m"][v].mean() * 1000) if v.any() else None,
            "residual_mm_max": float(res["residual_m"][v].max() * 1000) if v.any() else None,
            "snap_ticks": int(res["snap_flags"][v].any(axis=1).sum()) if v.any() else 0,
        }
    return out, meta
