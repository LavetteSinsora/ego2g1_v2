"""b_calib (global): the fixed wrist->flange alignment rotation B, per side.

B is the convention "when the human wrist is oriented like this, the flange
is oriented like that". It conjugates every relative action (B^-1 Δ B), so it
must be ONE constant across the dataset - a per-episode B is a hidden
variable the policy cannot observe (kept only as a diagnostic mode).

Modes:
- geometric: human palm frame (wrist + index/middle/pinky knuckle landmarks,
  averaged over all valid ticks of all episodes) mapped onto the Revo2 palm
  frame through the flange->Revo2 mount rotation:  B = R_wp · G_r^T · R_mount^T.
  Solves "align the two palms" - the physically meaningful statement.
- dataset_mean: chordal mean of the per-episode t0 calibrations
  (B_e = R_wrist(t0)^T · R_flange_nominal, each computed after placement S).
- per_episode: downstream stages calibrate at each episode's t0 (diagnostic).

Whatever the mode, the manifest reports the per-episode calibrated B spread
around the chosen B - large angles mean the fixed convention leaves large
initial orientation offsets (they land in the state, where the policy can
see them, but they should be sane; tens of degrees is expected, ~180 means
the mount rotation is wrong).
"""

import numpy as np

from ...core import frames
from .. import io

# OpenXR landmark indices (Pico 26-joint layout)
XR_WRIST = 1
XR_INDEX_K, XR_MIDDLE_K, XR_PINKY_K = 7, 12, 22
SUBSAMPLE = 3   # every Nth valid tick feeds the palm-frame average


def chordal_mean(rots):
    """Rotation closest (Frobenius) to the arithmetic mean of rotations."""
    M = np.mean(np.asarray(rots), axis=0)
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def mount_rotation(cfg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", cfg.revo2_mount_rpy_deg, degrees=True).as_matrix()


def _wrist_to_palm_mean(cfg, ep_paths, side):
    """Mean rotation from the human wrist frame to the geometric palm frame,
    over all valid ticks of all episodes (raw Pico arrays from s001; local
    frames are invariant to the world remap)."""
    from ...core.hand.fk_tables import palm_frame_from_points
    from ...core.hand.retarget import _quat_to_rot

    pre = side[0]
    samples = []
    for p in ep_paths:
        arrays, _ = io.load_stage(cfg, p.stem, "s001")
        pos = arrays[f"{pre}_hand_pos"].astype(np.float64)         # (T,26,3)
        quat = arrays[f"{pre}_hand_wrist_quat"].astype(np.float64)  # (T,4) xyzw
        valid = arrays[f"{pre}_valid"].astype(bool)
        for k in np.flatnonzero(valid)[::SUBSAMPLE]:
            G_palm = palm_frame_from_points(pos[k, XR_WRIST], pos[k, XR_INDEX_K],
                                            pos[k, XR_MIDDLE_K], pos[k, XR_PINKY_K], side)
            R_wrist = _quat_to_rot(quat[k])
            samples.append(R_wrist.T @ G_palm)
    return chordal_mean(samples), len(samples)


def _per_episode_Bs(cfg, ep_paths, backend):
    from ...kin.placement import calibrate_alignment, first_valid_tick
    from ..s003_proprioception.grid_util import load_grid

    out = {"left": [], "right": []}
    for p in ep_paths:
        grid, _, _ = load_grid(cfg, p.stem, with_S=True)
        Bs = calibrate_alignment(grid, first_valid_tick(grid), backend)
        for side in out:
            out[side].append(Bs[side][:3, :3])
    return out


def run_global(cfg, ep_paths):
    from ...core.hand.fk_tables import load_tables
    from ...kin.g1 import G1Backend

    backend = G1Backend()
    B_eps = _per_episode_Bs(cfg, ep_paths, backend)

    B = {}
    arrays = {}
    meta = {"mode": cfg.b_alignment, "mount_rpy_deg": list(cfg.revo2_mount_rpy_deg)}
    for side in ("left", "right"):
        R_wp, n = _wrist_to_palm_mean(cfg, ep_paths, side)
        G_r = load_tables(side)["robot_palm"]
        meta[f"{side}_palm_samples"] = n
        if cfg.b_alignment == "geometric":
            B_R = R_wp @ G_r.T @ mount_rotation(cfg).T
        elif cfg.b_alignment == "dataset_mean":
            B_R = chordal_mean(B_eps[side])
        elif cfg.b_alignment == "per_episode":
            B_R = np.eye(3)   # placeholder; stages calibrate per episode
        else:
            raise SystemExit(f"unknown b_alignment: {cfg.b_alignment}")
        B[side] = frames.se3_from_rot(B_R)
        # Flange->Revo2-base rotation that makes the mounted hand's palm
        # coincide with the (mean) human palm whenever the flange sits at the
        # stored pose T_wrist·B:  R_flange·R_mount·G_r = R_wrist·R_wp with
        # R_flange = R_wrist·B  =>  R_mount = B^T · R_wp · G_r^T.
        # (For b_alignment=geometric this reduces to cfg's mount_rotation.)
        arrays[f"mount_R_{side}"] = B_R.T @ R_wp @ G_r.T
        spread = [frames.rot_geodesic_deg(B_R, Be) for Be in B_eps[side]]
        meta[f"{side}_spread_deg"] = {"mean": float(np.mean(spread)),
                                      "max": float(np.max(spread)),
                                      "min": float(np.min(spread))}
        # always report how far the geometric-free reference (dataset mean)
        # sits from the chosen B - a consistency cross-check between modes
        meta[f"{side}_vs_dataset_mean_deg"] = float(
            frames.rot_geodesic_deg(B_R, chordal_mean(B_eps[side])))
        print(f"  B[{side}] ({cfg.b_alignment}): per-episode spread "
              f"mean {meta[f'{side}_spread_deg']['mean']:.1f} deg, "
              f"max {meta[f'{side}_spread_deg']['max']:.1f} deg; "
              f"vs dataset-mean {meta[f'{side}_vs_dataset_mean_deg']:.1f} deg")

    arrays.update({"B_left": B["left"], "B_right": B["right"]})
    return arrays, meta
