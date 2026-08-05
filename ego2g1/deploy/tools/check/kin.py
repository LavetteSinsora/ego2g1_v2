"""Rungs 2/3/3b (`check fk` / `check ik` / `check tcp-orientation`):
offline kinematics validation against a dataset (joint order, frames,
vec9 encoding, IK budget) plus the human-eyeball TCP/flange-convention
rung (docs/relation_deploy_plan.md §4.4). No hardware, no checkpoint."""

from __future__ import annotations

import sys
import time

import numpy as np

from ego2g1.core import layout, se3
from ego2g1.deploy.tools.replay_dataset import load_episode

# --- 2. fk -------------------------------------------------------------------

def fk(dataset: str, episode: int = 0, tol: float = 1e-4) -> None:
    """FK the dataset's stored joints and compare to its stored state.

    Validates joint order, waist==0, the flange site, the pelvis frame, and the
    vec9 encoding in one shot. No hardware, no checkpoint."""
    import pandas as pd
    import pathlib

    from ego2g1.deploy.core.kinematics import Kinematics

    files = sorted(pathlib.Path(dataset).glob("data/*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset}/data/")
    df = pd.read_parquet(files[min(episode, len(files) - 1)])
    arm = np.stack(df["arm_qpos"].to_numpy())
    state = np.stack(df["state"].to_numpy())
    kin = Kinematics()
    print(f"{files[min(episode, len(files)-1)].name}: {len(arm)} frames\n")

    worst = 0.0
    for h in layout.HANDS:
        errs = []
        for t in range(0, len(arm), 5):
            got = se3.se3_to_vec9(kin.flange_poses(arm[t])[h])
            errs.append(np.abs(got - state[t, layout.EEF[h]]))
        e = np.stack(errs)
        worst = max(worst, float(e.max()))
        print(f"  {h:5s}  trans max {e[:, :3].max():.3e} m   "
              f"rot6d max {e[:, 3:].max():.3e}")

    print(f"\nworst {worst:.3e}")
    if worst < tol:
        print("PASS — FK reproduces the dataset state.")
    else:
        sys.exit("FAIL — joint order, frame, or flange is wrong. "
                 "Do NOT go to hardware.")


# --- 3. ik -------------------------------------------------------------------

def ik(dataset: str, episode: int = 0, n: int = 150, ik_iters: int = 25) -> None:
    """Track the dataset's stored poses with the deploy IK; compare to its
    stored joints and time the solve. Also where you learn whether one solve
    fits in a 30 Hz tick."""
    from ego2g1.deploy.core.kinematics import Kinematics

    ep = load_episode(dataset, episode)
    kin = Kinematics(ik_iters=ik_iters)
    kin.ground(ep["arm"][0])

    n = min(n, len(ep["arm"]))
    q_err, t_err, dur = [], [], []
    for t in range(n):
        targets = {h: se3.vec9_to_se3(ep["pose"][h][t]) for h in layout.HANDS}
        t0 = time.perf_counter()
        q = kin.solve(targets)
        dur.append((time.perf_counter() - t0) * 1000)
        q_err.append(np.abs(q - ep["arm"][t]))
        t_err.append(max(kin.tracking_error(targets).values()))

    q_err, t_err, dur = np.stack(q_err), np.array(t_err), np.array(dur)
    budget = 1000.0 / 30
    print(f"{ep['name']}: {n} ticks, warm-started, posture-tracks-last\n")
    print(f"  joint err    mean {q_err.mean():.4f} rad   max {q_err.max():.4f} rad")
    print("    (nonzero is EXPECTED: the smoothness cost trades exact replication "
          "for low accel)")
    print(f"  flange err   mean {t_err.mean()*1000:.2f} mm   max {t_err.max()*1000:.2f} mm")
    print(f"  solve time   mean {dur.mean():.2f} ms   p95 {np.percentile(dur, 95):.2f} ms")
    print(f"\n  30 Hz budget {budget:.1f} ms -> IK uses {dur.mean()/budget*100:.1f}%")
    if t_err.max() > 0.02:
        sys.exit("FAIL — IK cannot track the training poses. Frames are wrong.")
    print("PASS")


# --- 3b. tcp-orientation ------------------------------------------------------

# Cardinal directions in the PELVIS frame (== world frame; `pelvis` has no
# quat in assets/unitree_g1/g1_fixed_base.xml, so it's an identity rotation
# from world, and MuJoCo's world is Z-up by default with no <option gravity=...>
# override here). Confirmed, not assumed, by FKing the all-zero arm config:
# the flange sits at [+x forward, +y toward that arm's own side, +z above the
# pelvis origin] with an (to numerical noise) IDENTITY rotation relative to
# pelvis — i.e. at exactly zero, flange-local axes ARE pelvis-cardinal axes.
_CARDINAL = [
    ("+X (forward)",     np.array([1.0, 0.0, 0.0])),
    ("-X (backward)",    np.array([-1.0, 0.0, 0.0])),
    ("+Y (robot-left)",  np.array([0.0, 1.0, 0.0])),
    ("-Y (robot-right)", np.array([0.0, -1.0, 0.0])),
    ("+Z (up)",          np.array([0.0, 0.0, 1.0])),
    ("-Z (down)",        np.array([0.0, 0.0, -1.0])),
]


def _nearest_cardinal(v: np.ndarray) -> tuple[str, float]:
    """Unit vector `v` (pelvis frame) -> (nearest-cardinal label, angle off it)."""
    dots = [(label, float(np.dot(v, axis))) for label, axis in _CARDINAL]
    label, dot = max(dots, key=lambda t: t[1])
    return label, float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def _set_joint(q: np.ndarray, name: str, val: float) -> None:
    from ego2g1.core import layout as _layout
    for side in _layout.HANDS:
        joints = _layout.ARM_JOINTS[side]
        if name in joints:
            q[_layout.ARM_SLICE[side].start + joints.index(name)] = val
            return
    raise KeyError(f"unknown arm joint {name!r}")


def _tcp_check_configs() -> list[tuple[str, dict]]:
    """Ready pose + one +/-90 deg rotation per wrist axis, on top of ready.

    wrist_roll/pitch/yaw axes are literally "1 0 0" / "0 1 0" / "0 0 1" in
    assets/unitree_g1/g1_fixed_base.xml (local, pre-rotation) — and at ready,
    everything upstream of the wrist composes to identity (see _CARDINAL's
    comment), so rotating exactly one wrist joint by +/-90 deg rotates the
    flange by +/-90 deg about the matching pelvis-cardinal axis. That is what
    makes these configs "visibly different" and easy to eyeball against video.
    """
    half_pi = float(np.pi / 2)
    return [
        ("ready (NOMINAL_ARM_QPOS, all wrist joints at 0)", {}),
        ("wrist_roll +90 deg from ready",
         {"left_wrist_roll_joint": half_pi, "right_wrist_roll_joint": half_pi}),
        ("wrist_pitch +90 deg from ready",
         {"left_wrist_pitch_joint": half_pi, "right_wrist_pitch_joint": half_pi}),
        ("wrist_yaw +90 deg from ready",
         {"left_wrist_yaw_joint": half_pi, "right_wrist_yaw_joint": half_pi}),
        ("wrist_yaw -90 deg from ready",
         {"left_wrist_yaw_joint": -half_pi, "right_wrist_yaw_joint": -half_pi}),
    ]


# Per-axis physical meaning, derived (not guessed) from the sibling repo:
#   data_extraction_zh/src/ego_relation/s1_pico_mode2/tcp.py's
#   TCP_TO_INWARD_PALM matrices, combined with the palm-frame construction in
#   .../s1_pico_mode2/native_mode2.py's `_palm_frame` (read for context only —
#   ego2g1_v2 does not import from data_extraction_zh, that repo is a
#   separate uv project).
#
#   `palm_pose_to_tcp` computes result_rot = palm_rot @ TCP_TO_INWARD_PALM.T,
#   i.e. R_world_tcp = R_world_palm @ R_palm_tcp with R_palm_tcp =
#   TCP_TO_INWARD_PALM.T, so TCP axis i (expressed in palm coords) = ROW i of
#   TCP_TO_INWARD_PALM. `_palm_frame` returns columns (forward, lateral,
#   normal) where forward = wrist->middle-finger-knuckle (unit vector),
#   normal = cross(forward, across) (palm-plane normal), lateral =
#   cross(normal, forward). Reading off TCP_TO_INWARD_PALM's rows:
#     left:  row0=[1,0,0]->tcp_x=palm_x(forward)   row1=[0,0,1]->tcp_y=palm_z(normal)   row2=[0,-1,0]->tcp_z=-palm_y(-lateral)
#     right: row0=[1,0,0]->tcp_x=palm_x(forward)   row1=[0,0,-1]->tcp_y=-palm_z(-normal) row2=[0,1,0]->tcp_z=palm_y(lateral)
#   The per-side sign flips on y/z are exactly what's needed to keep the tcp_x
#   meaning ("forward", i.e. fingers direction) common to both hands, given
#   `_palm_frame`'s own left/right `across` sign flip.
#
# TCP +Z, re-derived directly from `_palm_frame`'s algebra (not guessed):
# `across` (right hand) = index_knuckle - pinky_knuckle, i.e. it POINTS FROM
# pinky TOWARD index. `lateral = cross(normal, forward)` is, by the BAC-CAB
# identity ((forward x across) x forward = across - (across.forward)forward),
# exactly `across` with its forward-component removed — i.e. lateral points
# the SAME direction as `across`: pinky -> index, for the right hand.
# `TCP_TO_INWARD_PALM["right"]` row 2 = [0,1,0] -> TCP+Z = +lateral = pinky ->
# index. For the left hand `across` is defined with the opposite sign
# (pinky_knuckle - index_knuckle, i.e. index -> pinky) AND
# TCP_TO_INWARD_PALM["left"] row 2 = [0,-1,0] -> TCP+Z = -lateral = -(index ->
# pinky) = pinky -> index again — the two per-hand sign choices are exactly
# what keeps TCP+Z meaning the SAME physical direction (pinky-side toward
# index/thumb-side) on both hands, mirroring how row 0 keeps TCP+X meaning
# "forward" on both hands. So TCP+Z is "pinky -> index", NOT "thumb -> pinky"
# — an earlier version of this rung had this backwards; if you see a printed
# "TCP +Z" axis pointing opposite to a naive "thumb toward pinky" read of the
# real hand, that is expected, not a bug (empirically confirmed against the
# real right hand at ready pose, 2026-07-31: predicted "up" via the OLD wrong
# label matched a real "down" thumb->pinky sweep exactly as this correction
# predicts).
#
# TCP +Y sign: empirically corroborated (not merely asserted) against the
# real right hand at ready pose, 2026-07-31 — predicted "robot-left", and the
# operator confirmed the hand orientation matches "toward the robot's own
# base," i.e. inward/robot-left. Re-check if a later observation on either
# hand disagrees; this was one data point (right hand only).
_TCP_AXIS_MEANING = [
    "fingers-forward: wrist -> middle-finger-knuckle direction (TCP +X, "
    "both hands)",
    "palm-plane normal, 'INWARD_PALM' per the convention's name — points "
    "toward the robot's own body/base (empirically confirmed on the real "
    "right hand at ready pose, 2026-07-31; TCP +Y, both hands after the "
    "per-side sign flip in TCP_TO_INWARD_PALM)",
    "pinky-knuckle -> index-knuckle direction (re-derived from "
    "_palm_frame's algebra, NOT 'thumb-to-pinky' — see the comment above "
    "this list; TCP +Z, both hands after the per-side sign flip in "
    "TCP_TO_INWARD_PALM)",
]
# Short forms of the above for the per-config, per-axis print lines (the full
# sentence is already printed once, in the header).
_TCP_AXIS_SHORT = [
    "fingers-forward, TCP +X",
    "palm-normal (toward robot base, confirmed), TCP +Y",
    "pinky-knuckle->index-knuckle, TCP +Z",
]


def tcp_orientation() -> None:
    """Print the FK flange orientation at a few illustrative arm configs,
    next to what the training-side TCP_TO_INWARD_PALM convention says the
    corresponding human-tracked TCP axis means. Offline, no dataset, no
    robot — only the MuJoCo model (same cost fk/ik already pay).

    This is NOT a pass/fail rung: there's no ground truth in this codebase to
    check against (the real correspondence between the flange and a
    physically mounted BrainCo hand's palm has never been measured — that's
    the open risk, see docs/relation_deploy_plan.md §1 item 1 / §4.4). Its
    output is meant to sit next to a photo or video of the real arm, in the
    same joint configuration, so a human can eyeball whether the printed
    "flange axis -> TCP meaning" correspondence actually looks right — the
    dangerous failure mode (a 90-degree-family axis mislabeling) is exactly
    the kind of thing that is obvious by eye and easy to miss in isolated
    unit math.
    """
    from ego2g1.core import layout
    from ego2g1.kin.g1 import NOMINAL_ARM_QPOS
    from ego2g1.deploy.core.kinematics import Kinematics

    kin = Kinematics()

    print("TCP-orientation sanity rung (offline, no robot, no dataset).\n")
    print("Flange = *_ee_site = wrist_yaw_link origin, ZERO offset "
          "(core/layout.py EE_SITES) — this is NOT a palm/hand-mount frame.")
    print("The fixed rotation (if any) between this flange and the actual "
          "mounted BrainCo palm has never been measured; this rung assumes "
          "IDENTITY (flange axis i == TCP axis i) — that assumption is "
          "exactly what a human must confirm by eye below.\n")
    print("Per-axis meaning, flange local axis -> hypothesized TCP meaning:")
    for i, meaning in enumerate(_TCP_AXIS_MEANING):
        print(f"  local +{'XYZ'[i]}  ->  {meaning}")
    print("\nPelvis-frame convention (confirmed from "
          "assets/unitree_g1/g1_fixed_base.xml's identity-quat pelvis body + "
          "numeric FK, not assumed): +X forward, +Y robot-left, +Z up. At "
          "the all-zero arm config the flange frame coincides with this "
          "pelvis frame exactly, so 'nearest pelvis axis' below is a direct "
          "physical-space reading there.\n")

    for name, overrides in _tcp_check_configs():
        q = np.zeros(layout.ARM_DOF)
        for jname, val in NOMINAL_ARM_QPOS.items():
            _set_joint(q, jname, val)
        for jname, val in overrides.items():
            _set_joint(q, jname, val)

        poses = kin.flange_poses(q)
        print(f"=== {name} ===")
        for h in layout.HANDS:
            T = poses[h]
            R, t = T[:3, :3], T[:3, 3]
            print(f"  {h:5s}  flange translation (pelvis frame, m): "
                  f"{np.round(t, 4)}")
            print(f"  {h:5s}  flange rotation matrix (pelvis frame):")
            for row in R:
                print(f"           {np.round(row, 4)}")
            for i in range(3):
                col = R[:, i]
                label, off_deg = _nearest_cardinal(col)
                print(f"    local +{'XYZ'[i]} ({_TCP_AXIS_SHORT[i]}): "
                      f"{np.round(col, 4)} -> nearest pelvis axis {label} "
                      f"(off by {off_deg:.1f} deg)")
        print()

    print("No automatic PASS/FAIL — this rung only prints. Compare this "
          "output against a photo/video of the real mounted BrainCo hand at "
          "the SAME joint configurations before trusting a relation_eef "
          "policy's rotations on hardware. A 90-degree-family mismatch "
          "(e.g. TCP +Y showing up where +Z was expected) is the failure "
          "mode this rung exists to catch.")
