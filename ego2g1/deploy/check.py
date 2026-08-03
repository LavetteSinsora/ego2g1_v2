"""The bring-up ladder. Walk it in order; each rung gates the next.

Adapted from the old deploy's check.py (third_party/openpi/ego2g1/deploy) to
the vendored-executor architecture. Rungs 1/4/5/6/7 touch the robot; 2/3/8 do
not.

    python -m ego2g1.deploy.check listen      # 1. DDS only, no commands   [robot]
    python -m ego2g1.deploy.check fk          # 2. FK vs dataset state     [offline]
    python -m ego2g1.deploy.check ik          # 3. IK vs dataset joints    [offline]
    python -m ego2g1.deploy.check tcp-orientation  # 3b. TCP/flange convention [offline]
    python -m ego2g1.deploy.check camera      # 4. one frame, to disk      [robot]
    python -m ego2g1.deploy.check stereo-capture --out-dir calib_images
                                               # 4b. one auto-numbered stereo
                                               #     pair, for calibration    [robot]
    python -m ego2g1.deploy.check hand-sweep  # 5. one finger at a time    [robot]
    python -m ego2g1.deploy.check hand-jog --hand right
                                               # 5b. interactively close a hand
                                               #     around a real object, for
                                               #     BRAINCO_CLOSED_POSE       [robot]
    python -m ego2g1.deploy.replay_dataset --dataset ...        # 6. stored JOINTS  [robot]
    python -m ego2g1.deploy.replay_dataset --dataset ... --from-eef  # 6b. eef->IK  [robot]
    python -m ego2g1.deploy.check replay-actions --dataset ...  # 7. ACTION labels  [robot]
    python -m ego2g1.deploy.check latency     # 8. round trip to the server [no robot]

Rungs 2 and 3 need no hardware and no checkpoint, and between them validate
joint order, the waist==0 assumption, the flange frame, the pelvis frame, the
vec9 encoding, and the IK — most of what can silently be wrong.

Rung 3b is `EgoRelationTrainConfig`-specific (docs/relation_deploy_plan.md
§4.4): it does not touch a dataset or a checkpoint at all, only the MuJoCo
model, and exists to let a human eyeball whether the fixed
`TCP_TO_INWARD_PALM` rotation convention the relational training pipeline
assumes for the human-tracked palm (in the sibling `data_extraction_zh`
repo — never imported here) actually matches this robot's own flange
orientation once a BrainCo hand is mounted on it. That correspondence is an
assumption (plan §1 item 1), not something FK/IK can confirm on their own —
a wrong 90°-family axis label here would make a relational policy's
predicted rotations move the real arm the wrong way.

Rungs 6 and 7 both drive the real arm from a recording with the policy out of
the loop, and they are NOT the same test. 6 streams stored joints straight to
the executor: it never touches an action label and proves the plumbing (order,
sign, units, rates, hands, e-stop). 7 feeds the episode's ACTION-shaped deltas
through the real conversion path — measured-FK anchor, delta composition,
OneEuroSE3, mink IK, JointFilter, clamp — and proves the TRANSFORMS. A frame
or anchor bug leaves 6 perfect and shows up only in 7; run 6 first so 7 is
interpretable.
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from ..core import layout, se3
from . import actions as _actions
from .replay_dataset import load_episode


# --- 1. listen ---------------------------------------------------------------

def listen(iface: str | None = None, domain: int = 0, seconds: float = 5.0,
           hands: bool = True) -> None:
    """Subscribe only. No publishers, nothing commanded. Proves the DDS domain,
    the topic names, and that the Brainco bridge is actually running."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    if iface:
        ChannelFactoryInitialize(domain, iface)
    else:
        ChannelFactoryInitialize(domain)

    state = {"msg": None, "t": 0.0}

    def on_state(msg):
        state["msg"], state["t"] = msg, time.monotonic()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    hand_state = {}
    if hands:
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        for h in layout.HANDS:
            hand_state[h] = {"q": None, "t": 0.0}

            def make_cb(hh):
                def cb(msg):
                    hand_state[hh]["q"] = np.array(
                        [msg.states[i].q for i in range(layout.HAND_DIM)], np.float32)
                    hand_state[hh]["t"] = time.monotonic()
                return cb

            s = ChannelSubscriber(f"rt/brainco/{h}/state", MotorStates_)
            s.Init(make_cb(h), 10)

    t0 = time.monotonic()
    while state["msg"] is None:
        if time.monotonic() - t0 > 5.0:
            sys.exit("no rt/lowstate in 5 s — check the link / DDS domain / iface.")
        time.sleep(0.05)
    print(f"lowstate OK (age {(time.monotonic()-state['t'])*1000:.0f} ms)\n")

    # arm slots 15..28 (legs 0-11, waist 12-14) — G1_29_JointArmIndex order.
    arm_idx = list(range(15, 29))
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        q = np.array([state["msg"].motor_state[i].q for i in arm_idx])
        print(f"  arm q  L {np.round(q[:7], 3)}  R {np.round(q[7:], 3)}")
        for h in layout.HANDS:
            if hands:
                if hand_state[h]["q"] is None:
                    print(f"  hand {h:5s} NO STATE — is the Brainco bridge running?")
                else:
                    age = time.monotonic() - hand_state[h]["t"]
                    print(f"  hand {h:5s} {np.round(hand_state[h]['q'], 3)}  "
                          f"(age {age*1000:.0f} ms)")
        time.sleep(0.5)
    print("\nlisten OK — no commands were sent.")


# --- 2. fk -------------------------------------------------------------------

def fk(dataset: str, episode: int = 0, tol: float = 1e-4) -> None:
    """FK the dataset's stored joints and compare to its stored state.

    Validates joint order, waist==0, the flange site, the pelvis frame, and the
    vec9 encoding in one shot. No hardware, no checkpoint."""
    import pandas as pd
    import pathlib

    from .kinematics import Kinematics

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
    from .kinematics import Kinematics

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
    from ..core import layout as _layout
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
    from ..core import layout
    from ..kin.g1 import NOMINAL_ARM_QPOS
    from .kinematics import Kinematics

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


# --- 4. camera ---------------------------------------------------------------

def camera(host: str = "192.168.123.164", eye: str = "left",
           out: str = "check_camera.png", auto_start_server: bool = True) -> None:
    """Grab one frame and write it out. Then LOOK AT IT next to a training frame.

    This is the highest-risk open item in the deployment: the model trained on
    Pico-headset egocentric video, and a systematically different viewpoint
    fails quietly and looks like a bad policy.

    `auto_start_server=True` (default): SSH into `host` and start
    `image_server.py` there if it isn't already reachable, instead of just
    failing with "is it running?" -- see `remote_image_server.py`'s module
    docstring. Set False to get the old fail-fast behavior (e.g. if you're
    intentionally testing that the server ISN'T up)."""
    import cv2

    from .camera import HeadCamera

    cam = HeadCamera(host=host, eye=eye, auto_start_server=auto_start_server)
    cam.connect()
    img = cam.read()
    cam.close()
    print(f"frame: {img.shape} {img.dtype}  range [{img.min()}, {img.max()}]")
    cv2.imwrite(out, img[..., ::-1])
    print(f"wrote {out} — compare against a training video frame before "
          "trusting a rollout.")


# --- 4b. stereo-capture -------------------------------------------------------

def _next_pair_index(out_path) -> int:
    """The lowest non-negative integer N such that neither left_{N:03d}.png
    nor right_{N:03d}.png exists yet in `out_path` — so repeated invocations
    never collide or need the operator to track a counter by hand, and a
    manually deleted/re-shot pair doesn't leave a permanent gap unfilled."""
    existing = set()
    for p in out_path.glob("*_*.png"):
        stem_suffix = p.stem.rsplit("_", 1)[-1]
        if stem_suffix.isdigit():
            existing.add(int(stem_suffix))
    idx = 0
    while idx in existing:
        idx += 1
    return idx


def stereo_capture(host: str = "192.168.123.164", out_dir: str = "calib_images",
                   timeout: float = 10.0, auto_start_server: bool = True) -> None:
    """Grab ONE stereo pair (both eyes, from the same wire frame) and save it
    as an auto-numbered `left_NNN.png`/`right_NNN.png` pair -- the capture
    half of stereo calibration (docs/relation_deploy_plan.md §9 task 6b;
    `perception/stereo_calib.py`'s CLI consumes exactly this naming).

    Run this once per checkerboard position/tilt: hold the board somewhere
    new, run this command, repeat 15-20 times, covering different positions
    across the frame (not just the center — that's what teaches the solver
    about lens distortion, worst at the edges), different distances, and
    some tilt (not always dead-on). The pair index is picked automatically
    from whatever `left_*.png` files already exist in `out_dir` (the lowest
    unused number), so you never have to track `_000`, `_001`, ... by hand,
    and re-running after deleting a bad pair reuses that slot rather than
    leaving a permanent gap.

    Both eyes come from `HeadCamera.read_stereo()` (one wire frame, split in
    two) rather than two separate single-eye captures — the board is static,
    so a few milliseconds apart is fine, but the point is the two files are
    GUARANTEED to be the same physical moment, never at risk of drifting out
    of sync the way two independent `check camera --eye ...` calls could
    (e.g. if the operator moves the board between them without noticing).

    `auto_start_server=True` (default): SSH into `host` and start
    `image_server.py` there if it isn't already reachable, rather than just
    failing -- see `remote_image_server.py`'s module docstring. This is the
    point of this flag existing: run this command repeatedly across a whole
    capture session without ever manually SSHing in yourself.
    """
    import pathlib

    import cv2

    from .camera import HeadCamera

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    idx = _next_pair_index(out_path)

    cam = HeadCamera(host=host, eye="left", auto_start_server=auto_start_server)
    cam.connect(timeout=timeout)
    t0 = time.monotonic()
    stereo = cam.read_stereo()
    while stereo is None:
        if time.monotonic() - t0 > timeout:
            cam.close()
            raise TimeoutError(
                f"connected to {host} but never got a full stereo pair (both "
                f"eyes) within {timeout}s — the image_server may be sending "
                "only one eye; check its cam_config.")
        time.sleep(0.05)
        stereo = cam.read_stereo()
    cam.close()

    left, right = stereo
    left_path = out_path / f"left_{idx:03d}.png"
    right_path = out_path / f"right_{idx:03d}.png"
    cv2.imwrite(str(left_path), left[..., ::-1])
    cv2.imwrite(str(right_path), right[..., ::-1])
    print(f"pair #{idx}: wrote {left_path} and {right_path}  ({left.shape} {left.dtype})")


# --- 5. hand sweep -------------------------------------------------------------

def hand_sweep(iface: str | None = None, domain: int = 0, hand: str = "right",
               motor: int = 2, lo: float = 0.0, hi: float = 0.6,
               seconds: float = 4.0) -> None:
    """Drive ONE Brainco motor slowly between two commands, watching the arm not
    at all. Commands are [0, 1] (0=open, 1=closed) — that much is settled. What
    this rung resolves is the ORDER: whether HAND_MOTOR_ORDER [thumb_flex,
    thumb_rot, index, middle, ring, pinky] maps 1:1 onto Brainco's [Thumb,
    ThumbAux, Index, Middle, Ring, Pinky]. If commanding `motor` moves a
    different finger, fix the mapping before any policy runs."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

    if iface:
        ChannelFactoryInitialize(domain, iface)
    else:
        ChannelFactoryInitialize(domain)

    name = layout.HAND_MOTOR_ORDER[motor]
    print(f"sweeping {hand} motor {motor} ({name}) between {lo} and {hi}")
    print("WATCH THE HAND. Which finger actually moves?\n")

    pubs, msgs = {}, {}
    for h in layout.HANDS:
        pubs[h] = ChannelPublisher(f"rt/brainco/{h}/cmd", MotorCmds_)
        pubs[h].Init()
        m = MotorCmds_()
        m.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(layout.HAND_DIM)]
        for i in range(layout.HAND_DIM):
            m.cmds[i].q = 0.0
            m.cmds[i].dq = 1.0   # vendor uses dq as a speed field here
        msgs[h] = m

    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            phase = 0.5 - 0.5 * np.cos(
                2 * np.pi * (time.monotonic() - t0) / seconds * 2)
            msgs[hand].cmds[motor].q = float(lo + (hi - lo) * phase)
            for h in layout.HANDS:
                pubs[h].Write(msgs[h])
            print(f"  cmd {msgs[hand].cmds[motor].q:.3f}", end="\r")
            time.sleep(1 / 200)
    finally:
        for h in layout.HANDS:
            for i in range(layout.HAND_DIM):
                msgs[h].cmds[i].q = 0.0
            pubs[h].Write(msgs[h])
        print("\n\nreturned to open.")


# --- 5b. hand jog (BRAINCO_CLOSED_POSE measurement) -----------------------------

def _read_key(timeout: float = 0.05):
    """Non-blocking single keypress from stdin, or None if nothing arrived
    within `timeout` -- the standard termios/select technique for
    interactive terminal control without a third-party dependency (matches
    this repo's existing no-new-deps-for-a-CLI-rung discipline)."""
    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else None


def hand_jog(iface: str | None = None, domain: int = 0, hand: str = "right",
            step: float = 0.02) -> None:
    """Interactively jog all 6 BrainCo motors for ONE hand via the keyboard,
    watching the real hand close around a real object, until it's exactly
    the grip you want -- then prints the final (6,) vector formatted ready
    to paste into `ego2g1/deploy/gripper_calib.py`'s
    `BRAINCO_CLOSED_POSE[hand]` (docs/relation_deploy_plan.md §7: there is
    no principled way to derive this from the binary training signal, a
    human has to measure it on the real hand around the real task objects).

    Keys (case-insensitive):
      1-6    select which motor to adjust -- HAND_MOTOR_ORDER order:
             1=thumb_flex 2=thumb_rot 3=index 4=middle 5=ring 6=pinky
      j/k    decrease / increase the SELECTED motor by `step` (clamped [0,1])
      o      open this hand fully (all motors -> 0.0) -- safety reset
      c      close this hand fully (all motors -> 1.0) -- coarse starting point
      p      print the current (6,) vector without quitting
      q      quit -- prints the final vector once more and exits
    Ctrl-C also exits cleanly, same as 'q'.

    The OTHER hand is held open throughout and is never touched. Commands
    publish continuously (not just on keypress) at the same rate
    `hand_sweep` uses -- the Brainco driver holds stale state if it isn't
    commanded every tick (see `actions.py`'s `JointChunks` docstring), so a
    live interactive tool must keep publishing even while idle between
    keypresses, not just when a key changes something.
    """
    import termios
    import tty

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

    if hand not in layout.HANDS:
        raise ValueError(f"hand must be one of {layout.HANDS}, got {hand!r}")

    if iface:
        ChannelFactoryInitialize(domain, iface)
    else:
        ChannelFactoryInitialize(domain)

    pubs, msgs = {}, {}
    for h in layout.HANDS:
        pubs[h] = ChannelPublisher(f"rt/brainco/{h}/cmd", MotorCmds_)
        pubs[h].Init()
        m = MotorCmds_()
        m.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(layout.HAND_DIM)]
        for i in range(layout.HAND_DIM):
            m.cmds[i].q = 0.0
            m.cmds[i].dq = 1.0
        msgs[h] = m

    selected = 0

    def _status_line() -> str:
        vals = ", ".join(
            f"{'*' if i == selected else ' '}{name}={msgs[hand].cmds[i].q:.3f}"
            for i, name in enumerate(layout.HAND_MOTOR_ORDER)
        )
        return f"  [{hand}] {vals}"

    def _print_vector() -> None:
        vec = [round(float(msgs[hand].cmds[i].q), 3) for i in range(layout.HAND_DIM)]
        print(f"\n\nBRAINCO_CLOSED_POSE[{hand!r}] = np.array({vec}, dtype=np.float32)"
              f"   # {layout.HAND_MOTOR_ORDER}")

    print(f"Jogging {hand} hand. Keys: 1-6 select motor, j/k -/+ step, "
          "o open all, c close all, p print, q quit.")
    print(f"Motor order: {layout.HAND_MOTOR_ORDER}\n")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key()
            if key:
                key = key.lower()
                if key == "q":
                    break
                if key in "123456":
                    selected = int(key) - 1
                elif key == "j":
                    msgs[hand].cmds[selected].q = float(
                        np.clip(msgs[hand].cmds[selected].q - step, 0.0, 1.0))
                elif key == "k":
                    msgs[hand].cmds[selected].q = float(
                        np.clip(msgs[hand].cmds[selected].q + step, 0.0, 1.0))
                elif key == "o":
                    for i in range(layout.HAND_DIM):
                        msgs[hand].cmds[i].q = 0.0
                elif key == "c":
                    for i in range(layout.HAND_DIM):
                        msgs[hand].cmds[i].q = 1.0
                elif key == "p":
                    _print_vector()

            for h in layout.HANDS:
                pubs[h].Write(msgs[h])
            print(_status_line(), end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        _print_vector()


# --- 7. replay the ACTION labels through the real conversion path ---------------

def replay_actions(dataset: str, episode: int = 0, fps: int = 30,
                   horizon: int = 50, ik_iters: int = 25,
                   posture_cost: float = 0.05, max_step: float = 0.15,
                   network_interface: str | None = None,
                   max_pos_speed: float | None = None,
                   dry_run: bool = False, yes: bool = False,
                   out: str = "replay_actions.npz") -> None:
    """Drive the arm from ACTION-shaped chunks with the policy replaced by the
    recording: at each chunk start, read the MEASURED arm, anchor there, build
    the chunk's deltas from the stored poses (delta_k = T(t0)⁻¹ T(t0+k) — what
    a perfect policy would output), and run the real conversion (OneEuroSE3 ->
    IK posture-tracks-last -> JointFilter) + clamp + executor. Rung 6 proves
    the plumbing; this proves the transforms."""
    from . import safety as _safety
    from .actions import RelativeEEFChunks

    ep = load_episode(dataset, episode)
    n = len(ep["arm"])
    print(f"{ep['name']}: {n} frames @ {fps} Hz, chunks of {horizon}")

    if dry_run:
        from .executor import MockExecutor
        executor = MockExecutor(fps=fps, initial_q=ep["arm"][0])
    else:
        from .executor import UnitreeExecutor
        executor = UnitreeExecutor(fps=fps, network_interface=network_interface,
                                   max_pos_speed=max_pos_speed)
        if not yes and input(
                "replay the action labels on the REAL arm? [y/N] "
                ).strip().lower() != "y":
            return
    executor.connect()

    converter = RelativeEEFChunks(fps=fps, ik_iters=ik_iters,
                                  posture_cost=posture_cost)
    clamp = _safety.Clamp(_safety.SafetyLimits(max_joint_step=max_step))

    dt = 1.0 / fps
    log_cmd, log_meas = [], []
    row = np.zeros(_actions.ROBOT_DIM)
    try:
        # soft-ramp to the start via the vendor's first-send drive_to_waypoint
        row[_actions.ARM] = ep["arm"][0]
        for h in layout.HANDS:
            row[_actions.HAND[h]] = ep["hand"][h][0]
        executor.send(row)
        if not dry_run:
            time.sleep(2.0)
        clamp.reset(executor.arm_q())

        from .runner import precise_wait
        t_wall = time.monotonic()
        for t0_idx in range(0, n - 1, horizon):
            k_max = min(horizon, n - 1 - t0_idx)
            arm_q = executor.arm_q()
            hand_cmds = {h: ep["hand"][h][t0_idx] for h in layout.HANDS}
            # what a perfect policy would output against this anchor
            chunk = np.zeros((k_max, layout.DIM))
            for h in layout.HANDS:
                T0 = se3.vec9_to_se3(ep["pose"][h][t0_idx])
                for k in range(k_max):
                    Tk = se3.vec9_to_se3(ep["pose"][h][t0_idx + 1 + k])
                    chunk[k, layout.EEF[h]] = se3.se3_to_vec9(se3.se3_inv(T0) @ Tk)
                    chunk[k, layout.HAND[h]] = ep["hand"][h][t0_idx + 1 + k]
            joints = converter.convert(chunk, arm_q, hand_cmds)
            print(f"  chunk @ {t0_idx}: IK worst "
                  f"{converter.last_tracking_error*1000:.1f} mm")
            for k in range(k_max):
                t_cycle_end = t_wall + dt
                joints[k, _actions.ARM] = clamp(joints[k, _actions.ARM], dt)
                executor.send(joints[k], t_cycle_end + dt)
                log_cmd.append(joints[k, _actions.ARM].copy())
                log_meas.append(executor.arm_q())
                precise_wait(t_cycle_end)
                t_wall = t_cycle_end
        print("replay complete.")
    except KeyboardInterrupt:
        print("\ninterrupted — DAMPING.")
        executor.damp()
    finally:
        executor.close()

    if log_cmd:
        cmd, meas = np.stack(log_cmd), np.stack(log_meas)
        err = np.abs(cmd - meas)
        print(f"\ntracking: mean {err.mean():.4f} rad   max {err.max():.4f} rad")
        print(f"clamped ticks: {clamp.clamped_ticks}  "
              f"(max step seen {clamp.max_seen:.3f} rad)")
        np.savez(out, q_cmd=cmd, q_meas=meas, episode=episode, fps=fps)
        print(f"wrote {out}")


# --- 8. policy-server latency ---------------------------------------------------

def latency(host: str = "127.0.0.1", port: int = 8000, n: int = 20,
            frame_hw: tuple[int, int] = (480, 640)) -> None:
    """Time the round trip to the policy server. No robot, no camera.

    Run it TWICE: on the server box (127.0.0.1) and on the deploy machine. The
    server-local number is pure inference; the difference is what the network
    costs. p95 is the number that matters — the budget is a cliff, not a
    gradient (latency.budget_for). The first call includes an XLA compile
    (minutes cold) and is reported separately: never let a policy's first-ever
    request happen with the robot in the loop."""
    from . import client as _client
    from . import latency as _latency

    c = _client.PolicyClient(host, port)
    frame = np.random.randint(0, 255, (*frame_hw, 3), dtype=np.uint8)
    state = np.zeros(30, dtype=np.float32)

    print(f"\nserver {host}:{port} | horizon {c.action_horizon} "
          f"dim {c.action_dim} fps {c.fps} control_mode {c.control_mode}")

    first, samples = _latency.measure_policy_latency(
        lambda: c.infer(frame, state, "latency check"), n)
    lat = np.array(samples)
    p95 = float(np.quantile(lat, 0.95))
    print(f"first call (includes XLA compile): {first:.1f} s")
    print(f"steady: mean {lat.mean()*1000:.0f} ms   p95 {p95*1000:.0f} ms   "
          f"max {lat.max()*1000:.0f} ms\n")

    for mode in ("sync", "async", "temporal_smoothing"):
        b = _latency.budget_for(mode, fps=c.fps, horizon=c.action_horizon,
                                inference_hz=4.0, max_latency_steps=8)
        verdict = ("no hard budget (holds during inference)" if b is None else
                   ("OK, %.0f ms headroom" % ((b - p95 * 1.15) * 1000)
                    if p95 * 1.15 <= b else
                    "OVER BUDGET — the runner will REFUSE this mode"))
        print(f"  {mode:20s} budget "
              f"{'—' if b is None else '%4.0f ms' % (b*1000)}   {verdict}")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    tyro.extras.subcommand_cli_from_dict({
        "listen": listen,
        "fk": fk,
        "ik": ik,
        "tcp-orientation": tcp_orientation,
        "camera": camera,
        "stereo-capture": stereo_capture,
        "hand-sweep": hand_sweep,
        "hand-jog": hand_jog,
        "replay-actions": replay_actions,
        "latency": latency,
    })
