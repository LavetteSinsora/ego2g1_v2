"""The bring-up ladder. Walk it in order; each rung gates the next.

Adapted from the old deploy's check.py (third_party/openpi/ego2g1/deploy) to
the vendored-executor architecture. Rungs 1/4/5/6/7 touch the robot; 2/3/8 do
not.

    python -m ego2g1.deploy.check listen      # 1. DDS only, no commands   [robot]
    python -m ego2g1.deploy.check fk          # 2. FK vs dataset state     [offline]
    python -m ego2g1.deploy.check ik          # 3. IK vs dataset joints    [offline]
    python -m ego2g1.deploy.check camera      # 4. one frame, to disk      [robot]
    python -m ego2g1.deploy.check hand-sweep  # 5. one finger at a time    [robot]
    python -m ego2g1.deploy.replay_dataset --dataset ...        # 6. stored JOINTS  [robot]
    python -m ego2g1.deploy.replay_dataset --dataset ... --from-eef  # 6b. eef->IK  [robot]
    python -m ego2g1.deploy.check replay-actions --dataset ...  # 7. ACTION labels  [robot]
    python -m ego2g1.deploy.check latency     # 8. round trip to the server [no robot]

Rungs 2 and 3 need no hardware and no checkpoint, and between them validate
joint order, the waist==0 assumption, the flange frame, the pelvis frame, the
vec9 encoding, and the IK — most of what can silently be wrong.

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


# --- 4. camera ---------------------------------------------------------------

def camera(host: str = "192.168.123.164", eye: str = "left",
           out: str = "check_camera.png") -> None:
    """Grab one frame and write it out. Then LOOK AT IT next to a training frame.

    This is the highest-risk open item in the deployment: the model trained on
    Pico-headset egocentric video, and a systematically different viewpoint
    fails quietly and looks like a bad policy."""
    import cv2

    from .camera import HeadCamera

    cam = HeadCamera(host=host, eye=eye)
    cam.connect()
    img = cam.read()
    cam.close()
    print(f"frame: {img.shape} {img.dtype}  range [{img.min()}, {img.max()}]")
    cv2.imwrite(out, img[..., ::-1])
    print(f"wrote {out} — compare against a training video frame before "
          "trusting a rollout.")


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
        "camera": camera,
        "hand-sweep": hand_sweep,
        "replay-actions": replay_actions,
        "latency": latency,
    })
