"""The deploy loop: observe -> strategy -> clamp -> executor, at fps, paced.

Loop semantics ported from zh_deploy_inference/examples/unitree_inference/
runner.py plus unitree_deploy's own UnitreeEnv.step (real_unitree_env.py):

  * busy-wait pacing (`precise_wait`): time.sleep overshoots by milliseconds
    (worse on macOS); the last ~1 ms is spun so ticks land where the
    interpolator expects them.
  * FUTURE-stamped targets: each waypoint is stamped t_cycle_end + control_dt
    — one period past the end of this cycle — so the vendored 500 Hz
    interpolator always interpolates toward a point ahead of it and never
    extrapolates. Copied exactly; do not "simplify" it to `now`.
  * the first commanded motion is unitree_deploy's own drive_to_waypoint soft
    ramp (its robot.send_action issues drive_to_waypoint on the first call,
    schedule_waypoint after — see robot/robot.py `initial_data_received`).

What this runner adds over zh's (each measured-motivated, see
docs/jitter_root_cause.md):

  * the startup latency self-check — REFUSES to start a timing strategy whose
    budget the measured inference latency cannot honor (latency.py),
  * the safety layer: per-tick joint-delta clamp, watchdog on state/camera
    staleness + plan starvation + IK tracking error, all tripping to damp(),
  * the recorder: every seam into events.jsonl.

Live entry point (see docs/deploy.md for the full rung ladder first):

    python -m ego2g1.deploy.runner --host <serve-box> --port 8000 \
        --prompt "put the bottle in the box" --mode sync

With `--dashboard` the start is GATED: the loop launches idle (holding) and the
web page's Start/Pause/Record/Reset/E-STOP buttons drive the lifecycle
(begin/pause/record_toggle/reset_to_episode/estop below); `--ungated` restores
the terminal Enter-to-start. `--dataset <lerobot root>` arms reset-to-episode.

Dry run, no robot, no camera (mock executor; needs only the policy server):

    python -m ego2g1.deploy.runner --host 127.0.0.1 --prompt "x" --dry-run
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time

import numpy as np

from ..core import layout
from . import actions as _actions
from . import latency as _latency
from . import recorder as _recorder
from . import safety as _safety
from . import strategies as _strategies

logger = logging.getLogger(__name__)


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    """Sleep coarsely, spin the last `slack_time`. Verbatim from
    unitree_deploy.robot_devices.robots_devices_utils.precise_wait (local copy
    so the runner logic imports without the vendored package)."""
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass


class DeployRunner:
    """One rollout. Everything is injected so tests run it hardware-free."""

    def __init__(self, *, adapter, strategy, executor, camera=None,
                 recorder=None, limits: _safety.SafetyLimits | None = None,
                 fps: int = 30, max_steps: int = 10_000_000,
                 gated: bool = False, dataset: str | None = None,
                 clock=time.monotonic, wait=precise_wait):
        self.adapter = adapter
        self.strategy = strategy
        self.executor = executor
        self.camera = camera
        self.recorder = recorder or _recorder.NullRecorder()
        self.limits = limits or _safety.SafetyLimits()
        self.fps = int(fps)
        self.dt = 1.0 / self.fps
        self.max_steps = int(max_steps)
        self.dataset = dataset     # lerobot dataset root; enables reset_to_episode
        self._clock = clock
        self._wait = wait

        self.clamp = _safety.Clamp(self.limits)
        self.watchdog = _safety.Watchdog(self.limits, self._on_trip)
        # Hands start OPEN and thereafter track what we last COMMANDED — the
        # model's state hand-block is the command stream, never encoders.
        self.last_hands = {h: np.zeros(layout.HAND_DIM) for h in layout.HANDS}
        self.steps_executed = 0
        self.running = False       # set/cleared around run(); read by telemetry()
        # Gate between idle (holding pose) and active (observe->infer->pop).
        # `gated` launches idle: the dashboard's Start button calls begin().
        self._active = threading.Event()
        if not gated:
            self._active.set()
        self._ctrl_lock = threading.Lock()   # serializes begin() vs a reset ramp

    # --- seams ----------------------------------------------------------------

    def _on_trip(self) -> None:
        self.recorder.log("estop", reason=self.watchdog.reason)
        self.executor.damp()

    def _observe(self) -> dict:
        arm_q = self.executor.arm_q()
        image = self.camera.read() if self.camera is not None else None
        return {"arm_q": arm_q,
                "hand_cmds": {h: self.last_hands[h].copy() for h in layout.HANDS},
                "image": image,
                "prompt": getattr(self.adapter, "prompt", "")}

    # --- the loop ---------------------------------------------------------------

    def run(self) -> None:
        self.clamp.reset(self.executor.arm_q())
        self.running = True
        was_idle = True     # first activation re-arms too (clears probe state)
        step = 0
        try:
            while step < self.max_steps:
                # 0. gate (dashboard Start/Pause). Idle is HOLDING, not stopped:
                # the vendored 500 Hz interpolator keeps the last waypoint. We
                # keep reading the state so a link that dies while idle still
                # trips instead of leaving a stiff arm unsupervised.
                if not self._active.is_set():
                    was_idle = True
                    self.executor.arm_q()      # keeps state_age() meaningful
                    self.watchdog.check_state_age(self.executor.state_age())
                    if self.watchdog.tripped:
                        break
                    time.sleep(0.05)
                    continue
                if was_idle:
                    was_idle = False
                    self._rearm("begin")

                t_cycle_end = self._clock() + self.dt
                t_command_target = t_cycle_end + self.dt   # future-stamped

                # 1. observe (+ staleness checks)
                self.watchdog.check_state_age(self.executor.state_age())
                if self.camera is not None:
                    self.watchdog.check_camera_age(self.camera.age())
                if self.watchdog.tripped:
                    break
                observation = self._observe()
                self.recorder.log("obs", step=step,
                                  state_age=self.executor.state_age(),
                                  arm_q=observation["arm_q"])
                self.strategy.update_observation(observation)

                # 2. wait for a plan (starvation is duration-based, not a spin)
                while not self.strategy.has_action():
                    self.watchdog.check_starvation(False, self._clock())
                    if self.watchdog.tripped:
                        break
                    time.sleep(0.001)
                if self.watchdog.tripped:
                    break
                self.watchdog.check_starvation(True, self._clock())

                # 3. pop -> sanity -> clamp -> send
                row = np.asarray(self.strategy.pop_action(), dtype=np.float64)
                if not _safety.sanity_check_joint_row(row):
                    self.watchdog.trip(f"insane joint row at step {step}")
                    break
                before = row[_actions.ARM].copy()
                row[_actions.ARM] = self.clamp(row[_actions.ARM], self.dt)
                if not np.array_equal(before, row[_actions.ARM]):
                    self.recorder.log("clamp", step=step,
                                      max_step=float(np.abs(
                                          before - row[_actions.ARM]).max()))
                self.executor.send(row, t_command_target)
                for i, h in enumerate(layout.HANDS):
                    self.last_hands[h] = row[_actions.HAND[h]].copy()
                self.recorder.log("action", step=step, row=row)
                self.steps_executed = step + 1

                # 4. per-chunk IK tracking error (relative_eef only)
                err = getattr(self.adapter, "last_tracking_error", None)
                if err is not None:
                    self.watchdog.check_tracking(float(err))
                    if err > 0:
                        self.recorder.log("tracking", step=step, worst_m=float(err))

                # 5. pace
                self._wait(t_cycle_end)
                step += 1
                if step % 50 == 0:
                    logger.info("executed %d steps", step)
        finally:
            self.running = False
            self.strategy.close()
            if self.watchdog.tripped:
                logger.error("stopped by watchdog: %s", self.watchdog.reason)

    def _rearm(self, why: str) -> None:
        """Drop stale plans and re-ground after time passed outside the loop
        (pause, reset ramp): the world moved on; the old chunk, the causal
        filters, and the clamp's last knot did not."""
        if hasattr(self.strategy, "clear"):
            self.strategy.clear()
        reset = getattr(self.adapter, "reset", None)
        if callable(reset):
            reset()
        self.clamp.reset(self.executor.arm_q())
        # starvation timing must restart from NOW, not from before the idle
        self.watchdog.check_starvation(True, self._clock())
        self.recorder.log("rearm", why=why)

    # --- lifecycle controls (deploy/dashboard.py POSTs) -------------------------

    @property
    def is_active(self) -> bool:
        return self._active.is_set()

    def begin(self) -> None:
        """Leave idle (the dashboard's Start): the loop resumes observing and
        inferring; the actual re-arm happens on the loop thread."""
        if self.watchdog.tripped:
            raise RuntimeError("watchdog is tripped; nothing runs until restart")
        if not self._ctrl_lock.acquire(blocking=False):
            raise RuntimeError("busy — a reset ramp is in progress")
        try:
            self._active.set()
        finally:
            self._ctrl_lock.release()
        logger.info("loop ACTIVE — inferring")

    def pause(self) -> None:
        """Return to idle. NOT an e-stop: the vendored interpolator keeps
        HOLDING the last waypoint (the arm stays stiff); planning just stops."""
        self._active.clear()
        logger.info("loop IDLE — holding pose")

    def record_toggle(self) -> dict:
        """Roll the recording session boundary (the dashboard's Record button).
        Needs the RecorderSwitch the CLI wires in; plain recorders can't swap."""
        toggle = getattr(self.recorder, "toggle", None)
        if not callable(toggle):
            raise RuntimeError("record toggle needs a RecorderSwitch "
                               "(run via `python -m ego2g1.deploy.runner`)")
        return toggle()

    def reset_to_episode(self, episode_index: int, *, ramp_s: float = 3.0,
                         max_speed: float = 0.5, settle_s: float = 0.4) -> dict:
        """Ramp arm+hands to an episode's FIRST posture and re-arm from rest.

        Must be idle: yanking the arm while a policy drives it would fight the
        planner. The ramp streams future-stamped waypoints through the SAME
        executor.send path the loop uses — linear interpolation at fps, ramp
        lengthened if `ramp_s` would break the `max_speed` (rad/s) cap. Blocks
        until settled, so the dashboard's response means 'done'."""
        if self._active.is_set():
            raise RuntimeError("pause before resetting — cannot reset while active")
        if self.watchdog.tripped:
            raise RuntimeError("watchdog is tripped; nothing runs until restart")
        if self.dataset is None:
            raise RuntimeError("no --dataset given; reset-to-episode needs one")
        if not self._ctrl_lock.acquire(blocking=False):
            raise RuntimeError("busy — another reset is in progress")
        try:
            from .replay_dataset import load_episode

            ep = load_episode(self.dataset, int(episode_index))
            q_start = np.asarray(ep["arm"][0], dtype=np.float64)
            hand_start = {h: np.clip(np.asarray(ep["hand"][h][0], np.float64), 0.0, 1.0)
                          for h in layout.HANDS}
            q0 = self.executor.arm_q()
            h0 = {h: self.last_hands[h].copy() for h in layout.HANDS}

            n = max(1, int(round(ramp_s * self.fps)))
            cap = max_speed * self.dt                    # rad per tick
            worst = float(np.abs(q_start - q0).max())
            if cap > 0:
                n = max(n, int(np.ceil(worst / cap)))
            row = np.zeros(_actions.ROBOT_DIM)
            for k in range(1, n + 1):
                if self.watchdog.tripped:
                    raise RuntimeError("tripped during the reset ramp")
                t_cycle_end = self._clock() + self.dt
                a = k / n
                row[_actions.ARM] = (1.0 - a) * q0 + a * q_start
                for h in layout.HANDS:
                    row[_actions.HAND[h]] = (1.0 - a) * h0[h] + a * hand_start[h]
                self.executor.send(row, t_cycle_end + self.dt)
                self._wait(t_cycle_end)
            # let the 500 Hz interpolator land + the arm settle, then re-ground
            self._wait(self._clock() + settle_s)

            arm_q = self.executor.arm_q()
            for h in layout.HANDS:
                self.last_hands[h] = hand_start[h].copy()
            self._rearm(f"reset_to_episode {int(episode_index)}")
            residual = float(np.abs(arm_q - q_start).max())
            self.recorder.log("reset", episode=int(episode_index),
                              q_start=q_start, residual=residual)
            logger.info("reset to episode %d (residual %.3f rad)",
                        episode_index, residual)
            return {"episode": int(episode_index), "residual": residual}
        finally:
            self._ctrl_lock.release()

    # --- observability (deploy/dashboard.py) -----------------------------------

    def estop(self, reason: str = "external") -> None:
        """Trip the watchdog from outside the loop (the dashboard's E-STOP
        button). Routes through the normal trip path: recorder + damp()."""
        self.watchdog.trip(reason)

    def telemetry(self) -> dict:
        """A JSON-serializable snapshot for the live dashboard. Called ONLY
        from the dashboard's HTTP thread (~10 Hz), never from the loop. Pure
        pull: the strategy/executor telemetry() below read existing state
        under their own existing locks and copy small arrays — the loop body,
        the inference worker, and the vendored 500 Hz thread gain no code."""
        now = self._clock()
        st = self.strategy.telemetry() if hasattr(self.strategy, "telemetry") else {}
        ex = self.executor.telemetry() if hasattr(self.executor, "telemetry") else {}
        ready = bool(st.get("ready"))
        horizon = int(st.get("horizon") or 0)
        index = int(st.get("index") or 0)
        budget = st.get("budget")
        groups = [{"label": "L-arm", "start": 0, "stop": 7},
                  {"label": "R-arm", "start": 7, "stop": _actions.ARM_DOF}]
        for h in layout.HANDS:
            groups.append({"label": f"{h[0].upper()}-hand",
                           "start": _actions.HAND[h].start,
                           "stop": _actions.HAND[h].stop})
        return {
            "now": now,
            "mode": st.get("mode", "?"),
            "server_rtc": bool(st.get("rtc")),
            "active": (self.running and self._active.is_set()
                       and not self.watchdog.tripped),
            "recording": bool(getattr(
                self.recorder, "recording",
                not isinstance(self.recorder, _recorder.NullRecorder))),
            "has_dataset": self.dataset is not None,   # enables Reset button
            "task": getattr(self.adapter, "prompt", ""),
            "horizon": horizon, "fps": self.fps, "dim": _actions.ROBOT_DIM,
            # --- the strategy's core data structure ---
            "ready": ready,
            "index": index,
            # pop and send happen in the SAME tick here: robot-now == pointer
            "wall_slot": index if ready else None,
            "trigger": st.get("trigger"),
            "d": st.get("d") if st.get("d") is not None
                 else (budget or {}).get("d"),
            "action_row": ex.get("last_row"),
            "row_slot": max(0, index - 1) if ready else None,
            "groups": groups,
            # --- inference lifecycle ---
            "inferring": bool(st.get("inferring")),
            "pending": bool(st.get("pending")),
            "worker_dead": bool(st.get("worker_dead")),
            "last_splice": {},
            # --- health / timing ---
            "stats": {"ticks": int(self.steps_executed),
                      "chunks": (budget or {}).get("n"),
                      "votes": st.get("votes")},
            "budget": budget,
            "runway_s": (horizon - index) / self.fps if ready else None,
            "camera_age": (float(self.camera.age())
                           if self.camera is not None else None),
            "clamped_ticks": int(self.clamp.clamped_ticks),
            "watchdog": {"tripped": bool(self.watchdog.tripped),
                         "reason": self.watchdog.reason},
            # --- executor ---
            "arm_q": ex.get("arm_q"),
            "state_age": ex.get("state_age"),
            "estopped": bool(ex.get("estopped")),
        }


# --- CLI assembly -----------------------------------------------------------------


@dataclasses.dataclass
class Args:
    # --- policy server ---
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = ""
    # --- strategy ---
    mode: str = "sync"                 # strategies.MODES
    inference_hz: float = 4.0
    exp_weight_m: float = 0.01
    max_latency_steps: int = 8
    min_smooth_steps: int = 10
    rtc_execute_horizon: int | None = None
    # --- action mode: "auto" reads the checkpoint's control_mode from the
    # server handshake; override only to test a mismatched pairing on purpose.
    action_mode: str = "auto"          # auto | joint | relative_eef
    ik_iters: int = 25
    posture_cost: float = 0.05         # the measured smoothness knob
    collision_min_dist: float = 0.005
    # --- robot ---
    network_interface: str | None = None   # DDS iface; None joins the default domain
    fps: int = 0                       # 0 = the checkpoint's own fps
    max_pos_speed: float | None = None # soften the interpolator cap for bring-up
    camera_host: str = "192.168.123.164"
    eye: str = "left"
    # --- observability ---
    dashboard: bool = False            # live web monitor (deploy/dashboard.py)
    dashboard_port: int = 8080
    ungated: bool = False              # --dashboard implies a gated start (idle
                                       # until the page's Start); this opts out
    # --- session ---
    record_dir: str = "recordings"
    no_record: bool = False            # don't auto-open a session (the
                                       # dashboard's Record button still can)
    dataset: str | None = None         # lerobot dataset root; enables the
                                       # dashboard's reset-to-episode
    max_steps: int = 10_000_000
    dry_run: bool = False              # mock executor + static camera, no robot
    skip_latency_check: bool = False   # ONLY for offline debugging; never live
    # --- safety ---
    max_joint_step: float = 0.15
    max_state_age: float = 0.2
    max_starvation: float = 2.0
    max_track_err: float = 0.10        # metres of IK tracking error before the e-stop


def main(args: Args) -> None:
    from . import client as _client
    from . import policy_adapter as _policy_adapter

    client = _client.PolicyClient(args.host, args.port)
    fps = args.fps or client.fps
    horizon = client.action_horizon

    action_mode = args.action_mode
    if action_mode == "auto":
        action_mode = "joint" if client.control_mode == "joint" else "relative_eef"
    adapter = (_policy_adapter.make_adapter(
        "relative_eef", client, args.prompt, ik_iters=args.ik_iters,
        posture_cost=args.posture_cost, collision_min_dist=args.collision_min_dist)
        if action_mode == "relative_eef"
        else _policy_adapter.make_adapter("joint", client, args.prompt))
    logger.info("action mode: %s (server control_mode=%s)", action_mode,
                client.control_mode)

    # --- executor + camera
    if args.dry_run:
        from .camera import StaticCamera
        from .executor import MockExecutor
        executor, cam = MockExecutor(fps=fps), StaticCamera()
    else:
        from .camera import HeadCamera
        from .executor import UnitreeExecutor
        executor = UnitreeExecutor(fps=fps, network_interface=args.network_interface,
                                   max_pos_speed=args.max_pos_speed)
        cam = HeadCamera(host=args.camera_host, eye=args.eye)
    cam.connect()

    # --- recorder: always a RecorderSwitch, so the dashboard's Record button
    # can roll session boundaries; --no-record just skips the auto-open.
    rec = _recorder.RecorderSwitch(
        args.record_dir, args.prompt or "deploy", cameras={"head": cam}, meta={
            "mode": args.mode, "action_mode": action_mode, "horizon": horizon,
            "fps": fps, "host": args.host, "port": args.port,
            "prompt": args.prompt, "dim": _actions.ROBOT_DIM,
            # strategy params, so replay_record.py can rebuild the exact buffer
            "inference_hz": args.inference_hz,
            "exp_weight_m": args.exp_weight_m,
            "max_latency_steps": args.max_latency_steps,
            "min_smooth_steps": args.min_smooth_steps,
        })
    if not args.no_record:
        rec.start()

    budget = _latency.DelayBudget(fps)
    strategy = _strategies.make_strategy(
        args.mode, adapter, chunk_size=horizon, inference_hz=args.inference_hz,
        exp_weight_m=args.exp_weight_m, max_latency_steps=args.max_latency_steps,
        min_smooth_steps=args.min_smooth_steps, control_hz=fps,
        rtc_execute_horizon=args.rtc_execute_horizon, recorder=rec, budget=budget)

    limits = _safety.SafetyLimits(max_joint_step=args.max_joint_step,
                                  max_state_age=args.max_state_age,
                                  max_starvation=args.max_starvation,
                                  max_tracking_error_m=args.max_track_err)

    try:
        # --- connect BEFORE the latency check: the check must see the real
        # state and a real-sized camera frame (the wire cost is the point).
        # NOTE unitree_deploy's connect() runs its own soft drive_to_waypoint
        # ramp to the configured init pose — expect the arm to move here.
        executor.connect()

        if args.skip_latency_check:
            logger.warning("latency self-check SKIPPED — never do this on hardware")
        else:
            probe = {"arm_q": executor.arm_q(),
                     "hand_cmds": {h: np.zeros(layout.HAND_DIM) for h in layout.HANDS},
                     "image": cam.read(), "prompt": args.prompt}
            report = _latency.startup_self_check(
                args.mode, lambda: adapter.infer(dict(probe)),
                fps=fps, horizon=horizon, inference_hz=args.inference_hz,
                max_latency_steps=args.max_latency_steps)
            print(report.summary())
            rec.log("latency_check", **dataclasses.asdict(report))
            adapter.reset()   # the probe chunks polluted the causal filters

        gated = args.dashboard and not args.ungated
        runner = DeployRunner(adapter=adapter, strategy=strategy,
                              executor=executor, camera=cam, recorder=rec,
                              limits=limits, fps=fps, max_steps=args.max_steps,
                              gated=gated, dataset=args.dataset)
        dash = None
        if args.dashboard:
            from .dashboard import Dashboard
            dash = Dashboard(runner, port=args.dashboard_port)
            dash.start()   # up before the loop, so the page shows the hold
        if gated:
            logger.info("gated start: robot holds until you press Start on "
                        "the dashboard (http://localhost:%d)", dash.port)
        else:
            input("Robot connected, latency OK. Press Enter to start inference...")
        try:
            runner.run()
            if dash is not None and runner.watchdog.tripped:
                # Stay up for the post-mortem: the operator can read the
                # telemetry/trip reason instead of losing it to process exit.
                # damp() is latched — there is no restart from here.
                logger.error("watchdog tripped — dashboard stays up for "
                             "inspection; Ctrl-C to exit")
                while True:
                    time.sleep(1.0)
        finally:
            if dash is not None:
                dash.stop()
    except KeyboardInterrupt:
        logger.info("interrupted")
    except _latency.LatencyBudgetError:
        rec.log("latency_check_refused")
        raise
    finally:
        strategy.close()
        reset = getattr(adapter, "reset", None)
        if callable(reset):
            reset()
        rec.stop()
        cam.close()
        executor.close()
        logger.info("delay budget: %s", budget.stats())


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
