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

Dry run, no robot, no camera (mock executor; needs only the policy server):

    python -m ego2g1.deploy.runner --host 127.0.0.1 --prompt "x" --dry-run
"""

from __future__ import annotations

import dataclasses
import logging
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
        self._clock = clock
        self._wait = wait

        self.clamp = _safety.Clamp(self.limits)
        self.watchdog = _safety.Watchdog(self.limits, self._on_trip)
        # Hands start OPEN and thereafter track what we last COMMANDED — the
        # model's state hand-block is the command stream, never encoders.
        self.last_hands = {h: np.zeros(layout.HAND_DIM) for h in layout.HANDS}
        self.steps_executed = 0

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
        try:
            for step in range(self.max_steps):
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
                                  state_age=self.executor.state_age())
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
                if (step + 1) % 50 == 0:
                    logger.info("executed %d steps", step + 1)
        finally:
            self.strategy.close()
            if self.watchdog.tripped:
                logger.error("stopped by watchdog: %s", self.watchdog.reason)


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
    # --- session ---
    record_dir: str = "recordings"
    no_record: bool = False
    max_steps: int = 10_000_000
    dry_run: bool = False              # mock executor + static camera, no robot
    skip_latency_check: bool = False   # ONLY for offline debugging; never live
    # --- safety ---
    max_joint_step: float = 0.15
    max_state_age: float = 0.2
    max_starvation: float = 2.0


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

    # --- recorder
    if args.no_record:
        rec = _recorder.NullRecorder()
    else:
        session = _recorder.new_session(args.record_dir, args.prompt or "deploy")
        rec = _recorder.Recorder(session, cameras={"head": cam}, meta={
            "mode": args.mode, "action_mode": action_mode, "horizon": horizon,
            "fps": fps, "host": args.host, "port": args.port,
            "prompt": args.prompt, "dim": _actions.ROBOT_DIM,
        })
    rec.start()

    budget = _latency.DelayBudget(fps)
    strategy = _strategies.make_strategy(
        args.mode, adapter, chunk_size=horizon, inference_hz=args.inference_hz,
        exp_weight_m=args.exp_weight_m, max_latency_steps=args.max_latency_steps,
        min_smooth_steps=args.min_smooth_steps, control_hz=fps,
        rtc_execute_horizon=args.rtc_execute_horizon, recorder=rec, budget=budget)

    limits = _safety.SafetyLimits(max_joint_step=args.max_joint_step,
                                  max_state_age=args.max_state_age,
                                  max_starvation=args.max_starvation)

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

        input("Robot connected, latency OK. Press Enter to start inference...")
        runner = DeployRunner(adapter=adapter, strategy=strategy,
                              executor=executor, camera=cam, recorder=rec,
                              limits=limits, fps=fps, max_steps=args.max_steps)
        runner.run()
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
