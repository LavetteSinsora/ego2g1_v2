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

from ..core import layout, relation_layout
from . import actions as _actions
from . import gripper_calib as _gripper_calib
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
                 relation_mode: bool = False,
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
        # relation_eef mode ONLY: the observation side changes (live
        # perception needs BOTH camera eyes + a scalar last-commanded
        # gripper FRACTION, not the old modes' single eye + (6,)-vector
        # hand command — docs/relation_deploy_plan.md §3.3/§5.5). The
        # actuation side (executor row, clamp, watchdog) is untouched either
        # way: everything below the adapter still only ever sees (26,)
        # joint rows.
        self.relation_mode = bool(relation_mode)
        self._clock = clock
        self._wait = wait

        self.clamp = _safety.Clamp(self.limits)
        self.watchdog = _safety.Watchdog(self.limits, self._on_trip)
        # Hands start OPEN and thereafter track what we last COMMANDED — the
        # model's state hand-block is the command stream, never encoders.
        # relation_mode: a scalar fraction per hand (0=open..1=closed, the
        # same convention RelativeEEFRotvecChunks decodes/gripper_calib
        # inverts), NOT the old modes' (6,)-motor-vector — there is no
        # "hand block" in the 56-dim relation state, only a rounded grasp bit.
        self.last_hands = ({h: 0.0 for h in relation_layout.HANDS} if self.relation_mode
                           else {h: np.zeros(layout.HAND_DIM) for h in layout.HANDS})
        self.steps_executed = 0
        self.running = False       # set/cleared around run(); read by telemetry()
        # relation_eef only: high-water mark for latch/hand-state events
        # already drained into the recorder (see run()'s step 4b) -- -inf so
        # the very first tick's events (which may predate this instant on
        # some monotonic-clock bases) are never silently skipped.
        self._last_drained_event_t = float("-inf")
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
        prompt = getattr(self.adapter, "prompt", "")
        if self.relation_mode:
            return self._observe_relation(arm_q, image, prompt)
        return {"arm_q": arm_q,
                "hand_cmds": {h: self.last_hands[h].copy() for h in layout.HANDS},
                "image": image,
                "prompt": prompt}

    def _observe_relation(self, arm_q, image, prompt) -> dict:
        """relation_eef variant: RelationPolicyAdapter's `perception=` path
        (docs/relation_deploy_plan.md §5.5) needs BOTH camera eyes (for
        `StereoSGBMDepthSource`) plus the last-commanded gripper FRACTION per
        hand, not the old modes' single eye + (6,)-vector hand command. The
        single `image` above is still sent — the model itself takes exactly
        ONE egocentric image either way (camera.py's own docstring); only the
        PERCEPTION input differs.
        """
        rgb_left = rgb_right = None
        if self.camera is not None:
            stereo = self.camera.read_stereo()
            if stereo is not None:
                rgb_left, rgb_right = stereo
        hand_cmds_last = {h: float(self.last_hands[h]) for h in relation_layout.HANDS}
        return {"arm_q": arm_q,
                # kept for RelationPolicyAdapter.infer's unconditional
                # `request["hand_cmds"]` read (unused by
                # RelativeEEFRotvecChunks.convert — see actions.py — so the
                # scalar shape here is harmless); hand_cmds_last is the one
                # perception.observe() actually consumes.
                "hand_cmds": dict(hand_cmds_last),
                "hand_cmds_last": hand_cmds_last,
                "image": image,
                "rgb_left": rgb_left,
                "rgb_right": rgb_right,
                "prompt": prompt}

    def _hand_frac_from_command(self, hand: str, cmd) -> float:
        """relation_mode only: recover the scalar open/closed fraction from
        the just-EXECUTED (6,)-motor command, for the NEXT tick's
        `hand_cmds_last` — see `gripper_calib.frac_from_command`'s docstring
        for why this inverts the executed row rather than reading the
        converter's internal per-chunk `frac` directly. `self.adapter` is a
        `RelationPolicyAdapter` (or test double exposing the same
        `closed_pose` property) whenever `relation_mode` is True."""
        closed_pose = self.adapter.closed_pose[hand]
        return _gripper_calib.frac_from_command(cmd, closed_pose)

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
                for h in layout.HANDS:
                    if self.relation_mode:
                        self.last_hands[h] = self._hand_frac_from_command(
                            h, row[_actions.HAND[h]])
                    else:
                        self.last_hands[h] = row[_actions.HAND[h]].copy()
                self.recorder.log("action", step=step, row=row)
                self.steps_executed = step + 1

                # 4. per-chunk IK tracking error (relative_eef only)
                err = getattr(self.adapter, "last_tracking_error", None)
                if err is not None:
                    self.watchdog.check_tracking(float(err))
                    if err > 0:
                        self.recorder.log("tracking", step=step, worst_m=float(err))

                # 4b. relation_eef only: drain new latch/hand-closed
                # transitions (RelationPerception's own bounded event log,
                # for the dashboard's timeline) into events.jsonl -- reuses
                # the existing recorder mechanism, no new file format.
                if self.relation_mode:
                    perception = getattr(self.adapter, "perception", None)
                    if perception is not None:
                        for ev in perception.recent_events(
                                since_t=self._last_drained_event_t):
                            kind = "latch" if ev["kind"] == "latch" else "hand_state"
                            # ev's own "kind"/"t" would collide with log()'s
                            # positional `kind` / self-stamped "t" -- keep the
                            # original event time distinctly as "event_t".
                            fields = {k: v for k, v in ev.items()
                                     if k not in ("kind", "t")}
                            self.recorder.log(kind, event_t=ev["t"], **fields)
                            self._last_drained_event_t = ev["t"]

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
        if self.relation_mode:
            # relation_eef's recorded dataset hand column (if/when one
            # exists) is a scalar grasp bit, not the old modes' (6,)-motor
            # vector this ramp's `hand_start`/`h0` interpolation assumes
            # (ep["hand"][h][0], clipped to (6,)) — untested, undefined
            # territory, not something this pass built. Fail loud rather
            # than silently ramp toward a misinterpreted 6-vector.
            raise NotImplementedError(
                "reset_to_episode is not implemented for relation_eef mode "
                "(the dataset hand-column format for this mode is not yet "
                "defined — docs/relation_deploy_plan.md's Phase 2 scope "
                "does not include dataset replay)")
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
            # --- relation_eef perception stack (detector/tracker/latch) ---
            # None for joint/relative_eef deploys, or before the first tick's
            # perception has run -- the dashboard treats both as "n/a".
            "relation": self._relation_telemetry() if self.relation_mode else None,
        }

    def _relation_telemetry(self) -> dict | None:
        """`relation_eef`-only: a JSON-safe snapshot of the last perception
        tick's per-object detections + per-hand grasp/latch state, plus the
        recent latch/hand-closed event history, for the dashboard's overlay
        panels and timeline strip. Reads only already-computed state off the
        adapter/`RelationPerception` (no new perception work happens here) --
        same pure-pull contract as the rest of `telemetry()`."""
        perception = getattr(self.adapter, "perception", None)
        percept = getattr(self.adapter, "last_percept", None)
        return build_relation_telemetry(perception, percept)


def build_relation_telemetry(perception, percept) -> dict | None:
    """`DeployRunner._relation_telemetry`'s body, factored out as a free
    function: a JSON-safe dashboard snapshot from a `RelationPerception`
    instance and its last `observe()` result. Shared by `DeployRunner
    .telemetry()` and any lighter-weight caller that also wants the
    dashboard's overlay/status panels fed correctly without a real robot/
    policy attached (e.g. a perception-only preview tool)."""
    if perception is None or percept is None:
        return None

    hands = tuple(perception.task_config.hands)
    state = np.asarray(percept["state"])
    grasp_bits = state[-len(hands):] if hands else np.zeros(0)
    hand_closed = {h: bool(grasp_bits[i] >= 0.5) for i, h in enumerate(hands)}

    objects = []
    for obj in perception.task_config.objects:
        debug = percept["objects"].get(obj.instance_id)
        detection = perception.last_detections.get(obj.instance_id)
        pose = debug.pose_pelvis if debug is not None else None
        objects.append({
            "instance_id": obj.instance_id,
            "detected_this_tick": bool(debug.detected_this_tick) if debug else False,
            "tracked": bool(debug.tracked) if debug else False,
            "depth_m": debug.depth_m if debug else None,
            "confidence": (float(detection.confidence)
                          if detection is not None else None),
            "box_xyxy": (detection.box_xyxy.tolist()
                        if detection is not None and detection.box_xyxy is not None
                        else None),
            "position_pelvis": pose[:3, 3].tolist() if pose is not None else None,
        })

    hand_states = []
    for h in hands:
        result = percept["latch"][h]
        hand_states.append({
            "hand": h,
            "hand_closed": hand_closed[h],
            "state": result.state.value,
            "candidate_object": result.candidate_object,
            "latched_object": result.latched_object,
            "ticks_in_candidate": result.ticks_in_candidate,
            "reason": result.reason,
        })

    return {"objects": objects, "hands": hand_states,
            "events": perception.recent_events()}


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
    action_mode: str = "auto"          # auto | joint | relative_eef | relation_eef
    ik_iters: int = 25
    posture_cost: float = 0.05         # the measured smoothness knob
    collision_min_dist: float = 0.005
    # --- relation_eef-only perception config (docs/relation_deploy_plan.md
    # §5/§6). Unused by joint/relative_eef deploys — left None there, no new
    # file is ever required for those modes. All three become REQUIRED the
    # moment action_mode resolves to "relation_eef" (main() fails loud,
    # naming exactly which is missing, before touching the robot/camera).
    task_config: str | None = None     # YAML for perception/task_config.py's
                                       # DeployTaskConfig (objects, in the
                                       # checkpoint's fixed order, + hands)
    stereo_calib: str | None = None    # .npz for perception/depth.py's
                                       # StereoCalibration.load (head-camera
                                       # stereo intrinsics/extrinsics)
    camera_extrinsic: str | None = None   # .npz (key "T_pelvis_camera") from
                                       # perception/touch_calib.py's
                                       # solve_camera_extrinsic / _cli_solve
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


def _resolve_action_mode(action_mode: str, control_mode: str) -> str:
    """"auto" reads the checkpoint's control_mode off the server handshake
    (`ego2g1/deploy/client.py`'s `PolicyClient.control_mode`) and picks the
    matching action mode; any other value passes through unchanged (useful
    only to deliberately test a mismatched pairing on purpose). Factored out
    of `main()` so the auto-selection logic — in particular, that
    `control_mode == "relation_eef"` selects `relation_eef` alongside the
    existing `joint`/`relative_eef` cases — is directly unit-testable
    without a real PolicyClient/websocket connection."""
    if action_mode != "auto":
        return action_mode
    if control_mode == "joint":
        return "joint"
    if control_mode == "relation_eef":
        return "relation_eef"
    return "relative_eef"


def _build_relation_adapter(args: "Args", client, fps: int):
    """relation_eef mode's adapter: load task config + calibration, cross-
    check against the server, wire a real detector/depth perception stack
    into `RelationPolicyAdapter` (docs/relation_deploy_plan.md §5). Imports
    are local to this function — joint/relative_eef deploys must never pay
    for perception's cv2/DINO/SAM2 imports just by importing this module.

    Fails loud, naming exactly what's missing, the moment `relation_eef` is
    selected and any of `--task-config`/`--stereo-calib`/`--camera-extrinsic`
    is absent — same "fail loud before it can silently mis-serve" philosophy
    as `ego2g1/train/stamp.py`'s `check_supported` and
    `perception/task_config.py`'s own `validate_against_server_metadata`.
    """
    from . import policy_adapter as _policy_adapter

    missing = []
    if not args.task_config:
        missing.append("--task-config")
    if not args.stereo_calib:
        missing.append("--stereo-calib")
    if not args.camera_extrinsic:
        missing.append("--camera-extrinsic")
    if missing:
        raise ValueError(
            f"action_mode=relation_eef needs {', '.join(missing)} (the "
            "connected checkpoint's server control_mode advertises "
            "'relation_eef' — see ego2g1/deploy/client.py's handshake). "
            "relation_eef mode drives LIVE perception (object detection + "
            "stereo depth + hand-relative geometry, "
            "docs/relation_deploy_plan.md §5) every tick and refuses to "
            "guess a task config, stereo calibration, or camera extrinsic "
            "silently: --task-config is a YAML for "
            "ego2g1.deploy.perception.task_config.load_task_config, "
            "--stereo-calib is a .npz for "
            "ego2g1.deploy.perception.depth.StereoCalibration.load, and "
            "--camera-extrinsic is a .npz (key 'T_pelvis_camera') produced "
            "by ego2g1.deploy.perception.touch_calib.solve_camera_extrinsic. "
            "joint/relative_eef modes never need any of these.")

    from .perception.depth import StereoCalibration, StereoSGBMDepthSource
    from .perception.detector import GroundingDinoSam2Detector
    from .perception.relation_perception import RelationPerception
    from .perception.task_config import load_task_config, validate_against_server_metadata

    task_config = load_task_config(args.task_config)
    validate_against_server_metadata(task_config, client.metadata["ego2g1"])
    calib = StereoCalibration.load(args.stereo_calib)
    # touch_calib.py's own save convention (_cli_solve): np.savez(...,
    # T_pelvis_camera=T, rms_residual_m=..., n_points=...) -- load the same
    # key, don't invent a second convention for the same asset.
    T_pelvis_camera = np.load(args.camera_extrinsic)["T_pelvis_camera"]

    detector = GroundingDinoSam2Detector()
    depth_source = StereoSGBMDepthSource(calib)
    perception = RelationPerception(
        task_config, detector, depth_source, calib, T_pelvis_camera, fps=fps)

    return _policy_adapter.make_adapter(
        "relation_eef", client, args.prompt, ik_iters=args.ik_iters,
        posture_cost=args.posture_cost, collision_min_dist=args.collision_min_dist,
        perception=perception)


def _build_probe(action_mode: str, executor, cam, prompt: str) -> dict:
    """The one-shot request dict `startup_self_check` runs the adapter
    against, mode-aware: relation_eef's `RelationPolicyAdapter` (wired with
    `perception=`) needs the stereo pair + a scalar hand fraction, not the
    old modes' single eye + (6,)-vector hand command — see
    `DeployRunner._observe_relation`, which this mirrors for the one probe
    call that happens before the loop ever starts."""
    arm_q = executor.arm_q()
    if action_mode == "relation_eef":
        hand_cmds_last = {h: 0.0 for h in relation_layout.HANDS}
        rgb_left = rgb_right = None
        stereo = cam.read_stereo() if cam is not None else None
        if stereo is not None:
            rgb_left, rgb_right = stereo
        return {"arm_q": arm_q, "hand_cmds": dict(hand_cmds_last),
                "hand_cmds_last": hand_cmds_last,
                "image": cam.read() if cam is not None else None,
                "rgb_left": rgb_left, "rgb_right": rgb_right, "prompt": prompt}
    return {"arm_q": arm_q,
            "hand_cmds": {h: np.zeros(layout.HAND_DIM) for h in layout.HANDS},
            "image": cam.read() if cam is not None else None, "prompt": prompt}


def main(args: Args) -> None:
    from . import client as _client
    from . import policy_adapter as _policy_adapter

    client = _client.PolicyClient(args.host, args.port)
    fps = args.fps or client.fps
    horizon = client.action_horizon

    action_mode = _resolve_action_mode(args.action_mode, client.control_mode)

    if action_mode == "relation_eef":
        adapter = _build_relation_adapter(args, client, fps)
    elif action_mode == "relative_eef":
        adapter = _policy_adapter.make_adapter(
            "relative_eef", client, args.prompt, ik_iters=args.ik_iters,
            posture_cost=args.posture_cost, collision_min_dist=args.collision_min_dist)
    else:
        adapter = _policy_adapter.make_adapter("joint", client, args.prompt)
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
            probe = _build_probe(action_mode, executor, cam, args.prompt)
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
                              gated=gated, dataset=args.dataset,
                              relation_mode=(action_mode == "relation_eef"))
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
