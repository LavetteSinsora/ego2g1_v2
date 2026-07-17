"""The teleop control loop. DeployLoop's skeleton, with the policy layer removed.

Three threads, the same shape as `ego2g1.deploy.loop`:

  T1 arm emitter   500 Hz   traj_arm.eval(now)  -> rt/lowcmd
  T2 hand emitter  200 Hz   traj_hand.eval(now) -> rt/brainco/*/cmd
  T3 control       source   read hands -> retarget -> IK -> clamp -> push knots

What is deliberately NOT here: ChunkQueue, RTC, DelayBudget, the async inference
thread. All of that exists to hide ~400 ms of policy latency behind a 50-step action
chunk. A human hand has no such latency -- the sample IS the action, and the correct
thing to do with it is execute it now. Routing teleop through the chunk machinery would
add a chunk's worth of lag for nothing, which is the "30 Hz feels laggy" trap.

The 30 Hz in deploy was never the robot's command rate anyway: the arm has always been
emitted at 500 Hz and the hands at 200 Hz, interpolated between knots. 30 Hz was the
rate the POLICY produced knots at. Here the tracker produces them, at ~60 Hz, and the
budget is comfortable: the hand retarget costs ~0.45 ms for both hands and the mink IK
~0.84 ms, so ~1.3 ms of a 16 ms tick.

`Clamp` needs no retuning for the faster tick -- it already takes dt and caps at
min(max_joint_step, max_joint_vel * dt), so raising the rate makes it velocity-bound at
5 rad/s rather than more permissive.
"""

import dataclasses
import logging
import threading
import time

import numpy as np

from ._vendor.eg.common import layout
from ._vendor.eg.deploy import safety as _safety
from ._vendor.eg.deploy.trajectory import TrajectoryBuffer

logger = logging.getLogger(__name__)

SIDES = layout.HANDS


@dataclasses.dataclass
class TeleopConfig:
    arm_hz: float = 500.0
    hand_hz: float = 200.0
    control_hz: float = 60.0
    # How far ahead of the emitter we keep joint knots. One tick plus slack: a teleop
    # knot is only valid at the instant it was sampled, so queueing more of them just
    # buys latency.
    lookahead_s: float = 0.03
    # Both hands out of view: hold position past stream_hold_s, and if they stay gone past
    # stream_disengage_s treat the operator as absent and drop to IDLE (a safe hold that
    # needs a deliberate re-engage) rather than damping. Damp is for FAULTS, not for a
    # glance away. A single hand out is handled per-arm by the retargeter, not here.
    stream_hold_s: float = 0.15
    stream_disengage_s: float = 5.0
    # Seconds to blend the fingers from where they are to the operator's first command.
    # The arm needs no such ramp -- its delta is identity at engage, by construction.
    hand_engage_ramp_s: float = 0.5


class TeleopLoop:
    """Drives the robot from a HandSource. Owns no kinematics or safety of its own --
    every one of those is the deploy stack's, unchanged."""

    def __init__(self, cfg: TeleopConfig, *, dds, kinematics, source, retargeter,
                 limits: _safety.SafetyLimits = _safety.SafetyLimits()):
        self.cfg = cfg
        self.dds = dds
        self.kin = kinematics
        self.src = source
        self.rt = retargeter
        self.limits = limits

        self.traj_arm = TrajectoryBuffer(layout.ARM_DOF)
        self.traj_hand = TrajectoryBuffer(layout.HAND_DIM * len(SIDES))
        self.clamp = _safety.Clamp(limits)
        self.watchdog = _safety.Watchdog(limits, on_trip=self.dds.damp)

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._engaged = threading.Event()
        self._engage_t = 0.0
        self._quiet_since: float | None = None

        self.stats = {"ticks": 0, "dropped": 0, "held": 0, "engages": 0}

    # --- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        arm_q = self.dds.arm_q()
        now = time.monotonic()
        self.traj_arm.seed(now, arm_q)
        self.traj_hand.seed(now, np.zeros(layout.HAND_DIM * len(SIDES)))
        self.clamp.reset(arm_q)
        self.kin.ground(arm_q)

        for fn, name in ((self._emit_arm, "emit-arm"),
                         (self._emit_hand, "emit-hand"),
                         (self._control, "control")):
            t = threading.Thread(target=self._guard(fn), name=name, daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("teleop loop up: %.0f Hz control, %.0f Hz arm, %.0f Hz hand "
                    "(IDLE -- press 'e' to engage)",
                    self.cfg.control_hz, self.cfg.arm_hz, self.cfg.hand_hz)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)

    def _guard(self, fn):
        def run():
            try:
                fn()
            except Exception as e:
                logger.exception("thread %s died", threading.current_thread().name)
                self.watchdog.trip(f"{threading.current_thread().name} died: {e}")
        return run

    # --- the clutch ---------------------------------------------------------

    def engage(self) -> bool:
        """Latch the anchor and start following the operator.

        The anchor is the MEASURED FK of the real joints, so the first target equals the
        current pose exactly and the arm cannot jump. Refuses if a hand is not being
        tracked: engaging on a dead hand would anchor to a garbage wrist and then snap
        the arm the moment tracking returned.
        """
        if self.watchdog.tripped:
            logger.error("cannot engage: e-stop is latched")
            return False
        sample = self.src.latest()
        if sample is None or not all(sample.active[s] for s in self.rt.hands):
            logger.error("cannot engage: hands are not tracked "
                         "(hold both hands in view of the headset)")
            return False

        arm_q = self.dds.arm_q()
        self.kin.ground(arm_q)

        now = time.monotonic()
        first_heading = self.rt.orientation == "absolute" and not self.rt.heading_set
        self.rt.engage(sample, self.kin.flange_poses(arm_q), now)
        if first_heading:
            logger.info("heading set: left/right estimates agree to %.1f deg "
                        "(a large spread means B is off — see `check measure-c`)",
                        self.rt.heading_spread_deg)

        self._engage_t = now
        # Re-seed both trajectories at what the emitter is sending RIGHT NOW, so the
        # first pushed knot spans exactly one tick and the clamp is a genuine rate limit
        # rather than a licence to cross a stale gap in one period.
        q_now, h_now = self.traj_arm.eval(now), self.traj_hand.eval(now)
        if q_now is not None:
            self.traj_arm.reseed(now, q_now)
            self.clamp.reset(q_now)
        if h_now is not None:
            self.traj_hand.reseed(now, h_now)

        self._engaged.set()
        self.stats["engages"] += 1
        logger.info("ENGAGED (anchor #%d) -- the robot is following your hands",
                    self.stats["engages"])
        return True

    def disengage(self) -> None:
        if self._engaged.is_set():
            self._engaged.clear()
            self.rt.disengage()
            logger.info("IDLE -- robot holding; press 'e' to re-anchor")

    @property
    def engaged(self) -> bool:
        return self._engaged.is_set()

    def estop(self, reason: str = "manual") -> None:
        self._engaged.clear()
        self.watchdog.trip(reason)

    # --- T1 / T2: emitters (identical to DeployLoop) -------------------------

    def _emit_arm(self) -> None:
        period = 1.0 / self.cfg.arm_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            now = time.monotonic()
            if not self.watchdog.tripped:
                self.watchdog.check_state_age(self.dds.lowstate_age())
                q = self.traj_arm.eval(now)
                if q is not None and not self.watchdog.tripped:
                    self.dds.send_arm(q)
                self.traj_arm.prune(now)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _emit_hand(self) -> None:
        period = 1.0 / self.cfg.hand_hz
        n = layout.HAND_DIM
        while not self._stop.is_set():
            t0 = time.perf_counter()
            now = time.monotonic()
            if not self.watchdog.tripped:
                v = self.traj_hand.eval(now)
                if v is not None:
                    self.dds.send_hands({h: v[i * n:(i + 1) * n]
                                         for i, h in enumerate(SIDES)})
                self.traj_hand.prune(now)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    # --- T3: control --------------------------------------------------------

    def _control(self) -> None:
        period = 1.0 / self.cfg.control_hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            if self.watchdog.tripped:
                time.sleep(period)
                continue

            now = time.monotonic()
            if self._engaged.is_set() and self._stream_ok(now):
                self._tick(now)

            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _stream_ok(self, now: float) -> bool:
        """Both hands out of view: hold, and if it lasts, disengage -- do not damp.

        Losing the operator is not like losing the robot: the arm holding its last knot is
        a perfectly safe state. A brief loss (both hands dip out of the neck mount's FOV,
        the session hiccups) should just HOLD and then resume seamlessly when the hands
        return -- the retargeter re-anchors each hand on the way back, so nothing jumps. A
        PROLONGED loss means the operator is gone (walked off, headset slept), and the
        right response is to drop to IDLE -- a safe hold that requires a deliberate
        re-engage -- not to damp the robot. Damp is reserved for genuine faults (stale
        robot state, IK divergence, a thrown thread), which the watchdog still handles.
        """
        age = self.src.age()
        if age < self.cfg.stream_hold_s:
            self._quiet_since = None
            return True

        if self._quiet_since is None:
            self._quiet_since = now
            logger.warning("both hands out of view (%.0f ms) — holding position", age * 1e3)
        self.stats["held"] += 1

        if now - self._quiet_since > self.cfg.stream_disengage_s:
            logger.warning("hands lost for %.1fs — disengaging to IDLE; press 'e' to resume",
                           now - self._quiet_since)
            self.disengage()
            self._quiet_since = None
        return False

    def _tick(self, now: float) -> None:
        sample = self.src.latest()
        if sample is None:
            return

        targets, hand_cmds, info = self.rt.step(sample, now)
        if info["dropped"]:
            self.stats["dropped"] += 1

        q = self.kin.solve(targets)
        self.watchdog.check_tracking(self.kin.tracking_error(targets))
        if self.watchdog.tripped:
            return
        q = self.clamp(q, 1.0 / self.cfg.control_hz)

        # One tick ahead: the knot describes where the operator's hand IS, so the sooner
        # the emitter gets there the lower the felt lag. lookahead_s only has to cover a
        # late control tick, not buy a plan.
        t_k = now + max(self.cfg.lookahead_s, 1.0 / self.cfg.control_hz)
        self.traj_arm.push(t_k, q)
        self.traj_hand.push(t_k, np.concatenate([self._blend(h, hand_cmds[h], now)
                                                 for h in SIDES]))
        self.stats["ticks"] += 1

    def _blend(self, side: str, cmd: np.ndarray, now: float) -> np.ndarray:
        """Ease the fingers into the operator's grip over the first half-second.

        The arm's delta is identity at engage, so it starts where it is. The FINGER
        command is absolute, though: if the robot is holding a fist and the operator
        engages with an open hand, without this the hand flies open in one 200 Hz step.
        """
        u = (now - self._engage_t) / max(self.cfg.hand_engage_ramp_s, 1e-6)
        if u >= 1.0:
            return cmd
        i = SIDES.index(side)
        held = self.traj_hand.eval(self._engage_t)
        if held is None:
            return cmd
        start = held[i * layout.HAND_DIM:(i + 1) * layout.HAND_DIM]
        return (1.0 - u) * start + u * cmd
