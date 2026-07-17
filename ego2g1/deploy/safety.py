"""The safety layer. Ported from the old deploy's safety.py (provenance:
third_party/openpi/ego2g1/deploy/safety.py) and adapted to the new executor.

The vendored lowcmd path ships with none of this, so it is not garnish — it is
what stands between an IK glitch and the hardware. Three independent gates,
because they fail for different reasons:

  1. Joint-delta clamp — between the strategy and the executor. Catches a bad
     IK solve, a garbage row from a mis-normalized chunk, or a discontinuous
     strategy seam BEFORE it becomes a 500 Hz setpoint. A single 30 Hz knot
     that jumps 1 rad is a violent motion; clamping turns a lurch into a lag.
  2. Watchdog — state staleness and plan starvation. If we cannot SEE the
     robot, we must not command it.
  3. E-stop — one latched call to the executor's damp(). Stopping the
     publisher is NOT a stop: the firmware holds the last setpoint forever.

Everything trips to damp(), never to "stop sending".

G1-D reality baked into the numbers: no lower body, fixed/suspended base,
`rt/lowcmd` direct (never `rt/arm_sdk`), legs/waist held at measured with high
gains by the vendored controller — the arm is the only thing that moves, and
damp() is the only real stop.
"""

import dataclasses
import logging
import threading

import numpy as np

from ..core import layout
from . import actions as _actions

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SafetyLimits:
    # Max joint motion per 30 Hz action tick. 0.15 rad @ 30 Hz ~ 4.5 rad/s:
    # brisk but not violent for the G1 arm.
    max_joint_step: float = 0.15
    # Absolute joint-velocity ceiling used for the same check when ticks are late.
    max_joint_vel: float = 5.0
    # Trip the e-stop if the measured state goes quiet for this long.
    max_state_age: float = 0.2
    # Trip if the camera stops delivering frames — otherwise the policy is fed
    # a frozen image forever while the arm keeps moving on it.
    max_camera_age: float = 0.5
    # Trip if the strategy has no action for this long (holding the last
    # waypoint is safe, but it means the planner has died).
    max_starvation: float = 2.0
    # Trip if the IK cannot reach targets by this much — the policy is asking
    # for something the arm cannot do, or our frames are wrong.
    max_tracking_error_m: float = 0.10
    # Consecutive violations tolerated before tripping (one spike is noise).
    trip_after: int = 3


class Clamp:
    """Rate-limits the arm dims of the joint rows entering the executor."""

    def __init__(self, limits: SafetyLimits):
        self.limits = limits
        self._last = None
        self.clamped_ticks = 0
        self.max_seen = 0.0

    def reset(self, q14) -> None:
        self._last = np.asarray(q14, dtype=np.float64).copy()

    def __call__(self, q14, dt: float) -> np.ndarray:
        """Return a joint vector no further than the limit from the last one."""
        q = np.asarray(q14, dtype=np.float64)
        if self._last is None:
            self._last = q.copy()
            return q.copy()

        step = q - self._last
        mag = float(np.abs(step).max())
        self.max_seen = max(self.max_seen, mag)

        cap = min(self.limits.max_joint_step,
                  self.limits.max_joint_vel * max(dt, 1e-3))
        if mag > cap:
            self.clamped_ticks += 1
            step = step * (cap / mag)
            logger.warning("joint step %.3f rad clamped to %.3f rad", mag, cap)

        out = self._last + step
        self._last = out.copy()
        return out


class Watchdog:
    """Trips the e-stop when the world stops making sense."""

    def __init__(self, limits: SafetyLimits, on_trip):
        self.limits = limits
        self._on_trip = on_trip
        self._strikes = {}
        self._starving_since: float | None = None
        self._tripped = False
        self._lock = threading.Lock()
        self.reason: str | None = None

    @property
    def tripped(self) -> bool:
        return self._tripped

    def trip(self, reason: str) -> None:
        with self._lock:
            if self._tripped:
                return
            self._tripped = True
            self.reason = reason
        logger.error("E-STOP: %s", reason)
        try:
            self._on_trip()
        except Exception:
            logger.exception("e-stop handler raised")

    def _strike(self, key: str, bad: bool, reason: str) -> None:
        n = self._strikes.get(key, 0)
        if not bad:
            self._strikes[key] = 0
            return
        n += 1
        self._strikes[key] = n
        if n >= self.limits.trip_after:
            self.trip(reason)

    def check_state_age(self, age: float) -> None:
        self._strike("state", age > self.limits.max_state_age,
                     f"robot state stale for {age:.3f}s — cannot see the robot")

    def check_camera_age(self, age: float) -> None:
        self._strike("camera", age > self.limits.max_camera_age,
                     f"camera frame stale for {age:.3f}s — the policy would act "
                     "on a frozen image")

    def check_starvation(self, has_action: bool, now: float) -> None:
        """DURATION-based, not strike-based: a briefly empty buffer is normal
        (episode start, first inference in flight). Only a SUSTAINED absence of
        plan means the planner died — a strike counter would fire on every
        startup."""
        if has_action:
            self._starving_since = None
            return
        if self._starving_since is None:
            self._starving_since = now
        elif now - self._starving_since > self.limits.max_starvation:
            self.trip(f"no action for {now - self._starving_since:.1f}s — "
                      "the strategy/planner is dead")

    def check_tracking(self, worst_error_m: float) -> None:
        self._strike("track", worst_error_m > self.limits.max_tracking_error_m,
                     f"IK tracking error {worst_error_m*1000:.0f} mm — target "
                     "unreachable or the frames are wrong")


def sanity_check_model_action(action) -> bool:
    """Cheap guard on a (30,) relative_eef row before it reaches the IK.

    A mis-normalized or corrupted chunk shows up as non-finite values or a
    delta metres long. Catching it here means it never becomes a pose."""
    a = np.asarray(action)
    if a.shape != (layout.DIM,) or not np.all(np.isfinite(a)):
        return False
    for h in layout.HANDS:
        if np.linalg.norm(a[layout.EEF[h]][:3]) > 1.5:  # a 1.5 m single-chunk delta is nonsense
            return False
    return True


def sanity_check_joint_row(row) -> bool:
    """Guard on a (26,) executor row popped from a strategy."""
    r = np.asarray(row)
    if r.shape != (_actions.ROBOT_DIM,) or not np.all(np.isfinite(r)):
        return False
    if np.abs(r[_actions.ARM]).max() > np.pi + 0.5:   # outside any G1 joint range
        return False
    return True
