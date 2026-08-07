"""ExecutorSession: the ONE road rows take to the executor
(docs/deploy_refactor_plan.md §3).

Before this class, the observe→clamp→future-stamp→send→pace→damp-on-Ctrl-C
sequence was written four times (replay_dataset, replay_diag,
check.replay-actions, replay_relation_openloop) plus the canonical copy in
runner.py — and two of the four had NO clamp and NO row sanity check at all.
Now the invariants are constructor-owned and no caller can forget them:

  * every row is `safety.sanity_check_joint_row`-checked (an insane row
    raises `InsaneRowError` — the runner turns that into a watchdog trip,
    the replay tools damp and stop);
  * the arm slice goes through the per-tick `safety.Clamp` (a bad row
    becomes lag, never a lurch) with `clamp`/`action` events recorded;
  * every waypoint is future-stamped `t_cycle_end + dt` (the vendored
    500 Hz interpolator must interpolate toward a point ahead of it,
    never extrapolate — docs/deploy.md);
  * pacing is `_util.precise_wait` on an absolute schedule (`stream`) or
    the caller's own cycle clock (`send_row`);
  * Ctrl-C inside `stream()` damps BEFORE propagating.

The clamp deliberately stays in the loop for dataset replays too: a
legitimate replay never hits 0.209 rad/tick (measured over 20
put_bottle_in_box_ego episodes / 3920 ticks: median 0.024, p99 0.101,
worst 0.156 rad — and that is the OLD unsmoothed extraction), so the
A/B semantics of rung 6
are untouched — but a corrupt row or wrong-episode indexing now becomes lag
plus a printed count instead of a lurch. `clamped_ticks` is surfaced so any
interference is visible, not silent.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ...core import layout
from .. import _util
from .. import actions as _actions
from ..record import recorder as _recorder
from . import safety as _safety

logger = logging.getLogger(__name__)


class InsaneRowError(RuntimeError):
    """A row failed `safety.sanity_check_joint_row` — wrong shape, NaN, or an
    out-of-range arm value. Never sent."""


class ExecutorSession:
    """Wraps an executor with the clamp/sanity/stamp/pace/damp invariants.
    Everything is injected (clock/wait) so tests run it hardware-free, same
    discipline as DeployRunner."""

    def __init__(self, executor, *, fps: int,
                 limits: _safety.SafetyLimits | None = None,
                 recorder=None, clock=time.monotonic, wait=_util.precise_wait):
        self.executor = executor
        self.fps = int(fps)
        self.dt = 1.0 / self.fps
        self.limits = limits or _safety.SafetyLimits()
        self.clamp = _safety.Clamp(self.limits)
        self.recorder = recorder or _recorder.NullRecorder()
        self._clock = clock
        self._wait = wait

    # --- grounding ----------------------------------------------------------

    def ground(self) -> None:
        """Re-anchor the clamp at the measured arm — call after any gap in
        commanding (connect, pause, ramp) so the first delta isn't measured
        against a stale knot."""
        self.clamp.reset(self.executor.arm_q())

    def soft_start(self, row: np.ndarray, *, settle_s: float = 2.0) -> None:
        """First send, unstamped: unitree_deploy's own drive_to_waypoint soft
        ramp takes the arm from wherever it is to `row`. `settle_s` gives the
        ramp time to land (pass 0.0 for a MockExecutor); the clamp is then
        grounded at the measured result."""
        self.executor.send(np.asarray(row, dtype=np.float64))
        if settle_s > 0:
            time.sleep(settle_s)
        self.ground()

    # --- the per-row invariant ---------------------------------------------

    def send_row(self, row, t_target: float, *, step: int | None = None) -> np.ndarray:
        """sanity → clamp (arm slice only) → record → send. Returns the row
        actually sent (post-clamp copy)."""
        row = np.asarray(row, dtype=np.float64).copy()
        if not _safety.sanity_check_joint_row(row):
            raise InsaneRowError(f"insane joint row at step {step}")
        before = row[_actions.ARM].copy()
        row[_actions.ARM] = self.clamp(row[_actions.ARM], self.dt)
        if not np.array_equal(before, row[_actions.ARM]):
            self.recorder.log("clamp", step=step,
                              max_step=float(np.abs(
                                  before - row[_actions.ARM]).max()))
        self.executor.send(row, t_target)
        self.recorder.log("action", step=step, row=row)
        return row

    # --- the replay-tool loop ----------------------------------------------

    def stream(self, rows, *, on_tick=None, start_step: int = 0) -> bool:
        """Pace an iterable of (26,) rows at fps on an ABSOLUTE schedule
        (t0 + (k+1)·dt — drift-free, matching what the replay tools always
        did). `on_tick(k, sent_row)` runs after each send, before the wait —
        the instrumentation hook replay_diag/replay-actions capture
        measurements from. Returns True if the iterable completed, False on
        Ctrl-C (after DAMPING — the arm is limp and latched when this
        returns False)."""
        t0 = self._clock()
        try:
            for k, row in enumerate(rows):
                t_cycle_end = t0 + (k + 1) * self.dt
                sent = self.send_row(row, t_cycle_end + self.dt,
                                     step=start_step + k)
                if on_tick is not None:
                    on_tick(k, sent)
                self._wait(t_cycle_end)
            return True
        except (KeyboardInterrupt, InsaneRowError) as exc:
            print(f"\n{'interrupted' if isinstance(exc, KeyboardInterrupt) else exc}"
                  " — DAMPING.")
            self.executor.damp()
            if isinstance(exc, InsaneRowError):
                raise
            return False

    # --- the reset ramp ------------------------------------------------------

    def ramp_to(self, q_target: np.ndarray, hand_targets: dict,
                hand_start: dict, *, ramp_s: float = 3.0,
                max_speed: float = 0.5, settle_s: float = 0.4,
                abort=None) -> None:
        """Linear ramp of arm+hands to a target posture, `max_speed` (rad/s)
        capped, streamed future-stamped through the same executor.send path,
        then a settle wait. Deliberately BYPASSES the clamp — the ramp's own
        speed cap is the limit here, and the clamp would fight a long
        reposition. `abort()` (e.g. `lambda: watchdog.tripped`) is checked
        every tick; raising RuntimeError mid-ramp leaves the interpolator
        holding the last waypoint."""
        q_target = np.asarray(q_target, dtype=np.float64)
        q0 = self.executor.arm_q()
        n = max(1, int(round(ramp_s * self.fps)))
        cap = max_speed * self.dt                    # rad per tick
        worst = float(np.abs(q_target - q0).max())
        if cap > 0:
            n = max(n, int(np.ceil(worst / cap)))
        row = np.zeros(_actions.ROBOT_DIM)
        for k in range(1, n + 1):
            if abort is not None and abort():
                raise RuntimeError("aborted during the ramp")
            t_cycle_end = self._clock() + self.dt
            a = k / n
            row[_actions.ARM] = (1.0 - a) * q0 + a * q_target
            for h in layout.HANDS:
                row[_actions.HAND[h]] = ((1.0 - a) * hand_start[h]
                                         + a * hand_targets[h])
            self.executor.send(row, t_cycle_end + self.dt)
            self._wait(t_cycle_end)
        # let the 500 Hz interpolator land + the arm settle, then re-ground
        self._wait(self._clock() + settle_s)
        self.ground()
