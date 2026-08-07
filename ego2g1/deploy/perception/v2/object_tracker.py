"""Per-object position filter for the free-running loop (plan S2).

v1's `../tracker.py` held two separable things: a constant-velocity Kalman
filter, and a causal median+k·MAD outlier gate feeding OneEuroSE3
smoothing. Only one and a half of them survive.

WHY THE PREDICTION GOES
    The Kalman existed to carry ticks BETWEEN detector refreshes — the v1
    cascade ran the detector at ~2 Hz and the loop at 30 Hz, so 14 of every
    15 ticks had no measurement at all and something had to extrapolate. v2
    has a fresh measurement every round by construction (M3: detection and
    tracking on every frame), so there are no in-between ticks to carry. What
    is left of the velocity model is a state that drifts on its own between
    the measurements it is supposed to be filtering.

WHY THE GATE STAYS
    It is the only thing between one bad mask/depth sample and a garbage
    state vector. A mask that bleeds onto the gripper gives a median depth
    several centimetres near, and that lands directly in the policy's input.

WHY THE SMOOTHER IS POSITION-ONLY
    v1 smoothed the full SE(3) pose with `OneEuroSE3`, which slerps rotation
    toward each new sample. Three things make that wrong here:

      * rotation updates are already sparse (only on a usable crop, S1) and
        already gated, so each one is the freshest trustworthy information
        available — slerping it adds lag to the one quantity that has none
        to spare;
      * `OrientationRefiner` picks each new rotation's symmetry branch
        relative to ITS OWN last output, so a smoothed rotation reported
        downstream would be a different reference than the snap uses — two
        clocks on one quantity;
      * the MAD gate below is position-only. Smoothing a quantity that has no
        outlier gate is the wrong pairing: a bad rotation still arrives, just
        more slowly.

    So position is smoothed and rotation passes through. This narrows the
    plan's "keep OneEuroSE3" to "keep OneEuro"; see
    docs/perception_v2_notes.md.

WHY EVERY WINDOW IS IN SECONDS OR METRES
    Every tick-based constant in v1 was tuned for `observe()` at 30 Hz and is
    wrong at 2-4.5 Hz — `max_track_loss_ticks: 3` meant 0.1 s and now means
    0.7 s. The loop is free-running (T1), so its rate is not merely different,
    it VARIES: any constant expressed in samples is wrong the moment the scene
    changes. So the residual history is trimmed by age, not by count, and the
    outlier statistics are gathered on SPEED (m/s) rather than displacement
    (m). That last part is a deviation from the plan's `min_residual_m`
    wording, and it is the same argument the plan makes about tick counts: a
    displacement threshold silently means a different speed at every rate.
    See docs/perception_v2_notes.md.
"""

from __future__ import annotations

import numpy as np

from ....kin.filters import OneEuro

__all__ = ["ObjectTracker", "robust_speed_threshold"]


def robust_speed_threshold(history, *, minimum: float, scale: float = 8.0
                           ) -> float:
    """median + `scale` * 1.4826 * MAD over PAST speeds only, in m/s.

    1.4826 is the constant that makes MAD a consistent estimator of the
    standard deviation for Gaussian noise, so this is a robust z-score
    threshold. Causal by construction: `history` must contain only speeds
    already observed at or before the current round — never anything derived
    from a later frame. (v1's reference implementation in the extraction
    pipeline is acausal, using a Savitzky-Golay window and a (t-1, t+1)
    midpoint; that is fine offline and impossible here.)

    Under two samples MAD is undefined, so fall back to `minimum` alone:
    accept almost anything until there is enough history to judge by.
    """
    if len(history) < 2:
        return float(minimum)
    values = np.asarray(history, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(float(minimum), median + scale * 1.4826 * mad)


class ObjectTracker:
    """One object's position estimate: causal outlier gate + OneEuro.

    Per round, exactly one of:

        update(position, dt) -> (pose, accepted)   a measurement arrived
        hold(dt)             -> pose               none did

    `hold` genuinely holds — it does not extrapolate, and it does not run the
    smoother. The pose is unchanged, because with no motion model there is
    nothing honest to say about where the object went. A held pose that is
    stale is visible as staleness (`since_update_s`); a extrapolated one that
    is wrong is invisible.

    A rejected measurement is not necessarily noise: the object may have
    genuinely moved, or been re-detected somewhere new after an occlusion. A
    gate that rejects forever once fooled is worse than no gate. So a lone
    rejection is held, but `reacquire_window` rejections IN A ROW that agree
    with EACH OTHER (rather than with the stale estimate) are treated as a
    real change — the state resets onto them and the speed history clears, so
    the threshold adapts to the new regime instead of judging it against the
    old one forever. A single transient outlier surrounded by good samples
    never accumulates the run.
    """

    def __init__(
        self,
        initial_pose: np.ndarray,
        *,
        min_residual_m: float = 0.01,
        max_speed_m_s: float = 1.5,
        mad_scale: float = 8.0,
        history_s: float = 6.0,
        reacquire_window: int = 3,
        reacquire_consistency_m: float | None = None,
        reacquire_max_gap_s: float = 1.5,
        one_euro_kwargs: dict | None = None,
    ):
        initial_pose = np.asarray(initial_pose, dtype=np.float64)
        if initial_pose.shape != (4, 4):
            raise ValueError(f"expected (4, 4) pose, got {initial_pose.shape}")

        self._position = initial_pose[:3, 3].copy()
        self._rotation = initial_pose[:3, :3].copy()
        self._smoother = OneEuro(**(one_euro_kwargs or {}))
        self._smoothed_position = self._position.copy()

        # Absolute floor, so a very short dt cannot make the gate arbitrarily
        # tight — the depth quantisation alone is worth several millimetres
        # regardless of how fast the loop happens to be running.
        self._min_residual_m = float(min_residual_m)
        # Speed floor: nothing in this task moves faster than a hand-over.
        # Expressed in m/s so it means the same thing at 2 Hz and at 4.5 Hz.
        self._max_speed = float(max_speed_m_s)
        self._mad_scale = float(mad_scale)
        self._history_s = float(history_s)
        self._history: list[tuple[float, float]] = []   # (clock, speed m/s)

        self._reacquire_window = int(reacquire_window)
        self._reacquire_consistency_m = float(
            reacquire_consistency_m if reacquire_consistency_m is not None
            else 3.0 * min_residual_m)
        self._reacquire_max_gap_s = float(reacquire_max_gap_s)
        self._pending: list[tuple[float, np.ndarray]] = []

        # Internal monotonic clock, advanced by the caller's dt rather than
        # read from the wall. Makes every time-based window deterministic
        # under test, and keeps the tracker usable for replaying a recording
        # faster than real time.
        self._clock = 0.0
        self._since_update_s = 0.0
        self._last_accepted = True
        self._smoother.reset(self._position)            # seed; no tick-0 snap

    # -- read-only state ----------------------------------------------------

    @property
    def pose(self) -> np.ndarray:
        """(4, 4) reported pose: SMOOTHED position, rotation as last set."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._rotation
        T[:3, 3] = self._smoothed_position
        return T

    @property
    def position(self) -> np.ndarray:
        """The UNSMOOTHED accepted position — what the gate and the
        reacquisition run compare against. Smoothing belongs to what is
        reported, not to what is remembered: filtering the state the gate
        judges residuals from would let the gate slowly chase its own lag."""
        return self._position.copy()

    @property
    def last_accepted(self) -> bool:
        return self._last_accepted

    @property
    def since_update_s(self) -> float:
        """Seconds since the last ACCEPTED measurement. This is the staleness
        a caller should threshold on — not a tick count, and not
        "was there a measurement", since a rejected one leaves the estimate
        exactly as stale as no measurement at all."""
        return self._since_update_s

    # -- one round ----------------------------------------------------------

    def hold(self, dt: float) -> np.ndarray:
        """No measurement this round. Hold, do not extrapolate.

        The smoother is deliberately NOT advanced: with nothing new to filter
        it would only creep toward a value it has already reached, turning a
        held estimate into a slowly drifting one."""
        self._advance(dt)
        self._last_accepted = False
        return self.pose

    def update(self, measured_position, dt: float) -> tuple[np.ndarray, bool]:
        """Feed a measured 3D position. Returns (smoothed_pose, accepted).

        `accepted=False` means the implied SPEED exceeded the causal robust
        threshold: the measurement is discarded, never folded into the
        estimate and never added to the history, and the estimate holds — the
        same outcome as a round with no measurement at all. The exception is
        the reacquisition run described in the class docstring.
        """
        measured = np.asarray(measured_position, dtype=np.float64)
        if measured.shape != (3,):
            raise ValueError(f"expected (3,), got {measured.shape}")
        dt = max(float(dt), 1e-6)
        self._advance(dt)

        residual = float(np.linalg.norm(measured - self._position))
        speed = residual / dt
        limit_speed = robust_speed_threshold(
            [s for _, s in self._history],
            minimum=self._max_speed, scale=self._mad_scale)
        # Compare in metres so the absolute floor and the speed limit combine
        # cleanly: whichever is more permissive at this dt wins.
        threshold_m = max(self._min_residual_m, limit_speed * dt)
        accepted = residual <= threshold_m

        if accepted:
            self._accept(measured, speed)
        elif self._reacquire(measured):
            self._accept(measured, speed, clear_history=True)
            accepted = True

        self._smoothed_position = self._smoother.filter(self._position, dt)
        self._last_accepted = accepted
        return self.pose, accepted

    def set_orientation(self, rotation_matrix) -> None:
        """Refresh the rotation component. Called only when a fresh, USABLE
        orientation exists (S1) — this module has no clock and no opinion
        about when that is. Between calls the rotation simply holds, which is
        the whole point: an orientation from an occluded crop can be wrong by
        180 degrees, so no update beats a bad update."""
        R = np.asarray(rotation_matrix, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"expected (3, 3), got {R.shape}")
        self._rotation = R.copy()

    # -- internals ----------------------------------------------------------

    def _advance(self, dt: float) -> None:
        dt = max(float(dt), 0.0)
        self._clock += dt
        self._since_update_s += dt
        # Trim by AGE, not by count: at a varying rate a fixed-length deque
        # spans a different amount of real time every round, so the threshold
        # would silently retune itself whenever the loop sped up or slowed.
        cutoff = self._clock - self._history_s
        if self._history and self._history[0][0] < cutoff:
            self._history = [(t, s) for t, s in self._history if t >= cutoff]

    def _accept(self, measured: np.ndarray, speed: float, *,
                clear_history: bool = False) -> None:
        self._position = measured.copy()
        self._since_update_s = 0.0
        self._pending.clear()
        if clear_history:
            self._history.clear()
        else:
            self._history.append((self._clock, speed))

    def _reacquire(self, measured: np.ndarray) -> bool:
        """True when this rejection completes a run of mutually-consistent
        rejections, i.e. the object really did move and the estimate is what
        is wrong."""
        if self._pending and (self._clock - self._pending[-1][0]
                              > self._reacquire_max_gap_s):
            # Rejections separated by a long gap are not a "run" — they are
            # two unrelated bad samples, and treating them as evidence would
            # let noise minutes apart reset the estimate.
            self._pending.clear()
        self._pending.append((self._clock, measured.copy()))
        if len(self._pending) > self._reacquire_window:
            del self._pending[:-self._reacquire_window]
        if len(self._pending) < self._reacquire_window:
            return False
        points = np.asarray([p for _, p in self._pending], dtype=np.float64)
        spread = float(np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)))
        return spread <= self._reacquire_consistency_m
