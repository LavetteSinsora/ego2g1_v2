"""The fast (~20-30 Hz) tracker stage between detector refreshes (docs/relation_deploy_plan.md §5.3).

Per the plan, this stage must be LIGHTWEIGHT -- CoTracker (the training-side
tool `data_extraction_zh` uses) is explicitly out of scope here; this is a
constant-velocity Kalman filter over each object's 3D position, which is
simpler to make fully deterministic and unit-testable without real video than
an optical-flow re-projection would be (the plan explicitly allows either and
leans this way for exactly that reason).

Two more pieces, both required by the plan and both causal (no future
frames, unlike the offline reference this technique is adapted from):

  Causal outlier rejection   `data_extraction_zh/src/ego_relation/
                              s1_pico_mode2/smoothing.py::_robust_threshold`
                              is the reference for the STATISTICAL technique
                              (median + k * MAD, i.e. a robust z-score
                              threshold) -- but that file's callers
                              (`_repair_translation_outliers` etc.) are
                              acausal: they use a Savitzky-Golay window and a
                              (frame-1, frame+1) midpoint, i.e. future frames.
                              Here the threshold is built ONLY from past
                              accepted residuals (a running history), and a
                              rejected measurement is never "repaired" from a
                              future sample -- the tracker just holds/
                              extrapolates the Kalman prediction instead.
  OneEuroSE3 smoothing        Reused verbatim from `ego2g1.kin.filters`
                              (already used for EEF target smoothing in
                              `deploy/actions.py`) -- not reimplemented here.
                              It smooths the full SE(3) pose (position +
                              rotation); this module owns position (the fast
                              part), `orientation.py` owns rotation refresh
                              (the slow part) via `ObjectTracker.set_orientation`.

A rejected measurement is not necessarily noise, though -- the object may
have genuinely moved (or been re-detected somewhere new after an occlusion),
and a gate that rejects forever once fooled once is worse than no gate at
all (the exact failure mode `latch.py`'s design in
docs/relation_deploy_plan.md §5.4 calls out for the grasp/latch decision,
and the identical risk applies here). So a single rejected sample is held
against the Kalman prediction as usual, but if several rejected samples IN A
ROW agree with each other (not with the stale prediction) -- `reacquire_window`
consecutive measurements within `reacquire_consistency_m` of one another --
that run is treated as a genuine change: the Kalman state is reset to it and
the residual history clears, so the threshold starts adapting to the new
regime instead of judging it forever against the old one. A single transient
outlier (one bad frame surrounded by good ones) never accumulates enough
consecutive rejections to trigger this and is simply held/extrapolated.
"""

from __future__ import annotations

import collections

import numpy as np

from ...kin.filters import OneEuroSE3


def _causal_robust_threshold(
    residual_history: collections.deque[float] | list[float],
    *,
    minimum: float,
    scale: float = 8.0,
) -> float:
    """Median + `scale` * 1.4826 * MAD over PAST residuals only.

    Adapted from `smoothing.py::_robust_threshold`'s idea (robust z-score via
    MAD, `1.4826` is the constant that makes MAD a consistent estimator of
    the standard deviation for Gaussian noise) -- but causal: `residual_history`
    must contain only residuals already observed at or before the current
    tick, never anything computed from a later frame. With fewer than 2
    samples, MAD is undefined/zero, so fall back to `minimum` alone (accept
    almost anything until there's enough history to judge by).
    """
    if len(residual_history) < 2:
        return float(minimum)
    values = np.asarray(residual_history, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(float(minimum), median + scale * 1.4826 * mad)


class ConstantVelocityKalman3D:
    """Textbook linear KF over a 3D point: state = [position(3), velocity(3)].

    Deliberately the simplest thing that works: constant-velocity process
    model, direct position measurement. `predict()` and `correct()` are
    separate calls (not a combined `predict_and_update`) so a caller can
    advance the state without ever correcting it -- that separation is
    exactly what causal outlier rejection needs (reject == predict, don't
    correct; the state is left at the a-priori extrapolation).
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        *,
        process_noise_pos: float = 1e-5,
        process_noise_vel: float = 1e-3,
        measurement_noise: float = 1e-4,
        initial_velocity_variance: float = 1.0,
    ):
        initial_position = np.asarray(initial_position, dtype=np.float64)
        if initial_position.shape != (3,):
            raise ValueError(f"expected (3,), got {initial_position.shape}")
        self.x = np.concatenate([initial_position, np.zeros(3)])  # (6,)
        self.P = np.diag(
            [1e-6, 1e-6, 1e-6, initial_velocity_variance,
             initial_velocity_variance, initial_velocity_variance]
        )
        self._q_pos = float(process_noise_pos)
        self._q_vel = float(process_noise_vel)
        self._r = np.eye(3) * float(measurement_noise)
        self._H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def predict(self, dt: float) -> np.ndarray:
        """Advance state by `dt` under the constant-velocity model. Returns
        the predicted (a-priori) 3D position -- this IS the "hold the last
        good estimate" behavior when no correction follows."""
        dt = max(float(dt), 0.0)
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        Q = np.diag(
            [self._q_pos, self._q_pos, self._q_pos,
             self._q_vel, self._q_vel, self._q_vel]
        ) * max(dt, 1e-3)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[:3].copy()

    def correct(self, measured_position: np.ndarray) -> np.ndarray:
        """Bayesian update against a real position measurement. Only call
        this for an ACCEPTED measurement -- rejected ones must skip this."""
        z = np.asarray(measured_position, dtype=np.float64)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._r
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self._H) @ self.P
        return self.x[:3].copy()

    def reset(self, position: np.ndarray, *, initial_velocity_variance: float = 1.0) -> None:
        """Hard re-seed at a new position with zero velocity and inflated
        velocity uncertainty -- used for re-acquisition after a confirmed
        (not transient) change, so the filter doesn't spend many ticks
        fighting its own stale velocity estimate."""
        position = np.asarray(position, dtype=np.float64)
        self.x = np.concatenate([position, np.zeros(3)])
        self.P = np.diag(
            [1e-6, 1e-6, 1e-6, initial_velocity_variance,
             initial_velocity_variance, initial_velocity_variance]
        )

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()


class ObjectTracker:
    """Per-object fast tracker: Kalman position + causal outlier gate +
    OneEuroSE3 pose smoothing. One instance per tracked object instance_id.

    Usage per tick:
      - a new position measurement is available (detector centroid this
        cycle, or an optical-flow re-projection on an in-between tick):
        call `update(measured_position, dt)` -> (smoothed_pose, accepted).
      - no measurement this tick (pure extrapolation between detector/flow
        updates): call `predict(dt)` -> smoothed_pose.
      - the (independently, slowly-paced) orientation stage has a fresh
        rotation: call `set_orientation(rotation_matrix)` -- this module
        does not decide WHEN that happens, per the plan's "a caller
        controls the cadence, not this module" requirement.
    """

    def __init__(
        self,
        initial_pose: np.ndarray,
        *,
        min_residual_m: float = 0.01,
        mad_scale: float = 8.0,
        residual_history_len: int = 30,
        process_noise_pos: float = 1e-5,
        process_noise_vel: float = 1e-3,
        measurement_noise: float = 1e-4,
        reacquire_window: int = 3,
        reacquire_consistency_m: float | None = None,
        one_euro_kwargs: dict | None = None,
    ):
        initial_pose = np.asarray(initial_pose, dtype=np.float64)
        if initial_pose.shape != (4, 4):
            raise ValueError(f"expected (4, 4) pose, got {initial_pose.shape}")
        self._kalman = ConstantVelocityKalman3D(
            initial_pose[:3, 3],
            process_noise_pos=process_noise_pos,
            process_noise_vel=process_noise_vel,
            measurement_noise=measurement_noise,
        )
        self._rotation = initial_pose[:3, :3].copy()
        self._smoother = OneEuroSE3(**(one_euro_kwargs or {}))
        self._min_residual_m = float(min_residual_m)
        self._mad_scale = float(mad_scale)
        self._residual_history: collections.deque[float] = collections.deque(
            maxlen=int(residual_history_len)
        )
        # Reacquisition: consecutive rejected measurements that agree with
        # EACH OTHER (not with the stale prediction) beyond this window are
        # treated as a genuine change rather than noise. Default consistency
        # radius is a few multiples of min_residual_m -- tight enough that
        # actual sensor noise/jitter still won't trigger it by accident.
        self._reacquire_window = int(reacquire_window)
        self._reacquire_consistency_m = float(
            reacquire_consistency_m
            if reacquire_consistency_m is not None
            else 3.0 * min_residual_m
        )
        self._pending_rejections: collections.deque[np.ndarray] = collections.deque(
            maxlen=self._reacquire_window
        )
        self._last_smoothed = initial_pose.copy()
        self._last_accepted = True
        # Seed the smoother so tick 0 doesn't snap from a zero state.
        self._smoother.filter(initial_pose, dt=0.0)

    def _pose_from(self, position: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self._rotation
        T[:3, 3] = position
        return T

    def predict(self, dt: float) -> np.ndarray:
        """Advance with no new measurement this tick (pure extrapolation)."""
        position = self._kalman.predict(dt)
        self._last_smoothed = self._smoother.filter(self._pose_from(position), dt)
        self._last_accepted = False
        return self._last_smoothed.copy()

    def update(self, measured_position: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
        """Feed a new 3D position measurement. Returns (smoothed_pose, accepted).

        `accepted=False` means the residual from the Kalman's own a-priori
        prediction exceeded the causal MAD-robust threshold -- the
        measurement is discarded (never `correct()`-ed, never added to the
        residual history) and the tracker instead holds/extrapolates the
        prediction, exactly like a tick with no measurement at all -- UNLESS
        this is the `reacquire_window`-th consecutive rejection and all of
        them agree with each other (see class docstring), in which case it
        is treated as a genuine change: the Kalman state is reset onto it
        and `accepted=True` is returned for this tick.
        """
        measured_position = np.asarray(measured_position, dtype=np.float64)
        predicted = self._kalman.predict(dt)
        residual = float(np.linalg.norm(measured_position - predicted))
        threshold = _causal_robust_threshold(
            self._residual_history,
            minimum=self._min_residual_m,
            scale=self._mad_scale,
        )
        accepted = residual <= threshold
        if accepted:
            position = self._kalman.correct(measured_position)
            self._residual_history.append(residual)
            self._pending_rejections.clear()
        else:
            self._pending_rejections.append(measured_position)
            if len(self._pending_rejections) == self._reacquire_window and (
                self._reacquire_window == 1 or self._mutually_consistent(self._pending_rejections)
            ):
                self._kalman.reset(measured_position)
                position = self._kalman.position
                self._residual_history.clear()
                self._pending_rejections.clear()
                accepted = True
            else:
                position = predicted  # hold the a-priori extrapolation
        self._last_smoothed = self._smoother.filter(self._pose_from(position), dt)
        self._last_accepted = accepted
        return self._last_smoothed.copy(), accepted

    def _mutually_consistent(self, points) -> bool:
        points = np.asarray(points, dtype=np.float64)
        centroid = points.mean(axis=0)
        spread = float(np.max(np.linalg.norm(points - centroid, axis=1)))
        return spread <= self._reacquire_consistency_m

    def set_orientation(self, rotation_matrix: np.ndarray) -> None:
        """Refresh the rotation component. Call this whenever the caller's
        own (much slower, e.g. ~0.2 Hz) orientation cadence produces a new
        estimate -- typically `orientation.OrientationRefiner.refresh(...)`'s
        return value. This module has no timer of its own."""
        self._rotation = np.asarray(rotation_matrix, dtype=np.float64).copy()

    @property
    def pose(self) -> np.ndarray:
        """Latest smoothed (4, 4) pose estimate."""
        return self._last_smoothed.copy()

    @property
    def last_accepted(self) -> bool:
        return self._last_accepted
