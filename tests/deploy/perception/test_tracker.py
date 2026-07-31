"""tracker.py: the fast (~20-30 Hz) per-object position tracker.

Two things this file must nail down without any real video/GPU:
  1. Causal outlier rejection -- feed a smooth synthetic 3D trajectory with a
     few injected single-tick jumps; the tracker must stay close to the TRUE
     trajectory and must NOT follow the injected jumps. This is the main
     behavior worth verifying here, so it gets the most scrutiny.
  2. OneEuroSE3 smoothing behavior, reused (not reimplemented) from
     `ego2g1.kin.filters` -- a genuine, sustained position change should lag
     in (not jump discontinuously) and eventually converge, mirroring
     `tests/test_deploy_conversion.py`'s own noisy-vs-smoothed comparison
     style for the same filter.
"""

import numpy as np
import pytest

from ego2g1.deploy.perception.tracker import (
    ConstantVelocityKalman3D,
    ObjectTracker,
    _causal_robust_threshold,
)

DT = 1.0 / 20.0  # 20 Hz, within the plan's 20-30 Hz fast-tracker band


def _pose(position, rotation=None):
    T = np.eye(4)
    T[:3, 3] = position
    if rotation is not None:
        T[:3, :3] = rotation
    return T


def _smooth_true_trajectory(n, *, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * DT
    true_pos = np.stack(
        [
            0.02 * np.sin(2 * np.pi * 0.5 * t),
            0.01 * np.cos(2 * np.pi * 0.3 * t),
            0.30 + 0.005 * t,
        ],
        axis=1,
    )
    return true_pos, rng


class TestCausalRobustThreshold:
    def test_few_samples_falls_back_to_minimum(self):
        assert _causal_robust_threshold([], minimum=0.01) == 0.01
        assert _causal_robust_threshold([0.5], minimum=0.01) == 0.01

    def test_grows_with_spread_but_never_below_minimum(self):
        tight = _causal_robust_threshold([0.001, 0.0012, 0.0009, 0.0011], minimum=0.01)
        assert tight == pytest.approx(0.01)
        spread = _causal_robust_threshold([0.01, 0.05, 0.02, 0.08, 0.03], minimum=0.01)
        assert spread > 0.01

    def test_only_uses_the_values_given_no_hidden_future_lookup(self):
        # a pure function of its argument -- there is nothing else it could
        # possibly be causal/acausal about, but pin the property explicitly.
        # (minimum kept small so the MAD term, not the floor, governs here.)
        history = [0.010, 0.012, 0.009, 0.011]
        t1 = _causal_robust_threshold(history, minimum=0.001)
        t2 = _causal_robust_threshold(list(history), minimum=0.001)
        assert t1 == t2
        history_with_future = history + [10.0]  # simulates "a future sample"
        assert t1 != _causal_robust_threshold(history_with_future, minimum=0.001)


class TestConstantVelocityKalman3D:
    def test_predict_without_correct_extrapolates_constant_velocity(self):
        kf = ConstantVelocityKalman3D(np.array([0.0, 0.0, 0.0]))
        kf.x[3:] = [1.0, 0.0, 0.0]  # seed velocity directly: 1 m/s in x
        p1 = kf.predict(dt=0.1)
        p2 = kf.predict(dt=0.1)
        np.testing.assert_allclose(p1, [0.1, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(p2, [0.2, 0.0, 0.0], atol=1e-9)

    def test_correct_pulls_state_toward_measurement(self):
        kf = ConstantVelocityKalman3D(np.array([0.0, 0.0, 0.0]))
        kf.predict(dt=0.1)
        before = kf.position
        after = kf.correct(np.array([1.0, 0.0, 0.0]))
        assert np.linalg.norm(after - [1.0, 0.0, 0.0]) < np.linalg.norm(
            before - [1.0, 0.0, 0.0]
        )

    def test_reset_zeroes_velocity_and_snaps_position(self):
        kf = ConstantVelocityKalman3D(np.array([0.0, 0.0, 0.0]))
        kf.x[3:] = [5.0, 5.0, 5.0]
        kf.reset(np.array([9.0, -1.0, 2.0]))
        np.testing.assert_allclose(kf.position, [9.0, -1.0, 2.0])
        np.testing.assert_allclose(kf.velocity, [0.0, 0.0, 0.0])


class TestCausalOutlierRejection:
    """The main behavior: injected single-tick jumps must be rejected and
    must not pull the tracked pose toward them."""

    OUTLIER_FRAMES = {40, 41, 70}
    OUTLIER_OFFSET = np.array([0.15, -0.10, 0.05])  # ~19 cm jump

    def _run(self):
        n = 100
        true_pos, rng = _smooth_true_trajectory(n)
        tracker = ObjectTracker(_pose(true_pos[0]), min_residual_m=0.01, mad_scale=8.0)

        accepted_flags = {}
        errors_to_true = np.zeros(n)
        errors_to_outlier = {}
        for i in range(1, n):
            measured = true_pos[i] + rng.normal(scale=0.001, size=3)
            if i in self.OUTLIER_FRAMES:
                measured = measured + self.OUTLIER_OFFSET
            pose, accepted = tracker.update(measured, DT)
            accepted_flags[i] = accepted
            errors_to_true[i] = np.linalg.norm(pose[:3, 3] - true_pos[i])
            if i in self.OUTLIER_FRAMES:
                errors_to_outlier[i] = np.linalg.norm(pose[:3, 3] - measured)
        return accepted_flags, errors_to_true, errors_to_outlier

    def test_injected_outliers_are_all_rejected(self):
        accepted_flags, _, _ = self._run()
        for frame in self.OUTLIER_FRAMES:
            assert accepted_flags[frame] is False, (
                f"frame {frame}: injected outlier was accepted, expected rejection"
            )

    def test_tracked_pose_stays_close_to_true_trajectory_throughout(self):
        _, errors_to_true, _ = self._run()
        # Non-outlier ticks: tight tracking of the smooth true trajectory
        # (OneEuroSE3 lag on this trajectory's own curvature tops out
        # around ~1.8 cm even with zero outliers -- see
        # TestSmoothTrackingWithoutOutliers -- so the bound here allows for
        # that inherent smoothing lag, not just measurement noise).
        non_outlier_errors = np.delete(errors_to_true[1:], sorted(
            i - 1 for i in self.OUTLIER_FRAMES
        ))
        assert non_outlier_errors.max() < 0.025, (
            "tracking error on ordinary ticks should stay small (smoothing "
            "lag only, no outlier contamination)"
        )
        # Outlier ticks: error to the TRUE trajectory must stay comparably
        # small too -- this is the crux of the test, i.e. it did not chase
        # the ~15-19 cm injected jump (bound is far below the jump size).
        for frame in self.OUTLIER_FRAMES:
            assert errors_to_true[frame] < 0.03, (
                f"frame {frame}: tracker drifted {errors_to_true[frame]:.4f} m "
                "toward an injected outlier it should have rejected"
            )

    def test_tracked_pose_does_not_follow_the_injected_outliers(self):
        _, errors_to_true, errors_to_outlier = self._run()
        for frame in self.OUTLIER_FRAMES:
            # far from the (rejected) corrupted measurement...
            assert errors_to_outlier[frame] > 0.10
            # ...and, by construction, much closer to the true trajectory.
            assert errors_to_true[frame] < errors_to_outlier[frame] / 3.0


class TestSmoothTrackingWithoutOutliers:
    def test_kalman_plus_smoothing_tracks_a_clean_trajectory_tightly(self):
        n = 100
        true_pos, rng = _smooth_true_trajectory(n, seed=7)
        tracker = ObjectTracker(_pose(true_pos[0]), min_residual_m=0.01, mad_scale=8.0)
        errors = []
        for i in range(1, n):
            measured = true_pos[i] + rng.normal(scale=0.001, size=3)
            pose, accepted = tracker.update(measured, DT)
            assert accepted, f"frame {i}: a clean measurement should never be rejected"
            errors.append(np.linalg.norm(pose[:3, 3] - true_pos[i]))
        # OneEuroSE3's own smoothing lag on a continuously-curving trajectory
        # (not measurement noise) is the dominant error source here.
        assert max(errors) < 0.025


class TestReacquisitionAndSmoothingLag:
    """A SUSTAINED, mutually-consistent change (unlike a transient one-tick
    outlier) is a genuine change -- the tracker must eventually follow it,
    but via OneEuroSE3's smoothing (lag in gradually), never in one
    discontinuous jump."""

    def test_sustained_step_is_not_a_single_tick_jump_and_converges(self):
        rng = np.random.default_rng(1)
        tracker = ObjectTracker(_pose(np.zeros(3)), min_residual_m=0.01, mad_scale=8.0)

        # Settle near the origin first, so there is a real (small) residual
        # history / threshold in place before the step -- a more realistic
        # scenario than an untrained filter.
        for _ in range(40):
            measured = rng.normal(scale=0.001, size=3)
            tracker.update(measured, DT)

        step_target = np.array([0.05, 0.0, 0.0])  # 5 cm sustained step
        positions = []
        for _ in range(40):
            measured = step_target + rng.normal(scale=0.001, size=3)
            pose, _accepted = tracker.update(measured, DT)
            positions.append(pose[:3, 3].copy())
        positions = np.asarray(positions)

        # No discontinuous jump: every single tick moves only a small
        # fraction of the full 5 cm step, even across the reacquisition tick.
        per_tick_deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        assert per_tick_deltas.max() < 0.02, (
            "a single tick moved a large fraction of the whole step -- "
            "that is a discontinuous jump, not smoothing"
        )
        # It genuinely lags at first (does not snap to the target immediately)...
        first_tick_error = np.linalg.norm(positions[0] - step_target)
        assert first_tick_error > 0.03
        # ...but does eventually converge close to the new true position.
        final_error = np.linalg.norm(positions[-1] - step_target)
        assert final_error < 0.005

    def test_transient_single_tick_outlier_does_not_trigger_reacquisition(self):
        """Sanity check that the two behaviors don't fight each other: a
        single-frame outlier surrounded by good data must NOT be treated as
        the start of a genuine change, even though it looks identical to a
        step for exactly one tick."""
        rng = np.random.default_rng(3)
        n = 60
        true_pos, _ = _smooth_true_trajectory(n, seed=3)
        tracker = ObjectTracker(_pose(true_pos[0]), min_residual_m=0.01, mad_scale=8.0)

        outlier_frame = 30
        for i in range(1, n):
            measured = true_pos[i] + rng.normal(scale=0.001, size=3)
            if i == outlier_frame:
                measured = measured + np.array([0.2, 0.0, 0.0])
            pose, accepted = tracker.update(measured, DT)
            if i == outlier_frame:
                assert not accepted
            elif i == outlier_frame + 1:
                # back on the true trajectory immediately after -- confirms
                # the tracker never reset onto the single bad frame.
                assert np.linalg.norm(pose[:3, 3] - true_pos[i]) < 0.01


class TestOrientationDecoupledFromPositionTracking:
    def test_set_orientation_changes_only_rotation_not_position(self):
        tracker = ObjectTracker(_pose(np.array([1.0, 2.0, 3.0])))
        pose_before = tracker.pose
        new_rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        tracker.set_orientation(new_rotation)
        # OneEuroSE3 smooths rotation too (slerp), so convergence takes a
        # few ticks at a stationary position -- exactly the same lag
        # behavior as position, just applied to the rotation component.
        for _ in range(60):
            pose, _ = tracker.update(np.array([1.0, 2.0, 3.0]), DT)
        np.testing.assert_allclose(pose[:3, 3], pose_before[:3, 3], atol=1e-6)
        np.testing.assert_allclose(pose[:3, :3], new_rotation, atol=1e-3)


class TestPredictOnly:
    def test_predict_advances_without_a_measurement(self):
        tracker = ObjectTracker(_pose(np.zeros(3)))
        # give it some velocity via a couple of accepted updates first
        for i in range(5):
            tracker.update(np.array([0.01 * i, 0.0, 0.0]), DT)
        pose_before = tracker.pose
        pose_after = tracker.predict(DT)
        assert not tracker.last_accepted
        # constant-velocity extrapolation should move forward, not stand still
        assert not np.allclose(pose_after[:3, 3], pose_before[:3, 3])
