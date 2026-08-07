"""The S2 tracker: MAD gate kept, extrapolation deleted, windows in seconds."""

from __future__ import annotations

import numpy as np
import pytest

from ego2g1.deploy.perception.v2.object_tracker import (
    ObjectTracker, robust_speed_threshold,
)

from .conftest import pose


def _tracker(xyz=(0.5, 0.0, 0.0), **kw):
    # Smoothing off by default so the tests assert on the FILTER's decisions
    # rather than on OneEuro's lag, which is tuned separately.
    kw.setdefault("one_euro_kwargs", {"min_cutoff": 1e6, "beta": 0.0})
    return ObjectTracker(pose(xyz), **kw)


def test_hold_does_not_extrapolate():
    """The whole point of deleting the constant-velocity model: with a fresh
    measurement every round there are no in-between ticks to carry, and a
    state that drifts on its own between measurements is not filtering, it is
    inventing."""
    t = _tracker()
    for _ in range(5):
        t.update(t.position + np.array([0.02, 0, 0]), 0.22)
    moving = t.position.copy()
    for _ in range(10):
        t.hold(0.22)
    np.testing.assert_allclose(t.position, moving, atol=1e-12)


def test_hold_accumulates_visible_staleness():
    t = _tracker()
    t.hold(0.22)
    t.hold(0.22)
    assert t.since_update_s == pytest.approx(0.44)
    t.update(t.position, 0.22)
    assert t.since_update_s == 0.0


def test_outlier_is_rejected_and_the_estimate_holds():
    t = _tracker()
    for i in range(10):
        t.update(np.array([0.5 + 0.001 * i, 0, 0]), 0.22)
    before = t.position.copy()
    _, accepted = t.update(np.array([2.0, 0, 0]), 0.22)   # 1.5 m teleport
    assert not accepted
    np.testing.assert_allclose(t.position, before, atol=1e-12)


def test_consistent_rejections_reacquire():
    """A gate that rejects forever once fooled is worse than no gate: the
    object may genuinely have moved, or been re-detected somewhere new."""
    t = _tracker(reacquire_window=3, reacquire_consistency_m=0.02)
    for i in range(10):
        t.update(np.array([0.5 + 0.001 * i, 0, 0]), 0.22)
    new = np.array([2.0, 0.0, 0.0])
    accepted = [t.update(new, 0.22)[1] for _ in range(3)]
    assert accepted == [False, False, True]
    np.testing.assert_allclose(t.position, new, atol=1e-12)


def test_a_lone_outlier_never_reacquires():
    t = _tracker(reacquire_window=3)
    good = np.array([0.5, 0.0, 0.0])
    for _ in range(6):
        t.update(good, 0.22)
    for _ in range(4):
        assert t.update(np.array([2.0, 0, 0]), 0.22)[1] is False
        assert t.update(good, 0.22)[1] is True      # interleaved good sample
    np.testing.assert_allclose(t.position, good, atol=1e-12)


def test_rejections_far_apart_are_not_a_run():
    t = _tracker(reacquire_window=2, reacquire_max_gap_s=1.0)
    for _ in range(6):
        t.update(np.array([0.5, 0, 0]), 0.22)
    assert t.update(np.array([2.0, 0, 0]), 0.22)[1] is False
    for _ in range(10):                              # 2.2 s of holding
        t.hold(0.22)
    assert t.update(np.array([2.0, 0, 0]), 0.22)[1] is False, (
        "two bad samples seconds apart are not evidence of a real move")


def test_the_gate_is_rate_invariant():
    """S2's argument applied to metres as well as ticks: a DISPLACEMENT
    threshold silently means a different speed at every rate, and the loop is
    free-running. A 0.3 m/s object must be tracked at 30 Hz and at 4.5 Hz; a
    0.5 m jump must be rejected at both."""
    for dt in (1 / 30, 1 / 4.5):
        t = _tracker()
        step = 0.3 * dt                              # 0.3 m/s, plausible
        for i in range(1, 8):
            _, accepted = t.update(np.array([0.5 + step * i, 0, 0]), dt)
            assert accepted, f"legitimate motion rejected at dt={dt:.3f}"
        _, accepted = t.update(t.position + np.array([0.5, 0, 0]), dt)
        assert not accepted, f"0.5 m jump accepted at dt={dt:.3f}"


def test_history_is_trimmed_by_age_not_by_count():
    t = _tracker(history_s=1.0)
    for _ in range(20):
        t.update(t.position + np.array([0.001, 0, 0]), 0.05)   # 1.0 s of data
    n_fast = len(t._history)                          # noqa: SLF001
    t2 = _tracker(history_s=1.0)
    for _ in range(4):
        t2.update(t2.position + np.array([0.001, 0, 0]), 0.25)  # 1.0 s again
    assert n_fast > len(t2._history)                  # noqa: SLF001
    # ...but both span the same amount of REAL TIME, which is the invariant
    # that makes the threshold mean the same thing at both rates.
    for tracker in (t, t2):
        h = tracker._history                          # noqa: SLF001
        assert h[-1][0] - h[0][0] <= 1.0 + 1e-9


def test_orientation_holds_between_refreshes():
    """S1's asymmetry: position keeps updating while rotation holds, because
    an orientation from an occluded crop can be wrong by 180 degrees."""
    t = _tracker()
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t.set_orientation(R)
    for _ in range(5):
        t.update(t.position + np.array([0.01, 0, 0]), 0.22)
    np.testing.assert_allclose(t.pose[:3, :3], R, atol=1e-12)


def test_robust_threshold_falls_back_before_it_has_history():
    assert robust_speed_threshold([], minimum=1.5) == 1.5
    assert robust_speed_threshold([0.1], minimum=1.5) == 1.5
    assert robust_speed_threshold([0.1] * 10, minimum=0.0) == pytest.approx(0.1)


def test_shape_errors_are_loud():
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        ObjectTracker(np.eye(3))
    with pytest.raises(ValueError, match=r"\(3,\)"):
        _tracker().update(np.zeros(4), 0.1)
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        _tracker().set_orientation(np.eye(4))
