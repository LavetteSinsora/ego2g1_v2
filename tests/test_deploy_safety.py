"""Clamp, Watchdog, and the action sanity checks."""

import numpy as np

from ego2g1.deploy import actions as _actions
from ego2g1.deploy.safety import (
    Clamp,
    SafetyLimits,
    Watchdog,
    sanity_check_joint_row,
    sanity_check_model_action,
)
from ego2g1.core import layout


def test_clamp_limits_the_step_and_counts():
    limits = SafetyLimits(max_joint_step=0.1)
    clamp = Clamp(limits)
    clamp.reset(np.zeros(14))
    q = np.zeros(14)
    q[3] = 1.0                            # a 1 rad jump in one tick
    out = clamp(q, dt=1 / 30)
    assert out[3] <= 0.1 + 1e-9
    assert clamp.clamped_ticks == 1
    assert clamp.max_seen >= 1.0
    # small steps pass through untouched
    q2 = out.copy()
    q2[3] += 0.05
    np.testing.assert_allclose(clamp(q2, 1 / 30), q2)


def test_clamp_velocity_cap_binds_on_tiny_dt():
    clamp = Clamp(SafetyLimits(max_joint_step=0.15, max_joint_vel=1.0))
    clamp.reset(np.zeros(14))
    q = np.zeros(14)
    q[0] = 0.15                           # under the step cap...
    out = clamp(q, dt=0.01)               # ...but 15 rad/s over 10 ms
    assert out[0] <= 1.0 * 0.01 + 1e-9


def test_watchdog_strikes_then_trips_and_latches():
    tripped = []
    w = Watchdog(SafetyLimits(max_state_age=0.1, trip_after=3),
                 on_trip=lambda: tripped.append(1))
    for _ in range(2):
        w.check_state_age(0.5)
    assert not w.tripped                  # two strikes: not yet
    w.check_state_age(0.05)               # recovery resets the count
    for _ in range(3):
        w.check_state_age(0.5)
    assert w.tripped and tripped == [1]
    w.trip("again")                       # latched: handler not re-run
    assert tripped == [1]
    assert "stale" in w.reason


def test_watchdog_starvation_is_duration_based():
    w = Watchdog(SafetyLimits(max_starvation=1.0), on_trip=lambda: None)
    w.check_starvation(False, now=0.0)
    w.check_starvation(False, now=0.5)
    assert not w.tripped                  # brief emptiness is normal (startup)
    w.check_starvation(True, now=0.6)     # plan arrived: window resets
    w.check_starvation(False, now=1.0)
    w.check_starvation(False, now=1.9)
    assert not w.tripped
    w.check_starvation(False, now=2.5)    # 1.5 s of sustained nothing
    assert w.tripped


def test_watchdog_tracking_error():
    w = Watchdog(SafetyLimits(max_tracking_error_m=0.1, trip_after=2),
                 on_trip=lambda: None)
    w.check_tracking(0.05)
    w.check_tracking(0.2)
    w.check_tracking(0.2)
    assert w.tripped


def test_sanity_model_action():
    good = np.zeros(layout.DIM)
    assert sanity_check_model_action(good)
    bad_nan = good.copy()
    bad_nan[0] = np.nan
    assert not sanity_check_model_action(bad_nan)
    bad_far = good.copy()
    bad_far[layout.EEF["left"]][:3] = [2.0, 0, 0]   # a 2 m delta
    assert not sanity_check_model_action(bad_far)
    assert not sanity_check_model_action(np.zeros(29))


def test_sanity_joint_row():
    good = np.zeros(_actions.ROBOT_DIM)
    assert sanity_check_joint_row(good)
    bad = good.copy()
    bad[2] = 7.0                          # outside any G1 joint range
    assert not sanity_check_joint_row(bad)
    bad2 = good.copy()
    bad2[20] = np.inf
    assert not sanity_check_joint_row(bad2)
    assert not sanity_check_joint_row(np.zeros(14))
