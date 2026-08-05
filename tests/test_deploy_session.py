"""ExecutorSession (deploy/session.py): the one road rows take to the executor
(docs/deploy_refactor_plan.md §3, §9 tasks 4-5).

What used to be four hand-rolled loops — two of them with NO clamp and NO
sanity check — now shares these invariants by construction. These tests pin
them: sanity → clamp → future-stamp → record, absolute-schedule pacing, and
damp on interrupt/insanity.
"""

import numpy as np
import pytest

from ego2g1.deploy import actions as _actions
from ego2g1.deploy import safety as _safety
from ego2g1.deploy.executor import MockExecutor
from ego2g1.deploy.session import ExecutorSession, InsaneRowError


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class LogRecorder:
    def __init__(self):
        self.events = []

    def log(self, kind, **fields):
        self.events.append({"kind": kind, **fields})


def _session(recorder=None, max_joint_step=0.15):
    clock = FakeClock()
    ex = MockExecutor(fps=30, clock=clock)
    ex.connect()
    sess = ExecutorSession(
        ex, fps=30,
        limits=_safety.SafetyLimits(max_joint_step=max_joint_step),
        recorder=recorder, clock=clock,
        wait=lambda t_end: setattr(clock, "t", max(clock.t, t_end)))
    return sess, ex, clock


def test_send_row_future_stamps_and_records():
    rec = LogRecorder()
    sess, ex, clock = _session(recorder=rec)
    row = np.zeros(_actions.ROBOT_DIM)
    row[0] = 0.05
    t_target = clock() + 2 * sess.dt
    sent = sess.send_row(row, t_target, step=7)
    assert ex.sent[-1][0] == t_target
    np.testing.assert_allclose(sent, ex.sent[-1][1])
    kinds = [e["kind"] for e in rec.events]
    assert kinds == ["action"]           # no clamp event: step was legal
    assert rec.events[0]["step"] == 7


def test_send_row_clamps_and_logs_clamp_event():
    rec = LogRecorder()
    sess, ex, clock = _session(recorder=rec, max_joint_step=0.10)
    sess.ground()
    row = np.zeros(_actions.ROBOT_DIM)
    row[3] = 0.5                          # 0.5 rad in one tick: way over
    sent = sess.send_row(row, clock() + sess.dt, step=0)
    assert sent[3] == pytest.approx(0.10)
    kinds = [e["kind"] for e in rec.events]
    assert kinds == ["clamp", "action"]
    assert rec.events[0]["max_step"] == pytest.approx(0.4)


def test_send_row_refuses_insane_row():
    sess, ex, _ = _session()
    bad = np.full(_actions.ROBOT_DIM, np.nan)
    with pytest.raises(InsaneRowError):
        sess.send_row(bad, 0.0, step=1)
    assert not ex.sent                    # nothing reached the executor


def test_stream_absolute_schedule_and_stamps():
    sess, ex, clock = _session()
    t0 = clock()
    rows = [np.zeros(_actions.ROBOT_DIM) for _ in range(5)]
    seen = []
    ok = sess.stream(rows, on_tick=lambda k, r: seen.append(k))
    assert ok and seen == [0, 1, 2, 3, 4]
    # every waypoint stamped one period past its cycle end: t0 + (k+2)*dt
    for k, (t_target, _row) in enumerate(ex.sent):
        assert t_target == pytest.approx(t0 + (k + 2) * sess.dt)


def test_stream_damps_on_insane_row_mid_stream():
    sess, ex, _ = _session()
    rows = [np.zeros(_actions.ROBOT_DIM),
            np.full(_actions.ROBOT_DIM, np.inf),
            np.zeros(_actions.ROBOT_DIM)]
    with pytest.raises(InsaneRowError):
        sess.stream(rows)
    assert ex.damped
    assert len(ex.sent) == 1              # sends stopped at the bad row


def test_stream_damps_on_keyboard_interrupt():
    sess, ex, _ = _session()

    def rows():
        yield np.zeros(_actions.ROBOT_DIM)
        raise KeyboardInterrupt

    assert sess.stream(rows()) is False
    assert ex.damped


def test_soft_start_sends_unstamped_and_grounds_clamp():
    sess, ex, clock = _session(max_joint_step=0.05)
    row = np.zeros(_actions.ROBOT_DIM)
    row[0] = 1.0                          # far away: the vendor ramp handles it
    sess.soft_start(row, settle_s=0.0)
    assert len(ex.sent) == 1              # un-clamped: drive_to_waypoint's job
    # clamp is grounded at the landed pose, so the next equal row is legal
    sent = sess.send_row(row, clock() + sess.dt)
    assert sent[0] == pytest.approx(1.0)


def test_ramp_to_caps_speed_and_lands():
    sess, ex, clock = _session()
    q_target = np.full(_actions.ARM_DOF, 0.9)
    hands0 = {h: np.zeros(6) for h in ("left", "right")}
    hands1 = {h: np.full(6, 0.5) for h in ("left", "right")}
    sess.ramp_to(q_target, hands1, hands0, ramp_s=0.1, max_speed=0.5,
                 settle_s=0.0)
    steps = np.diff([r[_actions.ARM] for _t, r in ex.sent], axis=0)
    assert np.abs(steps).max() <= 0.5 * sess.dt + 1e-9   # rad/tick cap held
    np.testing.assert_allclose(ex.sent[-1][1][_actions.ARM], q_target)
    np.testing.assert_allclose(ex.sent[-1][1][_actions.HAND["left"]], 0.5)


def test_ramp_to_aborts_on_callback():
    sess, ex, _ = _session()
    with pytest.raises(RuntimeError, match="aborted"):
        sess.ramp_to(np.ones(_actions.ARM_DOF),
                     {h: np.zeros(6) for h in ("left", "right")},
                     {h: np.zeros(6) for h in ("left", "right")},
                     ramp_s=1.0, abort=lambda: True)
    assert not ex.sent
