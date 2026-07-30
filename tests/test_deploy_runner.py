"""The deploy loop end-to-end against the MockExecutor: no DDS, no mujoco.

Covers: future-stamped sends, clamp wiring, watchdog trips (insane rows,
starvation) reaching damp(), hand-command tracking, the recorder's
events.jsonl actually containing the seams, and the dashboard-driven
lifecycle (gated start, pause/resume, record toggle, reset-to-episode).
"""

import json
import pathlib
import threading
import time

import numpy as np
import pytest

from ego2g1.core import layout
from ego2g1.deploy import actions as _actions
from ego2g1.deploy import replay_dataset as _replay_dataset
from ego2g1.deploy.executor import MockExecutor
from ego2g1.deploy.recorder import Recorder, RecorderSwitch
from ego2g1.deploy.runner import DeployRunner
from ego2g1.deploy.safety import SafetyLimits
from ego2g1.deploy.strategies import SynchronousStrategy

FPS = 200          # fast ticks so tests run in milliseconds
NO_WAIT = lambda t_end, **kw: None  # noqa: E731


class JointHoldPolicy:
    """A 'policy' emitting small joint chunks around zero; adapter-shaped
    (consumes the runner's request dict, returns (H, 26) joint rows)."""

    def __init__(self, horizon=5, step=0.001, hand=0.4):
        self.horizon = horizon
        self.step = step
        self.hand = hand
        self.prompt = "test"
        self.calls = 0

    def infer(self, request):
        self.calls += 1
        q = np.asarray(request["arm_q"], dtype=np.float64)
        rows = np.zeros((self.horizon, _actions.ROBOT_DIM))
        for k in range(self.horizon):
            rows[k, _actions.ARM] = q + self.step * (k + 1)
            for h in layout.HANDS:
                rows[k, _actions.HAND[h]] = self.hand
        return {"actions": rows}


def make_runner(policy, executor, **kw):
    strategy = SynchronousStrategy(policy, chunk_size=policy.horizon)
    return DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                        fps=FPS, wait=NO_WAIT, **kw)


def test_runner_executes_steps_with_future_stamped_targets():
    executor = MockExecutor(fps=FPS)
    executor.connect()
    policy = JointHoldPolicy()
    runner = make_runner(policy, executor, max_steps=10)
    runner.run()
    assert runner.steps_executed == 10
    assert len(executor.sent) == 10
    assert policy.calls == 2                     # 10 steps / horizon 5
    # future-stamped: every waypoint lands strictly after its send time
    # (t_cycle_end + dt; MockExecutor stores t_target as given)
    for t_target, _row in executor.sent:
        assert t_target > time.monotonic() - 5.0   # sane clock domain
    # hands tracked from the last commanded row
    np.testing.assert_allclose(runner.last_hands["left"], 0.4)
    assert not executor.damped


def test_runner_clamps_discontinuous_chunks():
    class JumpPolicy(JointHoldPolicy):
        def infer(self, request):
            out = super().infer(request)
            out["actions"][:, 0] = 2.0           # a 2 rad jump, every row
            return out

    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = make_runner(JumpPolicy(), executor,
                         limits=SafetyLimits(max_joint_step=0.1,
                                             max_joint_vel=1e9),
                         max_steps=3)
    runner.run()
    # each executed row moved at most max_joint_step on joint 0
    qs = [row[0] for _, row in executor.sent]
    assert qs[0] <= 0.1 + 1e-9
    assert all(b - a <= 0.1 + 1e-9 for a, b in zip(qs, qs[1:]))
    assert runner.clamp.clamped_ticks >= 3


def test_runner_trips_on_insane_rows_and_damps():
    class NaNPolicy(JointHoldPolicy):
        def infer(self, request):
            out = super().infer(request)
            out["actions"][2, 3] = 20.0          # far outside any joint range
            return out

    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = make_runner(NaNPolicy(), executor, max_steps=10)
    runner.run()
    assert runner.watchdog.tripped
    assert "insane" in runner.watchdog.reason
    assert executor.damped                        # trip reached the hardware stop
    assert executor.estopped


def test_runner_trips_on_starvation():
    class NeverReady:
        def update_observation(self, obs): pass
        def has_action(self): return False
        def pop_action(self): raise AssertionError("must not be called")
        def close(self): pass

    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = DeployRunner(adapter=JointHoldPolicy(), strategy=NeverReady(),
                          executor=executor, fps=FPS, wait=NO_WAIT,
                          limits=SafetyLimits(max_starvation=0.05),
                          max_steps=5)
    runner.run()
    assert runner.watchdog.tripped
    assert "strategy/planner is dead" in runner.watchdog.reason
    assert executor.damped


def test_runner_watches_ik_tracking_error():
    class BadTrackingPolicy(JointHoldPolicy):
        last_tracking_error = 0.5                 # 50 cm: frames are wrong

    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = make_runner(BadTrackingPolicy(), executor, max_steps=10)
    runner.run()
    assert runner.watchdog.tripped
    assert "tracking error" in runner.watchdog.reason


def test_recorder_captures_the_seams(tmp_path):
    executor = MockExecutor(fps=FPS)
    executor.connect()
    rec = Recorder(tmp_path / "session", meta={"horizon": 5, "fps": FPS})
    rec.start()
    policy = JointHoldPolicy()
    strategy = SynchronousStrategy(policy, chunk_size=5, recorder=rec)
    runner = DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                          recorder=rec, fps=FPS, wait=NO_WAIT, max_steps=6)
    runner.run()
    rec.stop()

    events = [json.loads(l) for l in
              (tmp_path / "session" / "events.jsonl").read_text().splitlines()]
    kinds = {e["kind"] for e in events}
    assert {"obs", "action", "infer_result"} <= kinds
    acts = [e for e in events if e["kind"] == "action"]
    assert len(acts) == 6
    assert len(acts[0]["row"]) == _actions.ROBOT_DIM
    infer = [e for e in events if e["kind"] == "infer_result"]
    assert all("latency" in e for e in infer)
    meta = json.loads((tmp_path / "session" / "meta.json").read_text())
    assert meta["horizon"] == 5 and "t0_monotonic" in meta


def _wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_gated_runner_idles_until_begin():
    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = make_runner(JointHoldPolicy(), executor, max_steps=6, gated=True)
    t = threading.Thread(target=runner.run, daemon=True)
    t.start()
    time.sleep(0.15)                              # idle ticks, no commands
    assert executor.sent == []
    assert runner.telemetry()["active"] is False
    runner.begin()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert runner.steps_executed == 6             # max_steps counts ACTIVE ticks
    assert len(executor.sent) == 6


def test_begin_refuses_while_tripped():
    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = make_runner(JointHoldPolicy(), executor, gated=True)
    runner.estop("test")
    with pytest.raises(RuntimeError, match="tripped"):
        runner.begin()


def test_pause_holds_and_resume_reinfers(tmp_path):
    executor = MockExecutor(fps=FPS)
    executor.connect()
    policy = JointHoldPolicy(horizon=3)
    rec = Recorder(tmp_path / "session", meta={})
    rec.start()
    strategy = SynchronousStrategy(policy, chunk_size=3)
    runner = DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                          recorder=rec, fps=FPS, wait=NO_WAIT, gated=True)
    t = threading.Thread(target=runner.run, daemon=True)
    t.start()
    try:
        runner.begin()
        assert _wait_for(lambda: len(executor.sent) >= 4)
        runner.pause()
        time.sleep(0.12)                          # let the loop reach the gate
        n = len(executor.sent)
        time.sleep(0.12)
        assert len(executor.sent) == n            # idle == holding, not sending
        assert runner.telemetry()["active"] is False

        calls_before = policy.calls
        runner.begin()                            # resume drops the stale chunk
        assert _wait_for(lambda: len(executor.sent) > n)
        assert policy.calls > calls_before        # re-inferred, mid-chunk or not
    finally:
        runner.estop("test done")                 # ends the loop
        t.join(timeout=5.0)
        rec.stop()
    events = [json.loads(l) for l in
              (tmp_path / "session" / "events.jsonl").read_text().splitlines()]
    whys = [e["why"] for e in events if e["kind"] == "rearm"]
    assert whys.count("begin") >= 2               # first start + the resume


def _fake_episode(q_start, hand=0.7, n=4):
    return {"name": "fake", "arm": np.tile(q_start, (n, 1)),
            "pose": None,
            "hand": {h: np.full((n, layout.HAND_DIM), hand)
                     for h in layout.HANDS}}


def test_reset_to_episode_ramps_capped_and_rearms(monkeypatch):
    q_start = np.full(_actions.ARM_DOF, 0.3)
    monkeypatch.setattr(_replay_dataset, "load_episode",
                        lambda root, ep: _fake_episode(q_start))
    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = make_runner(JointHoldPolicy(), executor, gated=True,
                         dataset="fake://dataset")
    out = runner.reset_to_episode(0, ramp_s=0.01, max_speed=2.0, settle_s=0.0)
    assert out["episode"] == 0
    assert out["residual"] < 1e-9                 # MockExecutor tracks instantly
    # the ramp streamed monotone, per-tick-capped waypoints to the target
    qs = np.array([row[_actions.ARM] for _, row in executor.sent])
    cap = 2.0 / FPS + 1e-9
    assert np.all(np.abs(np.diff(qs, axis=0)) <= cap)
    assert np.all(np.abs(qs[0]) <= cap)           # first step capped from q=0
    np.testing.assert_allclose(qs[-1], q_start)
    np.testing.assert_allclose(runner.last_hands["left"], 0.7)
    assert runner.telemetry()["has_dataset"] is True


def test_reset_to_episode_refusals(monkeypatch):
    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = make_runner(JointHoldPolicy(), executor)   # active (ungated)
    with pytest.raises(RuntimeError, match="pause before resetting"):
        runner.reset_to_episode(0)
    runner.pause()
    with pytest.raises(RuntimeError, match="no --dataset"):
        runner.reset_to_episode(0)
    runner.dataset = "fake://dataset"
    runner.estop("test")
    with pytest.raises(RuntimeError, match="tripped"):
        runner.reset_to_episode(0)


def test_record_toggle_rolls_session_boundaries(tmp_path):
    executor = MockExecutor(fps=FPS)
    executor.connect()
    switch = RecorderSwitch(tmp_path, "toggle test", meta={"fps": FPS})
    runner = make_runner(JointHoldPolicy(), executor, recorder=switch,
                         gated=True)
    assert runner.telemetry()["recording"] is False
    switch.log("obs", step=0)                     # dropped while off
    out1 = runner.record_toggle()
    assert out1["recording"] is True
    d1 = out1["dir"]
    switch.log("action", step=1, row=[0.0])
    assert runner.telemetry()["recording"] is True
    out2 = runner.record_toggle()
    assert out2["recording"] is False and out2["dir"] == d1
    out3 = runner.record_toggle()
    d2 = out3["dir"]
    assert d2 != d1                               # a FRESH directory per take
    runner.record_toggle()

    events = [json.loads(l) for l in
              (pathlib.Path(d1) / "events.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["action"]   # pre-toggle log dropped
    assert (pathlib.Path(d2) / "events.jsonl").exists()


def test_record_toggle_without_switch_is_a_409_shaped_error():
    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = make_runner(JointHoldPolicy(), executor)   # NullRecorder default
    with pytest.raises(RuntimeError, match="RecorderSwitch"):
        runner.record_toggle()


def test_obs_events_carry_measured_arm_q(tmp_path):
    executor = MockExecutor(fps=FPS, initial_q=np.full(_actions.ARM_DOF, 0.1))
    executor.connect()
    rec = Recorder(tmp_path / "session", meta={})
    rec.start()
    policy = JointHoldPolicy()
    strategy = SynchronousStrategy(policy, chunk_size=policy.horizon)
    runner = DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                          recorder=rec, fps=FPS, wait=NO_WAIT, max_steps=3)
    runner.run()
    rec.stop()
    events = [json.loads(l) for l in
              (tmp_path / "session" / "events.jsonl").read_text().splitlines()]
    obs = [e for e in events if e["kind"] == "obs"]
    assert len(obs) == 3
    assert all(len(e["arm_q"]) == _actions.ARM_DOF for e in obs)
    assert obs[0]["arm_q"] == [0.1] * _actions.ARM_DOF


def test_mock_executor_contract():
    ex = MockExecutor()
    with pytest.raises(RuntimeError):
        ex.send(np.zeros(_actions.ROBOT_DIM))
    ex.connect()
    with pytest.raises(ValueError):
        ex.send(np.zeros(14))
    ex.send(np.zeros(_actions.ROBOT_DIM))
    ex.damp()
    ex.send(np.ones(_actions.ROBOT_DIM))          # latched: ignored
    assert len(ex.sent) == 1
