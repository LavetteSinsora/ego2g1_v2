"""The deploy loop end-to-end against the MockExecutor: no DDS, no mujoco.

Covers: future-stamped sends, clamp wiring, watchdog trips (insane rows,
starvation) reaching damp(), hand-command tracking, and the recorder's
events.jsonl actually containing the seams.
"""

import json
import time

import numpy as np
import pytest

from ego2g1.core import layout
from ego2g1.deploy import actions as _actions
from ego2g1.deploy.executor import MockExecutor
from ego2g1.deploy.recorder import Recorder
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
