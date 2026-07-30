"""The live dashboard against the Mock stack: telemetry snapshots must be
JSON-serializable, consistent with the loop's state, and servable over one
localhost GET — no robot, no policy server.
"""

import json
import urllib.error
import urllib.request

import numpy as np

from ego2g1.core import layout
from ego2g1.deploy import actions as _actions
from ego2g1.deploy.dashboard import Dashboard, _DemoLoop
from ego2g1.deploy.executor import MockExecutor
from ego2g1.deploy.latency import DelayBudget
from ego2g1.deploy.runner import DeployRunner
from ego2g1.deploy.strategies import (
    NaiveAsyncBuffer,
    SynchronousStrategy,
    TemporalEnsemblingBuffer,
    TemporalSmoothingBuffer,
)

FPS = 200
NO_WAIT = lambda t_end, **kw: None  # noqa: E731


class JointHoldPolicy:
    """Adapter-shaped: consumes the runner's request dict, returns joint rows."""

    def __init__(self, horizon=5, step=0.001, hand=0.4):
        self.horizon = horizon
        self.step = step
        self.hand = hand
        self.prompt = "dashboard test"
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


def make_runner(max_steps=7):
    executor = MockExecutor(fps=FPS)
    executor.connect()
    policy = JointHoldPolicy()
    budget = DelayBudget(FPS)
    strategy = SynchronousStrategy(policy, chunk_size=policy.horizon,
                                   budget=budget)
    runner = DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                          fps=FPS, wait=NO_WAIT, max_steps=max_steps)
    return runner, executor, policy


def test_sync_runner_telemetry_snapshot():
    runner, executor, policy = make_runner(max_steps=7)
    runner.run()
    t = runner.telemetry()

    json.dumps(t)                                 # must be JSON-serializable

    assert t["mode"] == "sync"
    assert t["dim"] == _actions.ROBOT_DIM
    assert t["fps"] == FPS
    assert t["task"] == "dashboard test"
    assert t["ready"] is True
    assert t["horizon"] == policy.horizon
    assert t["index"] == 7 - policy.horizon       # 2 rows into chunk 2
    assert t["wall_slot"] == t["index"]           # pop-and-send is one tick
    # executor telemetry: last commanded row + measured q + state age
    assert t["action_row"] == executor.sent[-1][1].tolist()
    assert t["arm_q"] == executor.arm_q().tolist()
    assert t["state_age"] == 0.0
    assert t["estopped"] is False
    # groups tile the 26 executor dims contiguously
    stops = [(g["start"], g["stop"]) for g in t["groups"]]
    assert stops[0][0] == 0 and stops[-1][1] == _actions.ROBOT_DIM
    assert all(a[1] == b[0] for a, b in zip(stops, stops[1:]))
    # budget saw one observation per chunk
    assert t["budget"]["n"] == policy.calls == 2
    assert t["stats"]["ticks"] == 7
    assert t["watchdog"] == {"tripped": False, "reason": None}
    assert t["active"] is False                   # run() finished
    assert t["camera_age"] is None                # no camera attached


def test_sync_strategy_reports_inferring_when_drained():
    policy = JointHoldPolicy(horizon=2)
    s = SynchronousStrategy(policy, chunk_size=2)
    assert s.telemetry()["inferring"] is True     # no chunk yet
    s.update_observation({"arm_q": np.zeros(14)})
    assert s.telemetry()["inferring"] is False
    s.pop_action(), s.pop_action()
    assert s.telemetry()["inferring"] is True     # drained == blocking window


def test_buffer_telemetry_smoothing():
    buf = TemporalSmoothingBuffer(max_latency_steps=8, min_smooth_steps=10)
    assert buf.telemetry() == {"ready": False, "horizon": 0, "index": 0,
                               "global_t": 0, "steps_since_update": 0}
    chunk = np.arange(10)[:, None] * np.ones((1, 4))
    buf.add_chunk(chunk, 0)
    for _ in range(3):
        buf.pop_action()
    t = buf.telemetry()
    assert t["ready"] and t["horizon"] == 10 and t["index"] == 3


def test_buffer_telemetry_naive_async():
    buf = NaiveAsyncBuffer()
    chunk = np.arange(10)[:, None] * np.ones((1, 4))
    buf.add_chunk(chunk, 0)
    for _ in range(4):
        buf.pop_action()
    t = buf.telemetry()
    assert t == {"ready": True, "horizon": 10, "index": 4, "global_t": 4}


def test_buffer_telemetry_ensembling():
    buf = TemporalEnsemblingBuffer(exp_weight_m=0.1)
    chunk = np.ones((5, 4))
    buf.add_chunk(chunk, 0)
    buf.add_chunk(2 * chunk, 0)
    t = buf.telemetry()
    assert t["votes"] == 2 and t["chunks"] == 2 and t["horizon"] == 0


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read()


def test_http_state_page_and_estop():
    runner, executor, _policy = make_runner(max_steps=5)
    runner.run()                                   # populate state first
    dash = Dashboard(runner, port=0)               # ephemeral port
    dash.start()
    base = f"http://127.0.0.1:{dash.port}"
    try:
        code, body = _get(base + "/state")
        assert code == 200
        t = json.loads(body)
        assert t["mode"] == "sync" and t["ready"] is True

        code, body = _get(base + "/")
        assert code == 200 and b"deploy monitor" in body

        code, _ = _get(base + "/frame.jpg")        # no camera -> 204, not 500
        assert code == 204

        # POST /estop reaches the watchdog -> damp(); GET stays pure telemetry
        req = urllib.request.Request(base + "/estop", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.loads(r.read()) == {"tripped": True}
        assert runner.watchdog.tripped and executor.damped

        # unsupported control -> 409, server survives
        req = urllib.request.Request(base + "/reset", data=b"{}", method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
            raise AssertionError("expected HTTP 409")
        except urllib.error.HTTPError as e:
            assert e.code == 409

        code, _ = _get(base + "/state")            # still serving after estop
        assert code == 200
    finally:
        dash.stop()


def _post(url, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_http_lifecycle_controls(tmp_path, monkeypatch):
    """Start / Pause / Record / Reset drive a live gated runner end-to-end
    over HTTP — the dashboard buttons' actual code path."""
    import threading
    import time as _time

    from ego2g1.deploy import replay_dataset as _replay_dataset
    from ego2g1.deploy.recorder import RecorderSwitch

    q_start = np.full(_actions.ARM_DOF, 0.2)
    monkeypatch.setattr(
        _replay_dataset, "load_episode",
        lambda root, ep: {"name": "fake", "arm": np.tile(q_start, (3, 1)),
                          "pose": None,
                          "hand": {h: np.zeros((3, layout.HAND_DIM))
                                   for h in layout.HANDS}})

    executor = MockExecutor(fps=FPS)
    executor.connect()
    policy = JointHoldPolicy()
    strategy = SynchronousStrategy(policy, chunk_size=policy.horizon)
    switch = RecorderSwitch(tmp_path, "http test", meta={})
    runner = DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                          recorder=switch, fps=FPS, wait=NO_WAIT,
                          gated=True, dataset="fake://dataset")
    loop_thread = threading.Thread(target=runner.run, daemon=True)
    loop_thread.start()
    dash = Dashboard(runner, port=0)
    dash.start()
    base = f"http://127.0.0.1:{dash.port}"
    try:
        # gated: idle until Start
        _time.sleep(0.1)
        assert executor.sent == []

        code, out = _post(base + "/start")
        assert code == 200 and out == {"active": True}
        t0 = _time.monotonic()
        while not executor.sent and _time.monotonic() - t0 < 5:
            _time.sleep(0.005)
        assert executor.sent

        # reset refuses while active (409, server survives)
        try:
            _post(base + "/reset", {"episode": 0})
            raise AssertionError("expected HTTP 409")
        except urllib.error.HTTPError as e:
            assert e.code == 409
            assert "pause" in json.loads(e.read())["error"]

        code, out = _post(base + "/pause")
        assert code == 200 and out == {"active": False}
        _time.sleep(0.1)

        code, out = _post(base + "/reset", {"episode": 0})
        assert code == 200 and out["episode"] == 0
        np.testing.assert_allclose(executor.arm_q(), q_start)

        code, out = _post(base + "/record")
        assert code == 200 and out["recording"] is True
        code, out2 = _post(base + "/record")
        assert code == 200 and out2["recording"] is False
        assert out2["dir"] == out["dir"]

        code, body = _get(base + "/state")
        t = json.loads(body)
        assert t["has_dataset"] is True and t["active"] is False
    finally:
        runner.estop("test done")
        loop_thread.join(timeout=5.0)
        dash.stop()


def test_demo_loop_matches_runner_telemetry_shape():
    demo = _DemoLoop().telemetry()
    runner, _, _ = make_runner(max_steps=5)
    runner.run()
    real = runner.telemetry()
    json.dumps(demo)
    assert set(real) <= set(demo)                  # demo covers the real keys
    assert len(demo["action_row"]) == _actions.ROBOT_DIM
