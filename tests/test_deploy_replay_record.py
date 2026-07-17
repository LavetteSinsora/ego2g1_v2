"""Round-trip: record with the ACTUAL Recorder -> reconstruct with Session ->
assert the rebuilt structures match what the live loop held. No robot; the
sync path drives the full DeployRunner + MockExecutor, the async path drives
the real TemporalSmoothingBuffer so the reconstruction must reproduce the
production blend math exactly.
"""

import time

import numpy as np

from ego2g1.core import layout
from ego2g1.deploy import actions as _actions
from ego2g1.deploy.executor import MockExecutor
from ego2g1.deploy.recorder import Recorder
from ego2g1.deploy.replay_record import Session
from ego2g1.deploy.runner import DeployRunner
from ego2g1.deploy.strategies import SynchronousStrategy, TemporalSmoothingBuffer

FPS = 200
NO_WAIT = lambda t_end, **kw: None  # noqa: E731


class JointHoldPolicy:
    def __init__(self, horizon=5, step=0.001, hand=0.4):
        self.horizon = horizon
        self.step = step
        self.hand = hand
        self.prompt = "replay test"

    def infer(self, request):
        q = np.asarray(request["arm_q"], dtype=np.float64)
        rows = np.zeros((self.horizon, _actions.ROBOT_DIM))
        for k in range(self.horizon):
            rows[k, _actions.ARM] = q + self.step * (k + 1)
            for h in layout.HANDS:
                rows[k, _actions.HAND[h]] = self.hand
        return {"actions": rows}


def record_sync_session(tmp_path, max_steps=6, horizon=5):
    executor = MockExecutor(fps=FPS)
    executor.connect()
    rec = Recorder(tmp_path / "session",
                   meta={"mode": "sync", "horizon": horizon, "fps": FPS})
    rec.start()
    policy = JointHoldPolicy(horizon=horizon)
    strategy = SynchronousStrategy(policy, chunk_size=horizon, recorder=rec)
    runner = DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                          recorder=rec, fps=FPS, wait=NO_WAIT,
                          max_steps=max_steps)
    runner.run()
    rec.stop()
    return tmp_path / "session", executor, strategy


def test_sync_roundtrip(tmp_path):
    session_dir, executor, strategy = record_sync_session(tmp_path)
    s = Session(session_dir)
    t0, t1 = s.span()
    assert t0 is not None and t1 >= t0

    # 6 steps at horizon 5 -> two chunks, pointer one row into the second.
    t_end = t1 + 1.0
    chunk, index = s.chunk_at(t_end)
    assert chunk.shape == (5, _actions.ROBOT_DIM)
    assert index == 1
    live = strategy.telemetry()
    assert (index, len(chunk)) == (live["index"], live["horizon"])
    # the reconstructed chunk IS the live one (sync: recorded verbatim)
    np.testing.assert_allclose(chunk, strategy._chunk)

    snap = s.at(t_end)
    assert snap["ready"] is True
    assert snap["index"] == 1 and snap["horizon"] == 5
    assert snap["phase"] == "executing"
    assert snap["step"] == 5
    # the commanded row at t is exactly what the executor was sent
    np.testing.assert_allclose(snap["action_row"], executor.sent[-1][1])
    # and, no clamping here, exactly the chunk row before the pointer
    np.testing.assert_allclose(snap["action_row"], chunk[index - 1])

    # before anything happened: idle, nothing ready
    early = s.at(t0 - 1.0)
    assert early["ready"] is False and early["phase"] == "idle"
    assert early["action_row"] is None

    # mid-inference: the window is [t_result - latency, t_result)
    e = [ev for ev in s.events if ev["kind"] == "infer_result"][0]
    assert s._phase_at(e["t"] - e["latency"] / 2) == "inferring"

    # timeline carries one row per inference
    kinds = [k for _t, k, _d in s.timeline()]
    assert kinds.count("infer_result") == 2


def test_smoothing_roundtrip_reproduces_the_production_blend(tmp_path):
    """Feed a REAL TemporalSmoothingBuffer while logging the same seams the
    live AsyncStrategy logs; Session must rebuild the exact blended chunk."""
    dim = 4
    meta = {"mode": "temporal_smoothing", "horizon": 6, "fps": 30,
            "max_latency_steps": 8, "min_smooth_steps": 4}
    rec = Recorder(tmp_path / "session", meta=meta)
    rec.start()
    buf = TemporalSmoothingBuffer(max_latency_steps=8, min_smooth_steps=4)

    def install(chunk, start):
        info = buf.add_chunk(chunk, start)
        rec.log("infer_result", latency=0.01, start_timestep=start,
                horizon=len(chunk), rtc=False, splice=info, actions=chunk)

    def pop(n):
        rows = []
        for _ in range(n):
            row = buf.pop_action()
            rec.log("action", step=len(rows), row=row)
            rows.append(np.asarray(row))
        return rows

    A = np.arange(6, dtype=np.float64)[:, None] * np.ones((1, dim))
    B = 100.0 + np.arange(6, dtype=np.float64)[:, None] * np.ones((1, dim))
    install(A, 0)
    pop(3)                       # steps_since_update = 3 when B lands
    install(B, 3)                # blends old remainder with B[3:]
    popped_live = pop(2)
    rec.stop()
    t_end = time.monotonic()

    s = Session(tmp_path / "session")
    chunk, index = s.chunk_at(t_end)
    assert index == 2

    # the reconstructed combined chunk == live popped rows + live remainder
    live_rows = popped_live + [np.asarray(a) for a in buf._chunk]
    np.testing.assert_allclose(chunk, np.stack(live_rows))
    # and the blend really blended: middle row is neither pure A nor pure B
    assert not np.allclose(chunk[1], A) and 4.0 < chunk[1][0] < 104.0

    live = buf.telemetry()
    assert (index, len(chunk)) == (live["index"], live["horizon"])

    snap = s.at(t_end)
    assert snap["ready"] is True
    np.testing.assert_allclose(snap["action_row"], popped_live[-1])


def test_timeline_and_estop_visibility(tmp_path):
    """A tripped session must show the estop in the timeline."""
    executor = MockExecutor(fps=FPS)
    executor.connect()
    rec = Recorder(tmp_path / "session",
                   meta={"mode": "sync", "horizon": 3, "fps": FPS})
    rec.start()

    class NaNPolicy(JointHoldPolicy):
        def infer(self, request):
            out = super().infer(request)
            out["actions"][1, 0] = 20.0     # insane row on the second pop
            return out

    policy = NaNPolicy(horizon=3)
    strategy = SynchronousStrategy(policy, chunk_size=3, recorder=rec)
    runner = DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                          recorder=rec, fps=FPS, wait=NO_WAIT, max_steps=10)
    runner.run()
    rec.stop()
    assert executor.damped

    s = Session(tmp_path / "session")
    kinds = [k for _t, k, _d in s.timeline()]
    assert "estop" in kinds
