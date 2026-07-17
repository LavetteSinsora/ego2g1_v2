"""Splice math of the five chunk consumers (ported from zh_deploy_inference).

These pin the behaviors the runner depends on: the sync re-infer trigger, the
naive-async skip arithmetic, the ensembling weights, the smoothing blend, and
the RTC request plumbing through AsyncStrategy.
"""

import time

import numpy as np
import pytest

from ego2g1.deploy.strategies import (
    AsyncStrategy,
    NaiveAsyncBuffer,
    SynchronousStrategy,
    TemporalEnsemblingBuffer,
    TemporalSmoothingBuffer,
    make_strategy,
)

DIM = 4  # strategies are dimension-agnostic; small keeps the math legible


class FakePolicy:
    """Returns arange-valued chunks; counts calls; optionally sleeps."""

    def __init__(self, horizon=5, delay=0.0):
        self.horizon = horizon
        self.delay = delay
        self.calls = 0
        self.requests = []

    def infer(self, observation):
        self.calls += 1
        self.requests.append(dict(observation))
        if self.delay:
            time.sleep(self.delay)
        base = self.calls * 100.0
        return {"actions": base + np.arange(self.horizon)[:, None]
                * np.ones((1, DIM))}


def test_synchronous_reinfers_only_when_drained():
    policy = FakePolicy(horizon=3)
    s = SynchronousStrategy(policy, chunk_size=3)
    popped = []
    for _ in range(6):
        s.update_observation({"o": 1})
        popped.append(s.pop_action()[0])
    assert policy.calls == 2
    assert popped == [100, 101, 102, 200, 201, 202]


def test_naive_async_skips_steps_spent_inferring():
    buf = NaiveAsyncBuffer()
    chunk = np.arange(10)[:, None] * np.ones((1, DIM))
    buf.add_chunk(chunk, start_timestep=0)
    for _ in range(4):        # consume 4 rows -> global_t = 4
        buf.pop_action()
    # a chunk generated at timestep 2 lands now: 2 steps elapsed -> skip 2
    buf.add_chunk(chunk, start_timestep=2)
    assert buf.pop_action()[0] == 2.0


def test_naive_async_holds_last_action_when_exhausted():
    buf = NaiveAsyncBuffer()
    buf.add_chunk(np.ones((2, DIM)), start_timestep=0)
    buf.pop_action()
    buf.pop_action()
    assert buf.has_action()               # last action still servable
    np.testing.assert_allclose(buf.pop_action(), 1.0)


def test_temporal_ensembling_weights_older_predictions_down():
    m = 0.5
    buf = TemporalEnsemblingBuffer(exp_weight_m=m)
    buf.add_chunk(np.full((3, DIM), 10.0), start_timestep=0)   # inference 0 (oldest)
    buf.add_chunk(np.full((3, DIM), 20.0), start_timestep=0)   # inference 1
    w = np.exp(-m * np.arange(2))
    w /= w.sum()
    expected = w[0] * 10.0 + w[1] * 20.0
    np.testing.assert_allclose(buf.pop_action(), expected)


def test_temporal_smoothing_blends_the_overlap_linearly():
    buf = TemporalSmoothingBuffer(max_latency_steps=8, min_smooth_steps=4)
    buf.add_chunk(np.zeros((4, DIM)), start_timestep=0)         # old plan: all 0
    info = buf.add_chunk(np.ones((4, DIM)), start_timestep=0)   # new plan: all 1
    assert info["drop_count"] == 0 and not info["late"]
    got = [buf.pop_action()[0] for _ in range(4)]
    np.testing.assert_allclose(got, np.linspace(0.0, 1.0, 4))   # 1-w blend


def test_temporal_smoothing_drops_stale_rows_and_flags_late():
    buf = TemporalSmoothingBuffer(max_latency_steps=2, min_smooth_steps=2)
    buf.add_chunk(np.zeros((8, DIM)), start_timestep=0)
    for _ in range(5):                       # 5 ticks pass; budget is 2
        buf.pop_action()
    chunk = np.arange(8)[:, None] * np.ones((1, DIM))
    info = buf.add_chunk(chunk, start_timestep=0)
    assert info["drop_count"] == 2           # capped at max_latency_steps
    assert info["late"] is True              # 5 > 2: seam not guaranteed


def test_temporal_smoothing_whole_chunk_too_stale_is_dropped():
    buf = TemporalSmoothingBuffer(max_latency_steps=3, min_smooth_steps=2)
    buf.add_chunk(np.zeros((5, DIM)), start_timestep=0)
    for _ in range(5):
        buf.pop_action()
    info = buf.add_chunk(np.ones((3, DIM)), start_timestep=0)
    assert info.get("dropped_whole_chunk") is True
    assert not buf.has_action()


def test_async_strategy_produces_and_closes():
    policy = FakePolicy(horizon=4)
    s = AsyncStrategy(policy, NaiveAsyncBuffer(), inference_hz=100.0)
    try:
        s.update_observation({"o": 1})
        deadline = time.monotonic() + 2.0
        while not s.has_action():
            assert time.monotonic() < deadline, "worker never produced"
            time.sleep(0.005)
        assert s.pop_action().shape == (DIM,)
    finally:
        s.close()


def test_async_strategy_rtc_attaches_request_fields():
    policy = FakePolicy(horizon=4)
    s = AsyncStrategy(policy, TemporalSmoothingBuffer(2, 2), inference_hz=100.0,
                      rtc=True, execute_horizon=4, control_hz=30.0)
    try:
        s.update_observation({"o": 1})
        deadline = time.monotonic() + 2.0
        while policy.calls < 2:
            assert time.monotonic() < deadline
            time.sleep(0.005)
    finally:
        s.close()
    first, second = policy.requests[0], policy.requests[1]
    assert first["enable_rtc"] and "prev_action_chunk" not in first
    assert second["enable_rtc"] and "prev_action_chunk" in second
    assert isinstance(second["inference_delay"], int)


def test_async_strategy_surfaces_worker_errors():
    class Exploding:
        def infer(self, observation):
            raise RuntimeError("boom")

    s = AsyncStrategy(Exploding(), NaiveAsyncBuffer(), inference_hz=100.0)
    try:
        s.update_observation({"o": 1})
        deadline = time.monotonic() + 2.0
        with pytest.raises(RuntimeError):
            while time.monotonic() < deadline:
                s.has_action()
                time.sleep(0.005)
            raise AssertionError("worker error never surfaced")
    finally:
        s.close()


def test_make_strategy_covers_all_modes():
    policy = FakePolicy()
    for mode in ("sync", "async", "temporal_ensembling", "temporal_smoothing", "rtc"):
        s = make_strategy(mode, policy, chunk_size=5, inference_hz=100.0)
        s.close()
    with pytest.raises(ValueError):
        make_strategy("nope", policy, chunk_size=5)


def test_non_finite_chunk_is_rejected():
    class NaNPolicy:
        def infer(self, observation):
            return {"actions": np.full((3, DIM), np.nan)}

    s = SynchronousStrategy(NaNPolicy(), chunk_size=3)
    with pytest.raises(ValueError):
        s.update_observation({"o": 1})
