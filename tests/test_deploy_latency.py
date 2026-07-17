"""DelayBudget arithmetic and the startup latency self-check (fake clock).

The self-check is the guard against the measured failure mode: 1.3-5.1 s
inferences against a 0.4 s budget -> freeze-lurch (docs/jitter_root_cause.md,
Cause 2). Refusal must be loud and must happen before the robot moves.
"""

import numpy as np
import pytest

from ego2g1.deploy.latency import (
    DelayBudget,
    LatencyBudgetError,
    budget_for,
    measure_policy_latency,
    startup_self_check,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_infer(clock, latencies):
    """infer_fn that consumes one latency per call by advancing the clock."""
    it = iter(latencies)

    def infer():
        clock.advance(next(it))

    return infer


# --- DelayBudget ---------------------------------------------------------------


def test_delay_budget_tracks_the_quantile():
    b = DelayBudget(fps=30, initial=12, quantile=0.95, headroom=1.15, max_d=20)
    assert b.d == 12                      # the initial forecast, held
    for _ in range(10):
        b.observe(0.1)                    # steady 100 ms
    want = int(np.ceil(0.1 * 1.15 * 30))  # = 4 ticks
    assert b.d == want


def test_delay_budget_saturates_instead_of_unbounded():
    b = DelayBudget(fps=30, max_d=20)
    for _ in range(10):
        b.observe(2.0)                    # a 2 s inference is not exotic
    assert b.d == 20                      # clamped: lose RTC continuity, keep the plan
    assert b.saturated > 0
    b.note_violation()
    assert b.stats()["violations"] == 1


def test_delay_budget_initial_respects_max():
    assert DelayBudget(fps=30, initial=50, max_d=20).d == 20


# --- budgets --------------------------------------------------------------------


def test_budget_for_modes():
    assert budget_for("sync", fps=30, horizon=50) is None
    assert budget_for("async", fps=30, horizon=50) == pytest.approx(50 / 30)
    # the worker period tightens the async budget when it is the binding term
    assert budget_for("async", fps=30, horizon=50,
                      inference_hz=4.0) == pytest.approx(0.5)
    assert budget_for("temporal_smoothing", fps=30, horizon=50,
                      max_latency_steps=8) == pytest.approx(8 / 30)
    assert budget_for("rtc", fps=30, horizon=50,
                      max_latency_steps=12) == pytest.approx(12 / 30)
    with pytest.raises(ValueError):
        budget_for("temporal_smoothing", fps=30, horizon=50)
    with pytest.raises(ValueError):
        budget_for("nope", fps=30, horizon=50)


# --- measure ----------------------------------------------------------------------


def test_measure_separates_the_compile_call():
    clock = FakeClock()
    first, samples = measure_policy_latency(
        make_infer(clock, [60.0, 0.1, 0.1, 0.1]), n=3, clock=clock)
    assert first == pytest.approx(60.0)          # XLA compile, reported apart
    assert samples == pytest.approx([0.1, 0.1, 0.1])


# --- the self-check ------------------------------------------------------------------


def test_self_check_passes_a_fast_policy():
    clock = FakeClock()
    report = startup_self_check(
        "temporal_smoothing", make_infer(clock, [10.0] + [0.1] * 5),
        fps=30, horizon=50, max_latency_steps=8, clock=clock)
    assert report.verdict == "ok"
    assert report.p95_s == pytest.approx(0.1)
    assert "headroom" in report.detail


def test_self_check_refuses_the_measured_freeze_lurch():
    # the actual measured rollout: seconds-long inferences, 8-step budget
    clock = FakeClock()
    with pytest.raises(LatencyBudgetError) as exc:
        startup_self_check(
            "temporal_smoothing",
            make_infer(clock, [60.0, 1.31, 4.31, 5.09, 1.5, 2.0]),
            fps=30, horizon=50, max_latency_steps=8, clock=clock)
    assert "REFUSE" in str(exc.value)
    assert "freeze-lurch" in str(exc.value)


def test_self_check_sync_never_refuses_but_warns_on_slow():
    clock = FakeClock()
    report = startup_self_check(
        "sync", make_infer(clock, [60.0] + [5.0] * 5),
        fps=30, horizon=50, clock=clock)
    assert report.verdict == "warn"              # slow but SAFE: it holds
    assert report.budget_s is None


def test_self_check_async_budget_is_the_chunk_duration():
    clock = FakeClock()
    # 1.2 s p95 vs a 50/30=1.67 s chunk: ok without an inference_hz bound
    report = startup_self_check(
        "async", make_infer(clock, [10.0] + [1.2] * 5),
        fps=30, horizon=50, clock=clock)
    assert report.verdict == "ok"
    # but the same latency cannot honor a 1 Hz worker period budget of 2/1 s? it can;
    # tighten: horizon 30 -> budget 1.0 s -> refuse
    clock2 = FakeClock()
    with pytest.raises(LatencyBudgetError):
        startup_self_check(
            "async", make_infer(clock2, [10.0] + [1.2] * 5),
            fps=30, horizon=30, clock=clock2)
