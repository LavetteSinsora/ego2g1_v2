"""Latency accounting: the delay budget, and the startup self-check that
refuses to run a timing strategy the measured latency cannot honor.

Why this module exists (docs/jitter_root_cause.md, Cause 2): the live rollout
measured chunk inference latencies of 1.31 s, 4.31 s and 5.09 s against a
budget of 12 slots = 0.4 s. The result was play ~1.7 s -> FREEZE 8.7 s -> lurch
onto a chunk computed from a 5 s-stale observation -> watchdog. No timing
scheme survives 10x its latency budget, and it fails silently, mid-rollout,
with the robot moving. So latency is measured BEFORE the robot moves, against
the strategy actually selected, and a budget violation is a refusal to start —
a loud failure at the terminal instead of a freeze-lurch on the arm.

DelayBudget is ported from the old deploy's client.py (provenance:
third_party/openpi/ego2g1/deploy/client.py).
"""

import dataclasses
import threading
import time

import numpy as np


class DelayBudget:
    """How many fps-ticks an inference takes — `d` in the RTC papers.

    Deliberately a FIXED budget rather than a per-call prediction. `d` has to
    be sent WITH the request (the guidance mask depends on it), so it is
    necessarily a forecast, and the two errors cost differently:

      d too small -> we execute past the frozen prefix while inferring; the new
                     chunk's committed slots no longer match what the robot did.
                     That is a discontinuity: a lurch at the seam.
      d too large -> we over-constrain and lose a little reactivity. Harmless.

    So: high quantile of observed latency, with headroom, held rather than
    chased. Deterministic beats adaptive.
    """

    def __init__(self, fps: int, *, initial: int = 12, quantile: float = 0.95,
                 window: int = 50, headroom: float = 1.15, max_d: int = 20):
        self.fps = fps
        self.quantile = quantile
        self.headroom = headroom
        # d MUST be bounded. Unbounded, a slow inference (GPU contention — a
        # 2 s call is not exotic) yields d > H; the chunk installs with zero
        # usable actions and the replan trigger goes negative: the loop
        # silently stops planning. Saturating means we merely lose the RTC
        # continuity guarantee (and say so) rather than the whole plan.
        self.max_d = int(max_d)
        self._d = min(int(initial), self.max_d)
        self._samples: list[float] = []
        self._window = window
        self._lock = threading.Lock()
        self.violations = 0
        self.saturated = 0

    @property
    def d(self) -> int:
        with self._lock:
            return self._d

    def observe(self, latency_s: float) -> None:
        with self._lock:
            self._samples.append(latency_s)
            if len(self._samples) > self._window:
                self._samples.pop(0)
            if len(self._samples) >= 5:
                q = float(np.quantile(self._samples, self.quantile))
                want = max(1, int(np.ceil(q * self.headroom * self.fps)))
                if want > self.max_d:
                    self.saturated += 1
                self._d = min(want, self.max_d)

    def note_violation(self) -> None:
        """The chunk landed LATER than d ticks: we executed past the frozen
        prefix and the seam is not guaranteed continuous. Counted and surfaced
        — a steady stream of these means the budget is too tight."""
        with self._lock:
            self.violations += 1

    def stats(self) -> dict:
        with self._lock:
            base = {"d": self._d, "violations": self.violations,
                    "saturated": self.saturated}
            if not self._samples:
                return {**base, "n": 0}
            s = np.asarray(self._samples)
            return {**base, "n": len(s),
                    "mean_ms": float(s.mean() * 1000),
                    "p95_ms": float(np.quantile(s, 0.95) * 1000)}


# --- the startup self-check -----------------------------------------------------


class LatencyBudgetError(RuntimeError):
    """Raised when the measured inference latency cannot honor the selected
    strategy's timing budget. This is a refusal to start, on purpose."""


@dataclasses.dataclass(frozen=True)
class LatencyReport:
    mode: str
    first_call_s: float          # includes any server-side JIT compile
    samples_s: tuple             # steady-state samples, first call excluded
    p95_s: float
    budget_s: float | None       # the hard budget for this mode (None = sync)
    verdict: str                 # "ok" | "warn" | "refuse"
    detail: str

    def summary(self) -> str:
        lat = ", ".join(f"{s*1000:.0f}" for s in self.samples_s)
        b = "none (blocking)" if self.budget_s is None else f"{self.budget_s*1000:.0f} ms"
        return (f"[latency self-check] mode={self.mode}  first={self.first_call_s:.1f}s "
                f"(compile)  steady=[{lat}] ms  p95={self.p95_s*1000:.0f} ms  "
                f"budget={b}\n  -> {self.verdict.upper()}: {self.detail}")


def budget_for(mode: str, *, fps: float, horizon: int,
               inference_hz: float | None = None,
               max_latency_steps: int | None = None) -> float | None:
    """The hard timing budget each strategy's splice math assumes.

    sync                None — the loop blocks on inference; slow is slow, but
                        the robot HOLDS (the executor keeps interpolating to
                        the last waypoint), it never splices stale chunks.
    async (naive)       one chunk duration H/fps: any longer and the buffer
                        runs dry before the replacement lands — the freeze.
    temporal_ensembling H/fps too — an ensemble of expired chunks is a freeze
                        with extra steps.
    temporal_smoothing  max_latency_steps/fps: the blend drops that many stale
    rtc                 rows on arrival, so a chunk older than that splices
                        actions the robot already executed — the lurch.
    """
    if mode == "sync":
        return None
    if mode in ("async", "temporal_ensembling"):
        b = horizon / fps
        if inference_hz:
            b = min(b, 1.0 / inference_hz * 2.0)  # worker must roughly keep its period
        return b
    if mode in ("temporal_smoothing", "rtc"):
        if max_latency_steps is None:
            raise ValueError(f"{mode} needs max_latency_steps")
        return max_latency_steps / fps
    raise ValueError(f"unknown mode {mode!r}")


def measure_policy_latency(infer_fn, n: int = 5, *, clock=time.monotonic
                           ) -> tuple[float, list[float]]:
    """(first_call_s, steady_samples). The first call is timed separately —
    it may include a server-side XLA compile (minutes cold) and must never
    happen with the robot in the loop."""
    t0 = clock()
    infer_fn()
    first = clock() - t0
    samples = []
    for _ in range(n):
        t0 = clock()
        infer_fn()
        samples.append(clock() - t0)
    return first, samples


def startup_self_check(mode: str, infer_fn, *, fps: float, horizon: int,
                       inference_hz: float | None = None,
                       max_latency_steps: int | None = None,
                       n: int = 5, headroom: float = 1.15,
                       clock=time.monotonic) -> LatencyReport:
    """Measure warmup inferences and REFUSE to start an over-budget strategy.

    `infer_fn` must perform one real inference against a representative
    observation (real camera frame size, real state) — a toy request measures
    the wrong thing; the wire cost is part of the number.

    Raises LatencyBudgetError when p95 * headroom exceeds the mode's budget.
    """
    first, samples = measure_policy_latency(infer_fn, n, clock=clock)
    p95 = float(np.quantile(samples, 0.95))
    budget = budget_for(mode, fps=fps, horizon=horizon,
                        inference_hz=inference_hz,
                        max_latency_steps=max_latency_steps)

    if budget is None:
        chunk_s = horizon / fps
        verdict, detail = "ok", (
            f"blocking mode; each inference pauses the rollout ~{p95*1000:.0f} ms")
        if p95 > chunk_s:
            verdict = "warn"
            detail = (f"inference ({p95*1000:.0f} ms) exceeds a whole chunk "
                      f"({chunk_s*1000:.0f} ms); the robot will visibly pause "
                      "between chunks. Safe, but consider a closer server.")
        return LatencyReport(mode, first, tuple(samples), p95, None, verdict, detail)

    if p95 * headroom > budget:
        detail = (
            f"measured p95 {p95*1000:.0f} ms x{headroom:.2f} headroom exceeds the "
            f"{mode} budget of {budget*1000:.0f} ms. This is exactly the measured "
            "freeze-lurch failure (1.3-5.1 s against 0.4 s, jitter_root_cause.md) "
            "and it will not get better mid-rollout. Fix the serving latency "
            "(profile the server, move it closer, drop image size) or run --mode "
            "sync, which holds instead of splicing stale chunks."
        )
        report = LatencyReport(mode, first, tuple(samples), p95, budget, "refuse", detail)
        raise LatencyBudgetError(report.summary())

    slack = budget - p95 * headroom
    return LatencyReport(
        mode, first, tuple(samples), p95, budget, "ok",
        f"{slack*1000:.0f} ms of headroom under the {budget*1000:.0f} ms budget")
