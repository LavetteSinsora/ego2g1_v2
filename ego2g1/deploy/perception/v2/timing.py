"""The `d` arithmetic and tick binding (plan T3, T4).

`d` is how many 30 Hz control ticks elapse between the instant an observation
describes and the instant the resulting chunk starts executing. It is sent
WITH the request because the RTC guidance mask depends on it, so it is
necessarily a forecast — and the two errors cost very differently:

    d too small   we execute past the frozen prefix while inferring; the new
                  chunk's committed slots no longer match what the robot did.
                  A discontinuity — a lurch at the seam.
    d too large   we over-constrain and lose a little reactivity. Harmless.

So erring high is correct, and a CONSTANT d beats a freshly-predicted one.

WHAT MAKES IT CONSTANT
`DelayBudget.observe()` is currently fed policy latency alone, which would
report d ≈ 4 and silently under-commit by roughly a full perception round. The
observation does not become available when the request is sent; it becomes
available when the round that produced it STARTED. So the delay to budget is

    d = P + L        P = perception round, L = policy round trip

and it is constant precisely because a replan waits for the in-flight round
(T3) rather than grabbing the newest completed one — waiting pins
`t_send - t_capture` at exactly P instead of letting it wander over P..2P.

MEASURED, on the 4090 (plan §2.1):

    P   perception round   221 ms = 6.6 ticks
    L   policy             124 ms = 3.7 ticks
        d = P + L          345 ms = 10.4 ticks
        x 1.15 headroom    397 ms = 11.9   ->  d = 12

12 ticks (400 ms) fits the 500 ms async budget with ~100 ms spare. 16 ticks
(533 ms) would exceed it and `startup_self_check` would refuse the mode.
"""

from __future__ import annotations

import math

__all__ = ["delay_ticks", "usable_slots", "chunk_arithmetic_closes"]


def delay_ticks(perception_s: float, policy_s: float, *, fps: float = 30.0,
                headroom: float = 1.15, max_d: int = 20) -> int:
    """`d` in control ticks, from the two measured latencies.

    `headroom` matches `DelayBudget`'s own 1.15 default. `max_d` saturates
    rather than growing without bound: unbounded, a slow round yields d > H,
    the chunk installs with zero usable slots and the replan trigger goes
    negative — the loop silently stops planning. Saturating merely loses the
    continuity guarantee, and says so.
    """
    if perception_s < 0 or policy_s < 0:
        raise ValueError("latencies must be non-negative")
    want = math.ceil((perception_s + policy_s) * headroom * fps)
    return max(1, min(int(want), int(max_d)))


def usable_slots(horizon: int, d: int) -> int:
    """Action slots left after the frozen prefix. Slot k means control tick
    `n_capture + k`, so the first `d` of them are already in the past by the
    time the chunk lands."""
    return max(0, int(horizon) - int(d))


def chunk_arithmetic_closes(horizon: int, d: int, replan_period_s: float, *,
                            fps: float = 30.0) -> tuple[bool, float]:
    """(closes, usable_seconds). A chunk must cover the replan period with
    slack, or the buffer runs dry before its replacement lands — the measured
    freeze-lurch failure (docs/jitter_root_cause.md)."""
    seconds = usable_slots(horizon, d) / float(fps)
    return seconds > float(replan_period_s), seconds
