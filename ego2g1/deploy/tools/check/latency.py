"""Rung 8 (`check latency`): round trip to the policy server + per-mode
budget verdicts. Run on the server box AND the deploy machine — the
difference is the network."""

from __future__ import annotations

import numpy as np

# --- 8. policy-server latency ---------------------------------------------------

def latency(host: str = "127.0.0.1", port: int = 8000, n: int = 20,
            frame_hw: tuple[int, int] = (480, 640)) -> None:
    """Time the round trip to the policy server. No robot, no camera.

    Run it TWICE: on the server box (127.0.0.1) and on the deploy machine. The
    server-local number is pure inference; the difference is what the network
    costs. p95 is the number that matters — the budget is a cliff, not a
    gradient (latency.budget_for). The first call includes an XLA compile
    (minutes cold) and is reported separately: never let a policy's first-ever
    request happen with the robot in the loop."""
    from ego2g1.deploy.core import client as _client
    from ego2g1.deploy.core import latency as _latency

    c = _client.PolicyClient(host, port)
    frame = np.random.randint(0, 255, (*frame_hw, 3), dtype=np.uint8)
    state = np.zeros(30, dtype=np.float32)

    print(f"\nserver {host}:{port} | horizon {c.action_horizon} "
          f"dim {c.action_dim} fps {c.fps} control_mode {c.control_mode}")

    first, samples = _latency.measure_policy_latency(
        lambda: c.infer(frame, state, "latency check"), n)
    lat = np.array(samples)
    p95 = float(np.quantile(lat, 0.95))
    print(f"first call (includes XLA compile): {first:.1f} s")
    print(f"steady: mean {lat.mean()*1000:.0f} ms   p95 {p95*1000:.0f} ms   "
          f"max {lat.max()*1000:.0f} ms\n")

    for mode in ("sync", "async", "temporal_smoothing"):
        b = _latency.budget_for(mode, fps=c.fps, horizon=c.action_horizon,
                                inference_hz=4.0, max_latency_steps=8)
        verdict = ("no hard budget (holds during inference)" if b is None else
                   ("OK, %.0f ms headroom" % ((b - p95 * 1.15) * 1000)
                    if p95 * 1.15 <= b else
                    "OVER BUDGET — the runner will REFUSE this mode"))
        print(f"  {mode:20s} budget "
              f"{'—' if b is None else '%4.0f ms' % (b*1000)}   {verdict}")
