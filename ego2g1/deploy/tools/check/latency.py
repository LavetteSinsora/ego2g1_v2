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
    # Size the state to the CONNECTED checkpoint, not to a constant. A
    # relation_eef checkpoint takes the 56-dim hand-major relation vector; the
    # joint/relative_eef ones take 30-dim FK proprioception. Sending the wrong
    # width either errors or measures a payload the real rollout never sends.
    state_dim = 56 if c.control_mode == "relation_eef" else 30
    state = np.zeros(state_dim, dtype=np.float32)

    print(f"\nserver {host}:{port} | horizon {c.action_horizon} "
          f"dim {c.action_dim} fps {c.fps} control_mode {c.control_mode}")
    print(f"sending observation/state as ({state_dim},) for control_mode "
          f"{c.control_mode!r}")

    # Keep the last reply so the server's own timing can be separated from the
    # wire cost: total - server = network + serialisation + image encode.
    last: dict = {}

    def _one():
        last.clear()
        last.update(c.infer(frame, state, "latency check"))

    first, samples = _latency.measure_policy_latency(_one, n)
    lat = np.array(samples)
    p95 = float(np.quantile(lat, 0.95))
    print(f"first call (includes XLA compile): {first:.1f} s")
    print(f"steady: mean {lat.mean()*1000:.0f} ms   p95 {p95*1000:.0f} ms   "
          f"max {lat.max()*1000:.0f} ms")

    timing = last.get("policy_timing") or {}
    server_s = None
    for key in ("infer_ms", "policy_ms", "infer_s", "total_ms", "total_s"):
        if key in timing:
            v = float(timing[key])
            server_s = v / 1000.0 if key.endswith("_ms") else v
            break
    if server_s is not None:
        wire = lat.mean() - server_s
        print(f"        of which server-side {server_s*1000:.0f} ms, "
              f"wire+encode {wire*1000:.0f} ms "
              f"({100*wire/max(lat.mean(), 1e-9):.0f}% of the round trip)")
        print("        -> if wire dominates, move the server closer or shrink "
              "the image; if server-side dominates, profile the policy.")
    elif timing:
        print(f"        server reported policy_timing={timing}")
    print()

    # --- where does the non-inference half of the round trip go? -------------
    # The payload is tiny: a 224x224x3 uint8 image is 147 KB, state is 224 B,
    # and the reply is ~3 KB. On loopback that should be well under a
    # millisecond, so a 60 ms gap between total and server-side is anomalous
    # and worth splitting before optimising anything.
    #
    # Two image sizes give a line: the SLOPE is per-byte cost (serialisation,
    # copies, bandwidth), the INTERCEPT is fixed per-request overhead
    # (connection handling, Nagle/delayed-ACK, server-side deserialise and
    # transform work that policy_timing does not count). They call for
    # completely different fixes, so measuring which one dominates is the
    # whole point.
    import time as _time

    from ego2g1.deploy.core import client as _c

    def _probe(side: int, reps: int = 10) -> tuple[float, int]:
        probe = _c.PolicyClient(host, port, resize=(side, side))
        img = np.random.randint(0, 255, (*frame_hw, 3), dtype=np.uint8)
        nbytes = probe._prepare_image(img).nbytes
        probe.infer(img, state, "warm")                 # warm the path
        t0 = _time.monotonic()
        for _ in range(reps):
            probe.infer(img, state, "latency check")
        return (_time.monotonic() - t0) / reps, nbytes

    try:
        t_lo, b_lo = _probe(112)
        t_hi, b_hi = _probe(224)
        print(f"payload probe: {b_lo/1024:.0f} KB -> {t_lo*1000:.0f} ms | "
              f"{b_hi/1024:.0f} KB -> {t_hi*1000:.0f} ms")
        if b_hi > b_lo:
            slope_ms_per_kb = (t_hi - t_lo) * 1000 / ((b_hi - b_lo) / 1024)
            fixed_ms = (t_lo - (b_lo / 1024) * slope_ms_per_kb / 1000) * 1000
            print(f"  per-KB {slope_ms_per_kb:.3f} ms   fixed overhead "
                  f"{fixed_ms:.0f} ms")
            if fixed_ms > 0.5 * t_hi * 1000:
                print("  -> FIXED cost dominates. Shrinking the image will not "
                      "help. Suspect Nagle/delayed-ACK (set TCP_NODELAY on the\n"
                      "     websocket socket) or server-side deserialise+"
                      "transform work that policy_timing excludes.")
            else:
                print("  -> PAYLOAD cost dominates. Shrink the image "
                      "(resize=, or JPEG on the wire) or avoid a copy in "
                      "_prepare_image.")
    except Exception as exc:                                    # noqa: BLE001
        print(f"payload probe skipped ({type(exc).__name__}: {exc})")
    print()

    for mode in ("sync", "async", "temporal_smoothing"):
        b = _latency.budget_for(mode, fps=c.fps, horizon=c.action_horizon,
                                inference_hz=4.0, max_latency_steps=8)
        verdict = ("no hard budget (holds during inference)" if b is None else
                   ("OK, %.0f ms headroom" % ((b - p95 * 1.15) * 1000)
                    if p95 * 1.15 <= b else
                    "OVER BUDGET — the runner will REFUSE this mode"))
        print(f"  {mode:20s} budget "
              f"{'—' if b is None else '%4.0f ms' % (b*1000)}   {verdict}")
