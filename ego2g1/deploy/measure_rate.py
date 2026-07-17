"""Measure the TRUE rt/lowstate arrival rate and its gap distribution.

Ported near-verbatim from the old deploy (third_party/openpi/ego2g1/deploy/
measure_rate.py). `check listen` polls the age at 2 Hz — it samples, it does
not count, so it cannot see a 500 Hz stream or resolve where the stalls are.
This timestamps EVERY callback and reports the real rate and inter-arrival
histogram. That is the number the "arm lags seconds behind" diagnosis hinges
on: a link that starves lowstate almost certainly consumes lowcmd on the same
schedule.

Sends nothing (unless you ask it to). Safe to run anytime.

    python -m ego2g1.deploy.measure_rate --seconds 10
    python -m ego2g1.deploy.measure_rate --seconds 10 --flood 500   # under our command load
"""

import threading
import time

import numpy as np
import tyro


def measure_rate(seconds: float = 10.0, iface: str | None = None, domain: int = 0,
                 flood: float = 0.0) -> None:
    """Count rt/lowstate arrivals for `seconds` and report rate + gap histogram.

    `--flood HZ` publishes a hold-in-place no-op lowcmd (kp=0, kd=1 — never
    moves the arm) at HZ from a second thread while measuring, to see whether
    OUR command traffic degrades the state stream. Off by default."""
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    if iface:
        ChannelFactoryInitialize(domain, iface)
    else:
        ChannelFactoryInitialize(domain)

    stamps: list[float] = []
    first_state = threading.Event()

    def on_state(_msg):
        stamps.append(time.perf_counter())
        first_state.set()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    print("waiting for first rt/lowstate ...")
    if not first_state.wait(timeout=5.0):
        raise TimeoutError("no rt/lowstate in 5s — check the link / DDS domain.")

    stop = threading.Event()
    flood_stats = {"sent": 0}
    if flood > 0:
        pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        pub.Init()
        crc = CRC()
        msg = unitree_hg_msg_dds__LowCmd_()
        for i in range(35):
            msg.motor_cmd[i].mode = 1
            msg.motor_cmd[i].kp = 0.0
            msg.motor_cmd[i].kd = 1.0

        def flood_fn():
            period = 1.0 / flood
            nxt = time.perf_counter()
            while not stop.is_set():
                msg.crc = crc.Crc(msg)
                pub.Write(msg)
                flood_stats["sent"] += 1
                nxt += period
                d = nxt - time.perf_counter()
                if d > 0:
                    time.sleep(d)

        threading.Thread(target=flood_fn, daemon=True).start()
        print(f"flooding rt/lowcmd at {flood:.0f} Hz (no-op hold, kp=0) while measuring")

    print(f"measuring for {seconds:.0f}s ...")
    stamps.clear()
    t0 = time.perf_counter()
    time.sleep(seconds)
    stop.set()
    elapsed = time.perf_counter() - t0

    if len(stamps) < 2:
        print(f"only {len(stamps)} messages in {elapsed:.1f}s — the stream is nearly dead.")
        return

    s = np.array(stamps)
    gaps_ms = np.diff(s) * 1e3
    rate = (len(s) - 1) / (s[-1] - s[0])

    print("\n" + "=" * 60)
    print(f"rt/lowstate: {len(s)} msgs in {elapsed:.1f}s  ->  {rate:.0f} Hz")
    if flood > 0:
        print(f"(while we published {flood_stats['sent']} lowcmd at {flood:.0f} Hz)")
    print(f"inter-arrival ms: mean {gaps_ms.mean():.1f}  "
          f"p50 {np.percentile(gaps_ms, 50):.1f}  p95 {np.percentile(gaps_ms, 95):.1f}  "
          f"p99 {np.percentile(gaps_ms, 99):.1f}  max {gaps_ms.max():.1f}")
    stalls = int((gaps_ms > 20).sum())
    print(f"stalls >20 ms: {stalls} ({100*stalls/len(gaps_ms):.1f}%)   "
          f">50 ms: {int((gaps_ms > 50).sum())}")

    edges = [0, 2, 4, 8, 16, 32, 64, 128, np.inf]
    counts, _ = np.histogram(gaps_ms, bins=edges)
    print("gap histogram:")
    for k in range(len(counts)):
        hi = "inf" if not np.isfinite(edges[k + 1]) else f"{edges[k+1]:4.0f}"
        bar = "#" * int(round(50 * counts[k] / max(counts.sum(), 1)))
        print(f"    {edges[k]:4.0f}-{hi} ms | {counts[k]:6d} "
              f"{100*counts[k]/len(gaps_ms):5.1f}%  {bar}")

    print("\ninterpretation:")
    if rate > 300:
        print(f"  ~{rate:.0f} Hz — the link is healthy. Arm lag is NOT transport;")
        print("  look again at gains / feedforward / a foreign publisher (sniff_lowcmd).")
    else:
        print(f"  ~{rate:.0f} Hz is far below 500 — the robot/link publishes state slowly.")
        print("  A link this starved on state almost certainly consumes lowcmd on the")
        print("  same schedule, so commands back up and the arm lags/judders.")
    if flood > 0:
        print("  Compare to a --flood 0 run: if flooding drops the rate, our own")
        print("  command volume is degrading the link.")


if __name__ == "__main__":
    tyro.cli(measure_rate)
