"""Is something ELSE publishing rt/lowcmd while we think we own the arm?

Ported near-verbatim from the old deploy (third_party/openpi/ego2g1/deploy/
sniff_lowcmd.py). The firmware executes the LAST message it received, so two
publishers disagreeing at 500 Hz produce exactly the classic "arm barely
moves, then lurches" stick-slip — and no change to OUR sender fixes it.

Run it in two states:

  1. WITH OUR STACK IDLE. ANY traffic here is a foreign publisher (the robot's
     onboard hold/balance controller, a stale process, another terminal) —
     that is the bug. Put the robot in the mode that relinquishes lowcmd and
     confirm this goes silent.
  2. WHILE a replay/deploy RUNS (second terminal). A rate much higher than our
     send rate, or a target q that does not match ours => someone else is
     also writing.

Read-only. Never publishes.

    python -m ego2g1.deploy.sniff_lowcmd --seconds 5     # our stack IDLE
    python -m ego2g1.deploy.sniff_lowcmd --seconds 10    # again, while deploy runs
"""

import time

import numpy as np
import tyro

# Arm slots in the 35-motor LowCmd: left 15-21, right 22-28
# (G1_29_JointArmIndex; legs 0-11, waist 12-14).
ARM_IDX_FLAT = list(range(15, 29))


def sniff_lowcmd(seconds: float = 5.0, iface: str | None = None,
                 domain: int = 0) -> None:
    """Count rt/lowcmd traffic and sample the arm targets on it. Publishes nothing."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    if iface:
        ChannelFactoryInitialize(domain, iface)
    else:
        ChannelFactoryInitialize(domain)

    stamps: list[float] = []
    arm_q: list[np.ndarray] = []
    arm_kp: list[np.ndarray] = []

    def on_cmd(msg):
        stamps.append(time.perf_counter())
        arm_q.append(np.array([msg.motor_cmd[i].q for i in ARM_IDX_FLAT], np.float32))
        arm_kp.append(np.array([msg.motor_cmd[i].kp for i in ARM_IDX_FLAT], np.float32))

    sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub.Init(on_cmd, 50)

    print(f"listening on rt/lowcmd for {seconds:.0f}s (publishing NOTHING) ...")
    stamps.clear()
    time.sleep(seconds)

    n = len(stamps)
    if n == 0:
        print("\n  rt/lowcmd is SILENT. Good — no foreign publisher.")
        print("  If our stack was idle, WE will be the only writer, so a judder is")
        print("  in our command after all. If you expected our stack to be running,")
        print("  it is not reaching the wire.")
        return

    s = np.array(stamps)
    rate = (n - 1) / (s[-1] - s[0]) if n > 1 else 0.0
    q = np.stack(arm_q)
    kp = np.stack(arm_kp)
    print("\n" + "=" * 60)
    print(f"rt/lowcmd: {n} msgs in {seconds:.0f}s  ->  {rate:.0f} Hz  "
          "<-- SOMEONE is publishing")
    print(f"  arm target q moves over window: {np.round(q.max(0) - q.min(0), 3)}")
    print(f"  arm kp seen on wire: {np.round(kp.mean(0), 1)}  "
          "(unitree_deploy's: shoulder/elbow 80, wrist 40)")
    print("\ninterpretation:")
    print("  If our stack was IDLE and this is not silent => a foreign publisher")
    print("  owns the wire and fights every command we send. Fixes: release the")
    print("  onboard arm/motion controller; kill the stale process; if the kp on")
    print("  the wire differs from 80/40, another config is commanding the arm.")
    print("  Re-run until silent with our stack idle; then replay should track.")


if __name__ == "__main__":
    tyro.cli(sniff_lowcmd)
