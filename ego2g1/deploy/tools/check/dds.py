"""Rung 1 (`check listen`): DDS subscribe only — proves the domain, the
topic names, and the Brainco bridge. Publishes nothing."""

from __future__ import annotations

import sys
import time

import numpy as np

from ego2g1.core import layout
from ego2g1.deploy._util import dds_init

# --- 1. listen ---------------------------------------------------------------

def listen(iface: str | None = None, domain: int = 0, seconds: float = 5.0,
           hands: bool = True) -> None:
    """Subscribe only. No publishers, nothing commanded. Proves the DDS domain,
    the topic names, and that the Brainco bridge is actually running."""
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    dds_init(iface, domain)

    state = {"msg": None, "t": 0.0}

    def on_state(msg):
        state["msg"], state["t"] = msg, time.monotonic()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    hand_state = {}
    if hands:
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
        for h in layout.HANDS:
            hand_state[h] = {"q": None, "t": 0.0}

            def make_cb(hh):
                def cb(msg):
                    hand_state[hh]["q"] = np.array(
                        [msg.states[i].q for i in range(layout.HAND_DIM)], np.float32)
                    hand_state[hh]["t"] = time.monotonic()
                return cb

            s = ChannelSubscriber(f"rt/brainco/{h}/state", MotorStates_)
            s.Init(make_cb(h), 10)

    t0 = time.monotonic()
    while state["msg"] is None:
        if time.monotonic() - t0 > 5.0:
            sys.exit("no rt/lowstate in 5 s — check the link / DDS domain / iface.")
        time.sleep(0.05)
    print(f"lowstate OK (age {(time.monotonic()-state['t'])*1000:.0f} ms)\n")

    # arm slots 15..28 (legs 0-11, waist 12-14) — G1_29_JointArmIndex order.
    arm_idx = list(range(15, 29))
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        q = np.array([state["msg"].motor_state[i].q for i in arm_idx])
        print(f"  arm q  L {np.round(q[:7], 3)}  R {np.round(q[7:], 3)}")
        for h in layout.HANDS:
            if hands:
                if hand_state[h]["q"] is None:
                    print(f"  hand {h:5s} NO STATE — is the Brainco bridge running?")
                else:
                    age = time.monotonic() - hand_state[h]["t"]
                    print(f"  hand {h:5s} {np.round(hand_state[h]['q'], 3)}  "
                          f"(age {age*1000:.0f} ms)")
        time.sleep(0.5)
    print("\nlisten OK — no commands were sent.")
