"""Rungs 5/5b (`check hand-sweep` / `check hand-jog`): resolve the Brainco
motor order one finger at a time, and interactively measure
BRAINCO_CLOSED_POSE around a real object (gripper_calib.py)."""

from __future__ import annotations

import sys
import time

import numpy as np

from ego2g1.core import layout
from ego2g1.deploy._util import dds_init

# --- 5. hand sweep -------------------------------------------------------------

def hand_sweep(iface: str | None = None, domain: int = 0, hand: str = "right",
               motor: int = 2, lo: float = 0.0, hi: float = 0.6,
               seconds: float = 4.0) -> None:
    """Drive ONE Brainco motor slowly between two commands, watching the arm not
    at all. Commands are [0, 1] (0=open, 1=closed) — that much is settled. What
    this rung resolves is the ORDER: whether HAND_MOTOR_ORDER [thumb_flex,
    thumb_rot, index, middle, ring, pinky] maps 1:1 onto Brainco's [Thumb,
    ThumbAux, Index, Middle, Ring, Pinky]. If commanding `motor` moves a
    different finger, fix the mapping before any policy runs."""
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

    dds_init(iface, domain)

    name = layout.HAND_MOTOR_ORDER[motor]
    print(f"sweeping {hand} motor {motor} ({name}) between {lo} and {hi}")
    print("WATCH THE HAND. Which finger actually moves?\n")

    pubs, msgs = {}, {}
    for h in layout.HANDS:
        pubs[h] = ChannelPublisher(f"rt/brainco/{h}/cmd", MotorCmds_)
        pubs[h].Init()
        m = MotorCmds_()
        m.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(layout.HAND_DIM)]
        for i in range(layout.HAND_DIM):
            m.cmds[i].q = 0.0
            m.cmds[i].dq = 1.0   # vendor uses dq as a speed field here
        msgs[h] = m

    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            phase = 0.5 - 0.5 * np.cos(
                2 * np.pi * (time.monotonic() - t0) / seconds * 2)
            msgs[hand].cmds[motor].q = float(lo + (hi - lo) * phase)
            for h in layout.HANDS:
                pubs[h].Write(msgs[h])
            print(f"  cmd {msgs[hand].cmds[motor].q:.3f}", end="\r")
            time.sleep(1 / 200)
    finally:
        for h in layout.HANDS:
            for i in range(layout.HAND_DIM):
                msgs[h].cmds[i].q = 0.0
            pubs[h].Write(msgs[h])
        print("\n\nreturned to open.")


# --- 5b. hand jog (BRAINCO_CLOSED_POSE measurement) -----------------------------

def _read_key(timeout: float = 0.05):
    """Non-blocking single keypress from stdin, or None if nothing arrived
    within `timeout` -- the standard termios/select technique for
    interactive terminal control without a third-party dependency (matches
    this repo's existing no-new-deps-for-a-CLI-rung discipline)."""
    import select

    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else None


def hand_jog(iface: str | None = None, domain: int = 0, hand: str = "right",
            step: float = 0.02) -> None:
    """Interactively jog all 6 BrainCo motors for ONE hand via the keyboard,
    watching the real hand close around a real object, until it's exactly
    the grip you want -- then prints the final (6,) vector formatted ready
    to paste into `ego2g1/deploy/gripper_calib.py`'s
    `BRAINCO_CLOSED_POSE[hand]` (docs/relation_deploy_plan.md §7: there is
    no principled way to derive this from the binary training signal, a
    human has to measure it on the real hand around the real task objects).

    Keys (case-insensitive):
      1-6    select which motor to adjust -- HAND_MOTOR_ORDER order:
             1=thumb_flex 2=thumb_rot 3=index 4=middle 5=ring 6=pinky
      j/k    decrease / increase the SELECTED motor by `step` (clamped [0,1])
      o      open this hand fully (all motors -> 0.0) -- safety reset
      c      close this hand fully (all motors -> 1.0) -- coarse starting point
      p      print the current (6,) vector without quitting
      q      quit -- prints the final vector once more and exits
    Ctrl-C also exits cleanly, same as 'q'.

    The OTHER hand is held open throughout and is never touched. Commands
    publish continuously (not just on keypress) at the same rate
    `hand_sweep` uses -- the Brainco driver holds stale state if it isn't
    commanded every tick (see `actions.py`'s `JointChunks` docstring), so a
    live interactive tool must keep publishing even while idle between
    keypresses, not just when a key changes something.
    """
    import termios
    import tty

    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_

    if hand not in layout.HANDS:
        raise ValueError(f"hand must be one of {layout.HANDS}, got {hand!r}")

    dds_init(iface, domain)

    pubs, msgs = {}, {}
    for h in layout.HANDS:
        pubs[h] = ChannelPublisher(f"rt/brainco/{h}/cmd", MotorCmds_)
        pubs[h].Init()
        m = MotorCmds_()
        m.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(layout.HAND_DIM)]
        for i in range(layout.HAND_DIM):
            m.cmds[i].q = 0.0
            m.cmds[i].dq = 1.0
        msgs[h] = m

    selected = 0

    def _status_line() -> str:
        vals = ", ".join(
            f"{'*' if i == selected else ' '}{name}={msgs[hand].cmds[i].q:.3f}"
            for i, name in enumerate(layout.HAND_MOTOR_ORDER)
        )
        return f"  [{hand}] {vals}"

    def _print_vector() -> None:
        vec = [round(float(msgs[hand].cmds[i].q), 3) for i in range(layout.HAND_DIM)]
        print(f"\n\nBRAINCO_CLOSED_POSE[{hand!r}] = np.array({vec}, dtype=np.float32)"
              f"   # {layout.HAND_MOTOR_ORDER}")

    print(f"Jogging {hand} hand. Keys: 1-6 select motor, j/k -/+ step, "
          "o open all, c close all, p print, q quit.")
    print(f"Motor order: {layout.HAND_MOTOR_ORDER}\n")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key()
            if key:
                key = key.lower()
                if key == "q":
                    break
                if key in "123456":
                    selected = int(key) - 1
                elif key == "j":
                    msgs[hand].cmds[selected].q = float(
                        np.clip(msgs[hand].cmds[selected].q - step, 0.0, 1.0))
                elif key == "k":
                    msgs[hand].cmds[selected].q = float(
                        np.clip(msgs[hand].cmds[selected].q + step, 0.0, 1.0))
                elif key == "o":
                    for i in range(layout.HAND_DIM):
                        msgs[hand].cmds[i].q = 0.0
                elif key == "c":
                    for i in range(layout.HAND_DIM):
                        msgs[hand].cmds[i].q = 1.0
                elif key == "p":
                    _print_vector()

            for h in layout.HANDS:
                pubs[h].Write(msgs[h])
            print(_status_line(), end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        _print_vector()
