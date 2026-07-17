# Interactive up/down control of the G1-D lift column (sheng jiang) over DDS.
#
# This file is executed INLINE on the robot via `python -c "<contents>"` (see
# move_g1d.sh). It is never copied onto the robot -- the source is streamed as a
# command-line argument, so no file is written to the robot disk.
#
# IMPORTANT: this source must contain NO single-quote characters, because the
# launcher wraps the whole program in single quotes when handing it to the remote
# shell. Use double quotes and chr() for control characters everywhere.
#
# Interface (verified on the robot):
#   command  topic rt/cmd_hispeed   (geometry_msgs/Point32): .z = vertical velocity
#            in [-1, 1], + = up, - = down, 0 = hold. Publish continuously at ~30 Hz.
#   feedback topic rt/hispeed_state (geometry_msgs/Point32): .y = height (meters).
#
# Machine specifics come in as env vars (set by move_g1d.sh on the ssh command):
#   LIFT_IFACE  DDS network interface on the robot (default eth0)
#   LIFT_DOMAIN DDS domain id (default 0)
import os, sys, time, tty, termios, select
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_
from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__Point32_ as MkP

STEP = 0.25    # velocity magnitude commanded while an arrow key is held (|z| <= 1)
CLAMP = 0.30   # hard cap on published velocity (safety)
HOLD = 0.15    # seconds since last key-repeat before auto-stop (release detection)
FPS = 30.0
ESC = chr(27)
ETX = chr(3)   # Ctrl-C

IFACE = os.environ.get("LIFT_IFACE", "eth0")
DOMAIN = int(os.environ.get("LIFT_DOMAIN", "0"))
ChannelFactoryInitialize(DOMAIN, IFACE)   # real robot: DDS domain 0, internal iface eth0
pub = ChannelPublisher("rt/cmd_hispeed", Point32_); pub.Init()
sub = ChannelSubscriber("rt/hispeed_state", Point32_); sub.Init()
msg = MkP()

v = 0.0
last = 0.0
fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
print("Lift ready. Up=raise  Down=lower  (release=stop)  q or Ctrl-C=quit", flush=True)
try:
    tty.setraw(fd)
    while True:
        # Drain any pending key bytes without blocking.
        while select.select([sys.stdin], [], [], 0)[0]:
            c = sys.stdin.read(1)
            if c == ETX or c == "q":
                raise KeyboardInterrupt
            if c == ESC:
                seq = sys.stdin.read(2)
                if seq == "[A":
                    v = +STEP; last = time.time()
                elif seq == "[B":
                    v = -STEP; last = time.time()
        # Auto-stop shortly after the held key stops repeating (i.e. on release).
        if time.time() - last > HOLD:
            v = 0.0
        msg.z = max(-CLAMP, min(CLAMP, v))
        pub.Write(msg)
        s = sub.Read()
        h = None if s is None else s.y
        hs = ("%.3f" % h) if h is not None else "?"
        sys.stdout.write("\r vel=%+.2f  height=%s m    " % (msg.z, hs))
        sys.stdout.flush()
        time.sleep(1.0 / FPS)
except KeyboardInterrupt:
    pass
finally:
    msg.z = 0.0
    for _ in range(5):     # guarantee a stop command lands
        pub.Write(msg); time.sleep(0.01)
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print("\nstopped (z=0 sent).")
