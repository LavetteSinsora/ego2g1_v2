"""The operator entrypoint.

    .venv/bin/python -m tools.teleop \
        --B data_extraction/work/_global/b_calib.npz \
        --dataset lerobot_datasets/ego2g1/put_bottle_in_box --start-from-episode 0

Keys (this terminal, no Enter needed):

    e      ENGAGE / re-anchor   the robot starts following your hands, from where it is
    space  DISENGAGE            the robot holds; your hands are free to move back
    q      DAMP and quit        the e-stop

The clutch is not a nicety. A human's reach is bigger than the G1's, so you WILL run the
arm out of its workspace; disengage, bring your hands back to a comfortable pose, and
re-engage. Re-engaging re-anchors, so nothing jumps.

--start-from-episode is close to mandatory, and the reason is worth knowing. Offline, the
placement fit S is what maps the human's hands into the robot's reachable set. The
relative action cancels S, so teleop does not have it -- the ANCHOR takes its place.
Engage with the robot's arms hanging at its sides and a normal reach lands ~190 mm
outside the workspace (measured, `check replay`). Ramping to a real episode's start pose
puts the robot where the demonstrator's hands were, and the same motion then tracks to
0.2 mm. Start from a pose that matches the posture you intend to engage in.
"""

import argparse
import logging
import pathlib
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

from ._vendor.eg.deploy import dds as _dds
from ._vendor.eg.deploy import kinematics as _kin
from ._vendor.eg.deploy import ramp as _ramp
from tools.teleop import calib as _calib
from tools.teleop.loop import TeleopConfig, TeleopLoop
from tools.teleop.retarget import SIDES, TeleopRetargeter, load_B
from tools.teleop.source import Hdf5Source, VuerSource, wait_for_hands

logger = logging.getLogger("teleop")


class RawKeys:
    """Unbuffered single-key reads from the terminal, restored on the way out.

    The clutch has to be reachable in one keystroke with no Enter -- if disengaging
    takes two keys the operator will not do it in time.
    """

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def get(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--B", required=True,
                   help="b_calib npz or LeRobot dataset root — the SAME wrist->flange "
                        "alignment the training labels were built with")
    p.add_argument("--dataset", default=None, help="LeRobot dataset, for the start pose")
    p.add_argument("--start-from-episode", type=int, default=None,
                   help="ramp to this episode's first arm pose before engaging")
    p.add_argument("--control-hz", type=float, default=60.0)
    p.add_argument("--ramp-s", type=float, default=3.0)
    p.add_argument("--network-interface", default=None)
    p.add_argument("--display-mode", default="pass-through",
                   choices=["pass-through", "ego", "immersive"],
                   help="pass-through: you watch the real robot with your own eyes and "
                        "the PICO is a pure hand-tracking sensor (neck-mounted). "
                        "ego: robot's view inset in the real world (head-worn).")
    p.add_argument("--cert", default=None)
    p.add_argument("--key", default=None)
    p.add_argument("--hands", nargs="+", default=list(SIDES))
    p.add_argument("--sim", action="store_true",
                   help="teleoperate a DYNAMIC MuJoCo G1 (with a table and a bottle) "
                        "instead of the real robot. macOS: run under mjpython.")
    p.add_argument("--replay-episode", default=None, metavar="HDF5",
                   help="drive the hands from a RECORDED episode instead of the PICO, so "
                        "you can WATCH the sim with no headset. Implies --sim, and "
                        "auto-engages at the episode's first frame.")
    p.add_argument("--sim-realtime", type=float, default=1.0,
                   help="sim speed multiplier (0.5 = half speed, easier to grasp)")
    p.add_argument("--fingers-only", dest="arm_follow", action="store_false",
                   help="FINGERS only: the arm stays pinned at the engage anchor and your "
                        "wrist motion is ignored; just your fingers drive the Revo2. The "
                        "arm is what leaves the workspace, so this cannot trip the IK "
                        "watchdog — the way to exercise the hand retarget on its own.")
    p.add_argument("--no-finger-smooth", dest="finger_smooth", action="store_false",
                   help="disable the One-Euro finger smoother (on by default)")
    p.add_argument("--smooth-min-cutoff", type=float, default=1.5,
                   help="One-Euro min cutoff (Hz): lower = smoother when still")
    p.add_argument("--smooth-beta", type=float, default=0.05,
                   help="One-Euro speed coefficient: higher = less lag when moving fast")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    hands = tuple(args.hands)
    if args.replay_episode:
        args.sim = True     # replaying onto the real robot is what `deploy` is for

    # --- robot (real, or a physics-stepped MuJoCo stand-in; everything above this line
    #     is identical either way -- only the six-method DDS seam is swapped)
    if args.sim:
        from .sim import SimDDS
        dds = SimDDS(realtime=args.sim_realtime)
    else:
        dds = _dds.G1DDS(network_interface=args.network_interface)
    kin = _kin.Kinematics()
    if not args.sim:
        dds.connect()
        # (sim connects inside run_viewer's on_ready: starting physics before the window
        #  exists races launch_passive's scene build and segfaults -- see sim.run_viewer)

    # Everything from the ramp to the first engage BLOCKS -- `wait_for_hands` until the
    # PICO is in view, `collect_open_hand` for the calibration hold, `ramp_to` for its
    # duration. Under --sim that must not run on the main thread: the viewer has to own
    # it (mjpython's GUI loop does), and a window that only appears after you have already
    # held the calibration pose is a window you cannot use to hold it.
    state = {"src": None, "loop": None, "error": None}

    def bring_up():
        if args.start_from_episode is not None:
            if not args.dataset:
                raise SystemExit("--start-from-episode needs --dataset")
            import pandas as pd
            files = sorted(pathlib.Path(args.dataset).glob("data/*/*.parquet"))
            if not files:
                raise SystemExit(f"no parquet under {args.dataset}/data/")
            df = pd.read_parquet(files[min(args.start_from_episode, len(files) - 1)])
            q_start = np.asarray(df["arm_qpos"].to_numpy()[0], dtype=float)
            logger.info("ramping to episode %d start pose", args.start_from_episode)
            _ramp.ramp_to(dds, q_start, np.zeros(len(hands) * 6), ramp_s=args.ramp_s)
        else:
            logger.warning(
                "no --start-from-episode: engaging from wherever the arms are. If that is "
                "the arms-down pose, a normal reach will leave the workspace and trip the "
                "IK watchdog. See the module docstring.")

        # --- operator (a live headset, or a recorded episode standing in for one)
        if args.replay_episode:
            src = Hdf5Source(args.replay_episode, loop=True)
            src.start()
            print(f"\n  replaying {args.replay_episode} into the sim (no headset).")
        else:
            src = VuerSource(display_mode=args.display_mode, cert=args.cert, key=args.key)
            src.start()
            print("\n  On the PICO: open  https://<this-host>:8012/?ws=wss://<this-host>:8012")
            print("  and press 'Virtual Reality'. Waiting for BOTH hands in view...")
            wait_for_hands(src, hands)
            print("  hands tracked.\n")
        state["src"] = src

        rt = TeleopRetargeter(load_B(args.B), hands=hands,
                              finger_smooth=args.finger_smooth,
                              smooth_min_cutoff=args.smooth_min_cutoff,
                              smooth_beta=args.smooth_beta,
                              arm_follow=args.arm_follow)
        if not args.arm_follow:
            print("  FINGERS-ONLY: the arm holds at the engage anchor; only your fingers "
                  "drive the hands.")
        if args.replay_episode:
            # the episode's own most-open frame, exactly as the offline retarget
            # calibrates. collect_open_hand wants a human holding a pose at a wall clock;
            # a replay has no human, and its opening frames may not be an open hand.
            rt.calibrate({s: Hdf5Source(args.replay_episode).pose[s] for s in hands})
        else:
            rt.calibrate(_calib.collect_open_hand(src, hands=hands))

        loop = TeleopLoop(TeleopConfig(control_hz=args.control_hz),
                          dds=dds, kinematics=kin, source=src, retargeter=rt)
        state["loop"] = loop
        loop.start()

        if args.replay_episode:
            # Rewind to frame 0 before engaging. Hdf5Source indexes by elapsed wall time,
            # so calibration has already advanced it; engaging at whatever frame it is on
            # anchors the robot's START POSE against a mid-episode hand posture, and the
            # mismatch walks the targets out of the workspace (measured: a 108 mm IK trip).
            src.start()
            time.sleep(0.2)
            loop.engage()
            print("  auto-engaged at the episode's first frame.")

    def bring_up_guarded():
        try:
            bring_up()
        except BaseException as e:              # noqa: BLE001 - reported on the main thread
            state["error"] = e

    try:
        if args.sim:
            # Viewer FIRST, on the main thread, so you can watch the robot while the
            # bring-up (which needs your hands) happens behind it. Physics and bring-up
            # both start from on_ready, i.e. only once the window exists.
            from .sim import run_viewer

            def on_ready():
                dds.connect()      # starts the physics thread
                threading.Thread(target=bring_up_guarded, name="bring-up",
                                 daemon=True).start()

            run_viewer(dds, state, on_ready=on_ready)
        else:
            bring_up()
            loop = state["loop"]
            print("\n  e = ENGAGE / re-anchor    space = DISENGAGE    q = DAMP and quit\n")
            with RawKeys() as keys:
                while not loop.watchdog.tripped:
                    k = keys.get()
                    if k == "e":
                        loop.engage()
                    elif k == " ":
                        loop.disengage()
                    elif k == "q":
                        loop.estop("operator pressed q")
                        break
                    time.sleep(0.02)
    except KeyboardInterrupt:
        if state["loop"]:
            state["loop"].estop("KeyboardInterrupt")
    finally:
        loop, src = state["loop"], state["src"]
        if loop is not None:
            loop.stop()
        if src is not None:
            src.close()
        if loop is None or not loop.watchdog.tripped:
            dds.damp()          # never leave the arm stiff on a held setpoint
        if hasattr(dds, "close"):
            dds.close()         # sim only: stop the physics thread
        if loop is not None:
            s = loop.stats
            print(f"\n  {s['ticks']} ticks, {s['engages']} engages, "
                  f"{s['dropped']} tracking dropouts, {s['held']} held ticks")
            if loop.watchdog.tripped:
                print(f"  E-STOP: {loop.watchdog.reason}")
        if state["error"] is not None:
            raise state["error"]

    return 1 if (loop is not None and loop.watchdog.tripped) else 0


if __name__ == "__main__":
    sys.exit(main())
