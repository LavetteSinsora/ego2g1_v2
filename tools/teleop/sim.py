"""`--sim`: teleoperate a DYNAMIC MuJoCo G1 instead of the real robot.

Rung 4 of the bring-up ladder. Everything above the robot is the REAL teleop path --
same VuerSource, same TeleopRetargeter, same mink IK, same TrajectoryBuffer, same
watchdog -- and only `G1DDS` is swapped for `SimDDS`. So what you feel here (including
the One-Euro finger smoothing) is what the robot will do, and a bug found here is a
real bug rather than a simulator artifact.

DYNAMIC, not kinematic. `G1HandsBackend` writes qpos and calls mj_forward, which can
never grasp anything: the fingers pass through the object. Here the composite model is
INTEGRATED -- gravity, contacts, and the position servos the model already ships
(`<position kp=500 dampratio=1>` on the G1, the vendor's servos on the Revo2). Two
consequences worth knowing:

  * `arm_q()` returns the MEASURED qpos, so the arm genuinely lags its command and the
    watchdog sees real tracking error, exactly as on the robot. A perfect-tracking fake
    (tests/test_loop.py FakeDDS) cannot show you that.
  * the Revo2's actuator force limits are the vendor's and they are WEAK (thumb
    metacarpal +-0.5 Nm; hand/screen.py already notes those servos cannot sustain the
    URDF-rated speeds in sim). Grip strength here is a lower bound on the real hand.

THREADING. MuJoCo's data is not thread-safe, and the teleop loop pushes commands from
its 500 Hz / 200 Hz emitter threads. So `send_arm`/`send_hands` only STORE the latest
command under a lock; every mj_step and every qpos read happens on the single physics
thread. The viewer syncs under the same lock. Nothing else touches `data`.

macOS: the interactive viewer must run under `mjpython` (it owns the main-thread GUI
loop); `mjpython` ships in the mujoco wheel and is already in .venv/bin/. On Linux
plain python is fine.

This module is the one part of the package that is NOT self-contained: it imports
`data_extraction.sim.g1_hands` to build the composite model, so `--sim` only works from
the repo, never from the robot PC (which does not need it -- it has a robot).
"""

import threading
import time

import numpy as np

SIDES = ("left", "right")

# Workspace, measured against the model rather than guessed. At the nominal pose the
# shoulders sit at [0, +-0.10, 1.085] with max_reach 0.375 m, the flanges at
# [0.20, +-0.149, 0.888], and the lowest hand geom at z=0.816.
#
#   bottle x = 0.28  -> 0.366 m from the shoulder. x=0.32 measures 0.397 and is OUT OF
#                       REACH; the margin here is small because it genuinely is small,
#                       which is the same fact the placement fit `S` solves offline and
#                       the engage anchor solves at teleop time.
#   table top 0.78   -> 3.6 cm under the resting hands, and the near edge is pushed out
#                       to x=0.22 so the arms hang in FRONT of it, not over it. Get this
#                       wrong and the hand rests in the table: the servo fights it and
#                       the wrist sits at a permanent tracking error.
TABLE_TOP_Z = 0.78
TABLE_HALF = (0.20, 0.40, TABLE_TOP_Z / 2.0)
TABLE_POS = (0.42, 0.0, TABLE_TOP_Z / 2.0)      # spans x 0.22 .. 0.62
BOTTLE_R, BOTTLE_HALF_H = 0.033, 0.075
BOTTLE_POS = (0.28, -0.15, TABLE_TOP_Z + BOTTLE_HALF_H)   # right hand, in reach
BOTTLE_MASS = 0.35


def build_scene(mount=None):
    """The composite G1+Revo2 model with a table and a graspable bottle, compiled for
    DYNAMICS. Reuses data_extraction's builder for the robot itself (one definition of
    the hand mount) and only adds the scene."""
    import warnings

    import mujoco

    from data_extraction.sim.g1_hands import build_g1_hands_spec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        spec = build_g1_hands_spec(mount)

        # The scene's floor ships as contype=0/conaffinity=0 -- a visual backdrop, since
        # a fixed-base G1 never touches it. Here things get dropped, and a bottle knocked
        # off the table would otherwise fall forever (measured: z=-386 m).
        for g in spec.geoms:
            if g.name == "wr_floor":
                g.contype, g.conaffinity = 1, 1

        world = spec.worldbody
        table = world.add_body(name="table", pos=list(TABLE_POS))
        table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=list(TABLE_HALF), rgba=[0.55, 0.45, 0.35, 1.0],
                       friction=[1.0, 0.005, 0.0001])

        bottle = world.add_body(name="bottle", pos=list(BOTTLE_POS))
        bottle.add_freejoint()
        # condim=4 (torsional friction) and a high slide friction are what stop a
        # cylinder spinning out of the fingers; without them a "grasp" slips
        # immediately even when the contact forces look right.
        bottle.add_geom(name="bottle_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[BOTTLE_R, BOTTLE_HALF_H, 0.0], mass=BOTTLE_MASS,
                        rgba=[0.15, 0.55, 0.85, 1.0], condim=4,
                        friction=[1.6, 0.05, 0.002])
        return spec.compile()


class SimDDS:
    """A physics-stepped stand-in for `G1DDS`.

    Implements exactly the six methods the live path calls -- connect, arm_q,
    lowstate_age, send_arm, send_hands, damp -- and nothing else, because that is the
    whole seam between the teleop loop and the robot (see loop.py).
    """

    def __init__(self, *, mount=None, realtime: float = 1.0):
        import mujoco

        self._mj = mujoco
        self.model = build_scene(mount)
        self.data = mujoco.MjData(self.model)
        self.realtime = realtime

        self.lock = threading.RLock()      # guards `data`; shared with the viewer
        self._damped = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.steps = 0

        # actuator ids, by name, on the COMPOSITE model. Arm actuators keep the G1's
        # names; the Revo2's are prefixed lh_/rh_ by the attach.
        from ._vendor.de.hand.constants import ACTUATOR_NAME, MOTOR_ORDER
        from ._vendor.eg.common import layout

        def act(name):
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if i < 0:
                raise RuntimeError(f"no actuator {name!r} on the composite model")
            return i

        # left 7 then right 7 -- the flat order layout.ARM_JOINTS / DualArmIK / dds use
        arm_names = [n for h in layout.HANDS for n in layout.ARM_JOINTS[h]]
        self._arm_act = np.array([act(n) for n in arm_names])
        pre = {"left": "lh_", "right": "rh_"}
        self._hand_act = {s: np.array([act(pre[s] + ACTUATOR_NAME[s][m])
                                       for m in MOTOR_ORDER]) for s in SIDES}
        self._hand_ctrl_max = {
            s: np.array([self.model.actuator(i).ctrlrange[1] for i in self._hand_act[s]])
            for s in SIDES}
        self._arm_qadr = np.array([self.model.joint(n).qposadr[0] for n in arm_names])

        # Hold everything where it starts. The fixed-base model still carries leg and
        # waist joints; they are position servos, so ctrl=qpos pins them instead of
        # letting them sag under gravity (the G1-D has no lower body anyway).
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = 0.0
        for i in range(self.model.nu):
            j = self.model.actuator(i).trnid[0]
            self.data.ctrl[i] = self.data.qpos[self.model.joint(j).qposadr[0]]

    # ------------------------------------------------------------- DDS interface
    def connect(self, *, timeout: float = 5.0) -> None:
        self.start()

    def arm_q(self) -> np.ndarray:
        with self.lock:
            return self.data.qpos[self._arm_qadr].copy()

    def lowstate_age(self) -> float:
        # the sim never goes stale while it is stepping; if it stops, say so loudly so
        # the watchdog trips exactly as it would on a dead robot
        return 0.001 if (self._thread and self._thread.is_alive()) else 9.9

    def send_arm(self, arm_q14, *, waist=None) -> None:
        if self._damped:
            return
        with self.lock:
            self.data.ctrl[self._arm_act] = np.asarray(arm_q14, dtype=np.float64)

    def send_hands(self, cmds: dict) -> None:
        if self._damped:
            return
        with self.lock:
            for s, v in cmds.items():
                c = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
                self.data.ctrl[self._hand_act[s]] = c * self._hand_ctrl_max[s]

    def damp(self) -> None:
        """Kill the servos: hold the arm where it currently IS and stop accepting
        commands. The sim keeps stepping so you can see the result."""
        with self.lock:
            self._damped = True
            self.data.ctrl[self._arm_act] = self.data.qpos[self._arm_qadr]

    # ------------------------------------------------------------------ physics
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="sim-physics",
                                        daemon=True)
        self._thread.start()

    def _run(self) -> None:
        dt = self.model.opt.timestep
        t_next = time.perf_counter()
        while not self._stop.is_set():
            with self.lock:
                self._mj.mj_step(self.model, self.data)
                self.steps += 1
            if self.realtime <= 0.0:
                time.sleep(0)          # free-run (headless/tests): just yield the GIL
                continue
            t_next += dt / self.realtime
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                t_next = time.perf_counter()   # fell behind; do not spiral

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ---------------------------------------------------------------- diagnostics
    def bottle_pose(self) -> np.ndarray:
        with self.lock:
            return self.data.body("bottle").xpos.copy()

    def arm_tracking_error(self) -> float:
        """max |commanded - measured| over the arm joints (rad)."""
        with self.lock:
            return float(np.abs(self.data.ctrl[self._arm_act]
                                - self.data.qpos[self._arm_qadr]).max())


def run_viewer(dds: SimDDS, state: dict, on_ready=None) -> None:
    """Passive viewer on the MAIN thread (macOS requires that, hence mjpython), physics
    on SimDDS's thread, teleop on the loop's threads.

    Opens IMMEDIATELY and does not wait for the bring-up: `state["loop"]` is filled in by
    the bring-up thread once the hands are tracked and calibrated, and until then the
    window just shows the robot standing there. That ordering is the point -- you need to
    see the robot WHILE you hold the calibration pose, not after.

    Keys arrive here rather than through `RawKeys`: once the viewer window has focus the
    terminal no longer sees keystrokes, so the same e/space/q contract is served from
    here. Sync holds `dds.lock`, the same mutex the physics thread takes around mj_step --
    MuJoCo's data is not thread-safe and the viewer reads all of it.

    `on_ready` is called AFTER the window exists, and is what starts the physics and the
    bring-up. That ordering is load-bearing, not tidiness: `launch_passive` builds its
    scene by reading `model`/`data` on its own, WITHOUT taking our lock, so if physics is
    already stepping when it is called the two race and the process SEGFAULTS on the spot
    (measured). Nothing may touch `data` until the viewer is constructed.
    """
    import mujoco.viewer

    quit_ = {"v": False}

    def key_cb(keycode: int) -> None:
        loop = state.get("loop")
        if loop is None:
            return                               # still bringing up; nothing to drive yet
        k = chr(keycode).lower() if 0 < keycode < 0x110000 else ""
        if k == "e":
            loop.engage()
            print("  ENGAGED")
        elif keycode == 32:                      # space
            loop.disengage()
            print("  disengaged")
        elif k == "q":
            loop.estop("operator pressed q")
            quit_["v"] = True

    print("\n  Viewer keys:  e = ENGAGE / re-anchor   space = DISENGAGE   q = DAMP and quit")
    print("  (click the viewer window first so it has keyboard focus)\n")
    with mujoco.viewer.launch_passive(dds.model, dds.data,
                                      key_callback=key_cb) as v:
        if on_ready is not None:
            on_ready()             # physics + bring-up start only now (see docstring)
        while v.is_running() and not quit_["v"]:
            loop = state.get("loop")
            if (loop is not None and loop.watchdog.tripped) or state.get("error"):
                break
            with dds.lock:
                v.sync()
            time.sleep(1.0 / 60.0)
