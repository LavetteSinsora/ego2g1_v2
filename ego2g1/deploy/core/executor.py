"""The executor: unitree_deploy's proven 500 Hz arm controller, wrapped once.

Adapted from the old deploy's unitree_backend.py (third_party/openpi/ego2g1/
deploy/unitree_backend.py — read it for the history of WHY this path and not a
raw lowcmd publisher).

WHY unitree_deploy. Its G1 arm controller (third_party/unitree_deploy,
robot_devices/arm/g1_arm.py) is the executor jitter_root_cause.md proved
innocent: a 500 Hz `_ctrl_motor_state` thread interpolating scheduled waypoints
(scipy interp1d with a max_pos_speed slew cap), soft `drive_to_waypoint` ramps
for the first move, and — the part our raw path was missing — gravity-comp tau
feedforward (`g1_arm_ik.solve_tau` = pin.rnea) on every command. Without tau,
kp=80 alone holds the arm against gravity and the kd term brakes every motion:
the measured stick-slip. zh's stack tracks smoothly because it drives through
here. We do the same and change nothing inside it.

What this wrapper adds:
  * ego2g1's row layout (actions.ROBOT_DIM = 14 arm + 6+6 Brainco, exactly
    unitree_deploy's `unitree_g1_brainco` order — no reindexing),
  * future-stamped scheduling: each send targets now + 2*control_dt, copied
    from unitree_deploy's own UnitreeEnv.step (real_unitree_env.py) — the
    interpolator reaches the waypoint one tick after the next and NEVER
    extrapolates,
  * `damp()` — the e-stop the vendor path lacks. G1-D has no balance
    controller and no safe "release": the firmware executes the last message
    forever, so the only real stop is a published damping command (kp=0,
    small kd). It stops the vendor's publish thread first so nothing
    re-stiffens the arm after us. LATCHED: send() becomes a no-op afterwards.
  * `state_age()` for the watchdog, tracked as time-since-the-measured-arm-
    last-CHANGED — a frozen lowstate and a dead link look identical from here,
    and both mean "stop".

MockExecutor at the bottom implements the same surface with no DDS, no torch,
no robot — every test and dry-run drives through it.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ...core import layout
from .. import actions as _actions

logger = logging.getLogger(__name__)


class UnitreeExecutor:
    """ego2g1-facing wrapper over unitree_deploy's G1 arm + Brainco controller."""

    # End-effector wire layouts, keyed by robot_type. The deploy layer's
    # canonical row stays (26,) = 14 arm + 6+6 for EVERY robot (actions.py's
    # ROBOT_DIM; strategies, clamp, session, safety and the recorder are all
    # written against it and stay mode-blind). Only the translation to the
    # vendor's own motor vector differs, and it differs HERE, in one place.
    #
    #   brainco  6 motors/hand, absolute [0, 1], 0=open 1=closed.
    #   dex1     ONE motor/hand ("kLeftGripper"/"kRightGripper", a
    #            z1_gripper-joint — unitree_deploy's robot_configs.py
    #            dex1_default_factory). Its command is the gear ROTATION IN
    #            RADIANS, which is the same quantity `UmiTrainConfig` trains on
    #            and the model emits natively, so it is passed through
    #            unchanged. Clipping it to [0, 1] like a Brainco fraction would
    #            crush every command to 1.0 and the gripper would never open —
    #            which is exactly why the limits live in this table instead of
    #            being hard-coded in send().
    _EE_LAYOUTS = {
        "unitree_g1_brainco": {"per_hand": 6, "limits": (0.0, 1.0)},
        # Range measured on red_block_on_yellow_block_umi: 1.20 (fully closed,
        # a hard floor in the data) .. 5.40 (fully open). Widened slightly so a
        # legitimate command at either extreme is not silently trimmed; this is
        # a corruption guard, not a calibration.
        "unitree_g1_dex1": {"per_hand": 1, "limits": (0.0, 6.0)},
    }

    def __init__(self, *, fps: int = 30, network_interface: str | None = None,
                 robot_type: str = "unitree_g1_brainco",
                 max_pos_speed: float | None = None):
        try:
            import torch  # noqa: F401  (unitree_deploy's send_action wants tensors)
            from unitree_deploy.robot.robot_utils import (
                make_robot_config, make_robot_from_config,
            )
        except ImportError as exc:
            raise RuntimeError(
                "executor needs the vendored unitree_deploy package. Install it:\n"
                "  VIRTUAL_ENV=$PWD/../.venv uv pip install -e third_party/unitree_deploy\n"
                "(plus unitree_sdk2py — see docs/deps-deploy.md)"
            ) from exc

        import torch
        self._torch = torch
        self.fps = int(fps)
        self.control_dt = 1.0 / self.fps
        self._iface = network_interface

        try:
            self._ee = self._EE_LAYOUTS[robot_type]
        except KeyError:
            raise ValueError(
                f"robot_type={robot_type!r} has no end-effector wire layout here; "
                f"known: {sorted(self._EE_LAYOUTS)}. Add one rather than letting "
                "send() guess how the (26,) row maps onto the vendor's motors."
            ) from None
        self.robot_type = robot_type
        cfg = make_robot_config(robot_type)
        # We own the camera path (deploy/camera.py); drop unitree's image
        # client so there is no second consumer on the head image server.
        cfg.cameras = {}
        if max_pos_speed is not None:
            for arm_cfg in cfg.arm.values():
                if hasattr(arm_cfg, "max_pos_speed"):
                    arm_cfg.max_pos_speed = float(max_pos_speed)
        self._robot = make_robot_from_config(cfg)
        self._connected = False
        self._estopped = False
        self._first_send = True
        self._lock = threading.Lock()
        self._last_q: np.ndarray | None = None
        self._last_ee: dict | None = None   # filled by arm_q(), read by ee_q()
        self._last_q_change_t = time.monotonic()
        self._last_sent: tuple[float, np.ndarray] | None = None  # telemetry only

    # --- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        # macOS has no native unitree CRC -> ~1 ms/LowCmd pure-Python inside
        # the 500 Hz loop (39-50% of the tick budget, measured). Bit-identical
        # zlib replacement, 187x faster; on a working Linux install this is a
        # no-op (native branch still used) -- but if that install is missing
        # its compiled crc_amd64.so/crc_aarch64.so, this is what keeps
        # CRC() from crashing at all (see fast_crc.py's module docstring).
        from . import fast_crc
        fast_crc.install()
        # Pre-init the DDS factory on the requested interface BEFORE the
        # vendor's own ChannelFactoryInitialize(0): the sdk's init is a
        # singleton, so ours wins. Best-effort — zh never sets one either.
        if self._iface is not None:
            try:
                from .._util import dds_init
                dds_init(self._iface)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "could not pre-init DDS on %s (%s); falling back to "
                    "unitree_deploy's ChannelFactoryInitialize(0)", self._iface, exc)

        self._robot.connect()
        self._connected = True

    def close(self) -> None:
        """Normal release: unitree_deploy drives to its init pose, then lets go.
        After an e-stop, do NOT let it re-energise the arm — damp was final."""
        if self._connected and not self._estopped:
            self._robot.disconnect()
        self._connected = False

    # --- read ---------------------------------------------------------------

    def arm_q(self) -> np.ndarray:
        """(14,) measured arm joints, DualArmIK order (left 7, right 7)."""
        obs = self._robot.capture_observation()
        raw = np.asarray(obs["observation.state"].numpy(), dtype=np.float64)
        # Cache the end-effector half from the SAME capture: the runner already
        # calls arm_q() every tick, so ee_q() becomes free instead of doubling
        # the number of round trips to the robot.
        per_hand = self._ee["per_hand"]
        self._last_ee = {h: float(raw[_actions.ARM_DOF + i * per_hand])
                         for i, h in enumerate(layout.HANDS)}
        q = raw[: _actions.ARM_DOF]
        with self._lock:
            if self._last_q is None or not np.array_equal(q, self._last_q):
                self._last_q = q.copy()
                self._last_q_change_t = time.monotonic()
        return q

    def ee_q(self) -> dict:
        """{hand: MEASURED end-effector value}. Dex1: the gripper's gear
        rotation in radians. Cached from the last `arm_q()` capture.

        NOT what the policy is fed. The training gripper signal saturates at a
        hard 1.20 whenever the gripper is closed — measured across the whole
        dataset, every one of the 117 episodes plateaus at bit-identical
        1.2000, and 38% of all frames sit there. A real jaw cannot close to its
        limit around a solid block, so that column is a commanded/clamped
        value, not a physical position. Feeding the true encoder at deploy
        would therefore read ~the block's width while gripping — a value the
        policy has never seen in that phase.

        So this is used for two things only: seeding the very first history
        sample of a rollout, and the command-vs-measured deviation, which is
        the honest "did the grasp actually take" signal the policy's own
        gripper channel cannot provide.
        """
        if self._last_ee is None:
            self.arm_q()
        return dict(self._last_ee)

    def state_age(self) -> float:
        """Seconds since the measured arm last changed. A live 500 Hz lowstate
        always changes (encoder noise); frozen == stale == the watchdog's
        business. Call arm_q() at loop rate for this to mean anything."""
        with self._lock:
            return time.monotonic() - self._last_q_change_t

    # --- write --------------------------------------------------------------

    def send(self, row26, t_target: float | None = None) -> None:
        """Schedule one (26,) joint row as a waypoint.

        The target is stamped in the FUTURE — default now + 2*control_dt,
        matching unitree_deploy's UnitreeEnv.step: t_cycle_end lands one
        control period out and the command target one more period beyond it,
        so the 500 Hz interpolator always interpolates, never extrapolates,
        and chunk boundaries stay velocity-consistent. Copy this exactly; a
        target stamped 'now' makes every waypoint a step function.
        """
        if not self._connected:
            raise RuntimeError("send before connect()")
        if self._estopped:
            return  # latched
        row = np.asarray(row26, dtype=np.float64).reshape(-1)
        if row.shape != (_actions.ROBOT_DIM,):
            raise ValueError(f"expected ({_actions.ROBOT_DIM},), got {row.shape}")
        row = row.copy()
        lo, hi = self._ee["limits"]
        row[_actions.ARM_DOF:] = np.clip(row[_actions.ARM_DOF:], lo, hi)
        if t_target is None:
            t_target = time.monotonic() + 2 * self.control_dt
        self._robot.send_action(self._torch.from_numpy(self._wire_row(row)),
                                t_target)
        # `row` is our private copy and never mutated again: a plain reference
        # swap is enough for the dashboard's pull-side read. No lock, no copy.
        self._last_sent = (float(t_target), row)
        self._first_send = False

    def _wire_row(self, row) -> np.ndarray:
        """Canonical (26,) row -> the vendor's motor vector for this robot.

        Brainco is the identity (its 6+6 block IS the wire layout, so the
        existing path is bit-identical). Dex1 takes ONE motor per hand, and by
        convention that value lives in slot 0 of each hand's block; the other
        five slots are unused padding kept so the deploy layer's row width is
        the same for every robot.
        """
        per_hand = self._ee["per_hand"]
        if per_hand == _actions.HAND_DOF:
            return row.astype(np.float32)
        out = np.empty(_actions.ARM_DOF + per_hand * len(layout.HANDS),
                       dtype=np.float32)
        out[:_actions.ARM_DOF] = row[_actions.ARM]
        for i, h in enumerate(layout.HANDS):
            block = row[_actions.HAND[h]]
            base = _actions.ARM_DOF + i * per_hand
            out[base:base + per_hand] = block[:per_hand]
        return out

    def hold(self) -> None:
        """Re-send the measured pose (a safe no-op waypoint)."""
        q = self.arm_q()
        row = np.zeros(_actions.ROBOT_DIM)
        row[_actions.ARM] = q
        self.send(row)

    # --- telemetry (dashboard pull side; reads only, existing lock only) -----

    def telemetry(self) -> dict:
        last = self._last_sent            # reference swap; safe plain read
        with self._lock:
            q = None if self._last_q is None else self._last_q.copy()
            age = time.monotonic() - self._last_q_change_t
        return {"last_row": None if last is None else last[1].tolist(),
                "last_t_target": None if last is None else last[0],
                "arm_q": None if q is None else q.tolist(),
                "state_age": age,
                "estopped": self._estopped}

    # --- e-stop ---------------------------------------------------------------

    def damp(self) -> None:
        """THE E-STOP. Stop the vendor's publish thread, then publish zero
        stiffness / pure damping on every joint. Latches send() off.

        Publishing (not merely stopping) is essential: the firmware holds the
        last command forever, so silence leaves the robot stiff at its last
        target. kd stays small — the arm goes limp but does not freefall.

        Reaches into G1_29_ArmController internals (stop_event, msg, crc,
        lowcmd_publisher) — the vendor exposes no stop that is not "drive to
        init pose", which is exactly the motion an e-stop must not make.
        Pinned to the vendored copy; if the internals move, this raises loudly
        rather than pretending to stop.
        """
        self._estopped = True   # latch first: send() is off from here on
        if not self._connected:
            return
        try:
            arms = getattr(self._robot, "arm", {})
            for ctrl in arms.values():
                ctrl.stop_event.set()           # halts _ctrl_motor_state + subscriber
            time.sleep(0.01)
            for ctrl in arms.values():
                msg = ctrl.msg                  # carries mode_machine, gains, q
                state = ctrl.lowstate_buffer.get_data()
                for i, cmd in enumerate(msg.motor_cmd):
                    if state is not None and i < len(state.motor_state) \
                            and state.motor_state[i].q is not None:
                        cmd.q = float(state.motor_state[i].q)
                    cmd.dq = 0.0
                    cmd.tau = 0.0
                    cmd.kp = 0.0
                    cmd.kd = 2.0
                msg.crc = ctrl.crc.Crc(msg)
                for _ in range(5):              # a few times, in case one drops
                    ctrl.lowcmd_publisher.Write(msg)
                time.sleep(0.002)
            logger.error("E-STOP: damp published, executor latched off")
        except AttributeError as exc:
            raise RuntimeError(
                "damp(): unitree_deploy internals moved (expected "
                "G1_29_ArmController.{stop_event,msg,crc,lowcmd_publisher}). "
                "The e-stop is NOT armed against this vendored copy — fix this "
                "before any hardware run."
            ) from exc

    @property
    def estopped(self) -> bool:
        return self._estopped


class MockExecutor:
    """Same surface, no hardware. Tracks perfectly and instantly; records every
    send for assertions. `clock` is injectable so tests need no real time."""

    def __init__(self, *, fps: int = 30, initial_q=None, clock=time.monotonic):
        self.fps = int(fps)
        self.control_dt = 1.0 / self.fps
        self._clock = clock
        self._q = np.zeros(_actions.ARM_DOF) if initial_q is None \
            else np.asarray(initial_q, dtype=np.float64).copy()
        self.sent: list[tuple[float, np.ndarray]] = []   # (t_target, row26)
        self.damped = False
        self._connected = False
        self._estopped = False

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def arm_q(self) -> np.ndarray:
        return self._q.copy()

    def ee_q(self) -> dict:
        """Mirror of UnitreeExecutor.ee_q: the last-sent end-effector value per
        hand (slot 0 of each hand block), or 0.0 before the first send."""
        if not self.sent:
            return {h: 0.0 for h in layout.HANDS}
        row = self.sent[-1][1]
        return {h: float(row[_actions.HAND[h]][0]) for h in layout.HANDS}

    def state_age(self) -> float:
        return 0.0

    def send(self, row26, t_target: float | None = None) -> None:
        if not self._connected:
            raise RuntimeError("send before connect()")
        if self._estopped:
            return
        row = np.asarray(row26, dtype=np.float64).reshape(-1)
        if row.shape != (_actions.ROBOT_DIM,):
            raise ValueError(f"expected ({_actions.ROBOT_DIM},), got {row.shape}")
        if t_target is None:
            t_target = self._clock() + 2 * self.control_dt
        self.sent.append((t_target, row.copy()))
        self._q = row[_actions.ARM].copy()     # perfect, instant tracking

    def hold(self) -> None:
        row = np.zeros(_actions.ROBOT_DIM)
        row[_actions.ARM] = self._q
        self.send(row)

    def telemetry(self) -> dict:
        last = self.sent[-1] if self.sent else None
        return {"last_row": None if last is None else last[1].tolist(),
                "last_t_target": None if last is None else last[0],
                "arm_q": self._q.tolist(),
                "state_age": self.state_age(),
                "estopped": self._estopped}

    def damp(self) -> None:
        self._estopped = True
        self.damped = True

    @property
    def estopped(self) -> bool:
        return self._estopped
