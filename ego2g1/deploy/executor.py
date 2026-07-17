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

from . import actions as _actions

logger = logging.getLogger(__name__)


class UnitreeExecutor:
    """ego2g1-facing wrapper over unitree_deploy's G1 arm + Brainco controller."""

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
        self._last_q_change_t = time.monotonic()

    # --- lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        # Pre-init the DDS factory on the requested interface BEFORE the
        # vendor's own ChannelFactoryInitialize(0): the sdk's init is a
        # singleton, so ours wins. Best-effort — zh never sets one either.
        if self._iface is not None:
            try:
                from unitree_sdk2py.core.channel import ChannelFactoryInitialize
                ChannelFactoryInitialize(0, self._iface)
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
        q = np.asarray(obs["observation.state"].numpy()[: _actions.ARM_DOF],
                       dtype=np.float64)
        with self._lock:
            if self._last_q is None or not np.array_equal(q, self._last_q):
                self._last_q = q.copy()
                self._last_q_change_t = time.monotonic()
        return q

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
        row[_actions.ARM_DOF:] = np.clip(row[_actions.ARM_DOF:], 0.0, 1.0)
        if t_target is None:
            t_target = time.monotonic() + 2 * self.control_dt
        self._robot.send_action(self._torch.from_numpy(row.astype(np.float32)),
                                t_target)
        self._first_send = False

    def hold(self) -> None:
        """Re-send the measured pose (a safe no-op waypoint)."""
        q = self.arm_q()
        row = np.zeros(_actions.ROBOT_DIM)
        row[_actions.ARM] = q
        self.send(row)

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

    def damp(self) -> None:
        self._estopped = True
        self.damped = True

    @property
    def estopped(self) -> bool:
        return self._estopped
