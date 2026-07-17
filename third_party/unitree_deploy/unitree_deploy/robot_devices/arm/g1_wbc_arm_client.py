"""G1 WBC Arm ZMQ Client — runs on the LOCAL PC.

Full-featured drop-in replacement for ``G1_WBC_ArmController``.
All trajectory interpolation and IK live here; the robot side
(``g1_wbc_arm_server.py``) is a thin DDS bridge.

Data flow:
  ZMQ SUB  ← Robot PUB  (full LowState JSON, port 5556)
       → populates state buffer → read_arm_q / read_full_state / …

  write_arm() → trajectory interpolation → ZMQ PUSH → Robot PULL (port 5555)
       → Robot writes JSON directly to rt/whole_body_sdk

Public API is identical to ``G1_WBC_ArmController``:

    config = G1WbcZmqArmConfig(motors=g1_motors, robot_ip="192.168.123.1")
    ctrl   = G1WbcZmqArmClient(config)
    ctrl.connect()
    ctrl.write_arm(action)
    state = ctrl.read_full_state()
    ctrl.disconnect()
"""

import json
import threading
import time
from collections.abc import Callable

import numpy as np
import zmq

from unitree_deploy.robot_devices.arm.configs import G1WbcZmqArmConfig
from unitree_deploy.robot_devices.arm.g1_arm_ik import G1_29_ArmIK
from unitree_deploy.robot_devices.robots_devices_utils import (
    DataBuffer,
    RobotDeviceAlreadyConnectedError,
)
from unitree_deploy.utils.joint_trajcetory_inter import JointTrajectoryInterpolator
from unitree_deploy.utils.rich_logger import log_error, log_info, log_success, log_warning


class G1WbcZmqArmClient:
    """PC-side ZMQ arm controller, interface-identical to G1_WBC_ArmController.

    Internally replaces the DDS subscriber/publisher with ZMQ SUB/PUSH while
    keeping the same trajectory interpolation and IK logic that lives in the
    original controller.
    """

    ARM_DIM = 14
    BASE_CMD_DIM = 7
    BASE_CMD_NAMES = [
        "base_cmd.vx",
        "base_cmd.vy",
        "base_cmd.ang_z",
        "base_cmd.waist_yaw",
        "base_cmd.waist_pitch",
        "base_cmd.waist_roll",
        "base_cmd.body_height",
    ]
    DEFAULT_BASE_CMD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.74]

    def __init__(self, config: G1WbcZmqArmConfig):
        self.motors = config.motors
        self.control_dt = config.control_dt
        self.robot_ip = config.robot_ip
        self.cmd_port = config.cmd_port
        self.state_port = config.state_port
        self.init_pose = np.array(config.init_pose, dtype=np.float64)
        self.max_pos_speed = config.max_pos_speed
        self.state_timeout_s = config.state_timeout_s

        # Local IK solver (pinocchio on the PC)
        self.g1_arm_ik = G1_29_ArmIK(unit_test=False, visualization=False)

        # Thread-safe state buffer (populated from ZMQ SUB)
        self._state_buffer: DataBuffer = DataBuffer()
        self._command_lock = threading.Lock()

        # Command targets (written by write_arm, read by _publish_command)
        self.q_target = np.zeros(self.ARM_DIM)
        self.tauff_target = np.zeros(self.ARM_DIM)
        self.base_cmd_target = list(self.DEFAULT_BASE_CMD)
        self.time_target = time.monotonic()
        self.arm_cmd = "schedule_waypoint"

        self._can_publish = False
        self._stop_event = threading.Event()
        self.is_connected = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys()) + self.BASE_CMD_NAMES

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()] + ["base_cmd"] * self.BASE_CMD_DIM

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "G1WbcZmqArmClient is already connected. Do not call connect() twice."
            )
        try:
            self._ctx = zmq.Context()

            # SUB: receive robot state from server PUB
            self._state_sock = self._ctx.socket(zmq.SUB)
            self._state_sock.setsockopt(zmq.RCVTIMEO, 200)
            self._state_sock.setsockopt(zmq.RCVHWM, 2)
            self._state_sock.connect(f"tcp://{self.robot_ip}:{self.state_port}")
            self._state_sock.setsockopt_string(zmq.SUBSCRIBE, "")

            # PUSH: send interpolated arm commands to server PULL
            self._cmd_sock = self._ctx.socket(zmq.PUSH)
            self._cmd_sock.setsockopt(zmq.SNDHWM, 2)
            self._cmd_sock.connect(f"tcp://{self.robot_ip}:{self.cmd_port}")

            # Start background threads
            self._sub_thread = self._start_daemon("zmq_sub", self._state_recv_loop)
            self._pub_thread = self._start_daemon("zmq_pub", self._publish_command)

            # Wait for first state
            deadline = time.time() + self.state_timeout_s
            while self._state_buffer.get_data() is None:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for state from {self.robot_ip}:{self.state_port}. "
                        "Is g1_wbc_arm_server.py running on the robot?"
                    )
                time.sleep(0.05)
                log_warning("[G1WbcZmqArmClient] Waiting for state from robot…")

            self._can_publish = True
            self.is_connected = True
            log_success(
                f"[G1WbcZmqArmClient] Connected to {self.robot_ip}\n"
                f"  Current arm q: {self.read_current_arm_q().round(4)}"
            )
        except Exception as e:
            self.disconnect()
            log_error(f"❌ G1WbcZmqArmClient.connect error: {e}")
            raise

    def _start_daemon(self, name: str, target: Callable) -> threading.Thread:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        return t

    def disconnect(self):
        self.is_connected = False
        self._can_publish = False
        self._stop_event.set()
        try:
            self._cmd_sock.close(linger=0)
            self._state_sock.close(linger=0)
        except Exception:
            pass
        if hasattr(self, "_sub_thread"):
            self._sub_thread.join(timeout=0.5)
        if hasattr(self, "_pub_thread"):
            self._pub_thread.join(timeout=0.5)
        try:
            self._ctx.term()
        except Exception:
            pass
        log_success("[G1WbcZmqArmClient] Disconnected.")

    # ------------------------------------------------------------------
    # Background thread: receive state from robot ZMQ PUB
    # ------------------------------------------------------------------

    def _state_recv_loop(self):
        """Subscribe to robot state published by the server and store in buffer."""
        log_info("[G1WbcZmqArmClient] state_recv_loop started")
        _diag_frames = 0  # print first few raw frames for diagnosis
        while not self._stop_event.is_set():
            try:
                raw = self._state_sock.recv()  # raw bytes
            except zmq.Again:
                continue
            except Exception as e:
                if self._stop_event.is_set():
                    break  # normal shutdown — context was terminated
                log_error(f"[G1WbcZmqArmClient] state_recv_loop recv error: {e}")
                continue

            # Log first few frames to help diagnose encoding / port issues
            if _diag_frames < 3:
                _diag_frames += 1
                log_info(f"[G1WbcZmqArmClient] recv frame #{_diag_frames} "
                         f"len={len(raw)} hex_prefix={raw[:8].hex()}")

            # ZMQ PUB/SUB may deliver internal handshake frames (0xff…).
            # Decode with errors='ignore' and skip non-JSON frames silently.
            try:
                text = raw.decode("utf-8", errors="ignore")
                state = json.loads(text)
                parsed = {k: np.array(v, dtype=np.float64) for k, v in state.items()}
                self._state_buffer.set_data(parsed)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue  # skip malformed / non-JSON frames silently

    # ------------------------------------------------------------------
    # Background thread: interpolate & push commands to robot
    # (mirrors G1_WBC_ArmController._publish_command)
    # ------------------------------------------------------------------

    def _send_wbc_command(self, arm_q: np.ndarray, arm_tau: np.ndarray, base_cmd: list):
        """Serialize and PUSH one interpolated command frame to the server."""
        payload = json.dumps({
            "arm_q":    np.asarray(arm_q,   dtype=float).tolist(),
            "arm_tau":  np.asarray(arm_tau,  dtype=float).tolist(),
            "base_cmd": base_cmd,
        })
        try:
            self._cmd_sock.send_string(payload, zmq.NOBLOCK)
        except zmq.Again:
            log_warning("[G1WbcZmqArmClient] cmd PUSH HWM reached — frame dropped")

    def _drive_to_waypoint(
        self,
        target_pose: np.ndarray,
        t_insert_time: float,
        arm_tauff: np.ndarray,
        base_cmd: list,
    ):
        curr_time = time.monotonic() + self.control_dt
        t_insert = curr_time + t_insert_time
        self._pose_interp = self._pose_interp.drive_to_waypoint(
            pose=target_pose,
            time=t_insert,
            curr_time=curr_time,
            max_pos_speed=self.max_pos_speed,
        )
        while time.monotonic() < t_insert and not self._stop_event.is_set():
            t0 = time.perf_counter()
            clipped = self._pose_interp(time.monotonic())
            if self._can_publish:
                self._send_wbc_command(clipped, arm_tauff, base_cmd)
            time.sleep(max(0, self.control_dt - (time.perf_counter() - t0)))

    def _schedule_waypoint(
        self,
        arm_q_target: np.ndarray,
        arm_time_target: float,
        t_now: float,
        start_time: float,
        last_waypoint_time: float,
        arm_tauff: np.ndarray,
        base_cmd: list,
    ) -> float:
        target_time = time.monotonic() - time.perf_counter() + arm_time_target
        curr_time = t_now + self.control_dt
        target_time = max(target_time, curr_time + self.control_dt)

        self._pose_interp = self._pose_interp.schedule_waypoint(
            pose=arm_q_target,
            time=target_time,
            max_pos_speed=self.max_pos_speed,
            curr_time=curr_time,
            last_waypoint_time=last_waypoint_time,
        )
        last_waypoint_time = target_time

        clipped = self._pose_interp(t_now)
        if self._can_publish:
            self._send_wbc_command(clipped, arm_tauff, base_cmd)

        time.sleep(max(0, self.control_dt - (time.perf_counter() - start_time)))
        return last_waypoint_time

    def _publish_command(self):
        """Trajectory interpolation loop — runs at control_dt on the PC.

        Mirrors G1_WBC_ArmController._publish_command exactly, but pushes
        interpolated commands via ZMQ instead of writing to DDS directly.
        Waits until the state buffer is populated before initialising the
        trajectory interpolator.
        """
        # Wait until we have at least one valid state frame from the robot
        log_info("[G1WbcZmqArmClient] _publish_command: waiting for first state…")
        while not self._stop_event.is_set():
            if self._state_buffer.get_data() is not None:
                break
            time.sleep(0.05)

        if self._stop_event.is_set():
            return

        self._pose_interp = JointTrajectoryInterpolator(
            times=[time.monotonic()], joint_positions=[self.read_arm_q()]
        )

        arm_q_target = self.read_arm_q()
        arm_tauff_target = np.zeros(self.ARM_DIM)
        arm_time_target = time.monotonic()
        arm_cmd = "schedule_waypoint"
        base_cmd_target = list(self.DEFAULT_BASE_CMD)
        last_waypoint_time = time.monotonic()

        try:
            while not self._stop_event.is_set():
                start_time = time.perf_counter()
                t_now = time.monotonic()

                with self._command_lock:
                    arm_q_target = self.q_target
                    arm_tauff_target = self.tauff_target
                    arm_time_target = self.time_target
                    arm_cmd = self.arm_cmd
                    base_cmd_target = self.base_cmd_target

                if arm_cmd == "drive_to_waypoint":
                    self._drive_to_waypoint(
                        target_pose=arm_q_target,
                        t_insert_time=0.8,
                        arm_tauff=arm_tauff_target,
                        base_cmd=base_cmd_target,
                    )
                elif arm_cmd == "schedule_waypoint":
                    last_waypoint_time = self._schedule_waypoint(
                        arm_q_target=arm_q_target,
                        arm_time_target=arm_time_target,
                        t_now=t_now,
                        start_time=start_time,
                        last_waypoint_time=last_waypoint_time,
                        arm_tauff=arm_tauff_target,
                        base_cmd=base_cmd_target,
                    )
        except Exception as e:
            log_error(f"❌ G1WbcZmqArmClient._publish_command error: {e}")

    # ------------------------------------------------------------------
    # State readers (mirror G1_WBC_ArmController)
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        s = self._state_buffer.get_data()
        if s is None:
            raise RuntimeError("No state received from robot yet.")
        return s

    def read_arm_q(self) -> np.ndarray:
        """Return 14-DoF arm joint positions [left 0-6, right 7-13]."""
        return self._get_state()["arm_q"].copy()

    def read_current_arm_q(self) -> np.ndarray:
        """Return full obs vector: [arm_q(14), waist_q(3), leg_q(12)]."""
        s = self._get_state()
        return np.concatenate([s["arm_q"], s["waist_q"], s["leg_q"]])

    def read_current_arm_dq(self) -> np.ndarray:
        """Return 14-DoF arm joint velocities."""
        return self._get_state()["arm_dq"].copy()

    def read_full_state(self) -> dict:
        """Return comprehensive state dict (all keys are numpy arrays)."""
        return {k: v.copy() for k, v in self._get_state().items()}

    # ------------------------------------------------------------------
    # Command writer (mirror G1_WBC_ArmController)
    # ------------------------------------------------------------------

    def write_arm(
        self,
        action: np.ndarray,
        tauff_target: np.ndarray | None = None,
        time_target: float | None = None,
        cmd_target: str | None = None,
    ):
        """Queue an arm command for the interpolation thread to dispatch."""
        action = np.asarray(action, dtype=float)
        q_target = action[: self.ARM_DIM]
        base_cmd = (
            action[self.ARM_DIM : self.ARM_DIM + self.BASE_CMD_DIM].tolist()
            if len(action) >= self.ARM_DIM + self.BASE_CMD_DIM
            else list(self.DEFAULT_BASE_CMD)
        )
        with self._command_lock:
            self.q_target = q_target
            self.tauff_target = (
                tauff_target if tauff_target is not None else self.g1_arm_ik.solve_tau(q_target)
            )
            if self.tauff_target is None:
                self.tauff_target = np.zeros(self.ARM_DIM)
            self.time_target = time_target if time_target is not None else time.monotonic()
            self.arm_cmd = cmd_target if cmd_target is not None else "schedule_waypoint"
            self.base_cmd_target = base_cmd

    def stop_publish(self):
        """Pause command publishing."""
        self._can_publish = False

    def resume_publish(self):
        """Resume command publishing."""
        self._can_publish = True

    # ------------------------------------------------------------------
    # IK / FK helpers
    # ------------------------------------------------------------------

    def arm_ik(
        self,
        l_ee_target: np.ndarray,
        r_ee_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.g1_arm_ik.solve_ik(
            l_ee_target,
            r_ee_target,
            self.read_arm_q(),
            self.read_current_arm_dq(),
        )

    def arm_fk(self, q: np.ndarray | None = None):
        if q is None:
            q = self.read_arm_q()
        return self.g1_arm_ik.solve_fk_matrix(q)

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------

    def go_start(self):
        base_cmd = list(self.DEFAULT_BASE_CMD)
        tauff = self.g1_arm_ik.solve_tau(self.init_pose) or np.zeros(self.ARM_DIM)
        with self._command_lock:
            self.q_target = self.init_pose.copy()
            self.tauff_target = tauff
            self.base_cmd_target = base_cmd
        log_success("[G1WbcZmqArmClient] Go Start OK!")

    def go_home(self):
        self.go_start()
        log_success("[G1WbcZmqArmClient] Go Home OK!")
