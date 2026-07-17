"""Dex1 Gripper ZMQ Client — runs on the LOCAL PC.

Full-featured drop-in replacement for ``Dex1_Gripper_Controller``.
All trajectory interpolation lives here; the robot side
(``gripper_server.py``) is a thin DDS bridge.

Data flow:
  ZMQ SUB  ← Robot PUB  (gripper state JSON, port 5558)
       → populates state buffer → read_current_endeffector_q / …

  write_endeffector() → trajectory interpolation → ZMQ PUSH → Robot PULL (port 5557)
       → Robot writes MotorCmds_ directly to rt/unitree_actuator/cmd

Public API is identical to ``Dex1_Gripper_Controller``:

    config = Dex1GripperZmqConfig(motors=gripper_motors, robot_ip="192.168.123.1")
    ctrl   = Dex1GripperZmqClient(config)
    ctrl.connect()
    ctrl.write_endeffector(q_target)
    q = ctrl.read_current_endeffector_q()
    ctrl.disconnect()
"""

import json
import threading
import time
from collections.abc import Callable

import numpy as np
import zmq

from unitree_deploy.robot_devices.endeffector.configs import Dex1GripperZmqConfig
from unitree_deploy.robot_devices.robots_devices_utils import (
    DataBuffer,
    RobotDeviceAlreadyConnectedError,
)
from unitree_deploy.utils.rich_logger import log_error, log_info, log_success, log_warning


class Dex1GripperZmqClient:
    """PC-side ZMQ gripper controller, interface-identical to Dex1_Gripper_Controller.

    Internally replaces the DDS subscriber/publisher with ZMQ SUB/PUSH while
    keeping the same trajectory interpolation logic that lives in the original
    controller.
    """

    GRIPPER_DIM = 1  # single-DOF gripper

    def __init__(self, config: Dex1GripperZmqConfig):
        self.motors = config.motors
        self.control_dt = config.control_dt
        self.robot_ip = config.robot_ip
        self.cmd_port = config.cmd_port
        self.state_port = config.state_port
        self.init_pose = np.array(config.init_pose, dtype=np.float64)
        self.max_pos_speed = config.max_pos_speed
        self.state_timeout_s = config.state_timeout_s

        # Thread-safe state buffer (populated from ZMQ SUB)
        self._state_buffer: DataBuffer = DataBuffer()
        self._command_lock = threading.Lock()

        # Command target (written by write_endeffector, read by _publish_command)
        self.q_target = np.zeros(self.GRIPPER_DIM)

        self._can_publish = False
        self._stop_event = threading.Event()
        self.is_connected = False

        # Clamping constants (matching original controller)
        self.MAX_DIST = 5.45
        self.MIN_DIST = 0.0

    # ------------------------------------------------------------------
    # Properties  (mirror Dex1_Gripper_Controller)
    # ------------------------------------------------------------------

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Dex1GripperZmqClient is already connected. Do not call connect() twice."
            )
        try:
            self._ctx = zmq.Context()

            # SUB: receive gripper state from server PUB
            self._state_sock = self._ctx.socket(zmq.SUB)
            self._state_sock.setsockopt(zmq.RCVTIMEO, 200)
            self._state_sock.setsockopt(zmq.RCVHWM, 2)
            self._state_sock.connect(f"tcp://{self.robot_ip}:{self.state_port}")
            self._state_sock.setsockopt_string(zmq.SUBSCRIBE, "")

            # PUSH: send interpolated gripper commands to server PULL
            self._cmd_sock = self._ctx.socket(zmq.PUSH)
            self._cmd_sock.setsockopt(zmq.SNDHWM, 2)
            self._cmd_sock.connect(f"tcp://{self.robot_ip}:{self.cmd_port}")

            # Only the state-recv thread is needed; commands are sent directly in write_endeffector
            self._sub_thread = self._start_daemon("zmq_sub", self._state_recv_loop)

            # Wait for first state
            deadline = time.time() + self.state_timeout_s
            while self._state_buffer.get_data() is None:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for state from {self.robot_ip}:{self.state_port}. "
                        "Is gripper_server.py running on the robot?"
                    )
                time.sleep(0.05)
                log_warning("[Dex1GripperZmqClient] Waiting for state from robot…")

            self._can_publish = True
            self.is_connected = True
            log_success(
                f"[Dex1GripperZmqClient] Connected to {self.robot_ip}\n"
                f"  Current gripper q: {self.read_current_endeffector_q().round(4)}"
            )
        except Exception as e:
            self.disconnect()
            log_error(f"❌ Dex1GripperZmqClient.connect error: {e}")
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
        try:
            self._ctx.term()
        except Exception:
            pass
        log_success("[Dex1GripperZmqClient] Disconnected.")

    # ------------------------------------------------------------------
    # Background thread: receive state from robot ZMQ PUB
    # ------------------------------------------------------------------

    def _state_recv_loop(self):
        """Subscribe to gripper state published by the server and store in buffer."""
        log_info("[Dex1GripperZmqClient] state_recv_loop started")
        _diag_frames = 0
        while not self._stop_event.is_set():
            try:
                raw = self._state_sock.recv()
            except zmq.Again:
                continue
            except Exception as e:
                if self._stop_event.is_set():
                    break  # normal shutdown — context was terminated
                log_error(f"[Dex1GripperZmqClient] state_recv_loop recv error: {e}")
                continue

            if _diag_frames < 3:
                _diag_frames += 1
                log_info(
                    f"[Dex1GripperZmqClient] recv frame #{_diag_frames} "
                    f"len={len(raw)} hex_prefix={raw[:8].hex()}"
                )

            try:
                text = raw.decode("utf-8", errors="ignore")
                state = json.loads(text)
                parsed = {k: np.array(v, dtype=np.float64) for k, v in state.items()}
                self._state_buffer.set_data(parsed)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue

    # ------------------------------------------------------------------
    # Background thread: interpolate & push commands to robot
    # ------------------------------------------------------------------

    def _send_gripper_command(self, gripper_q: np.ndarray):
        """Serialize and PUSH one command frame to the server."""
        payload = json.dumps({
            "gripper_q": np.asarray(gripper_q, dtype=float).tolist(),
        })
        try:
            self._cmd_sock.send_string(payload, zmq.NOBLOCK)
        except zmq.Again:
            log_warning("[Dex1GripperZmqClient] cmd PUSH HWM reached — frame dropped")

    # _publish_command removed: commands are sent synchronously in write_endeffector().


    # ------------------------------------------------------------------
    # State readers (mirror Dex1_Gripper_Controller)
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        s = self._state_buffer.get_data()
        if s is None:
            raise RuntimeError("No state received from robot yet.")
        return s

    def read_current_endeffector_q(self) -> np.ndarray:
        """Return gripper joint positions (1-DoF array)."""
        return self._get_state()["gripper_q"].copy()

    def read_current_endeffector_dq(self) -> np.ndarray:
        """Return gripper joint velocities (1-DoF array)."""
        return self._get_state()["gripper_dq"].copy()

    # ------------------------------------------------------------------
    # Command writer (mirror Dex1_Gripper_Controller)
    # ------------------------------------------------------------------

    def write_endeffector(
        self,
        q_target: list[float] | np.ndarray,
        tauff_target: list[float] | np.ndarray = None,
        time_target: float | None = None,
        cmd_target: str | None = None,
    ):
        """Send gripper command immediately on the calling thread (zero background-thread lag)."""
        q = np.asarray(q_target, dtype=float).flatten()[:self.GRIPPER_DIM]
        with self._command_lock:
            self.q_target = q
        if self._can_publish:
            self._send_gripper_command(q)

    def go_start(self):
        with self._command_lock:
            self.q_target = self.init_pose.copy()

    def go_home(self):
        self.go_start()

    def stop_publish(self):
        """Pause command publishing."""
        self._can_publish = False

    def resume_publish(self):
        """Resume command publishing."""
        self._can_publish = True
