"""G1 Whole-Body Control (WBC) Arm Controller for unitree-deploy.

This controller communicates with the G1 robot through its built-in WBC
controller via the DDS topic ``rt/whole_body_sdk``. Commands are sent as
JSON strings and include both arm joint targets and whole-body base commands,
enabling simultaneous arm and locomotion control.

Unlike ``G1_29_ArmController`` (which writes LowCmd directly to arm joints),
this class is designed for the WBC teleoperation / imitation-learning scenario
where the robot's internal WBC handles all low-level stabilization.

Communication architecture:
  Subscribe : rt/lowstate  → LowStateHG  (29-DoF joint state + IMU + odom)
  Publish   : rt/whole_body_sdk → String_ (JSON: {arm_q, arm_tau, base_cmd})
"""

import json
import threading
import time
from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG

from unitree_deploy.robot_devices.arm.configs import G1WbcArmConfig
from unitree_deploy.robot_devices.arm.g1_arm_ik import G1_29_ArmIK
from unitree_deploy.robot_devices.robot_devices_index import G1_29_JointArmIndex
from unitree_deploy.robot_devices.robots_devices_utils import (
    DataBuffer,
    MotorState,
    Robot_Num_Motors,
    RobotDeviceAlreadyConnectedError,
)
from unitree_deploy.utils.joint_trajcetory_inter import JointTrajectoryInterpolator
from unitree_deploy.utils.rich_logger import log_error, log_info, log_success, log_warning

# ---------------------------------------------------------------------------
# Internal state snapshot
# ---------------------------------------------------------------------------


class G1_29_WbcLowState:
    """Snapshot of the full G1 body state from LowState."""

    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.G1_29_Num_Motors)]

        # IMU
        self.accel = np.zeros(3, dtype=np.float32)
        self.quaternion = np.zeros(4, dtype=np.float32)
        self.gyroscope = np.zeros(3, dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper: Cartesian interpolation utilities
# ---------------------------------------------------------------------------


def _ensure_homog_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert various pose representations to 4x4 homogeneous matrix."""
    pose = np.array(pose, dtype=float)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (16,):
        return pose.reshape(4, 4)
    if pose.shape == (6,):
        mat = np.eye(4)
        mat[:3, 3] = pose[:3]
        mat[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
        return mat
    if pose.shape == (7,):
        mat = np.eye(4)
        mat[:3, 3] = pose[:3]
        mat[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        return mat
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


def _interpolate_pose(start_mat: np.ndarray, end_mat: np.ndarray, alpha: float) -> np.ndarray:
    """SE3 linear interpolation (Slerp for rotation, linear for translation)."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    pos = (1 - alpha) * start_mat[:3, 3] + alpha * end_mat[:3, 3]
    key_rots = Rotation.concatenate(
        [Rotation.from_matrix(start_mat[:3, :3]), Rotation.from_matrix(end_mat[:3, :3])]
    )
    slerp = Slerp([0, 1], key_rots)
    rot = slerp([alpha])[0]
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = pos
    return mat


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------


class G1_WBC_ArmController:
    """G1 WBC arm controller following the unitree-deploy device interface."""

    def __init__(self, config: G1WbcArmConfig):
        self.motors = config.motors
        self.control_dt = config.control_dt
        self.topic_low_state = config.topic_low_state
        self.topic_wbc_command = config.topic_wbc_command
        self.network_interface = config.network_interface
        self.init_pose = np.array(config.init_pose, dtype=np.float64)

        # Arm IK solver (same as G1_29_ArmController)
        self.g1_arm_ik = G1_29_ArmIK(unit_test=False, visualization=False)

        # Thread-safe state & command buffers
        self.lowstate_buffer = DataBuffer()
        self._command_lock = threading.Lock()

        self.max_pos_speed = getattr(config, "max_pos_speed", 180 * (np.pi / 180) * 2)

        self.q_target = np.zeros(self.ARM_DIM)
        self.tauff_target = np.zeros(self.ARM_DIM)
        # Default base_cmd: from config if provided (lets a robot config set e.g.
        # body_height=0.75), else the class default (height=0.74). Back-compat:
        # configs that predate the base_cmd field fall through to DEFAULT_BASE_CMD.
        self.base_cmd_target = list(getattr(config, "base_cmd", None) or self.DEFAULT_BASE_CMD)
        self.time_target = time.monotonic()
        self.arm_cmd = "schedule_waypoint"

        self._can_publish = False

        # Stop signal for daemon threads
        self._stop_event = threading.Event()
        self.is_connected = False

    # Layout of the combined action passed to write_arm:
    #   action[:14]  = arm joint target q  (left 0-6, right 7-13)
    #   action[14:]  = base_cmd (7 dims)
    #     [vx, vy, ang_z, waist_yaw, waist_pitch, waist_roll, body_height]
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

    @property
    def motor_names(self) -> list[str]:
        """Return 21 names: 14 arm joints + 7 base_cmd dimensions."""
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
                "G1_WBC_ArmController is already connected. Do not call connect() twice."
            )
        try:
            # ChannelFactoryInitialize(0)
            ChannelFactoryInitialize(0, self.network_interface)
            # ---- LowState subscriber ----
            self._lowstate_sub = ChannelSubscriber(self.topic_low_state, LowStateHG)
            self._lowstate_sub.Init()

            # ---- WBC command publisher ----
            self._wbc_pub = ChannelPublisher(self.topic_wbc_command, String_)
            self._wbc_pub.Init()
            self._wbc_msg = String_(data="")

            # ---- Daemon threads ----
            self._sub_thread = self._start_daemon("_subscribe_motor_state", self._subscribe_motor_state)
            self._pub_thread = self._start_daemon("_publish_command", self._publish_command)

            # Wait until first LowState arrives
            timeout = 10.0
            deadline = time.time() + timeout
            while not self.lowstate_buffer.get_data():
                if time.time() > deadline:
                    raise TimeoutError("Timed out waiting for LowState data.")
                time.sleep(0.05)
                log_warning("[G1_WBC_ArmController] Waiting for LowState DDS...")

            log_info(f"[G1_WBC_ArmController] Connected.\n  Current arm q: {self.read_current_arm_q()}")

            self._can_publish = True
            self.is_connected = True
            log_success("[G1_WBC_ArmController] Ready ✅")

        except Exception as e:
            self.disconnect()
            log_error(f"❌ Error in G1_WBC_ArmController.connect: {e}")
            raise

    def _start_daemon(self, name: str, target: Callable) -> threading.Thread:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        return t

    def disconnect(self):
        self.is_connected = False
        self._can_publish = False
        self._stop_event.set()
        log_success("[G1_WBC_ArmController] Disconnected.")

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _subscribe_motor_state(self):
        """Continuously read LowState and store in buffer."""
        try:
            while not self._stop_event.is_set():
                msg = self._lowstate_sub.Read()
                if msg is not None:
                    snap = G1_29_WbcLowState()
                    n = min(len(msg.motor_state), Robot_Num_Motors.G1_29_Num_Motors)
                    for i in range(n):
                        snap.motor_state[i].q = msg.motor_state[i].q
                        snap.motor_state[i].dq = msg.motor_state[i].dq
                        # snap.motor_state[i].tau = msg.motor_state[i].tau
                    snap.accel = np.array(msg.imu_state.accelerometer, dtype=np.float32)
                    snap.quaternion = np.array(msg.imu_state.quaternion, dtype=np.float32)
                    snap.gyroscope = np.array(msg.imu_state.gyroscope, dtype=np.float32)
                    snap.rpy = np.array(msg.imu_state.rpy, dtype=np.float32)
                    self.lowstate_buffer.set_data(snap)
                time.sleep(self.control_dt)
        except Exception as e:
            log_error(f"❌ Error in G1_WBC_ArmController._subscribe_motor_state: {e}")

    def _update_wbc_command(self, arm_q_target: np.ndarray, arm_tauff_target: np.ndarray, base_cmd: list):
        payload = json.dumps(
            {
                "arm_q": np.asarray(arm_q_target, dtype=float).tolist(),
                "arm_tau": np.asarray(arm_tauff_target, dtype=float).tolist(),
                "base_cmd": base_cmd,
            }
        )
        self._wbc_msg.data = payload
        self._wbc_pub.Write(self._wbc_msg)

    def _drive_to_waypoint(
        self, target_pose: np.ndarray, t_insert_time: float, arm_tauff_target: np.ndarray, base_cmd: list
    ):
        curr_time = time.monotonic() + self.control_dt
        t_insert = curr_time + t_insert_time
        self.pose_interp = self.pose_interp.drive_to_waypoint(
            pose=target_pose,
            time=t_insert,
            curr_time=curr_time,
            max_pos_speed=self.max_pos_speed,
        )

        while time.monotonic() < t_insert and not self._stop_event.is_set():
            start_time = time.perf_counter()

            clipped_arm_q_target = self.pose_interp(time.monotonic())
            if self._can_publish:
                self._update_wbc_command(clipped_arm_q_target, arm_tauff_target, base_cmd)

            time.sleep(max(0, (self.control_dt - (time.perf_counter() - start_time))))

    def _schedule_waypoint(
        self,
        arm_q_target: np.ndarray,
        arm_time_target: float,
        t_now: float,
        start_time: float,
        last_waypoint_time: float,
        arm_tauff_target: np.ndarray,
        base_cmd: list,
    ) -> float:
        target_time = time.monotonic() - time.perf_counter() + arm_time_target
        curr_time = t_now + self.control_dt
        target_time = max(target_time, curr_time + self.control_dt)

        self.pose_interp = self.pose_interp.schedule_waypoint(
            pose=arm_q_target,
            time=target_time,
            max_pos_speed=self.max_pos_speed,
            curr_time=curr_time,
            last_waypoint_time=last_waypoint_time,
        )
        last_waypoint_time = target_time

        clipped_arm_q_target = self.pose_interp(t_now)
        if self._can_publish:
            self._update_wbc_command(clipped_arm_q_target, arm_tauff_target, base_cmd)

        time.sleep(max(0, (self.control_dt - (time.perf_counter() - start_time))))
        return last_waypoint_time

    def _publish_command(self):
        """Publish the latest command JSON and interpolate trajectories at the configured control rate."""
        # wait dds init done !!!
        time.sleep(2)

        self.pose_interp = JointTrajectoryInterpolator(
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
                        arm_tauff_target=arm_tauff_target,
                        base_cmd=base_cmd_target,
                    )
                elif arm_cmd == "schedule_waypoint":
                    last_waypoint_time = self._schedule_waypoint(
                        arm_q_target=arm_q_target,
                        arm_time_target=arm_time_target,
                        t_now=t_now,
                        start_time=start_time,
                        last_waypoint_time=last_waypoint_time,
                        arm_tauff_target=arm_tauff_target,
                        base_cmd=base_cmd_target,
                    )
        except Exception as e:
            log_error(f"❌ Error in G1_WBC_ArmController._publish_command: {e}")

    # ------------------------------------------------------------------
    # State readers
    # ------------------------------------------------------------------

    def read_arm_q(self) -> np.ndarray:
        """Return current 14-DoF arm joint positions [left 0-6, right 7-13]."""
        snap = self.lowstate_buffer.get_data()
        return np.array(
            [snap.motor_state[i].q for i in G1_29_JointArmIndex],
            dtype=np.float64,
        )

    def read_current_arm_q(self) -> np.ndarray:
        """Return full robot state as a single flattened vector."""
        snap = self.lowstate_buffer.get_data()
        ms = snap.motor_state

        leg_q = np.array([ms[i].q for i in range(12)], dtype=np.float64)
        waist_q = np.array([ms[i].q for i in range(12, 15)], dtype=np.float64)
        arm_q = np.array([ms[i].q for i in G1_29_JointArmIndex], dtype=np.float64)

        obs = np.concatenate([arm_q, waist_q, leg_q])
        return obs

    def read_current_arm_dq(self) -> np.ndarray:
        """Return current 14-DoF arm joint velocities."""
        snap = self.lowstate_buffer.get_data()
        return np.array(
            [snap.motor_state[i].dq for i in G1_29_JointArmIndex],
            dtype=np.float64,
        )

    def read_full_state(self) -> dict:
        """Return a comprehensive state dict including legs, waist, arms, and IMU."""
        snap = self.lowstate_buffer.get_data()
        ms = snap.motor_state
        return {
            "leg_q": np.array([ms[i].q for i in range(12)], dtype=np.float64),
            "waist_q": np.array([ms[i].q for i in range(12, 15)], dtype=np.float64),
            "arm_q": np.array([ms[i].q for i in G1_29_JointArmIndex], dtype=np.float64),
            "arm_dq": np.array([ms[i].dq for i in G1_29_JointArmIndex], dtype=np.float64),
            "imu_accel": snap.accel.copy(),
            "imu_quat": snap.quaternion.copy(),
            "imu_gyro": snap.gyroscope.copy(),
            "imu_rpy": snap.rpy.copy(),
        }

    # ------------------------------------------------------------------
    # Command writer
    # ------------------------------------------------------------------

    def write_arm(
        self,
        action: np.ndarray,
        tauff_target: np.ndarray | None = None,
        # Protocol compatibility (ignored in WBC mode)
        time_target: float | None = None,
        cmd_target: str | None = None,
    ):
        """Update the WBC command to publish targets."""
        action = np.asarray(action, dtype=float)
        q_target = action[: self.ARM_DIM]
        base_cmd = (
            action[self.ARM_DIM : self.ARM_DIM + self.BASE_CMD_DIM].tolist()
            if len(action) >= self.ARM_DIM + self.BASE_CMD_DIM
            else list(self.base_cmd_target)
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
        """Pause command publishing (motor overheat protection etc.)."""
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
        """Forward kinematics."""
        if q is None:
            q = self.read_arm_q()
        return self.g1_arm_ik.solve_fk_matrix(q)

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------
    def go_start(self):
        base_cmd_target = list(self.DEFAULT_BASE_CMD)
        tauff = self.g1_arm_ik.solve_tau(self.init_pose) or np.zeros(self.ARM_DIM)
        with self._command_lock:
            self.q_target = self.init_pose.copy()
            self.tauff_target = tauff
            self.base_cmd_target = base_cmd_target
        log_success("[G1_WBC_ArmController] Go Start OK!\n")

    def go_home(self):
        """Alias for go_start() — return arms to the configured init pose."""
        self.go_start()
        log_success("[G1_WBC_ArmController] Go Home OK!\n")

