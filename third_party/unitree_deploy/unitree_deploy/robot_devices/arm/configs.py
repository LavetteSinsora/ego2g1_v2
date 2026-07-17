import abc
from dataclasses import dataclass
from dataclasses import field

import draccus
import numpy as np


@dataclass
class ArmConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@ArmConfig.register_subclass("z1")
@dataclass
class Z1ArmConfig(ArmConfig):
    motors: dict[str, tuple[int, str]]

    init_pose: list = None
    unit_test: bool = False
    control_dt: float = 1 / 500.0

    robot_kp: np.ndarray = field(default_factory=lambda: np.array([4, 6, 6, 6, 6, 6]))
    robot_kd: np.ndarray = field(default_factory=lambda: np.array([350, 300, 300, 200, 200, 200]))
    max_pos_speed: float = 180 * (np.pi / 180) * 2
    log_level: str | int = "ERROR"

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")


@ArmConfig.register_subclass("z1_dual")
@dataclass
class Z1DualArmConfig(ArmConfig):
    left_robot_ip: str
    left_robot_port1: int
    left_robot_port2: int
    right_robot_ip: str
    right_robot_port1: int
    right_robot_port2: int
    motors: dict[str, tuple[int, str]]

    robot_kp: np.ndarray = field(default_factory=lambda: np.array([4, 6, 6, 6, 6, 6]))
    robot_kd: np.ndarray = field(default_factory=lambda: np.array([350, 300, 300, 200, 200, 200]))
    mock: bool = False
    unit_test: bool = False
    init_pose_left: list | None = None
    init_pose_right: list | None = None
    max_pos_speed: float = 180 * (np.pi / 180) * 2
    control_dt: float = 1 / 500.0

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")


@ArmConfig.register_subclass("g1")
@dataclass
class G1ArmConfig(ArmConfig):
    motors: dict[str, tuple[int, str]]
    mock: bool = False
    unit_test: bool = False
    init_pose: np.ndarray | list = field(default_factory=lambda: np.zeros(14))

    control_dt: float = 1 / 500.0
    max_pos_speed: float = 180 * (np.pi / 180) * 2

    topic_low_command: str = "rt/lowcmd"
    topic_low_state: str = "rt/lowstate"

    kp_high: float = 300.0
    kd_high: float = 3.0
    kp_low: float = 80.0  # 140.0
    kd_low: float = 3.0  # 3.0
    kp_wrist: float = 40.0
    kd_wrist: float = 1.5

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")


@ArmConfig.register_subclass("g1_wbc")
@dataclass
class G1WbcArmConfig(ArmConfig):
    """Configuration for G1 whole-body control (WBC) mode.

    Unlike G1ArmConfig (which sends LowCmd directly to arm joints),
    this config targets the rt/whole_body_sdk DDS topic and sends a
    JSON string {arm_q, arm_tau, base_cmd} to the robot's built-in
    WBC controller, enabling simultaneous arm + base motion control.
    """

    motors: dict[str, tuple[int, str]]
    # Network interface for DDS communication
    network_interface: str

    # Control loop
    control_dt: float = 1 / 100.0  # 100 Hz command publish rate

    # DDS topics
    topic_low_state: str = "rt/lowstate"
    topic_wbc_command: str = "rt/whole_body_sdk"

    # Initial 14-DoF arm pose (indices 0-6: left arm, 7-13: right arm)
    init_pose: np.ndarray = None
    max_pos_speed: float = 180 * (np.pi / 180) * 2

    # Default WBC base command used when the action stream carries only the 14
    # arm joints (no per-step base_cmd). Layout:
    #   [vx, vy, ang_z, waist_yaw, waist_pitch, waist_roll, body_height]
    # body_height 0.74 is the controller default; callers that want the robot to
    # stand at a different height (e.g. 0.75 for brainco make_sandwich) override
    # this in their robot config factory.
    base_cmd: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.74])

    def __post_init__(self):
        if self.init_pose is None:
            self.init_pose = np.zeros(14)
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")


@ArmConfig.register_subclass("g1_wbc_zmq")
@dataclass
class G1WbcZmqArmConfig(ArmConfig):
    """PC-side config for the ZMQ client proxy of G1 WBC arm.

    The matching server (g1_wbc_arm_server.py) must be running on the robot
    (or any host that has DDS access to the G1) before connecting.

    Transport:
      cmd   port  – PC PUSH  → Robot PULL  (JSON arm commands)
      state port  – Robot PUB → PC SUB      (JSON full LowState)
    """

    motors: dict[str, tuple[int, str]]
    # Network address of the robot running g1_wbc_arm_server.py
    robot_ip: str = "192.168.123.1"

    # ZMQ port for command channel (PUSH/PULL, PC → Robot)
    cmd_port: int = 5555
    # ZMQ port for state channel (PUB/SUB, Robot → PC)
    state_port: int = 5556

    # Control loop
    control_dt: float = 1 / 100.0  # 100 Hz

    # Initial 14-DoF arm pose (indices 0-6: left arm, 7-13: right arm)
    init_pose: np.ndarray = None
    max_pos_speed: float = 180 * (np.pi / 180) * 2

    # How long to wait for the first state message before raising TimeoutError
    state_timeout_s: float = 5.0

    def __post_init__(self):
        if self.init_pose is None:
            self.init_pose = np.zeros(14)
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")
