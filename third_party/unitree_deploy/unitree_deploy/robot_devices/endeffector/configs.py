import abc
from dataclasses import dataclass

import draccus
import numpy as np


@dataclass
class EndEffectorConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@EndEffectorConfig.register_subclass("dex_1")
@dataclass
class Dex1GripperConfig(EndEffectorConfig):
    motors: dict[str, tuple[int, str]]
    unit_test: bool = False
    init_pose: list | None = None
    control_dt: float = 1 / 200
    mock: bool = False
    max_pos_speed: float = 180 * (np.pi / 180) * 2
    topic_gripper_command: str = "rt/unitree_actuator/cmd"
    topic_gripper_state: str = "rt/unitree_actuator/state"

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")


@EndEffectorConfig.register_subclass("dex_1_zmq")
@dataclass
class Dex1GripperZmqConfig(EndEffectorConfig):
    """PC-side config for the ZMQ client proxy of Dex1 gripper.

    The matching server (gripper_server.py) must be running on the robot
    before connecting.

    Transport:
      cmd   port – PC PUSH  → Robot PULL  (JSON gripper commands)
      state port – Robot PUB → PC SUB     (JSON gripper state)
    """

    motors: dict[str, tuple[int, str]]

    # Network address of the robot running gripper_server.py
    robot_ip: str = "192.168.123.1"

    # ZMQ port for command channel (PUSH/PULL, PC → Robot)
    cmd_port: int = 5557
    # ZMQ port for state channel (PUB/SUB, Robot → PC)
    state_port: int = 5558

    # Control loop
    control_dt: float = 1 / 200.0  # 200 Hz

    # Initial gripper pose (1-DoF, 0 = fully open)
    init_pose: list | None = None
    max_pos_speed: float = 180 * (np.pi / 180) * 2

    # How long to wait for the first state message before raising TimeoutError
    state_timeout_s: float = 5.0

    def __post_init__(self):
        if self.init_pose is None:
            self.init_pose = [0.0]
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")


@EndEffectorConfig.register_subclass("dex_3")
@dataclass
class Dex3HandConfig(EndEffectorConfig):
    motors: dict[str, tuple[int, str]]
    unit_test: bool = False
    control_dt: float = 1 / 200
    mock: bool = False
    simulation_mode: bool = False

    k_topic_dex3_left_command = "rt/dex3/left/cmd"
    k_topic_dex3_right_command = "rt/dex3/right/cmd"
    k_topic_dex3_left_state = "rt/dex3/left/state"
    k_topic_dex3_right_state = "rt/dex3/right/state"

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")


@EndEffectorConfig.register_subclass("inspire")
@dataclass
class InspireHandConfig(EndEffectorConfig):
    motors: dict[str, tuple[int, str]]
    unit_test: bool = False
    control_dt: float = 1 / 200
    mock: bool = False
    simulation_mode: bool = False

    k_topic_inspire_command = "rt/inspire/cmd"
    k_topic_inspire_state = "rt/inspire/state"

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")


@EndEffectorConfig.register_subclass("brainco")
@dataclass
class BraincoHandConfig(EndEffectorConfig):
    motors: dict[str, tuple[int, str]]
    unit_test: bool = False
    control_dt: float = 1 / 200
    mock: bool = False
    simulation_mode: bool = False

    k_topic_brainco_left_command = "rt/brainco/left/cmd"
    k_topic_brainco_right_command = "rt/brainco/right/cmd"
    k_topic_brainco_left_state = "rt/brainco/left/state"
    k_topic_brainco_right_state = "rt/brainco/right/state"

    def __post_init__(self):
        if self.control_dt < 0.002:
            raise ValueError(f"`control_dt` must > 1/500 (got {self.control_dt})")
