import abc
from dataclasses import dataclass, field

import draccus
import numpy as np

from unitree_deploy.robot_devices.arm.configs import (
    ArmConfig,
    G1ArmConfig,
    G1WbcArmConfig,
    G1WbcZmqArmConfig,
    Z1ArmConfig,
    Z1DualArmConfig,
)
from unitree_deploy.robot_devices.cameras.configs import (
    CameraConfig,
    ImageClientCameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from unitree_deploy.robot_devices.endeffector.configs import (
    BraincoHandConfig,
    Dex1GripperConfig,
    Dex1GripperZmqConfig,
    Dex3HandConfig,
    EndEffectorConfig,
    InspireHandConfig,
)

# ======================== arm motors =================================
# name: (index, model)
g1_motors = {
    "kLeftShoulderPitch": [0, "g1-joint"],
    "kLeftShoulderRoll": [1, "g1-joint"],
    "kLeftShoulderYaw": [2, "g1-joint"],
    "kLeftElbow": [3, "g1-joint"],
    "kLeftWristRoll": [4, "g1-joint"],
    "kLeftWristPitch": [5, "g1-joint"],
    "kLeftWristyaw": [6, "g1-joint"],
    "kRightShoulderPitch": [7, "g1-joint"],
    "kRightShoulderRoll": [8, "g1-joint"],
    "kRightShoulderYaw": [9, "g1-joint"],
    "kRightElbow": [10, "g1-joint"],
    "kRightWristRoll": [11, "g1-joint"],
    "kRightWristPitch": [12, "g1-joint"],
    "kRightWristYaw": [13, "g1-joint"],
}

z1_motors = {
    "kWaist": [0, "z1-joint"],
    "kShoulder": [1, "z1-joint"],
    "kElbow": [2, "z1-joint"],
    "kForearmRoll": [3, "z1-joint"],
    "kWristAngle": [4, "z1-joint"],
    "kWristRotate": [5, "z1-joint"],
    "kGripper": [6, "z1-joint"],
}

z1_dual_motors = {
    "kLeftWaist": [0, "z1-joint"],
    "kLeftShoulder": [1, "z1-joint"],
    "kLeftElbow": [2, "z1-joint"],
    "kLeftForearmRoll": [3, "z1-joint"],
    "kLeftWristAngle": [4, "z1-joint"],
    "kLeftWristRotate": [5, "z1-joint"],
    "kRightWaist": [7, "z1-joint"],
    "kRightShoulder": [8, "z1-joint"],
    "kRightElbow": [9, "z1-joint"],
    "kRightForearmRoll": [10, "z1-joint"],
    "kRightWristAngle": [11, "z1-joint"],
    "kRightWristRotate": [12, "z1-joint"],
}

dex3_motors = {
    "kLeftHandThumb0",
    "kLeftHandThumb1",
    "kLeftHandThumb2",
    "kLeftHandMiddle0",
    "kLeftHandMiddle1",
    "kLeftHandIndex0",
    "kLeftHandIndex1",
    "kRightHandThumb0",
    "kRightHandThumb1",
    "kRightHandThumb2",
    "kRightHandIndex0",
    "kRightHandIndex1",
    "kRightHandMiddle0",
    "kRightHandMiddle1",
}

inspire_motors = {
    "kLeftHandPinky",
    "kLeftHandRing",
    "kLeftHandMiddle",
    "kLeftHandIndex",
    "kLeftHandThumbBend",
    "kLeftHandThumbRotation",
    "kRightHandPinky",
    "kRightHandRing",
    "kRightHandMiddle",
    "kRightHandIndex",
    "kRightHandThumbBend",
    "kRightHandThumbRotation",
}

brainco_motors = {
    "kLeftHandThumb": (0, "brainco"),
    "kLeftHandThumbAux": (1, "brainco"),
    "kLeftHandIndex": (2, "brainco"),
    "kLeftHandMiddle": (3, "brainco"),
    "kLeftHandRing": (4, "brainco"),
    "kLeftHandPinky": (5, "brainco"),
    "kRightHandThumb": (0, "brainco"),
    "kRightHandThumbAux": (1, "brainco"),
    "kRightHandIndex": (2, "brainco"),
    "kRightHandMiddle": (3, "brainco"),
    "kRightHandRing": (4, "brainco"),
    "kRightHandPinky": (5, "brainco"),
}
# =========================================================


# ======================== camera =================================


def z1_intelrealsense_camera_default_factory():
    return {
        "cam_high": IntelRealSenseCameraConfig(
            serial_number="044122071036",
            fps=30,
            width=640,
            height=480,
        ),
        # "cam_wrist": IntelRealSenseCameraConfig(
        #     serial_number="419122270615",
        #     fps=30,
        #     width=640,
        #     height=480,
        # ),
    }


def z1_dual_intelrealsense_camera_default_factory():
    return {
        # "cam_left_wrist": IntelRealSenseCameraConfig(
        #     serial_number="218722271166",
        #     fps=30,
        #     width=640,
        #     height=480,
        # ),
        # "cam_right_wrist": IntelRealSenseCameraConfig(
        #     serial_number="419122270677",
        #     fps=30,
        #     width=640,
        #     height=480,
        # ),
        "cam_high": IntelRealSenseCameraConfig(
            serial_number="947522071393",
            fps=30,
            width=640,
            height=480,
        ),
    }


def g1_image_client_default_factory():  # teleimager
    return {
        "imageclient": ImageClientCameraConfig(),
    }


def usb_camera_default_factory():
    return {
        "cam_high": OpenCVCameraConfig(
            camera_index="/dev/video2",
            fps=30,
            width=640,
            height=480,
        ),
        "cam_left_wrist": OpenCVCameraConfig(
            camera_index="/dev/video4",
            fps=30,
            width=640,
            height=480,
        ),
        "cam_right_wrist": OpenCVCameraConfig(
            camera_index="/dev/video0",
            fps=30,
            width=640,
            height=480,
        ),
    }


# =========================================================


# ======================== endeffector =================================


def dex1_default_factory():
    return {
        "left": Dex1GripperConfig(
            unit_test=True,
            motors={
                "kLeftGripper": [0, "z1_gripper-joint"],
            },
            topic_gripper_state="rt/dex1/left/state",
            topic_gripper_command="rt/dex1/left/cmd",
        ),
        "right": Dex1GripperConfig(
            unit_test=True,
            motors={
                "kRightGripper": [1, "z1_gripper-joint"],
            },
            topic_gripper_state="rt/dex1/right/state",
            topic_gripper_command="rt/dex1/right/cmd",
        ),
    }


def dex1_zmq_default_factory(robot_ip: str = "192.168.123.1"):
    return {
        "left": Dex1GripperZmqConfig(
            motors={"kLeftGripper": (0, "dex1")},
            robot_ip=robot_ip,
            cmd_port=5557,
            state_port=5558,
            init_pose=[0.0],
        ),
        "right": Dex1GripperZmqConfig(
            motors={"kRightGripper": (1, "dex1")},
            robot_ip=robot_ip,
            cmd_port=5559,
            state_port=5560,
            init_pose=[0.0],
        ),
    }


def dex3_default_factory():
    return {
        "dex_3": Dex3HandConfig(
            motors=dex3_motors,
            k_topic_dex3_left_command="rt/dex3/left/cmd",
            k_topic_dex3_right_command="rt/dex3/right/cmd",
            k_topic_dex3_left_state="rt/dex3/left/state",
            k_topic_dex3_right_state="rt/dex3/right/state",
        ),
    }


def inspire_default_factory():
    return {
        "inspire": InspireHandConfig(
            unit_test=True,
            motors=inspire_motors,
            k_topic_inspire_command="rt/inspire/cmd",
            k_topic_inspire_state="rt/inspire/state",
        ),
    }


def brainco_default_factory():
    # NOTE: the rt/brainco/* topics are class-level attributes on
    # BraincoHandConfig (not dataclass fields), so they must NOT be passed as
    # constructor kwargs — doing so raises TypeError. They already carry the
    # correct defaults on the class.
    return {
        "brainco": BraincoHandConfig(
            unit_test=True,
            motors=brainco_motors,
        ),
    }


# =========================================================

# ======================== arm =================================


def z1_arm_default_factory(init_pose=None):
    return {
        "z1": Z1ArmConfig(
            init_pose=np.zeros(7) if init_pose is None else init_pose,
            motors=z1_motors,
        ),
    }


def z1_dual_arm_single_config_factory(init_pose=None):
    return {
        "z1_dual": Z1DualArmConfig(
            left_robot_ip="127.0.0.1",
            left_robot_port1=8071,
            left_robot_port2=8072,
            right_robot_ip="127.0.0.1",
            right_robot_port1=8075,
            right_robot_port2=8076,
            init_pose_left=np.zeros(6) if init_pose is None else init_pose[:6],
            init_pose_right=np.zeros(6) if init_pose is None else init_pose[6:],
            control_dt=1 / 250.0,
            motors=z1_dual_motors,
        ),
    }


def g1_dual_arm_default_factory(init_pose=None):
    return {
        "g1": G1ArmConfig(
            init_pose=np.zeros(14) if init_pose is None else init_pose,
            motors=g1_motors,
            mock=False,
        ),
    }


def g1_wbc_dual_arm_default_factory(init_pose=None, network_interface="eth0", base_cmd=None):
    """Factory for G1 WBC arm – publishes JSON to rt/whole_body_sdk.

    base_cmd: optional [vx, vy, ang_z, waist_yaw, waist_pitch, waist_roll, body_height]
    used when the action stream carries only the 14 arm joints. None → config
    default (height 0.74). Pass e.g. [0,0,0,0,0,0,0.75] to stand at 0.75 m.
    """
    arm_cfg = G1WbcArmConfig(
        motors=g1_motors,
        network_interface=network_interface,
        init_pose=init_pose,  # None → uses G1WbcArmConfig default
    )
    if base_cmd is not None:
        arm_cfg.base_cmd = list(base_cmd)
    return {"g1_wbc": arm_cfg}


def g1_wbc_zmq_arm_default_factory(robot_ip: str = "192.168.123.1"):
    """Factory for G1 WBC arm via ZMQ – PC-side client connecting to g1_wbc_arm_server.py."""
    return {
        "g1_wbc_zmq": G1WbcZmqArmConfig(
            motors=g1_motors,
            robot_ip=robot_ip,
            cmd_port=5555,
            state_port=5556,
        ),
    }


# =========================================================


# robot_type:  arm devies _ endeffector devies _ camera devies
@dataclass
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@dataclass
class UnitreeRobotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})
    arm: dict[str, ArmConfig] = field(default_factory=lambda: {})
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=lambda: {})


# =============================== Single-arm:z1, Camera:Realsense ========================================
@RobotConfig.register_subclass("unitree_z1_realsense")
@dataclass
class Z1_Realsense_RobotConfig(UnitreeRobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=z1_intelrealsense_camera_default_factory)
    arm: dict[str, ArmConfig] = field(default_factory=z1_arm_default_factory)


# =============================== Dual-arm:z1, Endeffector:dex1, Camera:Realsense ========================================
@RobotConfig.register_subclass("unitree_z1_dual_dex1_realsense")
@dataclass
class Z1dual_Dex1_Realsense_RobotConfig(UnitreeRobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=z1_dual_intelrealsense_camera_default_factory)
    arm: dict[str, ArmConfig] = field(default_factory=z1_dual_arm_single_config_factory)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=dex1_default_factory)


# =============================== Dual-arm:z1, Endeffector:dex1, Camera:Realsense ========================================
@RobotConfig.register_subclass("unitree_z1_dual_dex1_opencv")
@dataclass
class Z1dual_Dex1_Opencv_RobotConfig(UnitreeRobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=usb_camera_default_factory)
    arm: dict[str, ArmConfig] = field(default_factory=z1_dual_arm_single_config_factory)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=dex1_default_factory)


# =============================== Arm:g1, Endeffector:dex1, Camera:imageclint ========================================
@RobotConfig.register_subclass("unitree_g1_dex1")
@dataclass
class G1_Dex1_Imageclint_RobotConfig(UnitreeRobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=g1_image_client_default_factory)
    arm: dict[str, ArmConfig] = field(default_factory=g1_dual_arm_default_factory)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=dex1_default_factory)


# =============================== Arm:g1, Endeffector:dex1, Camera:(none — sourced from policy server) ====================
@RobotConfig.register_subclass("unitree_g1_dex1_no_image")
@dataclass
class G1_Dex1_NoImage_RobotConfig(UnitreeRobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    arm: dict[str, ArmConfig] = field(default_factory=g1_dual_arm_default_factory)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=dex1_default_factory)


# =============================== Arm:g1, Endeffector:dex3, Camera:imageclint ========================================
@RobotConfig.register_subclass("unitree_g1_dex3")
@dataclass
class G1_Dex3_Imageclint_RobotConfig(UnitreeRobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=g1_image_client_default_factory)
    arm: dict[str, ArmConfig] = field(default_factory=g1_dual_arm_default_factory)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=dex3_default_factory)


# =============================== Arm:g1, Endeffector:brainco, Camera:imageclint ========================================
@RobotConfig.register_subclass("unitree_g1_brainco")
@dataclass
class G1_Brainco_Imageclint_RobotConfig(UnitreeRobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=g1_image_client_default_factory)
    arm: dict[str, ArmConfig] = field(default_factory=g1_dual_arm_default_factory)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=brainco_default_factory)


# =============================== Arm:g1, Endeffector:inspire, Camera:imageclint ========================================
@RobotConfig.register_subclass("unitree_g1_inspire")
@dataclass
class G1_Inspire_Imageclint_RobotConfig(UnitreeRobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=g1_image_client_default_factory)
    arm: dict[str, ArmConfig] = field(default_factory=g1_dual_arm_default_factory)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=inspire_default_factory)


# =============================== WBC mode: Arm:g1(wbc), Endeffector:dex1, Camera:imageclient ========================================
@RobotConfig.register_subclass("unitree_g1_wbc_dex1")
@dataclass
class G1Wbc_Dex1_Imageclient_RobotConfig(UnitreeRobotConfig):
    """G1 robot in whole-body control (WBC) mode with Dex1 grippers.

    - Arm commands are sent as JSON to ``rt/whole_body_sdk`` (WBC topic).
    - Dex1 dual grippers for end-effector control.
    - ImageClient camera for head-mounted RGB streaming.
    """

    network_interface: str = "eth0"

    cameras: dict[str, CameraConfig] = field(default_factory=g1_image_client_default_factory)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=dex1_default_factory)

    def __post_init__(self):
        # Build the arm dict after network_interface is known
        self.arm = g1_wbc_dual_arm_default_factory(network_interface=self.network_interface)


# =============================== WBC mode: Arm:g1(wbc), Endeffector:brainco, Camera:imageclient ====================
@RobotConfig.register_subclass("unitree_g1_wbc_brainco")
@dataclass
class G1Wbc_Brainco_Imageclient_RobotConfig(UnitreeRobotConfig):
    """G1 robot in whole-body control (WBC) mode with Brainco dexterous hands.

    - Arm commands are sent as JSON to ``rt/whole_body_sdk`` (WBC topic).
    - Brainco 12-DoF dual hands (left 6 + right 6) for end-effector control.
    - ImageClient camera for head-mounted RGB streaming.
    - Action layout (26-D, matches pi05_unitree_g1_brainco_make_sandwich):
        [0:14]  = arm joints (left 0-6, right 7-13)
        [14:26] = brainco hands (left 6 + right 6)
      The WBC base_cmd is NOT in the action stream; it is held at
      [0,0,0,0,0,0,0.75] (stand at 0.75 m, waist/legs still) by this config.
    """

    network_interface: str = "enp3s0"

    cameras: dict[str, CameraConfig] = field(default_factory=g1_image_client_default_factory)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=brainco_default_factory)

    def __post_init__(self):
        # Build the arm dict after network_interface is known. body_height=0.75.
        self.arm = g1_wbc_dual_arm_default_factory(
            network_interface=self.network_interface,
            base_cmd=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
        )


# =============================== WBC mode: Arm:g1(wbc), Endeffector:dex1, Camera:(none — sourced from policy server) ====
@RobotConfig.register_subclass("unitree_g1_wbc_dex1_no_image")
@dataclass
class G1Wbc_Dex1_NoImage_RobotConfig(UnitreeRobotConfig):
    """WBC variant with cameras disabled; the policy server pulls images directly
    from the robot's image_server."""

    network_interface: str = "eth0"

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    endeffector: dict[str, EndEffectorConfig] = field(default_factory=dex1_default_factory)

    def __post_init__(self):
        self.arm = g1_wbc_dual_arm_default_factory(network_interface=self.network_interface)


# =============================== WBC mode: Arm:g1(wbc,zmq), Endeffector:dex1(zmq), Camera:imageclient ========================================
@RobotConfig.register_subclass("unitree_g1_wbc_dex1_zmq")
@dataclass
class G1Wbc_Dex1Zmq_Imageclient_RobotConfig(UnitreeRobotConfig):
    """G1 robot in WBC mode with both arm and Dex1 grippers via ZMQ (PC-side clients).

    - Arm: G1WbcZmqArmClient connects to g1_wbc_arm_server.py (ports 5555/5556).
    - Dex1 grippers: Dex1GripperZmqClient connects to gripper_server.py
      (left 5557/5558, right 5559/5560).
    - ImageClient camera for head-mounted RGB streaming.
    - Both servers must be running on the robot before connecting.
    """

    robot_ip: str = "192.168.123.164"

    cameras: dict[str, CameraConfig] = field(default_factory=g1_image_client_default_factory)

    def __post_init__(self):
        self.arm = g1_wbc_zmq_arm_default_factory(robot_ip=self.robot_ip)
        self.endeffector = dex1_zmq_default_factory(robot_ip=self.robot_ip)
