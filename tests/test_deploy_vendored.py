"""The vendored unitree_deploy copy: assets present, 3.12-safe dataclasses.

The configs regression matters because Python 3.12 rejects mutable dataclass
defaults (np.ndarray/list) at class-definition time, and even where it does
not, a shared default array is a cross-instance aliasing bug. The vendored
copy must keep every such default behind field(default_factory=...).
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

VENDOR = (pathlib.Path(__file__).resolve().parent.parent
          / "third_party" / "unitree_deploy")


def _load_by_path(name: str, rel: str):
    """Import a vendored module by file path — bypasses the editable install
    of the OLD repo's copy that may shadow the namespace package. Bytecode
    writing is suppressed so the test never litters the vendored tree with
    the __pycache__ it asserts absent."""
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, VENDOR / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = prev


def test_vendor_tree_is_complete():
    # the executor itself
    assert (VENDOR / "unitree_deploy" / "robot_devices" / "arm" / "g1_arm.py").exists()
    assert (VENDOR / "unitree_deploy" / "utils" / "joint_trajcetory_inter.py").exists()
    # gravity comp needs the URDF
    assert (VENDOR / "unitree_deploy" / "robot_devices" / "assets" / "g1"
            / "g1_body29_hand14.urdf").exists()
    # installs as a path dep
    assert (VENDOR / "pyproject.toml").exists()
    # build junk must NOT have been vendored
    assert not (VENDOR / "unitree_deploy.egg-info").exists()
    assert not list(VENDOR.rglob("__pycache__"))


def test_arm_configs_are_312_safe_and_unshared():
    cfgs = _load_by_path("vendored_arm_configs",
                         "unitree_deploy/robot_devices/arm/configs.py")
    a = cfgs.G1ArmConfig(motors={})
    b = cfgs.G1ArmConfig(motors={})
    assert isinstance(a.init_pose, np.ndarray)
    a.init_pose[0] = 9.9
    assert b.init_pose[0] == 0.0, "G1ArmConfig.init_pose default is SHARED"

    z1 = cfgs.Z1ArmConfig(motors={})
    z2 = cfgs.Z1ArmConfig(motors={})
    z1.robot_kp[0] = -1
    assert z2.robot_kp[0] != -1, "Z1ArmConfig.robot_kp default is SHARED"

    w1 = cfgs.G1WbcArmConfig(motors={}, network_interface="eth0")
    w2 = cfgs.G1WbcArmConfig(motors={}, network_interface="eth0")
    w1.base_cmd[0] = 5.0
    assert w2.base_cmd[0] == 0.0, "G1WbcArmConfig.base_cmd default is SHARED"
    w1.init_pose[0] = 5.0
    assert w2.init_pose[0] == 0.0, "G1WbcArmConfig.init_pose default is SHARED"


def test_g1_config_wire_contract():
    """The facts the deploy layer assumes about the vendored G1 config."""
    cfgs = _load_by_path("vendored_arm_configs2",
                         "unitree_deploy/robot_devices/arm/configs.py")
    g1 = cfgs.G1ArmConfig(motors={})
    assert g1.topic_low_command == "rt/lowcmd"      # NEVER rt/arm_sdk on G1-D
    assert g1.topic_low_state == "rt/lowstate"
    assert g1.control_dt == pytest.approx(1 / 500)  # the 500 Hz executor
    # gains our dds notes document: shoulder/elbow 80/3, wrist 40/1.5
    assert (g1.kp_low, g1.kd_low) == (80.0, 3.0)
    assert (g1.kp_wrist, g1.kd_wrist) == (40.0, 1.5)


def test_brainco_and_g1_motor_order_match_ego2g1():
    """robot_configs' motor dicts define the send_action packing order the
    executor wrapper relies on: arm L7+R7, then brainco L6+R6."""
    src = (VENDOR / "unitree_deploy" / "robot" / "robot_configs.py").read_text()
    g1 = src.index("g1_motors")
    order = ["kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
             "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristyaw",
             "kRightShoulderPitch"]
    pos = [src.index(k, g1) for k in order]
    assert pos == sorted(pos), "g1_motors order changed — arm packing is wrong"
    b = src.index("brainco_motors")
    horder = ["kLeftHandThumb", "kLeftHandThumbAux", "kLeftHandIndex",
              "kLeftHandMiddle", "kLeftHandRing", "kLeftHandPinky",
              "kRightHandThumb"]
    hpos = [src.index(k, b) for k in horder]
    assert hpos == sorted(hpos), "brainco_motors order changed — hand packing is wrong"
