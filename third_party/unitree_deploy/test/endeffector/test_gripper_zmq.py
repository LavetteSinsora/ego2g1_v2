"""Test script for Dex1GripperZmqClient (ZMQ proxy version of Dex1_Gripper_Controller).

This mirrors the gripper test but uses the ZMQ client/server split instead
of DDS directly.  The robot must be running gripper_server.py.

Usage — static import check (no hardware):
    python test/endeffector/test_gripper_zmq.py --static-only

Usage — with robot server running:
    python test/endeffector/test_gripper_zmq.py --robot-ip 192.168.123.1
"""

import time
from dataclasses import dataclass

import tyro

from unitree_deploy.robot_devices.endeffector.configs import Dex1GripperZmqConfig
from unitree_deploy.robot_devices.endeffector.utils import make_endeffector_motors_buses_from_configs
from unitree_deploy.robot_devices.robots_devices_utils import precise_wait
from unitree_deploy.utils.rich_logger import log_info
from unitree_deploy.utils.trajectory_generator import sinusoidal_single_gripper_motion


def gripper_zmq_default_factory(robot_ip: str) -> dict[str, Dex1GripperZmqConfig]:
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


def test_static_import():
    """Verify that all new modules can be imported without hardware."""
    from unitree_deploy.robot_devices.endeffector.configs import Dex1GripperZmqConfig  # noqa: F401
    from unitree_deploy.robot_devices.endeffector.gripper_client import Dex1GripperZmqClient  # noqa: F401

    log_info(
        "✅ Static import OK — Dex1GripperZmqConfig and Dex1GripperZmqClient imported successfully."
    )

period = 2.0
motion_period = 2.0
motion_amplitude = 0.99


def test_with_hardware(robot_ip: str):
    """Live hardware test: connect via ZMQ, run sinusoidal motion on both grippers."""
    log_info("=" * 60)
    log_info(f"Gripper ZMQ hardware test  (robot_ip={robot_ip!r})")
    log_info("=" * 60)
    log_info("Make sure gripper_server.py is running on the robot!")

    control_dt = 1 / 30

    endeffectors = make_endeffector_motors_buses_from_configs(
        gripper_zmq_default_factory(robot_ip)
    )

    for name in endeffectors:
        log_info(f"Connecting to '{name}' gripper…")
        endeffectors[name].connect()
        log_info(f"Connected endeffector '{name}'.")

    try:
        while True:
            t_cycle_end = time.monotonic() + control_dt
            target_q = sinusoidal_single_gripper_motion(
                period=motion_period, amplitude=motion_amplitude, current_time=time.perf_counter()
            )
            for name in endeffectors:
                endeffectors[name].write_endeffector(q_target=target_q)
            # precise_wait(t_cycle_end)
            time.sleep(1/10)

    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C detected.")
    finally:
        for name in endeffectors:
            endeffectors[name].disconnect()
        print("✅ Clients disconnected.")


@dataclass
class Args:
    # IP address of the robot running gripper_server.py
    robot_ip: str = "127.0.0.1"

    # Only run static import checks (no hardware needed)
    static_only: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.static_only:
        test_static_import()
    else:
        test_with_hardware(args.robot_ip)
