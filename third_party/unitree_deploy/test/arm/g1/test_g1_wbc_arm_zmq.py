"""Test script for G1WbcZmqArmClient (ZMQ proxy version of G1_WBC_ArmController).

This mirrors test_g1_wbc_arm.py but uses the ZMQ client/server split
instead of DDS directly.  The robot must be running g1_wbc_arm_server.py.

Usage — static import check (no hardware):
    python test/arm/g1/test_g1_wbc_arm_zmq.py --static-only

Usage — with robot server running:
    python test/arm/g1/test_g1_wbc_arm_zmq.py --robot-ip 192.168.123.1

Action layout (write_arm / robot.send_action):
    action[:14] = arm joint target q  (left 0-6, right 7-13)
    action[14:] = base_cmd (7 dims)
        [vx, vy, ang_z, waist_yaw, waist_pitch, waist_roll, body_height]
"""

import time
from dataclasses import dataclass

import numpy as np
import tyro

from unitree_deploy.utils.rich_logger import log_info


def test_static_import():
    """Verify that all new modules can be imported without hardware."""
    from unitree_deploy.robot_devices.arm.configs import G1WbcZmqArmConfig  # noqa: F401
    from unitree_deploy.robot_devices.arm.g1_wbc_arm_client import G1WbcZmqArmClient  # noqa: F401

    log_info("✅ Static import OK — G1WbcZmqArmConfig and G1WbcZmqArmClient imported successfully.")


def test_with_hardware(robot_ip: str, control_base_height: bool = False):
    """Live hardware test: connect via ZMQ, read state, move arms, disconnect."""
    import pinocchio as pin

    from unitree_deploy.robot.robot_configs import g1_motors
    from unitree_deploy.robot_devices.arm.configs import G1WbcZmqArmConfig
    from unitree_deploy.robot_devices.arm.g1_wbc_arm_client import G1WbcZmqArmClient
    from unitree_deploy.robot_devices.robots_devices_utils import precise_wait

    log_info("=" * 60)
    log_info(f"ZMQ hardware test  (robot_ip={robot_ip!r})")
    log_info("=" * 60)
    log_info("Make sure g1_wbc_arm_server.py is running on the robot!")

    config = G1WbcZmqArmConfig(motors=g1_motors, robot_ip=robot_ip)
    ctrl = G1WbcZmqArmClient(config)
    ctrl.connect()
    time.sleep(1.5)
    log_info("✅ Client connected. Waiting to start...")

    log_info(f"motor_names[{len(ctrl.motor_names)}]: {ctrl.motor_names}")
    log_info(f"Current arm q: {ctrl.read_current_arm_q().round(4)}")

    # Define initial target poses for left and right arms
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )
    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    rotation_speed = 0.005  # radians per iteration
    control_dt = 1 / 50     # 50 Hz
    step = 0
    max_step = 240
    height_max_step = 240
    height_step = 0
    height_direction = 1

    user_input = input("Please enter the start signal (enter 's' to start): \n")
    if user_input.lower() == "s":
        try:
            while True:
                t_cycle_end = time.monotonic() + control_dt

                direction = 1 if step <= 120 else -1
                angle = rotation_speed * (step if step <= 120 else (240 - step))

                cos_half = np.cos(angle / 2)
                sin_half = np.sin(angle / 2)

                L_quat = pin.Quaternion(cos_half, 0, sin_half, 0)
                R_quat = pin.Quaternion(cos_half, 0, 0, sin_half)

                delta_l = np.array([0.001, 0.001, 0.001]) * direction
                delta_r = np.array([0.001, -0.001, 0.001]) * direction

                L_tf_target.translation += delta_l
                R_tf_target.translation += delta_r
                L_tf_target.rotation = L_quat.toRotationMatrix()
                R_tf_target.rotation = R_quat.toRotationMatrix()

                sol_q, sol_tauff = ctrl.arm_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)

                if control_base_height:
                    progress = height_step / float(height_max_step)
                    body_height = (
                        0.75 - 0.10 * progress if height_direction == 1 else 0.65 + 0.10 * progress
                    )
                    height_step += 1
                    if height_step > height_max_step:
                        height_step = 0
                        height_direction *= -1
                else:
                    body_height = 0.75

                base_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, body_height]
                action = np.concatenate([sol_q, base_cmd])
                ctrl.write_arm(action=action, tauff_target=sol_tauff)

                step = (step + 1) % (max_step + 1)
                # print("t_cycle_end", t_cycle_end)
                time.sleep(1/50)
                # precise_wait(t_cycle_end)

        except KeyboardInterrupt:
            print("\n🛑 Ctrl+C detected. Disconnecting…")
            ctrl.disconnect()
            print("✅ Client disconnected.")


@dataclass
class Args:
    # IP address of the robot running g1_wbc_arm_server.py
    robot_ip: str = "192.168.123.1"

    # Only run static import checks (no hardware needed)
    static_only: bool = False

    # Oscillate base height during the dynamic motion test
    control_base_height: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.static_only:
        test_static_import()
    else:
        test_with_hardware(args.robot_ip, args.control_base_height)
