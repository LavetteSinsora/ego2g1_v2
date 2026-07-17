"""Test script for G1_WBC_ArmController.

Usage (static import check, no hardware):
    cd /home/unitree/code/unitree/unitree-deploy
    python test/arm/g1/test_g1_wbc_arm.py --static-only

Usage (with hardware connected):
    python test/arm/g1/test_g1_wbc_arm.py --net wlp0s20f3

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


def test_with_hardware(net: str, control_base_height: bool = False):
    """Live hardware test: connect, read state, move dynamically like test_g1_arm, disconnect."""
    import pinocchio as pin

    from unitree_deploy.robot.robot_configs import g1_motors
    from unitree_deploy.robot_devices.arm.configs import G1WbcArmConfig
    from unitree_deploy.robot_devices.arm.g1_wbc_arm import G1_WBC_ArmController
    from unitree_deploy.robot_devices.robots_devices_utils import precise_wait

    log_info("=" * 60)
    log_info(f"Hardware test  (net={net!r})")
    log_info("=" * 60)

    config = G1WbcArmConfig(motors=g1_motors, network_interface=net)
    ctrl = G1_WBC_ArmController(config)
    ctrl.connect()
    time.sleep(1.5)
    log_info("✅ Arm connected. Waiting to start...")

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

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    # Motion parameters
    control_dt = 1 / 50  # Control cycle duration (20ms)
    step = 0
    max_step = 240
    height_max_step = 240
    height_step = 0
    height_direction = 1  # 1: descending, -1: ascending

    # Wait for user input to start the motion loop
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == "s":
        try:
            while True:
                # Define timing for the control cycle
                t_cycle_end = time.monotonic() + control_dt

                direction = 1 if step <= 120 else -1
                angle = rotation_speed * (step if step <= 120 else (240 - step))

                cos_half_angle = np.cos(angle / 2)
                sin_half_angle = np.sin(angle / 2)

                L_quat = pin.Quaternion(cos_half_angle, 0, sin_half_angle, 0)  # 绕 Y 轴旋转
                R_quat = pin.Quaternion(cos_half_angle, 0, 0, sin_half_angle)  # 绕 Z 轴旋转

                delta_l = np.array([0.001, 0.001, 0.001]) * direction
                delta_r = np.array([0.001, -0.001, 0.001]) * direction

                L_tf_target.translation += delta_l
                R_tf_target.translation += delta_r

                L_tf_target.rotation = L_quat.toRotationMatrix()
                R_tf_target.rotation = R_quat.toRotationMatrix()

                # Solve inverse kinematics for the arm
                sol_q, sol_tauff = ctrl.arm_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
                # print(f"Arm solution: q={sol_q.round(4)}, tauff={sol_tauff.round(4)}")

                # Calculate base height if enabled
                if control_base_height:
                    progress = height_step / float(height_max_step)
                    if height_direction == 1:
                        body_height = 0.75 - (0.10 * progress)  # 0.75 -> 0.65
                    else:
                        body_height = 0.65 + (0.10 * progress)  # 0.65 -> 0.75

                    height_step += 1
                    if height_step > height_max_step:
                        height_step = 0
                        height_direction *= -1
                else:
                    body_height = 0.75
                # Send joint target command to arm
                base_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, body_height]
                action = np.concatenate([sol_q, base_cmd])
                # print("read_arm_q",ctrl.read_arm_q()    )
                ctrl.write_arm(
                    action=action,
                    tauff_target=sol_tauff,  # Optional: send torque feedforward
                )

                # Update step and reset after full cycle
                step = (step + 1) % (max_step + 1)

                # Wait until end of control cycle
                precise_wait(t_cycle_end)

        except KeyboardInterrupt:
            # Handle Ctrl+C to safely disconnect
            print("\n🛑 Ctrl+C detected. Disconnecting arm...")
            ctrl.disconnect()
            print("✅ Arm disconnected. Exiting.")


@dataclass
class Args:
    # Network interface for DDS (e.g. enp5s0). Required for hardware test.
    net: str = ""

    # Only run static import checks (no hardware needed).
    static_only: bool = False

    # Oscillate base height during dynamic motion test.
    control_base_height: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)

    if not args.net:
        log_info("⚠️  --net not specified and --static-only not set.")
        log_info("    Run with --static-only or provide --net <iface>.")
    else:
        test_with_hardware(args.net, args.control_base_height)
