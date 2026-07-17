import collections
import time

import numpy as np
import torch

from unitree_deploy.robot.robot_utils import make_robot
from unitree_deploy.robot_devices.robots_devices_utils import precise_wait
from unitree_deploy.utils.rich_logger import log_success


class UnitreeEnv:
    def __init__(
        self,
        robot_type: str = "z1_realsense",
        dt: float = 1 / 30,
        init_pose_arm: np.ndarray | list[float] | None = None,
        network_interface: str | None = None,
    ):
        self.control_dt = dt
        self.init_pose_arm = init_pose_arm
        self.robot_type = robot_type
        robot_kwargs = {}
        if network_interface is not None:
            robot_kwargs["network_interface"] = network_interface
        self.robot = make_robot(self.robot_type, **robot_kwargs)

    def connect(self):
        self.robot.connect()

    def get_observation(self):
        observation = self.robot.capture_observation()

        # Process images
        images_observation = {
            key.split("observation.images.")[-1]: value.numpy()
            for key, value in observation.items()
            if key.startswith("observation.images.")
        }
        # for image_key, image in images_observation.items():
        #     cv2.imwrite(f"{image_key}.png", image)

        # Process state
        state = observation["observation.state"].numpy()

        # Construct observation dictionary
        obs = collections.OrderedDict(
            qpos=state,
            qvel=np.zeros_like(state),
            effort=np.zeros_like(state),
            images=images_observation,
        )
        return obs

    def step(self, action):
        t_cycle_end = time.monotonic() + self.control_dt
        t_command_target = t_cycle_end + self.control_dt
        self.robot.send_action(torch.from_numpy(action), t_command_target)
        precise_wait(t_cycle_end)

    def close(self) -> None:
        self.robot.disconnect()
        log_success("Robot disconnected successfully! 🎉")


def make_real_env(
    robot_type: str,
    dt: float | None,
    init_pose_arm: np.ndarray | list[float] | None = None,
    network_interface: str | None = None,
) -> UnitreeEnv:
    return UnitreeEnv(robot_type, dt, init_pose_arm, network_interface)
