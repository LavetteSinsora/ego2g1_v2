"""
Script: robot_client_gr00t_17_no_image.py

Variant of robot_client_gr00t_17.py for the architecture where the GR00T policy
server subscribes directly to the robot's image_server (ZMQ PUB). This client
ONLY sends state + language; the server merges the latest camera frames before
inference. Eliminates the duplicate image hop over WiFi.

Pair with:
    Unitree-Issac-GR00T/gr00t/eval/run_gr00t_server_direct_image.py
    robot_type: unitree_g1_dex1_no_image
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import logging_mp
import numpy as np
import torch
import tyro

from unitree_deploy.real_unitree_env import make_real_env

from .groot_adapter.server_client_17 import PolicyClient

logger = logging_mp.getLogger(__name__)
logger.setLevel(logging_mp.INFO)

from .robot_config import INIT_POSE, OBS_STATE_KEYS  # noqa: E402
from .robot_config import REAL_ACTION_KEYS as ACTION_KEYS  # noqa: E402

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


class Gr00tZMQAdapter:
    """Wrapper for ZMQ Policy Client. Sends state-only observations; the server
    injects images from its own ZMQ subscription to the robot image_server."""

    def __init__(
        self,
        host="127.0.0.1",
        port=5555,
        robot_type="unitree_g1_wbc_dex1_no_image",
        action_horizon=16,
    ):
        self.client = PolicyClient(host=host, port=port)
        self.robot_type = robot_type
        self.action_horizon = action_horizon
        if not self.client.ping():
            raise RuntimeError("Failed to ping GR00T Server!")
        logger.info("Connected to GR00T Policy Server (direct-image mode)")

    def predict_action(self, observation):
        action_dict, _ = self.client.get_action(observation)

        action = action_dict.get("action", action_dict)

        ref_arr = np.array(action.get("left_arm", [[[0]]]), dtype=np.float32)
        shape_prefix = ref_arr.shape[:-1]

        if self.robot_type in ACTION_KEYS:
            config = ACTION_KEYS[self.robot_type]
            raw_action = np.zeros((*shape_prefix, config["dim"]), dtype=np.float32)
            for key, slc in config["keys"].items():
                raw_action[..., slc] = action.get(key, 0)
        else:
            raw_action = np.array(list(action.values())[0], dtype=np.float32)

        action_tensor = torch.from_numpy(raw_action)
        if action_tensor.ndim == 3 and action_tensor.shape[0] == 1:
            action_tensor = action_tensor.squeeze(0)

        return action_tensor[: self.action_horizon].cpu()


def prepare_observation(args: "Args", obs: dict) -> dict:
    """Format observation as state + language only.

    Images are NOT included here; the policy server pulls them directly from
    the robot's image_server via its own ZMQ subscription.
    """
    qpos = obs["qpos"]
    qpos = (qpos.numpy() if torch.is_tensor(qpos) else qpos).astype(np.float32)[None, None, ...]

    mapping = OBS_STATE_KEYS.get(args.robot_type, OBS_STATE_KEYS["unitree_g1_wbc_dex1"])
    state_dict = {key: qpos[..., slc] for key, slc in mapping.items()}

    return {
        "state": state_dict,
        "language": {"annotation.human.task_description": [[args.language_instruction]]},
    }


def run_eval(args: "Args") -> None:
    client = Gr00tZMQAdapter(host=args.host, port=args.port, robot_type=args.robot_type)

    env = make_real_env(
        robot_type=args.robot_type,
        dt=1 / args.control_freq,
        network_interface=args.network_interface,
    )
    env.connect()

    init_pose = INIT_POSE.get(args.robot_type, INIT_POSE["unitree_g1_wbc_dex1"])
    if args.init_pose_file and Path(args.init_pose_file).exists():
        with open(args.init_pose_file) as f:
            data = json.load(f)
            if "mean_init_qpos" in data:
                init_pose = np.array(data["mean_init_qpos"], dtype=np.float32)
    env.step(init_pose)
    time.sleep(2.0)

    try:
        t = 0
        action_queue = deque()

        while True:
            obs = prepare_observation(args, env.get_observation())
            if len(action_queue) == 0:
                t0 = time.perf_counter()
                pred_actions = client.predict_action(obs)

                t1 = time.perf_counter()
                logger.info(f"[Inference] Step: {t}, Latency: {(t1 - t0) * 1000:.2f} ms")

                for i in range(pred_actions.shape[0]):
                    action_queue.append(pred_actions[i].cpu().numpy())

            action = action_queue.popleft()
            action = np.concatenate([action[:14], action[-7:], action[14:16]])
            action[18:20] = [0.0, 0.0]
            print(action)
            if env.step(action) is False:
                return
            t += 1

    finally:
        env.close()


@dataclass
class Args:
    """GR00T Policy Execution Client for Unitree Robots — state-only variant."""

    robot_type: str = "unitree_g1_wbc_dex1_no_image"
    """Robot embodiment. Defaults to the wbc no-image variant; the policy server
    pulls images directly from the robot's image_server."""
    host: str = "127.0.0.1"
    """Inference Server IP."""
    port: int = 5555
    """Inference Server ZMQ Port."""
    control_freq: float = 30.0
    """Control Loop Frequency (Hz)."""
    action_horizon: int = 16
    """Model Prediction Horizon (T)."""
    language_instruction: str = "Pack black camera into box"
    """Task Instruction."""
    init_pose_file: str | None = None
    """JSON file to override INIT_POSE."""
    network_interface: str | None = None
    """Network interface to use."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_eval(args)
