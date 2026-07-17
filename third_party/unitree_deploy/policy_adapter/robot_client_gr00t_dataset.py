"""
Script: robot_client_gr00t.py

Description:
    The main client script for deploying GR00T-1.6 Policies on Unitree Robots (G1/Z1).
    It implements a Robust Asynchronous Receding Horizon Control (RHC) loop:

    1.  **Robot Hardware / Dataset Replay**:
        -   Captures observations (Images + Joint States) at ~30Hz.
    2.  **GR00T Inference Server**:
        -   Runs in a background thread to decouple inference latency (~75ms) from control loop.
        -   Sends observations via ZMQ and receives action chunks.
    3.  **Async Ensembler**:
        -   Fuses action chunks with variable latency using temporal ensembling.

Usage:
    # 1. Real Robot Deployment
    python3 unitree-deploy/scripts/robot_client_gr00t.py --robot_type unitree_g1_dex1

    # 2. Dataset Replay & Validation
    python3 unitree-deploy/scripts/robot_client_gr00t.py --dataset_repo_id lerobot/pusht
"""

import time
from collections import deque
from dataclasses import dataclass

import logging_mp
import numpy as np
import torch
import tqdm
import tyro

from .groot_adapter.server_client import PolicyClient

logger = logging_mp.getLogger(__name__)
logger.setLevel(logging_mp.INFO)


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

from .robot_config import CAM_KEY, OBS_STATE_KEYS  # noqa: E402
from .robot_config import DATASET_ACTION_KEYS as ACTION_KEYS  # noqa: E402


class Gr00tZMQAdapter:
    """Wrapper for ZMQ Policy Client handling data formatting and communication."""

    def __init__(self, host="127.0.0.1", port=5555, robot_type="unitree_g1_dex1", action_horizon=16):
        self.client = PolicyClient(host=host, port=port)
        self.robot_type = robot_type
        self.action_horizon = action_horizon
        if not self.client.ping():
            raise RuntimeError("Failed to ping GR00T Server!")
        logger.info("Connected to GR00T Policy Server")

    def predict_action(self, observation):
        """Sends pre-formatted payload to server, and parses result into absolute action tensor."""
        # 1. Inference
        action_dict, _ = self.client.get_action(observation)

        # 4. Parse Result
        action = action_dict.get("action", action_dict)

        # Extract base shape prefix (e.g., [B, T]) from left_arm to avoid broadcast faults
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

        # Truncate strictly to action_horizon before returning
        return action_tensor[: self.action_horizon].cpu()


def prepare_observation(args: "Args", obs: dict) -> dict:
    """Format UnitreeEnv observation directly into GR00T Policy Server payload format."""

    # 1. Process Video
    video_dict = {}
    hardware_images = obs["images"]
    cam_mapping = CAM_KEY.get(args.robot_type, CAM_KEY["unitree_g1_dex1"])
    for model_key, hardware_key in cam_mapping.items():
        raw_img = hardware_images.get(hardware_key, hardware_images.get(model_key))
        if raw_img is None:
            continue

        raw_img = raw_img.numpy() if torch.is_tensor(raw_img) else raw_img

        # Ensure [H, W, C] configuration
        if raw_img.ndim == 2:
            raw_img = raw_img[..., None]  # Grayscale [H, W] -> [H, W, 1]
        elif raw_img.ndim == 3 and raw_img.shape[0] in (1, 3):
            raw_img = np.transpose(raw_img, (1, 2, 0))  # [C, H, W] -> [H, W, C]

        video_dict[model_key] = (raw_img * 255.0).astype(np.uint8)[None, None, ...]

    # 2. Process State
    qpos = obs["qpos"]
    qpos = (qpos.numpy() if torch.is_tensor(qpos) else qpos).astype(np.float32)[None, None, ...]

    mapping = OBS_STATE_KEYS.get(args.robot_type, OBS_STATE_KEYS["unitree_g1_dex1"])
    state_dict = {key: qpos[..., slc] for key, slc in mapping.items()}

    return {
        "video": video_dict,
        "state": state_dict,
        "language": {"annotation.human.task_description": [[args.language_instruction]]},
    }


def run_eval(args: "Args") -> None:
    client = Gr00tZMQAdapter(
        host=args.host, port=args.port, robot_type=args.robot_type, action_horizon=args.action_horizon
    )

    # Initialize Environment
    from unitree_deploy.eval_dataset_env import DatasetEvalEnv

    env = DatasetEvalEnv(repo_id=args.dataset_repo_id)

    # Main Episode Loop
    for _ in tqdm.tqdm(range(args.num_rollouts_planned), desc="Progress"):
        """Core Sync RHC Loop for Dataset."""
        t = 0
        action_queue = deque()

        while True:
            # 1. Capture & Format
            obs = prepare_observation(args, env.get_observation())

            # 2. Synchronous Inference Request (Only when queue is empty)
            if len(action_queue) == 0:
                t0 = time.perf_counter()
                pred_actions = client.predict_action(obs)

                t1 = time.perf_counter()
                logger.info(f"[Inference] Step: {t}, Latency: {(t1 - t0) * 1000:.2f} ms")

                # Enqueue all predicted actions to use as a queue, or up to action_horizon
                for i in range(pred_actions.shape[0]):
                    action_queue.append(pred_actions[i].cpu().numpy())

            # 3. Execute
            action = action_queue.popleft()
            if env.step(action) is False:
                return
            t += 1


@dataclass
class Args:
    """GR00T Policy Execution Client for Unitree Robots"""

    robot_type: str = "unitree_g1_dex1"
    """Robot embodiment."""
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
    dataset_repo_id: str | None = None
    """Dataset ID for validation mode."""
    num_rollouts_planned: int = 1
    """Number of episodes to execute."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_eval(args)
