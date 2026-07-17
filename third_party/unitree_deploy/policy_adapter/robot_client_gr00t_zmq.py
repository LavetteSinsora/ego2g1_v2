"""
Script: robot_client_gr00t_zmq.py

Description:
    GR00T-1.6 Policy deployment client for the full-ZMQ robot stack
    (unitree_g1_wbc_dex1_zmq).  Both the arm (g1_wbc_arm_server.py) and
    the grippers (gripper_server.py) run on the robot; this script runs
    entirely on the PC side.

    Control flow:
      - Receding-horizon action queue (RHC)
      - Synchronous inference when queue is empty
      - precise_wait for fixed-rate control
      - Per-phase latency logging every LOG_EVERY steps

Usage:
    python3 -m policy_adapter.robot_client_gr00t_zmq \\
        --robot_ip 192.168.123.164 \\
        --host 10.3.0.241 \\
        --port 14095 \\
        --language_instruction "your task"
"""

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import logging_mp
import numpy as np
import torch
import tyro

from unitree_deploy.real_unitree_env import make_real_env
from unitree_deploy.robot.robot_configs import G1Wbc_Dex1Zmq_Imageclient_RobotConfig
from unitree_deploy.robot.robot_utils import make_robot_from_config

from .groot_adapter.server_client import PolicyClient

logger = logging_mp.getLogger(__name__)
logger.setLevel(logging_mp.INFO)

import os  # noqa: E402

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

# ---------------------------------------------------------------------------
# Robot-type constants (ZMQ variant shares the same obs/action layout as wbc)
# ---------------------------------------------------------------------------

ROBOT_TYPE = "unitree_g1_wbc_dex1_zmq"

from .robot_config import (  # noqa: E402
    CAM_KEY,
    REAL_ACTION_KEYS,
)
from .robot_config import (  # noqa: E402
    INIT_POSE as _INIT_POSE,
)
from .robot_config import (  # noqa: E402
    OBS_STATE_KEYS as _OBS_STATE_KEYS,
)

INIT_POSE = _INIT_POSE["unitree_g1_wbc_dex1"]
CAM_MAPPING = CAM_KEY["unitree_g1_wbc_dex1"]
OBS_STATE_KEYS = _OBS_STATE_KEYS["unitree_g1_wbc_dex1"]

ACTION_DIM = REAL_ACTION_KEYS["unitree_g1_wbc_dex1"]["dim"]
ACTION_KEYS = REAL_ACTION_KEYS["unitree_g1_wbc_dex1"]["keys"]


# ---------------------------------------------------------------------------
# GR00T ZMQ adapter
# ---------------------------------------------------------------------------


class Gr00tZMQAdapter:
    def __init__(self, host: str, port: int, action_horizon: int):
        self.client = PolicyClient(host=host, port=port)
        self.action_horizon = action_horizon
        if not self.client.ping():
            raise RuntimeError("Failed to ping GR00T Server!")
        logger.info("Connected to GR00T Policy Server")

    def predict_action(self, observation) -> torch.Tensor:
        if not getattr(self, "_obs_logged", False):
            self._obs_logged = True
            logger.info("[Observation detail] ---")
            for section, data in observation.items():
                if isinstance(data, dict):
                    for k, v in data.items():
                        arr = np.asarray(v) if not isinstance(v, np.ndarray) else v
                        logger.info(
                            f"  [{section}] {k}: shape={arr.shape} dtype={arr.dtype} "
                            f"min={arr.min():.4f} max={arr.max():.4f}"
                        )
                else:
                    logger.info(f"  [{section}]: {data}")
            logger.info("[Observation detail] ---")
        action_dict, _ = self.client.get_action(observation)
        action = action_dict.get("action", action_dict)

        ref_arr = np.array(action.get("left_arm", [[[0]]]), dtype=np.float32)
        shape_prefix = ref_arr.shape[:-1]

        raw_action = np.zeros((*shape_prefix, ACTION_DIM), dtype=np.float32)
        for key, slc in ACTION_KEYS.items():
            raw_action[..., slc] = action.get(key, 0)

        tensor = torch.from_numpy(raw_action)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor[: self.action_horizon].cpu()


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------


def prepare_observation(obs: dict, language_instruction: str) -> dict:
    # Images
    video_dict = {}
    hardware_images = obs["images"]
    for model_key, hw_key in CAM_MAPPING.items():
        raw = hardware_images.get(hw_key)
        if raw is None:
            raw = hardware_images.get(model_key)
        if raw is None:
            continue
        raw = raw.numpy() if torch.is_tensor(raw) else raw
        video_dict[model_key] = raw[None, None, ...]

    # State
    qpos = obs["qpos"]
    qpos = (qpos.numpy() if torch.is_tensor(qpos) else qpos).astype(np.float32)[None, None, ...]
    state_dict = {key: qpos[..., slc] for key, slc in OBS_STATE_KEYS.items()}

    return {
        "video": video_dict,
        "state": state_dict,
        "language": {"annotation.human.task_description": [[language_instruction]]},
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def run_eval(args: "Args") -> None:
    client = Gr00tZMQAdapter(host=args.host, port=args.port, action_horizon=args.action_horizon)

    # Build config with robot_ip directly — avoids network_interface confusion
    config = G1Wbc_Dex1Zmq_Imageclient_RobotConfig(robot_ip=args.robot_ip)
    robot = make_robot_from_config(config)

    env = make_real_env(robot_type=ROBOT_TYPE, dt=1 / args.control_freq)
    env.robot = robot  # inject ZMQ robot

    env.connect()

    # Override init pose from file if provided
    init_pose = INIT_POSE.copy()
    if args.init_pose_file and Path(args.init_pose_file).exists():
        with open(args.init_pose_file) as f:
            data = json.load(f)
            if "mean_init_qpos" in data:
                init_pose = np.array(data["mean_init_qpos"], dtype=np.float32)

    env.step(init_pose)
    time.sleep(2.0)

    # -----------------------------------------------------------------------
    # Latency infrastructure
    # -----------------------------------------------------------------------
    # Phases measured every step (unit: ms):
    #   obs_capture     – env.get_observation()  (camera DMA + ZMQ state recv)
    #   obs_format      – prepare_observation()  (numpy reshape, dict build)
    #   inference       – GR00T ZMQ round-trip   (only on queue-refill steps)
    #   action_assemble – slice + np.concatenate
    #   env_step        – robot command send + precise_wait (≈ control_dt)
    #   loop_total      – wall time for the full step
    lat_keys = ("obs_capture", "obs_format", "inference", "action_assemble", "env_step", "loop_total")
    lat: dict[str, list[float]] = {k: [] for k in lat_keys}

    def _ms(t0: float) -> float:
        """Elapsed milliseconds since t0 (perf_counter)."""
        return (time.perf_counter() - t0) * 1_000

    def _log_stats(step: int) -> None:
        """Log mean/max for each phase and clear accumulators."""
        parts = []
        for k in lat_keys:
            vals = lat[k]
            if not vals:
                continue
            mean_ms = sum(vals) / len(vals)
            max_ms = max(vals)
            parts.append(f"{k}: mean={mean_ms:.1f} max={max_ms:.1f} ms")
            lat[k].clear()
        logger.info(f"[Latency @ step {step:5d}]  " + " | ".join(parts))

    try:
        t = 0
        action_queue: deque = deque()

        while True:
            t_step = time.perf_counter()

            # 2. Inference (only when queue empty)
            if len(action_queue) == 0:
                # 1. Capture + format observation
                t0 = time.perf_counter()
                try:
                    raw_obs = env.get_observation()
                    obs = prepare_observation(raw_obs, args.language_instruction)
                except Exception as e:
                    logger.error(f"[get_observation] {e}", exc_info=True)
                    raise
                logger.info(f"[Obs] Step: {t}, Latency: {(time.perf_counter() - t0) * 1000:.2f} ms")

                t0 = time.perf_counter()
                try:
                    pred_actions = client.predict_action(obs)
                except Exception as e:
                    logger.error(f"[predict_action] {e}", exc_info=True)
                    raise
                logger.info(f"[Inference] Step: {t}, Latency: {(time.perf_counter() - t0) * 1000:.2f} ms")
                for i in range(pred_actions.shape[0]):
                    action_queue.append(pred_actions[i].cpu().numpy())

            # 3. Action assembly + send
            action = action_queue.popleft()
            action = np.concatenate([action[:14], action[16:23], action[14:16]])

            if env.step(action) is False:
                return

            logger.info(f"[Step] {t}, Total: {(time.perf_counter() - t_step) * 1000:.2f} ms")
            t += 1

    except KeyboardInterrupt:
        logger.info("Ctrl+C — stopping.")
    except Exception as e:
        logger.error(f"[run_eval] fatal error: {e}", exc_info=True)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """GR00T Policy ZMQ Client — full ZMQ robot stack (arm + gripper)."""

    robot_ip: str = "192.168.123.164"
    """IP address of the robot running g1_wbc_arm_server.py and gripper_server.py."""
    host: str = "127.0.0.1"
    """Inference Server IP."""
    port: int = 5555
    """Inference Server ZMQ Port."""
    control_freq: float = 30.0
    """Control Loop Frequency (Hz)."""
    action_horizon: int = 16
    """Model Prediction Horizon (T)."""
    language_instruction: str = ""
    """Task instruction string."""
    init_pose_file: str | None = None
    """Optional JSON file to override INIT_POSE (expects key 'mean_init_qpos')."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_eval(args)
