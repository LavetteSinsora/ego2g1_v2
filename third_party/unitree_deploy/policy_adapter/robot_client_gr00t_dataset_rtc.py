"""
Script: robot_client_gr00t_dataset_rtc.py

Description:
    The main client script for deploying GR00T-1.6 Policies on Unitree Dataset Validation Environments WITH RTC.
    It implements a Robust Asynchronous Receding Horizon Control (RTC) loop seamlessly merging chunks:

    1.  **Robot Hardware Controller Thread (`actor_control`)**:
        -   Runs strictly without blocking to preserve temporal guarantees using interpolators against Datasets.
    2.  **GR00T Inference Thread (`get_actions`)**:
        -   Asynchronous loop polling ZMQ while measuring true network+inference latency automatically merging ActionQueue.

Usage:
    python3 unitree-deploy/scripts/robot_client_gr00t_dataset_rtc.py --dataset_repo_id lerobot/wbc_test
"""

import math
import os
import time
import traceback
from dataclasses import dataclass
from threading import Event, Lock, Thread

import logging_mp
import numpy as np
import torch
import tqdm
import tyro

from .groot_adapter.server_client import PolicyClient
from .rtc import ActionInterpolator, ActionQueue, LatencyTracker, RTCAttentionSchedule, RTCConfig

logger = logging_mp.getLogger(__name__)
logger.setLevel(logging_mp.DEBUG)  # Set to DEBUG to see detailed logs

from .robot_config import CAM_KEY, OBS_STATE_KEYS  # noqa: E402
from .robot_config import DATASET_ACTION_KEYS as ACTION_KEYS  # noqa: E402

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


class Gr00tZMQAdapter:
    """Wrapper for ZMQ Policy Client handling data formatting and communication."""

    def __init__(self, host="127.0.0.1", port=5555, robot_type="unitree_g1_dex1", action_horizon=16):
        self.client = PolicyClient(host=host, port=port)
        self.robot_type = robot_type
        self.action_horizon = action_horizon
        if not self.client.ping():
            raise RuntimeError("Failed to ping GR00T Server!")
        logger.info("Connected to GR00T Policy Server")

    def predict_action(self, observation, options=None):
        """Sends pre-formatted payload to server, and parses result into absolute action tensor."""
        # 1. Inference
        logger.debug(f"[Gr00tZMQAdapter] Requesting action from Policy Server for {self.robot_type}")
        action_dict, _ = self.client.get_action(observation, options=options)

        # 2. Parse Result
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

        # Ensure [H, W, C] configuration natively from datasets
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

    logger.debug("[prepare_observation] Video shapes: {k: v.shape for k, v in video_dict.items()}")
    logger.debug("[prepare_observation] State shapes: {k: v.shape for k, v in state_dict.items()}")

    return {
        "video": video_dict,
        "state": state_dict,
        "language": {"annotation.human.task_description": [[args.language_instruction]]},
    }


class RobotWrapper:
    """Thread-safe interface to the unitree underlying env."""

    def __init__(self, env):
        self.env = env
        self.lock = Lock()

    def get_observation(self):
        with self.lock:
            return self.env.get_observation()

    def step(self, action):
        with self.lock:
            return self.env.step(action)


def get_actions(
    client: Gr00tZMQAdapter,
    robot: RobotWrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    args: "Args",
):
    """
    Dedicated thread for grabbing predicted chunk sequences off network asynchronously safely.
    It guarantees action chunks arrive continuously allowing the interpolation mechanism to breathe.
    """
    try:
        logger.info("[GET_ACTIONS] Starting background request loop")

        latency_tracker = LatencyTracker()
        time_per_chunk = 1.0 / args.control_freq

        while not shutdown_event.is_set():
            # Demand loop size check
            if action_queue.qsize() <= args.action_queue_size_to_get_new_actions:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()

                logger.debug(
                    f"[GET_ACTIONS] Triggered inference. Current action_queue index: {action_index_before_inference}, qsize: {action_queue.qsize()}"
                )

                # Get robot camera and joint payload asynchronously
                obs_raw = robot.get_observation()
                obs = prepare_observation(args, obs_raw)

                # Predictive formulation of latency mimicking exact execution bounds needed inside RTCProcessor interpolation masks
                time_per_chunk = 1.0 / args.control_freq

                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_chunk)
                logger.debug(
                    f"[GET_ACTIONS_VARS] time_per_chunk: {time_per_chunk:.4f}s, inference_latency: {inference_latency:.4f}s, inference_delay: {inference_delay}"
                )

                # Extract frozen overlapping unexecuted predictions of the old chunk segment
                prev_actions = action_queue.get_left_over()
                prev_chunk = prev_actions.cpu().numpy() if prev_actions is not None else None
                logger.debug(
                    f"[GET_ACTIONS_VARS] prev_chunk: type={type(prev_chunk)}, shape={prev_chunk.shape if prev_chunk is not None else None}"
                )

                # Build dictionary targeting custom Gr00tPolicy bypass directly down into Model internals
                rtc_options = {
                    "inference_delay": int(inference_delay),
                    "prev_chunk_left_over": prev_chunk.tolist() if prev_chunk is not None else None,
                    "execution_horizon": int(args.execution_horizon),
                }
                logger.debug(
                    f"[GET_ACTIONS_VARS] rtc_options built with inference_delay={rtc_options['inference_delay']}, prev_chunk_left_over len={len(rtc_options['prev_chunk_left_over']) if rtc_options['prev_chunk_left_over'] else None}"
                )

                # Predict via ZMQ Server carrying manual chunk overlaps natively
                pred_actions = client.predict_action(obs, options={"rtc_options": rtc_options})
                logger.debug(
                    f"[GET_ACTIONS_VARS] pred_actions: type={type(pred_actions)}, shape={pred_actions.shape}"
                )

                # Tally reality delay
                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_chunk)
                latency_tracker.add(new_latency)
                logger.debug(f"[GET_ACTIONS_VARS] new_latency: {new_latency:.4f}s, new_delay: {new_delay}")

                logger.info(
                    f"[GET_ACTIONS] Inference complete. Latency: {new_latency * 1000:.2f}ms, Commanded Delay Chunks: {new_delay}, Received Chunk Shape: {pred_actions.shape}"
                )

                if args.action_queue_size_to_get_new_actions < args.action_horizon + new_delay:
                    logger.warning(
                        f"[GET_ACTIONS] action_queue thresholds relatively small compared to delays! latency: {new_latency * 1000:.2f}ms"
                    )

                # Queue merge: we use pred_actions directly because the wrapper runs dynamically server-side
                action_queue.merge(pred_actions, pred_actions, new_delay, action_index_before_inference)
            else:
                time.sleep(0.001)

        logger.info("[GET_ACTIONS] Shutting down.")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


def actor_control(
    robot: RobotWrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    args: "Args",
    actor_env,
):
    """
    Core manipulation thread which dictates the true frequency commands sent natively
    across dataset verification simulator safely using an ActionInterpolator.
    """
    try:
        logger.info("[ACTOR] Starting actor hardware bridge")

        action_count = 0
        interpolator = ActionInterpolator(multiplier=1)
        action_interval = interpolator.get_control_interval(args.control_freq)

        while not shutdown_event.is_set():
            start_time = time.perf_counter()

            if interpolator.needs_new_action():
                new_action = action_queue.get()
                if new_action is not None:
                    logger.debug("[ACTOR] ActionQueue provided new chunk for interpolation.")
                    interpolator.add(new_action.cpu())

            action = interpolator.get()
            if action is not None:
                action_np = action.cpu().numpy()

                if action_count % int(args.control_freq) == 0:
                    logger.debug(
                        f"[ACTOR] Step {action_count}: Executing action slice... {action_np[:4].flatten()}"
                    )

                # Datasets assume pure raw action representations correctly without native hardware concatenation
                if robot.step(action_np) is False:
                    logger.info("[ACTOR] Dataset Episode finished naturally.")
                    shutdown_event.set()
                    break
                action_count += 1

            dt_s = time.perf_counter() - start_time
            time.sleep(max(0, (action_interval - dt_s) - 0.0001))

        logger.info(f"[ACTOR] Closing. Total actions sent: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Hardware step exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


def run_eval(args: "Args") -> None:
    client = Gr00tZMQAdapter(
        host=args.host, port=args.port, robot_type=args.robot_type, action_horizon=args.action_horizon
    )

    from unitree_deploy.eval_dataset_env import DatasetEvalEnv

    env = DatasetEvalEnv(repo_id=args.dataset_repo_id)
    robot_wrapper = RobotWrapper(env)

    # Main Episode Loop Multi-Threaded Overlay
    for episode_idx in tqdm.tqdm(range(args.num_rollouts_planned), desc="Progress"):
        logger.info(f"\\n{'=' * 60}")
        logger.info(f"Episode {episode_idx + 1} / {args.num_rollouts_planned}")
        logger.info(f"{'=' * 60}")

        # Initialize lerobot's Real-Time Action Queue using config logic
        rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=args.execution_horizon,
            max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
        action_queue = ActionQueue(rtc_config)
        shutdown_event = Event()

        get_actions_thread = Thread(
            target=get_actions,
            args=(client, robot_wrapper, action_queue, shutdown_event, args),
            daemon=True,
            name=f"GetActions-{episode_idx}",
        )
        get_actions_thread.start()

        actor_thread = Thread(
            target=actor_control,
            args=(robot_wrapper, action_queue, shutdown_event, args, env),
            daemon=True,
            name=f"Actor-{episode_idx}",
        )
        actor_thread.start()

        try:
            while not shutdown_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected, shutting down properly...")
            shutdown_event.set()
            break
        finally:
            if not shutdown_event.is_set():
                shutdown_event.set()
            # Join local episodic threads
            if get_actions_thread.is_alive():
                get_actions_thread.join(timeout=2.0)
            if actor_thread.is_alive():
                actor_thread.join(timeout=2.0)

        logger.info(f"Episode {episode_idx + 1} finalized.\\n")

    logger.info("All Dataset Rollouts finished via Async RTC Loop.")


@dataclass
class Args:
    """GR00T Policy Execution Client for Unitree Robots [Dataset RTC Compatible]"""

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
    execution_horizon: int = 10
    """Action Execution Horizon chunk."""
    action_queue_size_to_get_new_actions: int = 30
    """Threshold index boundary to pull inference pipeline chunks over again."""
    language_instruction: str = "Pack black camera into box"
    """Task Instruction."""
    dataset_repo_id: str | None = None
    """Dataset ID for validation mode."""
    num_rollouts_planned: int = 1
    """Number of episodes to execute."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_eval(args)
