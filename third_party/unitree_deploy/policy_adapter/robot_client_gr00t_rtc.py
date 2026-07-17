"""
Script: robot_client_gr00t_rtc.py

Description:
    GR00T-1.6 Policy deployment client for Unitree Robots (G1/Z1) WITH RTC.
    Follows the canonical LeRobot RTC architecture (eval_with_real_robot.py):

    1.  **get_actions thread**: Captures obs, runs ZMQ inference with RTC options
        (inference_delay, prev_chunk_left_over, execution_horizon), and calls
        ActionQueue.merge() to manage chunk overlap.
    2.  **actor_control thread**: Pops from ActionQueue.get(), permutes for
        embodiment, and sends via env.step() which has built-in 33ms precise_wait.

Usage:
    python3 -m policy_adapter.robot_client_gr00t_rtc --robot_type unitree_g1_wbc_dex1
"""

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread

import logging
import logging_mp
import numpy as np
import torch
import tyro

from unitree_deploy.real_unitree_env import make_real_env

from .groot_adapter.server_client import PolicyClient
from .robot_config import CAM_KEY, INIT_POSE, OBS_STATE_KEYS
from .robot_config import REAL_ACTION_KEYS as ACTION_KEYS
from .rtc import ActionQueue, LatencyTracker, RTCAttentionSchedule, RTCConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging_mp.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(sh)

logger.info("GR00T RTC Client Logger Initialized.")

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


# ---------------------------------------------------------------------------
# ZMQ Adapter
# ---------------------------------------------------------------------------
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
        action_dict, _ = self.client.get_action(observation, options=options)

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

        return action_tensor[: self.action_horizon]


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------
def prepare_observation(args: "Args", obs: dict) -> dict:
    """Format UnitreeEnv observation directly into GR00T Policy Server payload format."""

    video_dict = {}
    hardware_images = obs["images"]
    cam_mapping = CAM_KEY.get(args.robot_type, CAM_KEY["unitree_g1_dex1"])
    for model_key, hardware_key in cam_mapping.items():
        raw_img = hardware_images.get(hardware_key, hardware_images.get(model_key))
        if raw_img is None:
            continue
        raw_img = raw_img.numpy() if torch.is_tensor(raw_img) else raw_img
        video_dict[model_key] = raw_img[None, None, ...]

    qpos = obs["qpos"]
    qpos = (qpos.numpy() if torch.is_tensor(qpos) else qpos).astype(np.float32)[None, None, ...]

    mapping = OBS_STATE_KEYS.get(args.robot_type, OBS_STATE_KEYS["unitree_g1_dex1"])
    state_dict = {key: qpos[..., slc] for key, slc in mapping.items()}

    return {
        "video": video_dict,
        "state": state_dict,
        "language": {"annotation.human.task_description": [[args.language_instruction]]},
    }


# ---------------------------------------------------------------------------
# Robot Wrapper (thread-safe, matches LeRobot pattern)
# ---------------------------------------------------------------------------
class RobotWrapper:
    """Thread-safe interface to the Unitree env.
    
    NOTE: env.step() has built-in precise_wait(33ms), so the lock is held
    for the full cycle. This is acceptable because get_observation() in the
    inference thread can tolerate ~33ms contention.
    """

    def __init__(self, env):
        self.env = env
        self.lock = Lock()

    def get_observation(self):
        with self.lock:
            return self.env.get_observation()

    def step(self, action):
        with self.lock:
            self.env.step(action)


# ---------------------------------------------------------------------------
# get_actions thread (follows LeRobot pattern)
# ---------------------------------------------------------------------------
def get_actions(
    client: Gr00tZMQAdapter,
    robot: RobotWrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    args: "Args",
):
    """Background thread for requesting action chunks via ZMQ with RTC.
    
    Follows the canonical LeRobot eval_with_real_robot.py pattern:
    1. Check qsize <= threshold
    2. Record action_index_before_inference
    3. Get prev_chunk_left_over from ActionQueue
    4. Estimate inference_delay from latency_tracker.max()
    5. Capture observation
    6. Run inference with RTC options
    7. action_queue.merge(original, processed, delay, action_index)
    """
    try:
        logger.info("[GET_ACTIONS] Starting background inference loop")

        latency_tracker = LatencyTracker()
        time_per_step = 1.0 / args.control_freq
        inference_count = 0

        while not shutdown_event.is_set():
            if action_queue.qsize() <= args.action_queue_size_to_get_new_actions:
                current_time = time.perf_counter()
                action_index_before_inference = action_queue.get_action_index()

                # Get prev_chunk_left_over for RTC guidance
                prev_actions = action_queue.get_left_over()
                prev_chunk = prev_actions.cpu().numpy() if prev_actions is not None else None

                # Estimate inference delay using historical max (matches LeRobot)
                inference_latency = latency_tracker.max()
                inference_delay = math.ceil(inference_latency / time_per_step)

                # Capture observation (thread-safe via RobotWrapper lock)
                obs_raw = robot.get_observation()
                obs = prepare_observation(args, obs_raw)

                # Build RTC options for server
                rtc_options = {
                    "inference_delay": int(inference_delay),
                    "prev_chunk_left_over": prev_chunk.tolist() if prev_chunk is not None else None,
                    "execution_horizon": int(args.execution_horizon),
                }

                # Run ZMQ inference
                pred_actions = client.predict_action(obs, options={"rtc_options": rtc_options})

                # Measure actual latency
                new_latency = time.perf_counter() - current_time
                new_delay = math.ceil(new_latency / time_per_step)
                latency_tracker.add(new_latency)
                inference_count += 1

                # Merge into ActionQueue (let ActionQueue handle RTC chunk replacement)
                action_queue.merge(pred_actions, pred_actions, new_delay, action_index_before_inference)

                qsize = action_queue.qsize()
                p95 = latency_tracker.p95() * 1000 if latency_tracker.p95() else 0
                hist_max = latency_tracker.max() * 1000 if latency_tracker.max() else 0
                s_max = latency_tracker.sliding_max() * 1000 if latency_tracker.sliding_max() else 0
                prev_left = len(prev_chunk) if prev_chunk is not None else 0

                logger.info(
                    f"[RTC] #{inference_count} | "
                    f"qsize: {qsize} | "
                    f"inf_delay: {inference_delay}, new_delay: {new_delay} | "
                    f"latency: {new_latency*1000:.0f}ms (p95: {p95:.0f}ms, max: {hist_max:.0f}ms, s_max: {s_max:.0f}ms) | "
                    f"prev_left: {prev_left}, idx_before: {action_index_before_inference}"
                )

                if args.action_queue_size_to_get_new_actions < args.execution_horizon + new_delay:
                    logger.warning(
                        f"[GET_ACTIONS] threshold ({args.action_queue_size_to_get_new_actions}) < "
                        f"execution_horizon+delay ({args.execution_horizon + new_delay})"
                    )
            else:
                time.sleep(0.001)

        logger.info(f"[GET_ACTIONS] Shutting down. Total inferences: {inference_count}")
    except Exception as e:
        logger.error(f"[GET_ACTIONS] Exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


# ---------------------------------------------------------------------------
# actor_control thread (follows LeRobot pattern)
# ---------------------------------------------------------------------------
def actor_control(
    robot: RobotWrapper,
    action_queue: ActionQueue,
    shutdown_event: Event,
    args: "Args",
):
    """Thread for executing actions on the robot at control frequency.
    
    NOTE: env.step() has built-in precise_wait(33ms), so NO manual
    time.sleep is needed. This is the critical difference from the
    LeRobot reference (which needs manual sleep because robot.send_action
    does not block).
    """
    try:
        logger.info("[ACTOR] Starting actor control thread")

        # Init pose
        if args.init_pose_file and Path(args.init_pose_file).exists():
            with open(args.init_pose_file) as f:
                data = json.load(f)
                if "mean_init_qpos" in data:
                    INIT_POSE[args.robot_type] = np.array(data["mean_init_qpos"], dtype=np.float32)

        logger.info("[ACTOR] Moving to INIT_POSE...")
        robot.step(INIT_POSE[args.robot_type])
        time.sleep(2.0)
        logger.info("[ACTOR] INIT_POSE reached, starting control loop")

        action_count = 0

        while not shutdown_event.is_set():
            # Pop next action from ActionQueue
            new_action = action_queue.get()

            if new_action is not None:
                action_np = new_action.cpu().numpy()

                # Embodiment-specific permutation
                if args.robot_type == "unitree_g1_wbc_dex1" and action_np.shape[-1] == 23:
                    action_np = np.concatenate([action_np[:14], action_np[-7:], action_np[14:16]])

                # env.step() sends action AND waits 33ms via precise_wait
                robot.step(action_np)
                action_count += 1

                if action_count % (int(args.control_freq) * 5) == 0:
                    logger.info(f"[ACTOR] actions={action_count}, queue={action_queue.qsize()}")
            else:
                # No action available yet — sleep one control period
                time.sleep(1.0 / args.control_freq)

        logger.info(f"[ACTOR] Shutting down. Total actions: {action_count}")
    except Exception as e:
        logger.error(f"[ACTOR] Exception: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_eval(args: "Args") -> None:
    logger.info(f"Starting GR00T RTC Eval with robot_type: {args.robot_type}")
    logger.info(
        f"Config: control_freq={args.control_freq}Hz, action_horizon={args.action_horizon}, "
        f"execution_horizon={args.execution_horizon}, threshold={args.action_queue_size_to_get_new_actions}"
    )

    # 1. Connect to ZMQ inference server
    client = Gr00tZMQAdapter(
        host=args.host, port=args.port, robot_type=args.robot_type, action_horizon=args.action_horizon
    )

    # 2. Initialize hardware environment
    env = make_real_env(
        robot_type=args.robot_type, dt=1 / args.control_freq, network_interface=args.network_interface
    )
    env.connect()
    robot = RobotWrapper(env)

    # 3. Initialize ActionQueue with RTC config (matches LeRobot)
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    action_queue = ActionQueue(rtc_config)
    shutdown_event = Event()

    # 4. Start threads (matches LeRobot two-thread pattern)
    get_actions_thread = Thread(
        target=get_actions,
        args=(client, robot, action_queue, shutdown_event, args),
        daemon=True,
        name="GetActions",
    )
    get_actions_thread.start()

    actor_thread = Thread(
        target=actor_control,
        args=(robot, action_queue, shutdown_event, args),
        daemon=True,
        name="Actor",
    )
    actor_thread.start()

    # 5. Main thread monitors
    try:
        while not shutdown_event.is_set():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Ctrl+C detected, shutting down...")
    finally:
        shutdown_event.set()
        if get_actions_thread.is_alive():
            get_actions_thread.join(timeout=2.0)
        if actor_thread.is_alive():
            actor_thread.join(timeout=2.0)
        env.close()
        logger.info("Environment closed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class Args:
    """GR00T Policy Execution Client for Unitree Robots [RTC Compatible]"""

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
    """Queue size threshold to trigger new inference (LeRobot default: 30)."""
    language_instruction: str = "Pack black camera into box"
    """Task Instruction."""
    init_pose_file: str | None = None
    """JSON file to override INIT_POSE."""
    network_interface: str | None = None
    """Network interface to use."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_eval(args)
