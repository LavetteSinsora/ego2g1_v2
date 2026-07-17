"""
Script: robot_client_gr00t_single_thread.py

Description:
    Single-threaded GR00T-1.6 Policy deployment client for Unitree Robots (G1/Z1) WITH RTC.
    Eliminates thread competition by running inference and execution in a single loop.

    Architecture:
        ┌────────────────────────────────────────┐
        │         Main Loop (30Hz cycle)         │
        │                                        │
        │  1. Queue low? → obs → infer → merge   │
        │  2. Pop action from ActionQueue         │
        │  3. Permute for robot embodiment        │
        │  4. env.step(action)                    │
        └────────────────────────────────────────┘

    Key difference from multi-threaded version:
    - No locks, no threads, no RobotWrapper
    - Inference blocks the loop (~130ms), compensated by RTC inference_delay
    - Deterministic execution order: observe → infer → act

Usage:
    python3 -m policy_adapter.robot_client_gr00t_single_thread --robot_type unitree_g1_wbc_dex1
"""

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

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

logger.info("GR00T Single-Thread RTC Client Logger Initialized.")

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


# ---------------------------------------------------------------------------
# ZMQ Adapter (reused from multi-threaded version)
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
# Observation formatting (reused from multi-threaded version)
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
# Action permutation (embodiment-specific)
# ---------------------------------------------------------------------------
def permute_action(action_np: np.ndarray, robot_type: str) -> np.ndarray:
    """Reorder action dimensions from unified model output to robot-specific layout."""
    if robot_type == "unitree_g1_wbc_dex1" and action_np.shape[-1] == 23:
        # G1 WBC: [Arms(14), Commands(7), Hands(2)]
        return np.concatenate([action_np[:14], action_np[-7:], action_np[14:16]])
    # Other embodiments: pass through
    return action_np


# ---------------------------------------------------------------------------
# Main single-threaded control loop with RTC
# ---------------------------------------------------------------------------
def run_eval(args: "Args") -> None:
    logger.info(f"Starting GR00T Single-Thread RTC Eval with robot_type: {args.robot_type}")

    # 1. Connect to ZMQ inference server
    client = Gr00tZMQAdapter(
        host=args.host, port=args.port, robot_type=args.robot_type, action_horizon=args.action_horizon
    )

    # 2. Initialize hardware environment (no wrapper, direct access)
    env = make_real_env(
        robot_type=args.robot_type, dt=1 / args.control_freq, network_interface=args.network_interface
    )
    env.connect()

    # 3. Initialize RTC ActionQueue
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=10.0,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
    )
    action_queue = ActionQueue(rtc_config)
    latency_tracker = LatencyTracker()

    # 4. Load and execute init pose
    if args.init_pose_file and Path(args.init_pose_file).exists():
        with open(args.init_pose_file) as f:
            data = json.load(f)
            if "mean_init_qpos" in data:
                INIT_POSE[args.robot_type] = np.array(data["mean_init_qpos"], dtype=np.float32)

    logger.info("[MAIN] Moving to INIT_POSE...")
    env.step(INIT_POSE[args.robot_type])
    time.sleep(2.0)
    logger.info("[MAIN] INIT_POSE reached, starting control loop")

    # 5. Control loop state
    action_count = 0
    inference_count = 0
    time_per_step = 1.0 / args.control_freq
    loop_times = []  # Track loop cycle times for frequency monitoring

    try:
        logger.info(
            f"[MAIN] Config: control_freq={args.control_freq}Hz, "
            f"action_horizon={args.action_horizon}, execution_horizon={args.execution_horizon}, "
            f"queue_threshold={args.action_queue_size_to_get_new_actions}"
        )

        while True:
            loop_start = time.perf_counter()

            # ── Step A: Check if we need new actions ──────────────────────
            current_qsize = action_queue.qsize()
            if current_qsize <= args.action_queue_size_to_get_new_actions:
                logger.debug(f"[INFER] Triggered: qsize={current_qsize} <= threshold={args.action_queue_size_to_get_new_actions}")

                # A1. Capture observation
                t_obs = time.perf_counter()
                obs_raw = env.get_observation()
                obs = prepare_observation(args, obs_raw)
                obs_time = (time.perf_counter() - t_obs) * 1000
                logger.debug(f"[OBS] Captured in {obs_time:.1f}ms")

                # A2. Estimate inference delay for RTC alignment
                inference_latency = latency_tracker.p95()
                inference_delay = math.ceil(inference_latency / time_per_step)

                # A3. Extract leftover actions for RTC guidance
                prev_actions = action_queue.get_left_over()
                prev_chunk = prev_actions.cpu().numpy() if prev_actions is not None else None
                prev_left_count = len(prev_chunk) if prev_chunk is not None else 0

                # A4. Build RTC options
                rtc_options = {
                    "inference_delay": int(inference_delay),
                    "prev_chunk_left_over": prev_chunk.tolist() if prev_chunk is not None else None,
                    "execution_horizon": int(args.execution_horizon),
                }
                logger.debug(
                    f"[RTC_OPT] inference_delay={inference_delay}, "
                    f"execution_horizon={args.execution_horizon}, "
                    f"prev_chunk_steps={prev_left_count}"
                )

                # A5. Run inference (this blocks ~130ms)
                inference_start = time.perf_counter()
                pred_actions = client.predict_action(obs, options={"rtc_options": rtc_options})
                new_latency = time.perf_counter() - inference_start
                new_delay = math.ceil(new_latency / time_per_step)
                latency_tracker.add(new_latency)

                # A6. Relative skip: server already aligned at t+inference_delay,
                #     only skip the extra time beyond that estimate
                skip_delay = max(0, new_delay - inference_delay)

                logger.debug(
                    f"[INFER] Completed in {new_latency*1000:.1f}ms, "
                    f"pred_shape={list(pred_actions.shape)}, "
                    f"new_delay={new_delay}, skip={skip_delay}, drift={new_delay - inference_delay}"
                )

                # A7. Merge into queue (no action_index_before_inference to prevent
                #     ActionQueue from overriding our alignment)
                action_queue.merge(pred_actions, pred_actions, skip_delay, action_index_before_inference=None)

                inference_count += 1
                qsize_after = action_queue.qsize()

                p95 = latency_tracker.p95() * 1000 if latency_tracker.p95() else 0
                s_max = latency_tracker.sliding_max() * 1000 if latency_tracker.sliding_max() else 0
                h_max = latency_tracker.max() * 1000 if latency_tracker.max() else 0
                total_cycle = (time.perf_counter() - loop_start) * 1000

                logger.info(
                    f"[RTC] #{inference_count} | "
                    f"qsize: {current_qsize}→{qsize_after} | "
                    f"obs: {obs_time:.0f}ms, infer: {new_latency*1000:.0f}ms, total: {total_cycle:.0f}ms | "
                    f"delay: inf={inference_delay} new={new_delay} skip={skip_delay} | "
                    f"latency p95={p95:.0f}ms s_max={s_max:.0f}ms h_max={h_max:.0f}ms | "
                    f"prev_left={prev_left_count}"
                )

            # ── Step B: Execute one action from the queue ─────────────────
            action = action_queue.get()
            if action is not None:
                action_np = action.cpu().numpy()
                orig_shape = action_np.shape
                action_np = permute_action(action_np, args.robot_type)

                if action_count % 30 == 0:  # Log every ~1s at 30Hz
                    logger.debug(
                        f"[ACTION] #{action_count} shape: {orig_shape}→{action_np.shape}, "
                        f"queue_left={action_queue.qsize()}, "
                        f"range=[{action_np.min():.3f}, {action_np.max():.3f}]"
                    )

                t_step = time.perf_counter()
                env.step(action_np)
                step_time = (time.perf_counter() - t_step) * 1000

                if step_time > 50:  # Warn if step takes too long
                    logger.warning(f"[STEP] Slow hardware step: {step_time:.1f}ms")

                action_count += 1
            else:
                if action_count > 0:  # Only warn after first action (not during startup)
                    logger.warning(f"[STEP] Queue empty! No action to execute (qsize={action_queue.qsize()})")

            # ── Step C: Maintain control frequency ────────────────────────
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, time_per_step - elapsed - 0.0001)
            time.sleep(sleep_time)

            # Track actual loop frequency
            actual_cycle = time.perf_counter() - loop_start
            loop_times.append(actual_cycle)
            if len(loop_times) > 300:
                loop_times.pop(0)

            # Periodic status log (every ~5 seconds)
            if action_count > 0 and action_count % (int(args.control_freq) * 5) == 0:
                avg_cycle = sum(loop_times) / len(loop_times)
                max_cycle = max(loop_times)
                actual_hz = 1.0 / avg_cycle if avg_cycle > 0 else 0
                logger.info(
                    f"[MAIN] actions={action_count}, inferences={inference_count}, "
                    f"queue={action_queue.qsize()}, "
                    f"freq={actual_hz:.1f}Hz (avg_cycle={avg_cycle*1000:.1f}ms, max={max_cycle*1000:.1f}ms)"
                )

    except KeyboardInterrupt:
        logger.info("Ctrl+C detected, shutting down...")
    except Exception as e:
        logger.error(f"[MAIN] Exception: {e}")
        logger.error(traceback.format_exc())
    finally:
        env.close()
        logger.info(
            f"[MAIN] Finished. Actions sent: {action_count}, Inferences: {inference_count}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class Args:
    """GR00T Policy Execution Client for Unitree Robots [Single-Threaded RTC]"""

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
    execution_horizon: int = 8
    """Action Execution Horizon chunk."""
    action_queue_size_to_get_new_actions: int = 8
    """Trigger new inference when queue drops to this size."""
    language_instruction: str = "Pack black camera into box"
    """Task Instruction."""
    init_pose_file: str | None = None
    """JSON file to override INIT_POSE."""
    network_interface: str | None = None
    """Network interface to use."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_eval(args)