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

import numpy as np
import torch
import tyro

import logging
import os
import time
import warnings

import logging_mp

_orig_listener = logging_mp._logging_mp_queue_listener
def _patched_listener(queue_proxy, config, prog_name):
    from rich.logging import RichHandler
    _orig_init = RichHandler.__init__
    def _init_with_date(self, *args, **kwargs):
        kwargs["log_time_format"] = "%Y-%m-%d %H:%M:%S.%f"
        _orig_init(self, *args, **kwargs)
    RichHandler.__init__ = _init_with_date
    return _orig_listener(queue_proxy, config, prog_name)
logging_mp._logging_mp_queue_listener = _patched_listener

from .groot_adapter.server_client_17 import PolicyClient
from unitree_deploy.real_unitree_env import make_real_env

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
        logger.info("✓ NET     connected to unitree_wholeboby_general_model Policy Server")

    def predict_action(self, observation):
        state_summary = " ".join(
            f"{k}{tuple(np.asarray(v).shape)}" for k, v in observation.get("state", {}).items()
        )
        logger.info(f"    ↳ NET →  send  state={{{state_summary}}}")

        t_send = time.perf_counter()
        action_dict, _ = self.client.get_action(observation)
        rtt = (time.perf_counter() - t_send) * 1000

        action = action_dict.get("action", action_dict)
        logger.info(f"    ↳ NET ←  recv  rtt={rtt:6.2f} ms  keys={list(action.keys())}")

        ref_arr = np.array(action.get("left_arm", [[[0]]]), dtype=np.float32)
        shape_prefix = ref_arr.shape[:-1]

        if self.robot_type in ACTION_KEYS:
            config = ACTION_KEYS[self.robot_type]
            raw_action = np.zeros((*shape_prefix, config["dim"]), dtype=np.float32)
            slots = []
            for key, slc in config["keys"].items():
                raw_action[..., slc] = action.get(key, 0)
                slots.append(f"{key}@[{slc.start}:{slc.stop}]")
            logger.info(f"    ↳ ASM    {' '.join(slots)} → dim={config['dim']}")
        else:
            raw_action = np.array(list(action.values())[0], dtype=np.float32)
            logger.warning(f"    ↳ ASM    fallback (robot_type={self.robot_type} not in ACTION_KEYS)")

        action_tensor = torch.from_numpy(raw_action)
        if action_tensor.ndim == 3 and action_tensor.shape[0] == 1:
            action_tensor = action_tensor.squeeze(0)

        out = action_tensor[: self.action_horizon].cpu()
        logger.info(f"    ↳ chunk  shape={tuple(out.shape)} horizon={self.action_horizon}")
        return out


def prepare_observation(args: "Args", obs: dict) -> dict:
    """Format observation as state + language only.

    Images are NOT included here; the policy server pulls them directly from
    the robot's image_server via its own ZMQ subscription.
    """
    qpos = obs["qpos"]
    qpos = (qpos.numpy() if torch.is_tensor(qpos) else qpos).astype(np.float32)[None, None, ...]

    mapping = OBS_STATE_KEYS.get(args.robot_type, OBS_STATE_KEYS["unitree_g1_wbc_dex1"])
    state_dict = {key: qpos[..., slc] for key, slc in mapping.items()}

    parts = " ".join(f"{k}({v.shape[-1]})" for k, v in state_dict.items())
    lang_short = args.language_instruction if len(args.language_instruction) <= 60 else args.language_instruction[:57] + "..."
    logger.info(f"    ↳ qpos{tuple(qpos.shape)} → {parts}  │ lang={lang_short!r}")

    return {
        "state": state_dict,
        "language": {"annotation.human.task_description": [[args.language_instruction]]},
    }


def run_eval(args: "Args") -> None:
    logger.info(f"◆ INIT    server={args.host}:{args.port}  robot={args.robot_type}  freq={args.control_freq}Hz")
    client = Gr00tZMQAdapter(host=args.host, port=args.port, robot_type=args.robot_type)

    env = make_real_env(
        robot_type=args.robot_type,
        dt=1 / args.control_freq,
        network_interface=args.network_interface,
    )
    env.connect()
    logger.info(f"◆ INIT    env connected (iface={args.network_interface})")

    init_pose = INIT_POSE.get(args.robot_type, INIT_POSE["unitree_g1_wbc_dex1"])
    if args.init_pose_file and Path(args.init_pose_file).exists():
        with open(args.init_pose_file) as f:
            data = json.load(f)
            if "mean_init_qpos" in data:
                init_pose = np.array(data["mean_init_qpos"], dtype=np.float32)
                logger.info(f"◆ INIT    override init_pose ← {args.init_pose_file}")
    env.step(init_pose)
    time.sleep(2.0)
    logger.info(f"◆ INIT    init_pose{tuple(np.asarray(init_pose).shape)} reached → entering loop")

    try:
        t = 0
        action_queue = deque()

        while True:
            logger.info(f"━━━━━━━━━━━━━━━━ ▶ STEP {t} ━━━━━━━━━━━━━━━━")

            # ── [1/4] READ OBS ─────────────────────────────────────────────
            t_obs0 = time.perf_counter()
            raw_obs = env.get_observation()
            dt_obs = (time.perf_counter() - t_obs0) * 1000
            logger.info(f"[1/4] ⏱ READ OBS    env.get_observation()  {dt_obs:6.2f} ms")
            obs = prepare_observation(args, raw_obs)

            # ── [2/4] POLICY (only when chunk is exhausted) ────────────────
            if len(action_queue) == 0:
                logger.info("[2/4] ⇄ POLICY      chunk EXHAUSTED → fetching new action chunk from server")
                t0 = time.perf_counter()
                pred_actions = client.predict_action(obs)
                dt_pol = (time.perf_counter() - t0) * 1000
                for i in range(pred_actions.shape[0]):
                    action_queue.append(pred_actions[i].cpu().numpy())
                logger.info(
                    f"[2/4] ✓ POLICY      inference {dt_pol:6.2f} ms  "
                    f"→ enqueued {pred_actions.shape[0]} actions  (queue size = {len(action_queue)})"
                )
            else:
                logger.info(
                    f"[2/4] ⏭ POLICY      SKIP — chunk still has {len(action_queue)} action(s); "
                    f"reusing cached chunk (no server call)"
                )

            # ── [3/4] DEQUEUE one action + rearrange ───────────────────────
            action = action_queue.popleft()
            remaining = len(action_queue)
            logger.info(
                f"[3/4] ▶ DEQUEUE     pop a[{action.shape[0]}]  "
                f"range=[{action.min():+.3f}, {action.max():+.3f}]  mean={action.mean():+.3f}  "
                f"→ {remaining} remaining in chunk"
            )

            orig_dim = action.shape[0]
            action = np.concatenate([action[:14], action[-7:], action[14:16]])
            action[18:20] = [0.0, 0.0]
            logger.info(
                f"    ↳ rearrange  [:14]+[-7:]+[14:16] zeroed[18:20]  "
                f"dim {orig_dim} → {action.shape[0]}"
            )

            # ── [4/4] EXEC on robot ────────────────────────────────────────
            logger.info(f"[4/4] ✓ EXEC        env.step(action)  shape={tuple(action.shape)}")
            if env.step(action) is False:
                return
            t += 1

    except KeyboardInterrupt:
        logger.info(f"◆ SHUTDOWN  KeyboardInterrupt (Ctrl+C) at step {t} — exiting cleanly")
    finally:
        logger.info("◆ SHUTDOWN  closing env …")
        env.close()
        logger.info("◆ SHUTDOWN  done ✓")


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
