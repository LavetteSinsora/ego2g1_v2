"""Dex1 Gripper ZMQ Server — runs ON THE ROBOT.

Wraps the **unchanged** ``Dex1_Gripper_Controller`` (gripper.py) with ZMQ
transport so that all DDS communication stays exactly as it was in the
working version.

Data-flow
─────────
  rt/unitree_actuator/state  (DDS subscribe, via Dex1_Gripper_Controller)
      ─── read_current_endeffector_q/dq ───▶  ZMQ PUB  (state_port, default 5558)
                                               JSON: {"gripper_q": [...], "gripper_dq": [...]}

  ZMQ PULL  (cmd_port, default 5557)
      ─── recv JSON {"gripper_q": [...]} ──▶  controller.write_endeffector()
                                          ──▶  rt/unitree_actuator/cmd  (DDS publish)

All trajectory interpolation lives on the PC side (gripper_client.py).
The server just passes raw position targets straight through to the motor.

Usage (on the robot)
────────────────────
    python gripper_server.py --net eth10
    python gripper_server.py --net eth10 --cmd-port 5557 --state-port 5558
"""

import argparse
import json
import signal
import sys
import threading
import time

import numpy as np
import zmq
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from unitree_deploy.robot_devices.endeffector.configs import Dex1GripperConfig
from unitree_deploy.robot_devices.endeffector.gripper import Dex1_Gripper_Controller

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

# ---------------------------------------------------------------------------
# Default motor config (metadata only — actual DDS uses Gripper_Single_JointIndex)
# ---------------------------------------------------------------------------

def gripper_default_config(dt: float) -> dict:
    return {
        "left": Dex1GripperConfig(
            unit_test=False,
            motors={"kLeftGripper": [0, "z1_gripper-joint"]},
            control_dt=dt,
            init_pose=[0.0],
            topic_gripper_state="rt/dex1/left/state",
            topic_gripper_command="rt/dex1/left/cmd",
        ),
        "right": Dex1GripperConfig(
            unit_test=False,
            motors={"kRightGripper": [1, "z1_gripper-joint"]},
            control_dt=dt,
            init_pose=[0.0],
            topic_gripper_state="rt/dex1/right/state",
            topic_gripper_command="rt/dex1/right/cmd",
        ),
    }


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class Dex1GripperZmqServer:
    """ZMQ ↔ DDS bridge that wraps the unchanged Dex1_Gripper_Controller.

    * Uses the controller's existing DDS subscribe/publish logic unchanged.
    * Adds a ZMQ PUB socket to stream gripper state to the PC.
    * Adds a ZMQ PULL socket to receive gripper commands from the PC and
      forwards them directly to the controller (no robot-side interpolation —
      the PC client already interpolates).
    """

    def __init__(
        self,
        name: str,
        config: Dex1GripperConfig,
        cmd_port: int,
        state_port: int,
        control_dt: float = 1 / 200.0,
    ):
        self.name = name
        self.config = config
        self.cmd_port = cmd_port
        self.state_port = state_port
        self.control_dt = control_dt

        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        # ---- Gripper controller (DDS logic completely unchanged) ----
        self._ctrl = Dex1_Gripper_Controller(self.config)

        # connect() starts the DDS subscriber thread and waits for the first
        # state message, then starts the motor control thread — identical to
        # the working gripper.py flow.
        self._ctrl.connect()
        logger_mp.info(f"[Gripper ZMQ Server | {self.name}] DDS controller connected ✅")

        # Initialise q_target to current position so the control thread does
        # not try to move the gripper to 0 before the first ZMQ command arrives.
        current_q = self._ctrl.read_current_endeffector_q()
        self._ctrl.write_endeffector(q_target=current_q, cmd_target=None)
        logger_mp.info(f"[Gripper ZMQ Server | {self.name}] Gripper q initialised to {current_q}")

        # ---- ZMQ ----
        self._ctx = zmq.Context()

        # PUB: broadcast state to PC
        self._zmq_state_sock = self._ctx.socket(zmq.PUB)
        self._zmq_state_sock.setsockopt(zmq.SNDHWM, 2)
        self._zmq_state_sock.bind(f"tcp://*:{self.state_port}")

        # PULL: receive gripper commands from PC
        self._zmq_cmd_sock = self._ctx.socket(zmq.PULL)
        self._zmq_cmd_sock.setsockopt(zmq.RCVTIMEO, 100)  # ms — makes recv non-blocking
        self._zmq_cmd_sock.bind(f"tcp://*:{self.cmd_port}")

        logger_mp.info(
            f"[Gripper ZMQ Server | {self.name}] ZMQ ready — "
            f"state=tcp://*:{self.state_port}  cmd=tcp://*:{self.cmd_port}"
        )

        # ---- ZMQ daemon threads ----
        self._state_thread = threading.Thread(
            target=self._state_publish_loop, name="zmq_pub", daemon=True
        )
        self._cmd_thread = threading.Thread(
            target=self._cmd_recv_loop, name="zmq_pull", daemon=True
        )
        self._state_thread.start()
        self._cmd_thread.start()

    def stop(self):
        logger_mp.warning("[Gripper ZMQ Server] Shutting down…")
        self._stop_event.set()
        self._ctrl.disconnect()
        try:
            self._zmq_cmd_sock.close(linger=0)
            self._zmq_state_sock.close(linger=0)
            self._ctx.term()
        except Exception:
            pass
        logger_mp.info("[Gripper ZMQ Server] Stopped.")

    def wait(self):
        while not self._stop_event.is_set():
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # ZMQ threads
    # ------------------------------------------------------------------

    def _state_publish_loop(self):
        """Read state from the controller and broadcast via ZMQ PUB."""
        logger_mp.info("[Gripper ZMQ Server] state_publish_loop started")
        while not self._stop_event.is_set():
            start = time.perf_counter()
            try:
                gripper_q = self._ctrl.read_current_endeffector_q()
                gripper_dq = self._ctrl.read_current_endeffector_dq()
                payload = json.dumps({
                    "gripper_q": gripper_q.tolist(),
                    "gripper_dq": gripper_dq.tolist(),
                })
                self._zmq_state_sock.send(payload.encode(), zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as e:
                logger_mp.error(f"[Gripper ZMQ Server] state_publish_loop error: {e}")
            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, self.control_dt - elapsed))

    def _cmd_recv_loop(self):
        """Receive gripper-command JSON from PC and pass to the controller."""
        logger_mp.info("[Gripper ZMQ Server] cmd_recv_loop started")
        while not self._stop_event.is_set():
            try:
                raw = self._zmq_cmd_sock.recv()
            except zmq.Again:
                continue
            except Exception as e:
                logger_mp.error(f"[Gripper ZMQ Server] cmd_recv_loop recv error: {e}")
                continue

            try:
                msg = json.loads(raw)
                gripper_q = msg.get("gripper_q", [0.0])
                # cmd_target=None → direct position mode (no robot-side interpolation;
                # the PC client (gripper_client.py) already interpolates).
                self._ctrl.write_endeffector(
                    q_target=np.array(gripper_q, dtype=float),
                    cmd_target=None,
                )
            except Exception as e:
                logger_mp.error(f"[Gripper ZMQ Server] cmd_recv_loop parse error: {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Dex1 Gripper ZMQ server (wraps Dex1_Gripper_Controller, run on robot)"
    )
    p.add_argument("--net", required=True, help="Network interface for DDS (e.g. eth10)")
    p.add_argument(
        "--cmd-port-left", type=int, default=5557, help="ZMQ PULL port (Left cmd)"
    )
    p.add_argument(
        "--state-port-left", type=int, default=5558, help="ZMQ PUB port (Left state)"
    )
    p.add_argument(
        "--cmd-port-right", type=int, default=5559, help="ZMQ PULL port (Right cmd)"
    )
    p.add_argument(
        "--state-port-right", type=int, default=5560, help="ZMQ PUB port (Right state)"
    )
    p.add_argument(
        "--dt", type=float, default=1 / 200.0, help="State publish period in seconds"
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # ---- ChannelFactoryInitialize must be called exactly once per process ----
    ChannelFactoryInitialize(0, args.net)
    
    configs = gripper_default_config(args.dt)

    server_left = Dex1GripperZmqServer(
        name="Left",
        config=configs["left"],
        cmd_port=args.cmd_port_left,
        state_port=args.state_port_left,
        control_dt=args.dt,
    )
    server_right = Dex1GripperZmqServer(
        name="Right",
        config=configs["right"],
        cmd_port=args.cmd_port_right,
        state_port=args.state_port_right,
        control_dt=args.dt,
    )

    def _sig(sig, frame):
        print()
        server_left.stop()
        server_right.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    server_left.start()
    server_right.start()
    
    logger_mp.info("[Gripper ZMQ Server] Two grippers Running — press Ctrl-C to stop")
    
    # Wait until interrupted
    server_left.wait()
    server_right.wait()


if __name__ == "__main__":
    main()
