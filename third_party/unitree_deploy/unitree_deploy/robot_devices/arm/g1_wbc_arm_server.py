"""G1 WBC Arm ZMQ Server — runs ON THE ROBOT.

Thin DDS bridge: no trajectory interpolation, no IK.

  rt/lowstate  (DDS subscribe)
      ─── parse full LowState ───▶  ZMQ PUB  (state port, default 5556)
                                    JSON: {arm_q, arm_dq, waist_q, leg_q,
                                           imu_accel, imu_quat, imu_gyro, imu_rpy}

  ZMQ PULL  (cmd port, default 5555)
      ─── recv JSON {arm_q, arm_tau, base_cmd} ──▶  rt/whole_body_sdk  (DDS publish)

All trajectory interpolation and IK live on the PC side (g1_wbc_arm_client.py).

Usage (on the robot):
    python g1_wbc_arm_server.py --net eth0
    python g1_wbc_arm_server.py --net eth0 --cmd-port 5555 --state-port 5556
"""

import argparse
import json
import signal
import sys
import threading
import time

import numpy as np
import zmq
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

# ---------------------------------------------------------------------------
# Inlined helpers (no unitree_deploy package dependency needed on the robot)
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from enum import IntEnum


class G1_29_JointArmIndex(IntEnum):
    kLeftShoulderPitch  = 15
    kLeftShoulderRoll   = 16
    kLeftShoulderYaw    = 17
    kLeftElbow          = 18
    kLeftWristRoll      = 19
    kLeftWristPitch     = 20
    kLeftWristyaw       = 21
    kRightShoulderPitch = 22
    kRightShoulderRoll  = 23
    kRightShoulderYaw   = 24
    kRightElbow         = 25
    kRightWristRoll     = 26
    kRightWristPitch    = 27
    kRightWristYaw      = 28


class _G1_29_Num_Motors(IntEnum):
    value = 35


@dataclass
class MotorState:
    q:   float | None = None
    dq:  float | None = None
    tau: float | None = None


class DataBuffer:
    def __init__(self):
        self._data = None
        self._lock = threading.Lock()

    def get_data(self):
        with self._lock:
            return self._data

    def set_data(self, data):
        with self._lock:
            self._data = data


_G1_29_NUM_MOTORS = 35

# ---------------------------------------------------------------------------
# Minimal LowState snapshot (mirrors G1_29_WbcLowState in g1_wbc_arm.py)
# ---------------------------------------------------------------------------


class _LowStateSnap:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(_G1_29_NUM_MOTORS)]
        self.accel = np.zeros(3, dtype=np.float32)
        self.quaternion = np.zeros(4, dtype=np.float32)
        self.gyroscope = np.zeros(3, dtype=np.float32)
        self.rpy = np.zeros(3, dtype=np.float32)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

_DEFAULT_BASE_CMD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.74]


class G1WbcArmZmqServer:
    """Thin ZMQ ↔ DDS bridge that runs on the robot.

    - Subscribes to rt/lowstate, serialises the state as JSON and broadcasts
      it via a ZMQ PUB socket.
    - Listens on a ZMQ PULL socket for arm-command JSON from the PC and
      writes it directly to rt/whole_body_sdk without any interpolation.
    """

    TOPIC_LOW_STATE = "rt/lowstate"
    TOPIC_WBC_CMD = "rt/whole_body_sdk"

    def __init__(
        self,
        network_interface: str,
        cmd_port: int = 5555,
        state_port: int = 5556,
        control_dt: float = 1 / 100.0,
    ):
        self.network_interface = network_interface
        self.cmd_port = cmd_port
        self.state_port = state_port
        self.control_dt = control_dt

        self._lowstate_buf = DataBuffer()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        # ---- DDS init ----
        ChannelFactoryInitialize(0, self.network_interface)

        # LowState subscriber
        self._lowstate_sub = ChannelSubscriber(self.TOPIC_LOW_STATE, LowStateHG)
        self._lowstate_sub.Init()

        # WBC command publisher
        self._wbc_pub = ChannelPublisher(self.TOPIC_WBC_CMD, String_)
        self._wbc_pub.Init()
        self._wbc_msg = String_(data="")

        # Wait for first LowState
        logger_mp.warning("[ZMQ Server] Waiting for DDS LowState…")
        deadline = time.time() + 10.0
        while self._lowstate_buf.get_data() is None:
            self._poll_dds_once()
            if time.time() > deadline:
                raise TimeoutError("Timed out waiting for LowState from DDS.")
            time.sleep(0.05)
        logger_mp.info("[ZMQ Server] DDS LowState received ✅")

        # ---- ZMQ ----
        self._ctx = zmq.Context()

        # PUB: broadcast state to PC
        self._state_sock = self._ctx.socket(zmq.PUB)
        self._state_sock.setsockopt(zmq.SNDHWM, 2)
        self._state_sock.bind(f"tcp://*:{self.state_port}")

        # PULL: receive arm commands from PC
        self._cmd_sock = self._ctx.socket(zmq.PULL)
        self._cmd_sock.setsockopt(zmq.RCVTIMEO, 100)
        self._cmd_sock.bind(f"tcp://*:{self.cmd_port}")

        logger_mp.info(
            f"[ZMQ Server] Listening — "
            f"state=tcp://*:{self.state_port}  cmd=tcp://*:{self.cmd_port}"
        )

        # ---- Daemon threads ----
        self._dds_thread = threading.Thread(
            target=self._dds_subscribe_loop, name="dds_sub", daemon=True
        )
        self._state_thread = threading.Thread(
            target=self._state_publish_loop, name="zmq_pub", daemon=True
        )
        self._cmd_thread = threading.Thread(
            target=self._cmd_recv_loop, name="zmq_pull", daemon=True
        )
        self._dds_thread.start()
        self._state_thread.start()
        self._cmd_thread.start()

    def stop(self):
        logger_mp.warning("[ZMQ Server] Shutting down…")
        self._stop_event.set()
        self._cmd_sock.close(linger=0)
        self._state_sock.close(linger=0)
        self._ctx.term()
        logger_mp.info("[ZMQ Server] Stopped.")

    def wait(self):
        while not self._stop_event.is_set():
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # DDS helpers
    # ------------------------------------------------------------------

    def _poll_dds_once(self):
        msg = self._lowstate_sub.Read()
        if msg is not None:
            self._lowstate_buf.set_data(self._parse_lowstate(msg))

    def _parse_lowstate(self, msg) -> _LowStateSnap:
        snap = _LowStateSnap()
        n = min(len(msg.motor_state), _G1_29_NUM_MOTORS)
        for i in range(n):
            snap.motor_state[i].q = msg.motor_state[i].q
            snap.motor_state[i].dq = msg.motor_state[i].dq
        snap.accel = np.array(msg.imu_state.accelerometer, dtype=np.float32)
        snap.quaternion = np.array(msg.imu_state.quaternion, dtype=np.float32)
        snap.gyroscope = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        snap.rpy = np.array(msg.imu_state.rpy, dtype=np.float32)
        return snap

    def _snap_to_dict(self, snap: _LowStateSnap) -> dict:
        ms = snap.motor_state
        return {
            "leg_q":    [ms[i].q for i in range(12)],
            "waist_q":  [ms[i].q for i in range(12, 15)],
            "arm_q":    [ms[i].q for i in G1_29_JointArmIndex],
            "arm_dq":   [ms[i].dq for i in G1_29_JointArmIndex],
            "imu_accel": snap.accel.tolist(),
            "imu_quat":  snap.quaternion.tolist(),
            "imu_gyro":  snap.gyroscope.tolist(),
            "imu_rpy":   snap.rpy.tolist(),
        }

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _dds_subscribe_loop(self):
        """Continuously read LowState from DDS and store in buffer."""
        logger_mp.info("[ZMQ Server] dds_subscribe_loop started")
        while not self._stop_event.is_set():
            try:
                msg = self._lowstate_sub.Read()
                if msg is not None:
                    self._lowstate_buf.set_data(self._parse_lowstate(msg))
            except Exception as e:
                logger_mp.error(f"[ZMQ Server] dds_subscribe_loop error: {e}")
            time.sleep(self.control_dt)

    def _state_publish_loop(self):
        """Publish robot state via ZMQ PUB at control_dt rate."""
        logger_mp.info("[ZMQ Server] state_publish_loop started")
        while not self._stop_event.is_set():
            start = time.perf_counter()
            try:
                snap = self._lowstate_buf.get_data()
                if snap is not None:
                    payload = json.dumps(self._snap_to_dict(snap))
                    self._state_sock.send(payload.encode(), zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as e:
                logger_mp.error(f"[ZMQ Server] state_publish_loop error: {e}")
            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, self.control_dt - elapsed))

    def _cmd_recv_loop(self):
        """Receive arm command JSON from PC and write directly to DDS."""
        logger_mp.info("[ZMQ Server] cmd_recv_loop started")
        while not self._stop_event.is_set():
            try:
                raw = self._cmd_sock.recv()
            except zmq.Again:
                continue
            except Exception as e:
                logger_mp.error(f"[ZMQ Server] cmd_recv_loop recv error: {e}")
                continue

            try:
                msg = json.loads(raw)
                arm_q   = msg.get("arm_q",   [0.0] * 14)
                arm_tau = msg.get("arm_tau",  [0.0] * 14)
                base_cmd = msg.get("base_cmd", _DEFAULT_BASE_CMD)

                payload = json.dumps({
                    "arm_q":    arm_q,
                    "arm_tau":  arm_tau,
                    "base_cmd": base_cmd,
                })
                self._wbc_msg.data = payload
                self._wbc_pub.Write(self._wbc_msg)
            except Exception as e:
                logger_mp.error(f"[ZMQ Server] cmd_recv_loop write error: {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="G1 WBC arm ZMQ server (thin DDS bridge, run on robot)")
    p.add_argument("--net", required=True, help="Network interface for DDS (e.g. eth0)")
    p.add_argument("--cmd-port",   type=int, default=55555, help="ZMQ PULL port (PC → Robot commands)")
    p.add_argument("--state-port", type=int, default=55556, help="ZMQ PUB  port (Robot → PC state)")
    p.add_argument("--dt", type=float, default=1 / 100.0, help="State publish period in seconds")
    return p.parse_args()


def main():
    args = _parse_args()
    server = G1WbcArmZmqServer(
        network_interface=args.net,
        cmd_port=args.cmd_port,
        state_port=args.state_port,
        control_dt=args.dt,
    )

    def _sig(sig, frame):
        print()
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    server.start()
    logger_mp.info("[ZMQ Server] Running — press Ctrl-C to stop")
    server.wait()


if __name__ == "__main__":
    main()
