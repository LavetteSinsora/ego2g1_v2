import threading
import time
from multiprocessing import Array

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # dds
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl

from unitree_deploy.robot_devices.endeffector.configs import BraincoHandConfig
from unitree_deploy.robot_devices.robot_devices_index import (
    Brainco_Left_Hand_JointIndex,
    Brainco_Right_Hand_JointIndex,
)
from unitree_deploy.robot_devices.robots_devices_utils import DataBuffer, MotorState, Robot_Num_Motors
from unitree_deploy.utils.rich_logger import log_info, log_warning


class Brainco_Left_Hand_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.Brainco_Num_Motors)]


class Brainco_Right_Hand_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.Brainco_Num_Motors)]


class Brainco_Hand_Controller:
    def __init__(self, config: BraincoHandConfig):
        log_info("Initialize Brainco_Hand_Controller...")

        self.motors = config.motors
        self.mock = config.mock
        self.control_dt = config.control_dt
        self.unit_test = config.unit_test

        self.simulation_mode = config.simulation_mode

        self.k_topic_brainco_left_command = config.k_topic_brainco_left_command
        self.k_topic_brainco_right_command = config.k_topic_brainco_right_command
        self.k_topic_brainco_left_state = config.k_topic_brainco_left_state
        self.k_topic_brainco_right_state = config.k_topic_brainco_right_state

        self.hand_indices_len = Robot_Num_Motors.Brainco_Num_Motors

        self.left_lowstate_buffer = DataBuffer()
        self.right_lowstate_buffer = DataBuffer()

        # set True by the subscribe thread once the first paired state arrives.
        # NOTE: gate on this flag, not on any(state_array) — an open hand is all
        # zeros, so any() would never fire even though data is flowing.
        self.hand_sub_ready = False

        # default targets so the control loop is safe before the first write_endeffector()
        self.q_target = np.zeros(2 * Robot_Num_Motors.Brainco_Num_Motors, dtype=float)
        self.tauff_target = None
        self.time_target = None

        self.ctrl_lock = threading.Lock()
        self.running = False
        self.is_connected = False

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]

    def connect(self):
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(self.k_topic_brainco_left_command, MotorCmds_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(self.k_topic_brainco_right_command, MotorCmds_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(self.k_topic_brainco_left_state, MotorStates_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(self.k_topic_brainco_right_state, MotorStates_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array("d", Robot_Num_Motors.Brainco_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Robot_Num_Motors.Brainco_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if self.hand_sub_ready:
                break
            time.sleep(0.1)
            log_warning("[Brainco_Hand_Controller] Waiting to subscribe dds...")
        log_info("[Brainco_Hand_Controller] Subscribe dds ok.")

        self.hand_control_thread = threading.Thread(target=self._ctrl_hand_motor)
        self.hand_control_thread.daemon = True
        self.hand_control_thread.start()

        self.is_connected = True

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                left_lowstate = Brainco_Left_Hand_LowState()
                right_lowstate = Brainco_Right_Hand_LowState()

                # Update left hand state
                for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
                    left_lowstate.motor_state[id].q = left_hand_msg.states[id].q
                    self.left_hand_state_array[idx] = left_hand_msg.states[id].q
                self.left_lowstate_buffer.set_data(left_lowstate)
                # Update right hand state
                for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
                    right_lowstate.motor_state[id].q = right_hand_msg.states[id].q
                    self.right_hand_state_array[idx] = right_hand_msg.states[id].q
                self.right_lowstate_buffer.set_data(right_lowstate)
                self.hand_sub_ready = True
            time.sleep(0.002)

    def read_current_endeffector_q(self) -> np.ndarray | None:
        left_motor_states = np.array(
            [self.left_lowstate_buffer.get_data().motor_state[id].q for id in Brainco_Left_Hand_JointIndex]
        )
        right_motor_states = np.array(
            [self.right_lowstate_buffer.get_data().motor_state[id].q for id in Brainco_Right_Hand_JointIndex]
        )
        return np.concatenate((left_motor_states, right_motor_states))

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """Set current left, right hand motor state target q"""
        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_hand_msg.cmds[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_hand_msg)
        self.RightHandCmb_publisher.Write(self.right_hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")

    def _ctrl_hand_motor(self):
        self.running = True

        left_q_target = np.full(Robot_Num_Motors.Brainco_Num_Motors, 0.0, dtype=float)
        right_q_target = np.full(Robot_Num_Motors.Brainco_Num_Motors, 0.0, dtype=float)

        # initialize brainco hand's cmd msg
        self.left_hand_msg = MotorCmds_()
        self.left_hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Left_Hand_JointIndex))
        ]
        self.right_hand_msg = MotorCmds_()
        self.right_hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Right_Hand_JointIndex))
        ]

        for _, id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_hand_msg.cmds[id].q = 0.0
            self.left_hand_msg.cmds[id].dq = 1.0
        for _, id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_hand_msg.cmds[id].q = 0.0
            self.right_hand_msg.cmds[id].dq = 1.0

        try:
            while self.running:
                start_time = time.perf_counter()

                with self.ctrl_lock:
                    left_q_target = self.q_target[: self.hand_indices_len]
                    right_q_target = self.q_target[self.hand_indices_len :]

                self.ctrl_dual_hand(left_q_target, right_q_target)
                time.sleep(max(0, (self.control_dt - (time.perf_counter() - start_time))))

        finally:
            log_info("[Brainco_Hand_Controller] has been closed.")

    def write_endeffector(
        self,
        q_target: int | float | np.ndarray,
        tauff_target: int | float | np.ndarray = None,
        time_target: float | None = None,
        cmd_target: str | None = None,
    ):
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
            self.time_target = time_target
            self.cmd_target = cmd_target

    def disconnect(self):
        self.running = False
        self.is_connected = False
