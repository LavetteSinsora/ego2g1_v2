# for dex3-1
import threading
import time
from multiprocessing import Array

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # dds
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_  # idl

from unitree_deploy.robot_devices.endeffector.configs import Dex3HandConfig
from unitree_deploy.robot_devices.robot_devices_index import (
    Dex3_Hand_Left_JointIndex,
    Dex3_Hand_Right_JointIndex,
)
from unitree_deploy.robot_devices.robots_devices_utils import DataBuffer, MotorState, Robot_Num_Motors
from unitree_deploy.utils.rich_logger import log_info, log_warning


class Dex3_Left_Hand_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.Dex3_Num_Motors)]


class Dex3_Right_Hand_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.Dex3_Num_Motors)]


class Dex3_Hand_Controller:
    def __init__(self, config: Dex3HandConfig):
        log_info("Initialize Dex3_1_Controller...")

        self.motors = config.motors
        self.mock = config.mock
        self.control_dt = config.control_dt
        self.unit_test = config.unit_test

        self.simulation_mode = config.simulation_mode

        self.k_topic_dex3_left_command = config.k_topic_dex3_left_command
        self.k_topic_dex3_right_command = config.k_topic_dex3_right_command
        self.k_topic_dex3_left_state = config.k_topic_dex3_left_state
        self.k_topic_dex3_right_state = config.k_topic_dex3_right_state

        self.hand_indices_len = Robot_Num_Motors.Dex3_Num_Motors

        self.left_lowstate_buffer = DataBuffer()
        self.right_lowstate_buffer = DataBuffer()

        self.ctrl_lock = threading.Lock()

        self.is_connected = False

    def connect(self):
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(self.k_topic_dex3_left_command, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(self.k_topic_dex3_right_command, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(self.k_topic_dex3_left_state, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(self.k_topic_dex3_right_state, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array("d", Robot_Num_Motors.Dex3_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Robot_Num_Motors.Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            log_warning("[Dex3_1_Controller] Waiting to subscribe dds...")

        self.hand_control_thread = threading.Thread(target=self._ctrl_hand_motor)
        self.hand_control_thread.daemon = True
        self.hand_control_thread.start()

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()

            if left_hand_msg is not None and right_hand_msg is not None:
                left_lowstate = Dex3_Left_Hand_LowState()
                right_lowstate = Dex3_Right_Hand_LowState()

                # Update left hand state
                for _, id in enumerate(Dex3_Hand_Left_JointIndex):
                    left_lowstate.motor_state[id].q = left_hand_msg.motor_state[id].q
                self.left_lowstate_buffer.set_data(left_lowstate)
                # Update right hand state
                for _, id in enumerate(Dex3_Hand_Right_JointIndex):
                    right_lowstate.motor_state[id].q = right_hand_msg.motor_state[id].q
                self.right_lowstate_buffer.set_data(right_lowstate)
            time.sleep(0.002)

    def read_current_endeffector_q(self) -> np.ndarray | None:
        left_motor_states = np.array(
            [self.left_lowstate_buffer.get_data().motor_state[id].q for id in Dex3_Hand_Left_JointIndex]
        )
        right_motor_states = np.array(
            [self.right_lowstate_buffer.get_data().motor_state[id].q for id in Dex3_Hand_Right_JointIndex]
        )
        return np.concatenate((left_motor_states, right_motor_states))

    def read_current_endeffector_dq(self) -> np.ndarray | None:
        pass

    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= self.id & 0x0F
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_Hand_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_Hand_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)

    def _ctrl_hand_motor(self):
        self.running = True

        left_q_target = np.full(Robot_Num_Motors.Dex3_Num_Motors, 0)
        right_q_target = np.full(Robot_Num_Motors.Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        self.left_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_Hand_Left_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q = q
            self.left_msg.motor_cmd[id].dq = dq
            self.left_msg.motor_cmd[id].tau = tau
            self.left_msg.motor_cmd[id].kp = kp
            self.left_msg.motor_cmd[id].kd = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_Hand_Right_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode
            self.right_msg.motor_cmd[id].q = q
            self.right_msg.motor_cmd[id].dq = dq
            self.right_msg.motor_cmd[id].tau = tau
            self.right_msg.motor_cmd[id].kp = kp
            self.right_msg.motor_cmd[id].kd = kd

        try:
            while self.running:
                start_time = time.perf_counter()

                if self.read_current_endeffector_q:
                    with self.ctrl_lock:
                        left_q_target = self.q_target[: self.hand_indices_len]
                        right_q_target = self.q_target[self.hand_indices_len :]

                self.ctrl_dual_hand(left_q_target, right_q_target)
                time.sleep(max(0, (self.control_dt - (time.perf_counter() - start_time))))
        finally:
            log_info("Dex3_1_Controller has been closed.")

    def write_endeffector(
        self,
        q_target: int | float | np.ndarray,
        tauff_target: int | float | np.ndarray = None,
        time_target: float | None = None,
    ):
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
            self.time_target = time_target
