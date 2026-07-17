import threading
import time
from multiprocessing import Array

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # dds
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl

from unitree_deploy.robot_devices.endeffector.configs import InspireHandConfig
from unitree_deploy.robot_devices.robot_devices_index import (
    Inspire_Left_Hand_JointIndex,
    Inspire_Right_Hand_JointIndex,
)
from unitree_deploy.robot_devices.robots_devices_utils import DataBuffer, MotorState, Robot_Num_Motors
from unitree_deploy.utils.rich_logger import log_info, log_warning


class Inspire_Left_Hand_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.Brainco_Num_Motors)]


class Inspire_Right_Hand_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Robot_Num_Motors.Brainco_Num_Motors)]


class Inspire_Hand_Controller:
    def __init__(self, config: InspireHandConfig):
        log_info("Initialize Inspire_Hand_Controller...")

        self.motors = config.motors
        self.mock = config.mock
        self.control_dt = config.control_dt
        self.unit_test = config.unit_test

        self.simulation_mode = config.simulation_mode

        self.k_topic_inspire_command = config.k_topic_inspire_command
        self.k_topic_inspire_state = config.k_topic_inspire_state

        self.hand_indices_len = Robot_Num_Motors.Brainco_Num_Motors

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
        self.HandCmb_publisher = ChannelPublisher(self.k_topic_inspire_command, MotorCmds_)
        self.HandCmb_publisher.Init()

        self.HandState_subscriber = ChannelSubscriber(self.k_topic_inspire_state, MotorStates_)
        self.HandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array("d", Robot_Num_Motors.Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array("d", Robot_Num_Motors.Inspire_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.right_hand_state_array):  # any(self.left_hand_state_array) and
                break
            time.sleep(0.01)
            log_warning("[Brainco_Hand_Controller] Waiting to subscribe dds...")

        self.hand_control_thread = threading.Thread(target=self._ctrl_hand_motor)
        self.hand_control_thread.daemon = True
        self.hand_control_thread.start()

    def _subscribe_hand_state(self):
        while True:
            hand_msg = self.HandState_subscriber.Read()
            left_lowstate = Inspire_Left_Hand_LowState()
            right_lowstate = Inspire_Right_Hand_LowState()
            if hand_msg is not None:
                for _, id in enumerate(Inspire_Left_Hand_JointIndex):
                    left_lowstate.motor_state[id].q = hand_msg.states[id].q
                self.left_lowstate_buffer.set_data(left_lowstate)
                for _, id in enumerate(Inspire_Right_Hand_JointIndex):
                    right_lowstate.motor_state[id].q = hand_msg.states[id].q
                self.right_lowstate_buffer.set_data(right_lowstate)

            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = left_q_target[idx]
        for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = right_q_target[idx]

        self.HandCmb_publisher.Write(self.hand_msg)
        # logger_mp.debug("hand ctrl publish ok.")

    def _ctrl_hand_motor(self):
        self.running = True

        left_q_target = np.full(Robot_Num_Motors.Inspire_Num_Motors, 1.0)
        right_q_target = np.full(Robot_Num_Motors.Inspire_Num_Motors, 1.0)

        # initialize inspire hand's cmd msg
        self.hand_msg = MotorCmds_()
        self.hand_msg.cmds = [
            unitree_go_msg_dds__MotorCmd_()
            for _ in range(len(Inspire_Right_Hand_JointIndex) + len(Inspire_Left_Hand_JointIndex))
        ]

        for _, id in enumerate(Inspire_Left_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0
        for _, id in enumerate(Inspire_Right_Hand_JointIndex):
            self.hand_msg.cmds[id].q = 1.0

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
    ):
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
            self.time_target = time_target
