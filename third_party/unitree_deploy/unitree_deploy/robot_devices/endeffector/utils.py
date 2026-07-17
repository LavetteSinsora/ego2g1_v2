from typing import Protocol

from unitree_deploy.robot_devices.endeffector.configs import (
    BraincoHandConfig,
    Dex1GripperConfig,
    Dex1GripperZmqConfig,
    Dex3HandConfig,
    EndEffectorConfig,
    InspireHandConfig,
)


class EndEffector(Protocol):
    def connect(self): ...
    def disconnect(self): ...
    def motor_names(self): ...

    def read_current_endeffector_q(self): ...
    def read_current_endeffector_dq(self): ...
    def write_endeffector(self): ...

    def retarget_to_endeffector(self): ...
    def endeffector_ik(self): ...

    def go_start(self): ...
    def go_home(self): ...


def make_endeffector_motors_buses_from_configs(
    endeffector_configs: dict[str, EndEffectorConfig],
) -> list[EndEffectorConfig]:
    endeffector_motors_buses = {}

    for key, cfg in endeffector_configs.items():
        if cfg.type == "dex_1":
            from unitree_deploy.robot_devices.endeffector.gripper import Dex1_Gripper_Controller

            endeffector_motors_buses[key] = Dex1_Gripper_Controller(cfg)

        elif cfg.type == "dex_3":
            from unitree_deploy.robot_devices.endeffector.dex3 import Dex3_Hand_Controller

            endeffector_motors_buses[key] = Dex3_Hand_Controller(cfg)

        elif cfg.type == "inspire":
            from unitree_deploy.robot_devices.endeffector.inspire import Inspire_Hand_Controller

            endeffector_motors_buses[key] = Inspire_Hand_Controller(cfg)

        elif cfg.type == "brainco":
            from unitree_deploy.robot_devices.endeffector.brainco import Brainco_Hand_Controller

            endeffector_motors_buses[key] = Brainco_Hand_Controller(cfg)

        elif cfg.type == "dex_1_zmq":
            from unitree_deploy.robot_devices.endeffector.gripper_client import Dex1GripperZmqClient

            endeffector_motors_buses[key] = Dex1GripperZmqClient(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return endeffector_motors_buses


def make_endeffector_motors_bus(endeffector_type: str, **kwargs) -> EndEffectorConfig:
    if endeffector_type == "dex_1":
        from unitree_deploy.robot_devices.endeffector.gripper import Dex1_Gripper_Controller

        config = Dex1GripperConfig(**kwargs)
        return Dex1_Gripper_Controller(config)

    elif endeffector_type == "dex_3":
        from unitree_deploy.robot_devices.endeffector.dex3 import Dex3_Hand_Controller

        config = Dex3HandConfig(**kwargs)
        return Dex3_Hand_Controller(config)
    elif endeffector_type == "dex_1":
        from unitree_deploy.robot_devices.endeffector.inspire import Inspire_Hand_Controller

        config = InspireHandConfig(**kwargs)
        return Inspire_Hand_Controller(config)

    elif endeffector_type == "dex_1":
        from unitree_deploy.robot_devices.endeffector.brainco import Brainco_Hand_Controller

        config = BraincoHandConfig(**kwargs)
        return Brainco_Hand_Controller(config)

    else:
        raise ValueError(f"The motor type '{endeffector_type}' is not valid.")
