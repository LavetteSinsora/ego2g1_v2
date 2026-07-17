#!/bin/bash
# Script to run the GR00T Robot Client with full ZMQ stack (arm + gripper)

# Set Python Path if needed (adjust as per your workspace)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the ZMQ client
python3 -m policy_adapter.robot_client_gr00t_zmq \
    --robot_ip 192.168.123.164 \
    --action_horizon 16 \
    --language_instruction "" \
    --control_freq 30 \
    --host 10.3.0.241 \
    --port 14095