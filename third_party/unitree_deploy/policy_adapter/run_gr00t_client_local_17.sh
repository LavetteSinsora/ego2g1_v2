#!/bin/bash
# Script to run the GR00T Robot Client in ACT RHC mode


# python test/test_replay.py  --repo-id hengguo/wbc_meeting_room_0317_58  --robot-type unitree_g1_wbc_dex1

# Set Python Path if needed (adjust as per your workspace)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the client
python3 -m policy_adapter.robot_client_gr00t_17 \
    --robot_type unitree_g1_dex1 \
    --action_horizon 8 \
    --language_instruction "fold clothes" \
    --control_freq 30 \
    --host 0.0.0.0  \
    --port 9999


# python3 -m policy_adapter.robot_client_gr00t_async \
#     --robot_type unitree_g1_wbc_dex1 \
#     --action_horizon 16 \
#     --language_instruction "" \
#     --control_freq 30 \
#     --host 0.0.0.0  \
#     --network_interface "enp3s0" \
#     --port 8095

