#!/bin/bash
# Script to run the GR00T Robot Client in Dataset Replay (Offline Validation) Mode

# Set Python Path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the client with --dataset_name
# Note: The dataset repo_id or path is specified here.
python3 -m policy_adapter.robot_client_gr00t_dataset \
    --robot_type unitree_g1_wbc_dex1 \
    --dataset_repo_id /home/unitree/.cache/huggingface/lerobot/hengguo/wbc_meeting_room_0317_58 \
    --action_horizon 16 \
    --language_instruction "" \
    --host 10.3.0.241 \
    --port 14096
