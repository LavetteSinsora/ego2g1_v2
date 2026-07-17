#!/bin/bash
# Script to run the GR00T Robot Client in Dataset Replay (RTC) Mode

# Set Python Path
export PYTHONPATH=$PYTHONPATH:$(pwd)

python3 -m policy_adapter.robot_client_gr00t_dataset_rtc \
    --robot_type unitree_g1_wbc_dex1 \
    --dataset_repo_id ~/.cache/huggingface/lerobot/hengguo/wbc_meeting_room_0317_58 \
    --language_instruction "" \
    --host 10.3.0.241 \
    --port 14096
