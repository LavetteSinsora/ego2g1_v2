#!/bin/bash
# Script to run the GR00T Robot Client in Dataset Replay (Offline Validation) Mode

# Set Python Path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the client with --dataset_name
# Note: The dataset repo_id or path is specified here.
# python3 -m policy_adapter.robot_client_gr00t_dataset_17 \
#     --robot_type unitree_g1_wbc_dex1 \
#     --dataset_repo_id /home/unitree/meeting_room_1-2_0415-2 \
#     --action_horizon 16 \
#     --language_instruction "Approach the desk and visually inspect all objects on and around it. Pick up the pen from the table and place it into the pen holder. Then grasp the remote control and move it to the center front of the desk. Locate the paper ball and dispose of it into the trash bin. Identify the empty bottle and discard it as well. Take a bottle of water from the supply box and place it on the desk in an upright position. Straighten and organize all brochures on the desk. Push the chair fully under the desk. Return to the final position after completing all steps." \
#     --host 10.3.0.241 \
#     --port 14096


python3 -m policy_adapter.robot_client_gr00t_dataset_17 \
    --robot_type unitree_g1_wbc_dex1 \
    --dataset_repo_id /home/unitree/meeting_room_1-2_0415-2 \
    --action_horizon 16 \
    --language_instruction "Approach the desk and visually inspect all objects on and around it. Pick up the pen from the table and place it into the pen holder. Then grasp the remote control and move it to the center front of the desk. Locate the paper ball and dispose of it into the trash bin. Identify the empty bottle and discard it as well. Take a bottle of water from the supply box and place it on the desk in an upright position. Straighten and organize all brochures on the desk. Push the chair fully under the desk. Return to the final position after completing all steps." \
    --host 0.0.0.0 \
    --port 8095
