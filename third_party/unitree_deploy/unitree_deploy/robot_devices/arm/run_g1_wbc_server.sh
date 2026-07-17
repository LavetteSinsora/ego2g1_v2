#!/usr/bin/env bash
# G1 WBC Arm ZMQ Server — launch script for the ROBOT
#
# Usage:
#   bash run_g1_wbc_server.sh                    # defaults: eth0, port 5555/5556
#   bash run_g1_wbc_server.sh eth0               # explicit network interface
#   bash run_g1_wbc_server.sh eth0 5555 5556     # explicit ports
#
# The script auto-detects the unitree-deploy repo root (assumes this file is
# inside it or its deploy root is in $UNITREE_DEPLOY_ROOT).

set -euo pipefail

NET="${1:-eth0}"
CMD_PORT="${2:-5555}"
STATE_PORT="${3:-5556}"

# Locate the repo root: walk up from this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
exec python "${SCRIPT_DIR}/g1_wbc_arm_server.py" \
    --net       "$NET"        \
    --cmd-port  "$CMD_PORT"   \
    --state-port "$STATE_PORT"
