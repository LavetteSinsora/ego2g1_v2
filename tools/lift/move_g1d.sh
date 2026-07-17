#!/usr/bin/env bash
# Interactive up/down control of the G1-D lift column (sheng jiang) over DDS.
#
# The controller (lift_control.py) is streamed to the robot and run INLINE via
# `python -c`. Nothing is written to the robot disk. Arrow keys drive the column:
#   Up   = raise      Down = lower      release = stop      q / Ctrl-C = quit
#
# On quit / disconnect the controller always publishes velocity 0 (stop).
#
# Machine specifics are env vars; the defaults are our current G1-D:
#   ROBOT_USER  (unitree)
#   ROBOT_IP    (192.168.123.164)
#   REMOTE_PY   (/home/unitree/miniconda3/envs/tv/bin/python — the robot env with unitree_sdk2py)
#   LIFT_IFACE  (eth0 — the robot's internal NIC, passed through to the remote program)
#   LIFT_DOMAIN (0)
set -euo pipefail

ROBOT_USER="${ROBOT_USER:-unitree}"
ROBOT_IP="${ROBOT_IP:-192.168.123.164}"
REMOTE_PY="${REMOTE_PY:-/home/unitree/miniconda3/envs/tv/bin/python}"
LIFT_IFACE="${LIFT_IFACE:-eth0}"
LIFT_DOMAIN="${LIFT_DOMAIN:-0}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cat "$DIR/lift_control.py")"

# lift_control.py deliberately contains no single quotes, so wrapping the whole
# program in single quotes for the remote shell is safe. Machine specifics ride
# along as env vars on the remote command line.
REMOTE_CMD="LIFT_IFACE=$LIFT_IFACE LIFT_DOMAIN=$LIFT_DOMAIN $REMOTE_PY -c '$SRC'"

echo "Connecting to $ROBOT_USER@$ROBOT_IP  (password: 123)" >&2
echo "Controls: Up=raise  Down=lower  (release=stop)  q or Ctrl-C=quit" >&2

# -tt forces a pseudo-tty so arrow-key escape sequences reach the remote program.
exec ssh -tt \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  "$ROBOT_USER@$ROBOT_IP" "$REMOTE_CMD"
