# tools/lift

Interactive up/down control of the **G1-D lift column (升降)** from your laptop, over the
robot's DDS bus. Nothing is written to the robot — the controller runs inline via
`python -c` on the robot side.

## Usage

```bash
./tools/lift/move_g1d.sh
```

It SSHes into the robot (prompts for the password `123`), then:

| key           | action                        |
|---------------|-------------------------------|
| **↑ Up**      | raise the column (hold)       |
| **↓ Down**    | lower the column (hold)       |
| *release*     | stop (auto-stop ~0.15 s)      |
| **q / Ctrl-C**| quit (always sends stop)      |

A live line shows the commanded velocity and the current height in meters
(read back from the robot).

## How it works

The lift is exposed as a DDS topic (not serial/GPIO):

- **Command** — `rt/cmd_hispeed` (`geometry_msgs/Point32`): `.z` = vertical **velocity**
  in `[-1, 1]` (`+` up, `−` down, `0` hold), published continuously at ~30 Hz. It is a
  velocity command, so you hold to move and it stops when you release.
- **Feedback** — `rt/hispeed_state` (`geometry_msgs/Point32`): `.y` = current height (m).

This mirrors Unitree's own `G1_Mobile_Lift_Controller` in
`/unitree/module/unitree_eai/xr_teleoperate/teleop/robot_control/mobile_control.py`
on the robot. The lift firmware lives on the control board, so it responds whenever the
robot is powered/enabled — teleop does **not** need to be running.

## Safety

- Velocity is hard-clamped to `±0.30`.
- The column auto-stops ~0.15 s after the arrow key stops repeating (on release).
- On quit / Ctrl-C / dropped SSH the controller publishes `z = 0` (stop) before exiting.
- Make sure the column's travel is physically clear before moving it.

## Config (env vars)

Machine specifics are all overridable; defaults are our current G1-D (see
`docs/robot.md` for the network map):

| var           | default                                       | what |
|---------------|-----------------------------------------------|------|
| `ROBOT_USER`  | `unitree`                                     | ssh user on the robot |
| `ROBOT_IP`    | `192.168.123.164`                             | robot board on the 192.168.123.x subnet |
| `REMOTE_PY`   | `/home/unitree/miniconda3/envs/tv/bin/python` | robot-side python that has `unitree_sdk2py` |
| `LIFT_IFACE`  | `eth0`                                        | robot-side DDS NIC (passed through to the inline program) |
| `LIFT_DOMAIN` | `0`                                           | DDS domain id |

`lift_control.py` intentionally contains **no single-quote characters** so the launcher
can wrap the whole program in single quotes for the remote shell — keep it that way when
editing (double quotes and `chr()` everywhere).
