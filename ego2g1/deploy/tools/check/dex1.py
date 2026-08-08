"""Rung 5c (`check dex1`): resolve the Dex1 observation/command layout.

Sends NO action and commands NO motion. It does have to `connect()` to read a
live observation, and be clear about what that means on this robot:
`G1_Arm.connect()` reads the current motor positions, applies the configured
gains, sets every command to WHERE THE JOINT ALREADY IS, and starts the
publish thread. So the arm does not travel anywhere — but it does become
STIFF, holding its present pose. If it was hanging limp it will take its own
weight the moment this runs. Nothing here ever calls `send_action`, so no
target other than "stay put" is ever published.

Run it before any `umi_eef` rollout.

`umi_eef` rests on two assumptions about `unitree_g1_dex1` that were derived
from unitree_deploy's `dex1_default_factory` (one motor per hand,
`kLeftGripper`/`kRightGripper`, a z1_gripper-joint) rather than observed on
hardware:

  1. `capture_observation()["observation.state"]` is 16 wide and laid out
     [arm(14) | left gripper | right gripper], in `layout.HANDS` order. That is
     what `UnitreeExecutor.arm_q`/`ee_q` slice, and what the state-history
     buffer records as the gripper column.
  2. `send_action` wants that same 16-wide vector, which is what
     `UnitreeExecutor._wire_row` builds by taking slot 0 of each hand block out
     of the deploy layer's canonical (26,) row.

Both are silent if wrong. A mis-sliced observation feeds the policy a joint
angle where it expects a gripper aperture; a mis-built command writes the
gripper value onto a wrist joint. Neither raises, and both produce a robot
that moves plausibly and wrongly.

The third thing this resolves is UNITS. `UmiTrainConfig` executes the model's
gripper output verbatim, in radians of gear rotation, because that is what the
training data stores (`observation.state`, named `kRightGripper`, spanning
1.20 fully closed .. 5.40 fully open). This prints the live encoder so you can
confirm the robot reports the same quantity on the same scale — open the
gripper by hand, run it, close it, run it again.
"""

from __future__ import annotations

import numpy as np

from ego2g1.core import layout
from ego2g1.deploy import actions as _actions

# What the training data spans, measured over all 117 episodes / 41207 frames
# of red_block_on_yellow_block_umi. Not a calibration — the yardstick this rung
# holds the live encoder up against.
TRAIN_GRIP_CLOSED = 1.20
TRAIN_GRIP_OPEN = 5.40


def dex1(iface: str | None = None, robot_type: str = "unitree_g1_dex1",
         fps: int = 30) -> None:
    """Print the Dex1 observation layout and the command vector we would build.

    Sends no action. `connect()` energizes the arm to hold its CURRENT pose
    (see the module docstring) — expect it to stiffen, not to move.
    """
    from unitree_deploy.robot.robot_utils import make_robot_config, make_robot_from_config

    from ego2g1.deploy._util import dds_init
    from ego2g1.deploy.core.executor import UnitreeExecutor

    if iface is not None:
        dds_init(iface)

    ee = UnitreeExecutor._EE_LAYOUTS.get(robot_type)
    if ee is None:
        raise SystemExit(
            f"robot_type={robot_type!r} has no end-effector wire layout in "
            f"UnitreeExecutor._EE_LAYOUTS (known: "
            f"{sorted(UnitreeExecutor._EE_LAYOUTS)})")
    per_hand = ee["per_hand"]
    expect = _actions.ARM_DOF + per_hand * len(layout.HANDS)

    cfg = make_robot_config(robot_type)
    cfg.cameras = {}          # we own the camera path; do not open a second client
    robot = make_robot_from_config(cfg)

    print(f"robot_type      : {robot_type}")
    print(f"end-effector    : {per_hand} motor(s)/hand, command limits {ee['limits']}")
    print(f"expected state  : {expect} = arm({_actions.ARM_DOF}) + "
          f"{per_hand}x{len(layout.HANDS)} gripper")
    print(f"motors declared : "
          f"{ {k: list(getattr(v, 'motors', {})) for k, v in cfg.endeffector.items()} }")
    print()

    robot.connect()
    try:
        obs = robot.capture_observation()
        raw = np.asarray(obs["observation.state"].numpy(), dtype=np.float64).reshape(-1)
    finally:
        robot.disconnect()

    # --- 1. observation width ------------------------------------------------
    print(f"observation.state shape : {raw.shape}")
    if raw.shape[0] != expect:
        print(f"  ✗ MISMATCH — expected {expect}. `UnitreeExecutor.arm_q`/`ee_q` "
              "slice this by position, so a different width means both are "
              "reading the wrong numbers. Fix _EE_LAYOUTS before going further.")
    else:
        print(f"  ✓ matches the assumed layout")

    arm = raw[: _actions.ARM_DOF]
    tail = raw[_actions.ARM_DOF:]
    print(f"\narm (first {_actions.ARM_DOF}), rad:")
    for h in layout.HANDS:
        print(f"  {h:5s}: " + " ".join(f"{v:7.3f}" for v in arm[layout.ARM_SLICE[h]]))
    print(f"\ntail ({tail.shape[0]} values): " + " ".join(f"{v:.4f}" for v in tail))

    # --- 2. gripper identification + units -----------------------------------
    print("\ngripper, as ego2g1 would read it (ee_q):")
    for i, h in enumerate(layout.HANDS):
        v = float(tail[i * per_hand]) if tail.shape[0] > i * per_hand else float("nan")
        span = TRAIN_GRIP_OPEN - TRAIN_GRIP_CLOSED
        frac = (v - TRAIN_GRIP_CLOSED) / span if span else float("nan")
        where = ("CLOSED-ish" if frac < 0.15 else
                 "OPEN-ish" if frac > 0.85 else "mid-travel")
        flag = "" if -0.2 <= frac <= 1.2 else "   <-- OUTSIDE the training range!"
        print(f"  {h:5s}: {v:8.4f}   ({frac * 100:6.1f}% of the training "
              f"{TRAIN_GRIP_CLOSED}..{TRAIN_GRIP_OPEN} span, {where}){flag}")
    print("\n  Move the gripper by hand (or with the vendor tool) and re-run: the "
          "\n  number must sweep the same 1.20..5.40 range the training data uses. "
          "\n  If it does not, the model's output is NOT in this robot's units and "
          "\n  umi_eef would command nonsense — stop here.")

    # --- 3. the command vector we would build --------------------------------
    row = np.zeros(_actions.ROBOT_DIM)
    row[_actions.ARM] = arm                       # hold the measured pose
    for i, h in enumerate(layout.HANDS):
        row[_actions.HAND[h]][0] = float(tail[i * per_hand]) if tail.shape[0] > i else 0.0
    probe = UnitreeExecutor.__new__(UnitreeExecutor)
    probe._ee = ee
    wire = probe._wire_row(row)
    print(f"\n_wire_row: canonical ({_actions.ROBOT_DIM},) -> {wire.shape} for send_action")
    print("  " + " ".join(f"{v:.3f}" for v in wire))
    if wire.shape[0] != expect:
        print(f"  ✗ MISMATCH — send_action expects {expect}")
    elif np.allclose(wire, raw, atol=1e-3):
        print("  ✓ round-trips: a hold-in-place command reproduces the measured "
              "state exactly, so the observation slicing and the command "
              "assembly agree on the layout")
    else:
        print("  ✗ the hold-in-place command does NOT match the measured state — "
              "one of the two mappings is transposed:")
        print("    measured: " + " ".join(f"{v:.3f}" for v in raw))
    print("\nNo action was sent. The arm was energized to hold its current pose "
          "by connect(), and released again by disconnect().")
