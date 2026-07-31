"""Binary gripper -> BrainCo motor-command mapping (`relation_eef` mode only).

`EgoRelationTrainConfig` trains one binary gripper dim per hand (open/closed),
but the real BrainCo hand takes a 6-motor command
(`ego2g1.core.hand.constants.MOTOR_ORDER`, 0=open, 1=closed each). There is no
principled way to derive "what fully closed around THIS task's objects looks
like, per finger" from a binary training signal that never specified
individual finger commands — a human has to measure it on the real hand
(docs/relation_deploy_plan.md, §7).

`BRAINCO_CLOSED_POSE` below is a PLACEHOLDER: "every motor at 1.0" (fully
closed, `core/hand/constants.py`'s own 0=open/1=closed convention), used only
because the real measured values do not exist yet. Replace the two arrays
here, in this module only, once measured — `RelativeEEFRotvecChunks` imports
this constant and does `frac * BRAINCO_CLOSED_POSE[hand]`
(`ego2g1/deploy/actions.py`); nothing about that call site needs to change.
"""

import numpy as np

# TODO(calibration): measure per hand, per task object, in MOTOR_ORDER
# (thumb_flex, thumb_rot, index, middle, ring, pinky). PLACEHOLDER until then.
BRAINCO_CLOSED_POSE: dict[str, np.ndarray] = {
    "left": np.ones(6, dtype=np.float32),
    "right": np.ones(6, dtype=np.float32),
}


def frac_from_command(cmd, closed_pose) -> float:
    """Invert `cmd = frac * closed_pose` (`RelativeEEFRotvecChunks.convert`'s
    own construction, `ego2g1/deploy/actions.py`) back to the scalar
    open/closed FRACTION, from an already-EXECUTED 6-motor command alone.

    Why from the executed command rather than from the converter's own
    per-chunk `frac` (which it computes internally but never surfaces):
    `runner.py`'s `relation_eef` mode needs "last commanded fraction" at pop
    time, one row at a time, and by then the async strategies (strategies.py)
    may have already blended/reindexed which chunk SLOT this row came from —
    reaching into the adapter/converter to recover which slot's `frac` this
    is would mean threading chunk-seam bookkeeping through the (deliberately
    mode-blind) executor path. Inverting the relationship directly from what
    was actually sent to the hand needs none of that.

    Least-squares projection (a plain dot-product ratio): the `frac`
    minimizing `||frac * closed_pose - cmd||^2`. EXACT whenever `cmd` really
    is `frac * closed_pose` — which it always is immediately downstream of
    `RelativeEEFRotvecChunks.convert`, since `runner.py`'s safety clamp only
    ever touches the ARM slice of an executed row, never a HAND slice.
    Clipped to [0, 1] (a hand fraction is never negative or over-closed).

    Returns 0.0 (treat as "open" / "no information") if `closed_pose` is the
    zero vector — nothing to project onto. That is a calibration problem
    (an all-zero `BRAINCO_CLOSED_POSE` entry), not a per-tick one, and
    `hand_cmds_last` degrading to "open" is the safe direction for it to
    fail in (the latch state machine in `perception/latch.py` treats "open"
    as "not gripping", never a false-positive grasp).
    """
    cmd = np.asarray(cmd, dtype=np.float64)
    closed_pose = np.asarray(closed_pose, dtype=np.float64)
    denom = float(np.dot(closed_pose, closed_pose))
    if denom <= 0.0:
        return 0.0
    frac = float(np.dot(cmd, closed_pose) / denom)
    return float(np.clip(frac, 0.0, 1.0))
