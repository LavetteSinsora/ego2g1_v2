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
