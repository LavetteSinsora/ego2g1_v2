"""The 14-dim relation-mode action layout, and the 56-dim relation state shape.

Sibling of `ego2g1.core.layout` (the 30-dim relative_eef layout) — NOT a
replacement. The two describe different, incompatible action contracts and
their slices must never be cross-used.

Action layout (`EgoRelationTrainConfig`, `ego2g1/train/relation_transforms.py`'s
`RelativeEEFRotvecActions`), per hand `[dx, dy, dz, rx, ry, rz]` (translation +
ROTATION-VECTOR, anchor-relative), grippers at the TAIL:

    [L_dx L_dy L_dz L_rx L_ry L_rz | R_dx ... R_rz | L_grip R_grip]
     \\_____________ 6 ____________/ \\____ 6 ____/   \\____ 2 ____/

This is DERIVED to stay consistent with `EgoRelationTrainConfig.gripper_dims`
(a property, not a constant) — see tests/core/test_relation_layout.py, which
imports the real config class and asserts the two agree, rather than trusting
the numbers below by eyeball.

State layout (`RelationPrompt.__call__`), hand-major, 3 objects fixed by this
checkpoint's `train_config.objects`:

    [left->obj0(9) left->obj1(9) left->obj2(9)
     right->obj0(9) right->obj1(9) right->obj2(9)
     grasp_left grasp_right]                        # 6*9 + 2 = 56
"""

import numpy as np

HANDS = ("left", "right")

EEF6_DIM = 6                                    # [dx, dy, dz, rx, ry, rz]
N_HANDS = len(HANDS)
ACTION_DIM = EEF6_DIM * N_HANDS + N_HANDS       # 14

# Per-hand 6-dim relative-EEF slice: hands packed first, in HANDS order.
EEF6 = {h: slice(i * EEF6_DIM, (i + 1) * EEF6_DIM) for i, h in enumerate(HANDS)}

# Tail gripper slice: one scalar per hand, immediately after both EEF6 blocks.
# Matches EgoRelationTrainConfig.gripper_dims = tuple(range(6*n_hands, 6*n_hands+n_hands)).
_GRIP_START = EEF6_DIM * N_HANDS
GRIP = {h: slice(_GRIP_START + i, _GRIP_START + i + 1) for i, h in enumerate(HANDS)}

# --- relation state (56-dim), for RelationPolicyAdapter's documented contract ---

RELATION_DIM_PER_OBJECT_PER_HAND = 9   # vec9: object pose in that hand's TCP frame
N_OBJECTS = 3                          # this checkpoint's fixed object count
RELATION_STATE_DIM = (
    RELATION_DIM_PER_OBJECT_PER_HAND * N_OBJECTS * N_HANDS + N_HANDS
)  # 9*3*2 + 2 = 56


def split_action_row(row):
    """(14,) -> {hand: {"eef6": (6,), "grip": (1,)}}."""
    row = np.asarray(row)
    if row.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected (..., {ACTION_DIM}), got {row.shape}")
    return {h: {"eef6": row[..., EEF6[h]], "grip": row[..., GRIP[h]]} for h in HANDS}
