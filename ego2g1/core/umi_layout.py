"""The 7-dim UMI action layout, and the state-history shape.

Sibling of `ego2g1.core.layout` (30-dim relative_eef) and
`ego2g1.core.relation_layout` (14-dim relation_eef) — NOT a replacement. The
three describe different, incompatible action contracts and their slices must
never be cross-used.

Action layout (`UmiTrainConfig`, `ego2g1/train/umi_transforms.py`'s
`UmiRelativeActions`). ONE acting arm, anchor-relative translation +
ROTATION-VECTOR, continuous gripper at the tail::

    [dx dy dz  rx ry rz | grip]
     \\_____ 6 _______/   \\_1_/

State history (`UmiStateHistory`), one row per lag, most recent FIRST::

    row_j = [dx dy dz  rx ry rz | grip]      # tick t - lag_ticks[j], in the
                                             # ANCHOR's frame

The deploy side sends ABSOLUTE poses and lets the server's own
`UmiStateHistory` do the anchor composition — see
`ego2g1/deploy/modes/umi_eef.py`. That is deliberate: the composition is then
literally the same code at train and at serve time, so it cannot drift.

This is DERIVED to stay consistent with `UmiTrainConfig.gripper_dims` and
`.history_dim` (properties, not constants) — see tests/core/test_umi_layout.py,
which imports the real config class and asserts the two agree rather than
trusting the numbers below by eyeball.
"""

import numpy as np

# Which arm the policy drives. The other one holds its pose and contributes
# only its camera (the workspace context view this setup uses in place of the
# head camera it does not have).
ACTING_HAND = "right"
IDLE_HAND = "left"
HANDS = ("left", "right")

EEF6_DIM = 6                       # [dx, dy, dz, rx, ry, rz]
ACTION_DIM = EEF6_DIM + 1          # 7, one arm

EEF6 = slice(0, EEF6_DIM)
GRIP = slice(EEF6_DIM, EEF6_DIM + 1)

# --- state history -------------------------------------------------------------

POSE_DIM = 9                       # vec9: [tx, ty, tz, R[:,0], R[:,1]]
HISTORY_DIM = EEF6_DIM + 1         # 7: the same [eef6 | grip] shape as an action

# Defaults matching UmiTrainConfig's; the SERVED values come from the handshake
# (`client.metadata["ego2g1"]["lag_ticks"]`), never from these.
DEFAULT_HISTORY_LAGS = 5
DEFAULT_HISTORY_STRIDE = 3


def default_lag_ticks() -> tuple[int, ...]:
    return tuple(j * DEFAULT_HISTORY_STRIDE for j in range(DEFAULT_HISTORY_LAGS + 1))


def split_action_row(row):
    """(7,) -> {"eef6": (6,), "grip": (1,)}."""
    row = np.asarray(row)
    if row.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected (..., {ACTION_DIM}), got {row.shape}")
    return {"eef6": row[..., EEF6], "grip": row[..., GRIP]}
