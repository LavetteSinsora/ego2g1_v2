"""Training-side action construction from stored absolute poses.

The dataset stores absolute per-tick poses (s002_01) and hand commands
(s002_02); action chunks are built at load time: gather H+1 poses (row 0 =
anchor) + H hand commands via delta_timestamps, then difference the poses
relative to the anchor (SPEC.md "Loader semantics"). numpy only - no torch
or jax imports here.
"""

import numpy as np

from .rot6d import se3_to_vec9, vec9_to_se3


def make_delta_timestamps(action_horizon, fps):
    """delta_timestamps for LeRobotDataset: pose keys gather [0..H]/fps
    (anchor + chunk), hand keys gather [1..H]/fps (commands only)."""
    h = int(action_horizon)
    pose_ts = [k / fps for k in range(h + 1)]
    hand_ts = [k / fps for k in range(1, h + 1)]
    return {"pose.left": pose_ts, "pose.right": pose_ts,
            "hand.left": hand_ts, "hand.right": hand_ts}


class RelativeChunkActions:
    """Turn gathered pose/hand chunks into relative action chunks.

    Input sample (numpy): `pose.<hand>` (H+1, 9) vec9, `hand.<hand>` (H, 6)
    for each hand in `hands` order. Output: same sample without those keys,
    plus `actions` (H, (9+6)*len(hands)) f32 where per hand
    delta_k = vec9_to_se3(pose_0)^-1 @ vec9_to_se3(pose_k), k = 1..H,
    re-encoded as vec9 and concatenated [eef 9 | hand 6].
    Everything else passes through unchanged.
    """

    def __init__(self, hands=("left", "right")):
        self.hands = tuple(hands)

    def __call__(self, sample):
        out = dict(sample)
        parts = []
        for hand in self.hands:
            pose = np.asarray(out.pop(f"pose.{hand}"), dtype=np.float64)
            hand_cmds = np.asarray(out.pop(f"hand.{hand}"), dtype=np.float64)
            if pose.ndim != 2 or pose.shape[-1] != 9:
                raise ValueError(f"pose.{hand}: expected (H+1, 9), got {pose.shape}")
            if hand_cmds.shape != (pose.shape[0] - 1, 6):
                raise ValueError(
                    f"hand.{hand}: expected ({pose.shape[0] - 1}, 6), got {hand_cmds.shape}")
            T = vec9_to_se3(pose)                       # (H+1, 4, 4)
            deltas = se3_to_vec9(np.linalg.inv(T[0]) @ T[1:])   # (H, 9)
            parts.append(np.concatenate([deltas, hand_cmds], axis=-1))
        out["actions"] = np.concatenate(parts, axis=-1).astype(np.float32)
        return out
