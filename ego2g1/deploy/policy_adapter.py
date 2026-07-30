"""The policy⇄execution boundary: everything downstream sees JOINT chunks.

Pattern from zh_deploy_inference/examples/unitree_inference/policy_adapter.py
(their EEFPolicyAdapter), rebuilt for ego2g1's model contract: the adapter owns
the model-facing observation (FK state), the action-mode conversion
(actions.py), and the RTC prefix translation — so strategies.py and runner.py
are byte-identical whether the checkpoint speaks joints or anchor-relative EEF.

Runner-side request dict (built by runner._observe):

    {"arm_q":     (14,) measured arm joints at the observation tick,
     "hand_cmds": {hand: (6,)} LAST COMMANDED hand values (not encoders),
     "image":     (H, W, 3) uint8 RGB or None,
     "prompt":    str,
     # attached by AsyncStrategy when rtc=True:
     "enable_rtc": bool, "inference_delay": int ticks,
     "prev_action_chunk": (K, 26) JOINT rows — the leftover plan}

Adapter reply: {"actions": (H, 26) joint rows, ...server extras}.

RTC prefix translation (relative_eef): the strategy's leftover plan is joint
rows, but the server wants anchor-relative vec9 deltas against the NEW anchor.
FK gives the absolute flange pose of every leftover row; delta = anchor_new⁻¹ @
pose. This is `core.se3.reanchor_chunk` composed through FK — one code path,
no second delta convention. Hand dims pass through (absolute in both spaces).
"""

from __future__ import annotations

import numpy as np

from ..core import layout, se3
from . import actions as _actions


class JointPolicyAdapter:
    """`joint` mode: the model space IS the executor space (ZH-style).

    State up: (26,) [arm14 | handL6 | handR6]. Actions down: (H, 14) or
    (H, 26), validated and hand-padded by actions.JointChunks. The RTC prefix,
    when present, passes through unchanged — joint rows are already model
    space.
    """

    mode = "joint"

    def __init__(self, client, prompt: str = ""):
        self._client = client
        self._converter = _actions.JointChunks()
        self.prompt = prompt
        self.action_horizon = int(client.action_horizon)
        self.fps = int(client.fps)

    def infer(self, request: dict) -> dict:
        arm_q = np.asarray(request["arm_q"], dtype=np.float64)
        hand_cmds = request["hand_cmds"]
        state = np.concatenate(
            [arm_q] + [np.asarray(hand_cmds[h], dtype=np.float64)
                       for h in layout.HANDS])

        prev, d, n_prefix = None, 0, None
        if request.get("enable_rtc") and request.get("prev_action_chunk") is not None:
            prev_rows = np.asarray(request["prev_action_chunk"], dtype=np.float32)
            prev = np.zeros((self.action_horizon, prev_rows.shape[1]), np.float32)
            k = min(len(prev_rows), self.action_horizon)
            prev[:k] = prev_rows[:k]
            n_prefix = k
            d = int(request.get("inference_delay", 0))

        out = self._client.infer(request.get("image"), state, request.get("prompt", self.prompt),
                                 prev_chunk=prev, d=d, n_prefix=n_prefix)
        out["actions"] = self._converter.convert(out["actions"], arm_q, hand_cmds)
        return out

    def reset(self) -> None:
        self._converter.reset()


class RelativeEEFPolicyAdapter:
    """`relative_eef` mode: current ego2g1 checkpoints.

    Up:   (30,) state = measured-FK flange vec9 per hand + last hand commands.
    Down: (H, 30) anchor-relative chunk -> actions.RelativeEEFChunks (OneEuroSE3
          -> DualArmIK posture-tracks-last @ 0.05 -> JointFilter) -> (H, 26).

    `last_tracking_error` (worst IK flange error over the last converted chunk,
    metres) is surfaced for the runner's watchdog.
    """

    mode = "relative_eef"

    def __init__(self, client, prompt: str = "", *, converter=None, kin=None,
                 ik_iters: int = 25, posture_cost: float = 0.05,
                 collision_min_dist: float = 0.005):
        self._client = client
        self.prompt = prompt
        self.action_horizon = int(client.action_horizon)
        self.fps = int(client.fps)
        self._converter = converter or _actions.RelativeEEFChunks(
            kin, fps=self.fps, ik_iters=ik_iters, posture_cost=posture_cost,
            collision_min_dist=collision_min_dist)
        self._kin = self._converter.kin

    @property
    def last_tracking_error(self) -> float:
        return self._converter.last_tracking_error

    def infer(self, request: dict) -> dict:
        arm_q = np.asarray(request["arm_q"], dtype=np.float64)
        hand_cmds = request["hand_cmds"]
        state = self._kin.state(arm_q, hand_cmds)

        prev, d, n_prefix = None, 0, None
        if request.get("enable_rtc") and request.get("prev_action_chunk") is not None:
            prev, n_prefix = self._reanchor_joint_rows(
                np.asarray(request["prev_action_chunk"], dtype=np.float64), arm_q)
            d = int(request.get("inference_delay", 0))

        out = self._client.infer(request.get("image"), state,
                                 request.get("prompt", self.prompt),
                                 prev_chunk=prev, d=d, n_prefix=n_prefix)
        # keep the raw model-space chunk + the request state for the recorder:
        # without them a bad served chunk is undiagnosable from the recording
        self.last_state = np.asarray(state, dtype=np.float64)
        self.last_raw_chunk = np.asarray(out["actions"], dtype=np.float64)
        out["actions"] = self._converter.convert(out["actions"], arm_q, hand_cmds)
        out["slot_errors_m"] = getattr(self._converter, "last_slot_errors", None)
        out["raw_chunk"] = self.last_raw_chunk
        out["request_state"] = self.last_state
        # per-slot flange target positions (pelvis frame) — replay_mujoco.py's
        # RED "where the policy wanted the hand" marker
        out["flange_targets"] = getattr(self._converter, "last_targets", None)
        return out

    def _reanchor_joint_rows(self, rows, arm_q_new) -> tuple[np.ndarray, int]:
        """(K, 26) joint rows -> (H, 30) model-space prefix vs the NEW anchor.

        FK every leftover row and difference against the new observation's
        anchor. Row 0 of the result must be the action for the instant the new
        chunk's slot 0 executes — the caller (AsyncStrategy) already sliced the
        leftover, so it is. Zero-padded to H with n_prefix marking the real
        rows: a zero vec9 decodes to a det-0 matrix, not a pose, and the server
        must know where to stop (serve/policy.py enforces the same cap).
        """
        anchor_new = self._kin.flange_poses(arm_q_new)
        k = min(len(rows), self.action_horizon)
        out = np.zeros((self.action_horizon, layout.DIM), dtype=np.float32)
        for i in range(k):
            arm, hands = _actions.split_row(rows[i])
            poses = self._kin.flange_poses(arm)
            for h in layout.HANDS:
                delta = se3.se3_inv(anchor_new[h]) @ poses[h]
                out[i, layout.EEF[h]] = se3.se3_to_vec9(delta)
                out[i, layout.HAND[h]] = np.clip(hands[h], 0.0, 1.0)
        return out, k

    def reset(self) -> None:
        self._converter.reset()


def make_adapter(action_mode: str, client, prompt: str = "", **kwargs):
    if action_mode == "joint":
        return JointPolicyAdapter(client, prompt)
    if action_mode == "relative_eef":
        return RelativeEEFPolicyAdapter(client, prompt, **kwargs)
    raise ValueError(f"unknown action mode {action_mode!r}")
