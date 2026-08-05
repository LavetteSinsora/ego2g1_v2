"""`relative_eef` mode, complete in one file (docs/deploy_refactor_plan.md
§1): 30-dim FK proprio state up, (H, 30) anchor-relative vec9 chunks down,
through the measured jitter-fix pipeline (modes/eef.py). The current ego2g1
checkpoints. `RelativeEEFChunks`/`RelativeEEFPolicyAdapter` are re-exported
under their historical `actions`/`policy_adapter` names."""

from __future__ import annotations

import numpy as np

from ...core import layout, se3
from .. import actions as _actions
from . import base, eef


class RelativeEEFChunks(eef.EEFChunksBase):
    """`relative_eef` mode: (H, 30) anchor-relative vec9 chunks -> (H, 26)
    joints, via `EEFChunksBase`'s measured pipeline. The decode is
    `core.se3.compose` (anchor @ vec9_to_se3(delta)); hand dims are absolute
    Revo2 commands read straight off the action row, clipped to [0, 1]."""

    mode = "relative_eef"
    chunk_dim = layout.DIM
    hands = layout.HANDS

    def _delta(self, row, hand):
        return se3.vec9_to_se3(row[layout.EEF[hand]])

    def _hand_block(self, row, hand):
        return np.clip(row[layout.HAND[hand]], 0.0, 1.0)

    def _row_ok(self, row):
        return _actions.sanity_check_model_action(row)


class RelativeEEFPolicyAdapter:
    """`relative_eef` mode: current ego2g1 checkpoints.

    Up:   (30,) state = measured-FK flange vec9 per hand + last hand commands.
    Down: (H, 30) anchor-relative chunk -> RelativeEEFChunks (OneEuroSE3
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
        self._converter = converter or RelativeEEFChunks(
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
        return eef.convert_with_diagnostics(self, out, state, arm_q, hand_cmds)

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


class RelativeEEFMode(base.ProprioModeBase):
    name = "relative_eef"
    supports_rtc = True
    supports_reset_to_episode = True

    def build_adapter(self, client, args, fps: int):
        return RelativeEEFPolicyAdapter(
            client, args.prompt, ik_iters=args.ik_iters,
            posture_cost=args.posture_cost,
            collision_min_dist=args.collision_min_dist)


RELATIVE_EEF = RelativeEEFMode()
base.register(RELATIVE_EEF)
