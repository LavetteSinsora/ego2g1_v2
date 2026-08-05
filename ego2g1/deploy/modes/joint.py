"""`joint` mode, complete in one file (docs/deploy_refactor_plan.md §1):
the model space IS the executor space (ZH-style) — absolute joint chunks
pass through validated, never IK'd. The slot a future joint-space ego2g1
policy drops into. `JointChunks`/`JointPolicyAdapter` are re-exported under
their historical `actions`/`policy_adapter` names."""

from __future__ import annotations

import numpy as np

from ...core import layout
from .. import actions as _actions
from . import base


class JointChunks:
    """`joint` mode: absolute joint chunks pass through, validated, never IK'd.

    Accepts (H, 26) rows or (H, 14) arm-only rows; the latter are padded with
    the observation's held hand command (absolute hand dims must still be
    COMMANDED every tick or the Brainco driver holds stale state).
    """

    mode = "joint"

    def convert(self, actions, arm_q14, hand_cmds: dict) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] not in (_actions.ARM_DOF,
                                                         _actions.ROBOT_DIM):
            raise ValueError(
                f"joint mode expects (H, {_actions.ARM_DOF}) or "
                f"(H, {_actions.ROBOT_DIM}), got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("joint chunk contains non-finite values")
        out = np.empty((len(actions), _actions.ROBOT_DIM), dtype=np.float64)
        out[:, _actions.ARM] = actions[:, :_actions.ARM_DOF]
        if actions.shape[1] == _actions.ROBOT_DIM:
            for h in layout.HANDS:
                out[:, _actions.HAND[h]] = np.clip(
                    actions[:, _actions.HAND[h]], 0.0, 1.0)
        else:
            for h in layout.HANDS:
                out[:, _actions.HAND[h]] = np.clip(
                    np.asarray(hand_cmds[h], dtype=np.float64), 0.0, 1.0)
        return out

    def reset(self) -> None:
        pass


class JointPolicyAdapter:
    """`joint` mode: the model space IS the executor space (ZH-style).

    State up: (26,) [arm14 | handL6 | handR6]. Actions down: (H, 14) or
    (H, 26), validated and hand-padded by JointChunks. The RTC prefix,
    when present, passes through unchanged — joint rows are already model
    space.
    """

    mode = "joint"

    def __init__(self, client, prompt: str = ""):
        self._client = client
        self._converter = JointChunks()
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


class JointMode(base.ProprioModeBase):
    name = "joint"
    supports_rtc = True            # joint rows are already model space
    supports_reset_to_episode = True

    def build_adapter(self, client, args, fps: int):
        return JointPolicyAdapter(client, args.prompt)


JOINT = JointMode()
base.register(JOINT)
