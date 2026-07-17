"""The action-mode boundary: policy chunks in, JOINT chunks out.

This is the core abstraction of the deploy layer. A policy chunk enters in one
of two modes and leaves as the SAME thing either way — an (H, 26) array of
absolute joint-space rows in executor order — so everything downstream
(strategies, clamp, executor) is mode-blind:

  joint          (H, 14) or (H, 26) absolute joint targets. Executed directly,
                 NO IK anywhere (ZH-style; their policy emits joints learned
                 from real robot joints — smooth by construction, and the mode
                 a future joint-space ego2g1 policy plugs into). 14-dim chunks
                 are padded with the held hand command. mujoco is never
                 imported on this path.
  relative_eef   (H, 30) anchor-relative EEF deltas + absolute hand commands
                 (current ego2g1 checkpoints, ego2g1.core.layout). Converted
                 HERE, by the measured jitter fix (docs/jitter_root_cause.md):
                     OneEuroSE3 on the composed targets      (before IK)
                     DualArmIK, posture-tracks-last @ 0.05   (the solver)
                     JointFilter 4-tap                       (after IK)
                 with the IK re-grounded at the measured joints each chunk and
                 warm-carried row to row within it.

Executor row layout (unitree_deploy `unitree_g1_brainco`, robot_configs
g1_motors + brainco_motors — verified against the old unitree_backend.py):

    [0:14]  arm, left shoulder p/r/y, elbow, wrist r/p/y, then right — the
            exact ARM_JOINTS / DualArmIK order, no reindexing
    [14:20] left Brainco  [thumb, thumbAux, index, middle, ring, pinky], [0,1]
    [20:26] right Brainco, same order

Why chunks convert WHOLE, at inference time (ZH's EEFPolicyAdapter pattern),
rather than row-by-row at pop time: the async strategies blend OLD and NEW
chunks at the seam, and that blend is only well-defined in joint space —
averaging two vec9 orientation encodings is not a rotation. The cost is that
the One-Euro state, which persists across chunks for continuity, re-traverses
the seam overlap; that is a bounded extra smoothing lag, not a discontinuity.
"""

import numpy as np

from ..core import layout, se3

# --- executor row layout ------------------------------------------------------

ARM_DOF = layout.ARM_DOF                    # 14
HAND_DOF = layout.HAND_DIM                  # 6 per hand
ROBOT_DIM = ARM_DOF + HAND_DOF * len(layout.HANDS)   # 26

ARM = slice(0, ARM_DOF)
HAND = {h: slice(ARM_DOF + i * HAND_DOF, ARM_DOF + (i + 1) * HAND_DOF)
        for i, h in enumerate(layout.HANDS)}


def split_row(row):
    """(26,) -> (arm (14,), {hand: (6,)})."""
    row = np.asarray(row)
    return row[ARM], {h: row[HAND[h]] for h in layout.HANDS}


# --- the boundary --------------------------------------------------------------


class JointChunks:
    """`joint` mode: absolute joint chunks pass through, validated, never IK'd.

    Accepts (H, 26) rows or (H, 14) arm-only rows; the latter are padded with
    the observation's held hand command (absolute hand dims must still be
    COMMANDED every tick or the Brainco driver holds stale state).
    """

    mode = "joint"

    def convert(self, actions, arm_q14, hand_cmds: dict) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] not in (ARM_DOF, ROBOT_DIM):
            raise ValueError(
                f"joint mode expects (H, {ARM_DOF}) or (H, {ROBOT_DIM}), got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("joint chunk contains non-finite values")
        out = np.empty((len(actions), ROBOT_DIM), dtype=np.float64)
        out[:, ARM] = actions[:, :ARM_DOF]
        if actions.shape[1] == ROBOT_DIM:
            for h in layout.HANDS:
                out[:, HAND[h]] = np.clip(actions[:, HAND[h]], 0.0, 1.0)
        else:
            for h in layout.HANDS:
                out[:, HAND[h]] = np.clip(
                    np.asarray(hand_cmds[h], dtype=np.float64), 0.0, 1.0)
        return out

    def reset(self) -> None:
        pass


class RelativeEEFChunks:
    """`relative_eef` mode: (H, 30) anchor-relative chunks -> (H, 26) joints.

    The full measured pipeline, per chunk:
        anchor = FK(measured arm q at the observation tick)   # pelvis frame
        ground the IK at the measured q                        # close the loop
        per row k: target_k = anchor @ delta_k                 # core.se3.compose
                   target_k = OneEuroSE3(target_k)             # before IK
                   q_k = DualArmIK(target_k)                   # posture->last, 0.05
                   q_k = JointFilter(q_k)                      # after IK
    Tracking error is monitored per row; rows the QP could not reach are
    reported via `last_tracking_error` (the runner's watchdog reads it) —
    the QP silently approximates, so somebody has to ask.
    """

    mode = "relative_eef"

    def __init__(self, kin=None, *, fps: int = 30, ik_iters: int = 25,
                 posture_cost: float = 0.05, collision_min_dist: float = 0.005,
                 one_euro_kwargs: dict | None = None):
        from ..kin.filters import OneEuroSE3   # numpy-only

        if kin is None:
            from .kinematics import Kinematics  # mujoco enters here, lazily
            kin = Kinematics(ik_iters=ik_iters, fps=fps,
                             posture_cost=posture_cost,
                             collision_min_dist=collision_min_dist)
        self.kin = kin
        self.dt = 1.0 / float(fps)
        kw = one_euro_kwargs or {}
        self._smoother = {h: OneEuroSE3(**kw) for h in layout.HANDS}
        self.last_tracking_error: float = 0.0

    def convert(self, actions, arm_q14, hand_cmds: dict) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != layout.DIM:
            raise ValueError(
                f"relative_eef mode expects (H, {layout.DIM}), got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("relative_eef chunk contains non-finite values")

        anchor = self.kin.flange_poses(arm_q14)
        self.kin.ground(arm_q14)

        out = np.empty((len(actions), ROBOT_DIM), dtype=np.float64)
        worst = 0.0
        for k, row in enumerate(actions):
            targets = {}
            for h in layout.HANDS:
                T = se3.compose(anchor[h], row[layout.EEF[h]])
                targets[h] = self._smoother[h].filter(T, self.dt)
            out[k, ARM] = self.kin.solve(targets)
            worst = max(worst, max(self.kin.tracking_error(targets).values()))
            for h in layout.HANDS:
                out[k, HAND[h]] = np.clip(row[layout.HAND[h]], 0.0, 1.0)
        self.last_tracking_error = worst
        return out

    def reset(self) -> None:
        """Episode start / after an e-stop: clear all causal filter state so the
        first chunk is not blended with a stale trajectory."""
        for s in self._smoother.values():
            s.reset()
        self.kin.reset()


def make_converter(action_mode: str, **kwargs):
    if action_mode == "joint":
        return JointChunks()
    if action_mode == "relative_eef":
        return RelativeEEFChunks(**kwargs)
    raise ValueError(f"unknown action mode {action_mode!r} "
                     "(expected 'joint' or 'relative_eef')")
