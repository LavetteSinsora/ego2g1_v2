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
  relation_eef   (H, 14) anchor-relative EEF deltas, per hand [dx,dy,dz,
                 rx,ry,rz] (translation + ROTATION-VECTOR, not 6D), plus one
                 RAW binary gripper dim per hand at the tail
                 (ego2g1.core.relation_layout, EgoRelationTrainConfig). Same
                 measured pipeline as relative_eef — OneEuroSE3 -> DualArmIK
                 posture-tracks-last @ 0.05 -> JointFilter — only the pose
                 decode (rotvec instead of vec9/6D) and the gripper expansion
                 (binary -> 6 motors via gripper_calib.BRAINCO_CLOSED_POSE,
                 a measurement placeholder) differ. See
                 docs/relation_deploy_plan.md §3.

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

from ..core import layout, relation_layout, rotvec, se3
from . import gripper_calib

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
        self.last_slot_errors = np.zeros(0)
        # per-slot flange target POSITIONS (pelvis frame, post-One-Euro — the
        # pose the IK is actually judged against), for the recorder / the
        # MuJoCo replay's "where the policy wanted the hand" marker
        self.last_targets: dict[str, np.ndarray] = {}

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
        slot_err = np.zeros(len(actions))
        tgt_pos = {h: np.empty((len(actions), 3)) for h in layout.HANDS}
        for k, row in enumerate(actions):
            targets = {}
            for h in layout.HANDS:
                T = se3.compose(anchor[h], row[layout.EEF[h]])
                targets[h] = self._smoother[h].filter(T, self.dt)
                tgt_pos[h][k] = targets[h][:3, 3]
            out[k, ARM] = self.kin.solve(targets)
            slot_err[k] = max(self.kin.tracking_error(targets).values())
            for h in layout.HANDS:
                out[k, HAND[h]] = np.clip(row[layout.HAND[h]], 0.0, 1.0)
        self.last_targets = tgt_pos
        # per-slot residual PROFILE, not just the max: a residual that grows
        # with slot index means inflated deltas (e.g. per-slot rescale missing
        # server-side); a flat offset from slot 0 means an anchor/frame bug
        # (the 138 mm E-STOP of 2026-07-17 was diagnosed blind for lack of it)
        self.last_slot_errors = slot_err
        self.last_tracking_error = float(slot_err.max()) if len(slot_err) else 0.0
        return out

    def reset(self) -> None:
        """Episode start / after an e-stop: clear all causal filter state so the
        first chunk is not blended with a stale trajectory."""
        for s in self._smoother.values():
            s.reset()
        self.kin.reset()


class RelativeEEFRotvecChunks:
    """`relation_eef` mode: (H, 14) anchor-relative rotvec chunks -> (H, 26) joints.

    Deploy-side analogue of `ego2g1.train.relation_transforms
    .RelativeEEFRotvecActions` (the training-side transform that built the
    ground truth this class inverts). Structurally identical to
    `RelativeEEFChunks` — same FK anchor -> OneEuroSE3 -> DualArmIK
    (posture-tracks-last @ 0.05) -> JointFilter pipeline, same
    `last_tracking_error`/`last_slot_errors`/`last_targets` telemetry contract
    — only two things differ:

      pose decode  rotvec instead of vec9/6D: per hand, per row k,
                       delta_T = core.rotvec.vec6_to_se3(row[EEF6[h]])   # [t(3), rotvec(3)]
                       target  = anchor[h] @ delta_T
                   exactly inverting `RelativeEEFRotvecActions`'s
                       delta_k = inv(T_current) @ T_target_k
                       row[k]  = [delta_k.t, mat_to_rotvec(delta_k.R)]

      gripper      one RAW binary dim per hand (not the 6-dim absolute
                   Revo2 command RelativeEEFChunks reads straight off the
                   action row). `PerSlotQuantizeActionsInverse` explicitly
                   EXEMPTS `gripper_dims` from its inverse-quantile transform
                   (ego2g1/train/relation_transforms.py), so the value here
                   arrives exactly as the model produced it, in the same
                   {-1,+1}-ish convention `RelativeEEFRotvecActions` encoded
                   {0 open, 1 closed} into (`grip = target * 2 - 1`):
                       frac = clip((raw_grip + 1) / 2, 0, 1)
                       cmd  = frac * closed_pose[hand]            # (6,) motor cmd
                   `closed_pose` is a PER-ROBOT MEASUREMENT that does not
                   exist yet; it defaults to
                   `gripper_calib.BRAINCO_CLOSED_POSE`, a documented
                   placeholder ("every motor fully closed") — swapping in the
                   real measured arrays means editing only
                   `gripper_calib.py`, nothing here.
    """

    mode = "relation_eef"

    def __init__(self, kin=None, *, fps: int = 30, ik_iters: int = 25,
                 posture_cost: float = 0.05, collision_min_dist: float = 0.005,
                 one_euro_kwargs: dict | None = None,
                 closed_pose: dict[str, np.ndarray] | None = None):
        from ..kin.filters import OneEuroSE3   # numpy-only

        if kin is None:
            from .kinematics import Kinematics  # mujoco enters here, lazily
            kin = Kinematics(ik_iters=ik_iters, fps=fps,
                             posture_cost=posture_cost,
                             collision_min_dist=collision_min_dist)
        self.kin = kin
        self.dt = 1.0 / float(fps)
        kw = one_euro_kwargs or {}
        self._smoother = {h: OneEuroSE3(**kw) for h in relation_layout.HANDS}
        self.closed_pose = (dict(closed_pose) if closed_pose is not None
                            else dict(gripper_calib.BRAINCO_CLOSED_POSE))
        self.last_tracking_error: float = 0.0
        self.last_slot_errors = np.zeros(0)
        # per-slot flange target POSITIONS (pelvis frame, post-One-Euro), same
        # meaning as RelativeEEFChunks.last_targets — the recorder / MuJoCo
        # replay's "where the policy wanted the hand" marker
        self.last_targets: dict[str, np.ndarray] = {}

    def convert(self, actions, arm_q14, hand_cmds: dict) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != relation_layout.ACTION_DIM:
            raise ValueError(
                f"relation_eef mode expects (H, {relation_layout.ACTION_DIM}), "
                f"got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("relation_eef chunk contains non-finite values")

        anchor = self.kin.flange_poses(arm_q14)
        self.kin.ground(arm_q14)

        out = np.empty((len(actions), ROBOT_DIM), dtype=np.float64)
        slot_err = np.zeros(len(actions))
        tgt_pos = {h: np.empty((len(actions), 3)) for h in relation_layout.HANDS}
        for k, row in enumerate(actions):
            targets = {}
            for h in relation_layout.HANDS:
                delta_T = rotvec.vec6_to_se3(row[relation_layout.EEF6[h]])
                T = anchor[h] @ delta_T
                targets[h] = self._smoother[h].filter(T, self.dt)
                tgt_pos[h][k] = targets[h][:3, 3]
            out[k, ARM] = self.kin.solve(targets)
            slot_err[k] = max(self.kin.tracking_error(targets).values())
            for h in relation_layout.HANDS:
                raw_grip = float(row[relation_layout.GRIP[h]][0])
                frac = float(np.clip((raw_grip + 1.0) / 2.0, 0.0, 1.0))
                out[k, HAND[h]] = frac * self.closed_pose[h]
        self.last_targets = tgt_pos
        self.last_slot_errors = slot_err
        self.last_tracking_error = float(slot_err.max()) if len(slot_err) else 0.0
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
    if action_mode == "relation_eef":
        return RelativeEEFRotvecChunks(**kwargs)
    raise ValueError(f"unknown action mode {action_mode!r} "
                     "(expected 'joint', 'relative_eef', or 'relation_eef')")
