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


# --- model-space row guards ----------------------------------------------------
# Defined HERE (next to the layouts they check) and re-exported by safety.py
# for its callers/tests; safety.py already imports this module, so the
# dependency can only point this way.


def sanity_check_model_action(action) -> bool:
    """Cheap guard on a (30,) relative_eef row before it reaches the IK.

    A mis-normalized or corrupted chunk shows up as non-finite values or a
    delta metres long. Catching it here means it never becomes a pose."""
    a = np.asarray(action)
    if a.shape != (layout.DIM,) or not np.all(np.isfinite(a)):
        return False
    for h in layout.HANDS:
        if np.linalg.norm(a[layout.EEF[h]][:3]) > 1.5:  # a 1.5 m single-chunk delta is nonsense
            return False
    return True


def sanity_check_relation_action(action) -> bool:
    """Same guard for a (14,) relation_eef row: translation delta ≤ 1.5 m,
    rotvec magnitude ≤ 2π (any legitimate Rodrigues vector is ≤ π; 2π leaves
    slack for an unwrapped encoding, beyond that it's garbage), raw gripper
    within a loose ±3 of its {-1,+1} convention. The 30-dim guard existed
    from day one; this one was missing until the refactor — the only mode
    whose state comes from live perception had no model-space check."""
    a = np.asarray(action)
    if a.shape != (relation_layout.ACTION_DIM,) or not np.all(np.isfinite(a)):
        return False
    for h in relation_layout.HANDS:
        eef = a[relation_layout.EEF6[h]]
        if np.linalg.norm(eef[:3]) > 1.5:
            return False
        if np.linalg.norm(eef[3:]) > 2.0 * np.pi:
            return False
        if abs(float(a[relation_layout.GRIP[h]][0])) > 3.0:
            return False
    return True


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


class _EEFChunksBase:
    """The measured jitter-fix pipeline, shared by both EEF modes
    (docs/deploy_refactor_plan.md §2.2 — these were two ~95%-identical
    classes before):

        anchor = FK(measured arm q at the observation tick)   # pelvis frame
        ground the IK at the measured q                        # close the loop
        per row k: target_k = anchor @ self._delta(row, hand)  # mode decode
                   target_k = OneEuroSE3(target_k)             # before IK
                   q_k = DualArmIK(target_k)                   # posture->last, 0.05
                   q_k = JointFilter(q_k)                      # after IK
                   hands  = self._hand_block(row, hand)        # mode expand

    Tracking error is monitored per row; rows the QP could not reach are
    reported via `last_tracking_error` (the runner's watchdog reads it) —
    the QP silently approximates, so somebody has to ask. The per-slot
    residual PROFILE (`last_slot_errors`), not just the max, is kept: a
    residual growing with slot index means inflated deltas (per-slot rescale
    missing server-side); a flat offset from slot 0 means an anchor/frame
    bug (the 138 mm E-STOP of 2026-07-17 was diagnosed blind for lack of it).

    Subclasses fix the model-space layout with four small members:
    `chunk_dim`, `hands`, `_delta(row, hand) -> (4, 4)` (the anchor-relative
    pose decode), `_hand_block(row, hand) -> (6,)` (the hand-command
    expansion), and `_row_ok(row) -> bool` (the model-space sanity guard).
    """

    mode: str
    chunk_dim: int
    hands: tuple

    def __init__(self, kin=None, *, fps: int = 30, ik_iters: int = 25,
                 posture_cost: float = 0.05, collision_min_dist: float = 0.005,
                 one_euro_kwargs: dict | None = None):
        from ..kin.filters import OneEuroSE3   # numpy-only

        if kin is None:
            from .core.kinematics import Kinematics  # mujoco enters here, lazily
            kin = Kinematics(ik_iters=ik_iters, fps=fps,
                             posture_cost=posture_cost,
                             collision_min_dist=collision_min_dist)
        self.kin = kin
        self.dt = 1.0 / float(fps)
        kw = one_euro_kwargs or {}
        self._smoother = {h: OneEuroSE3(**kw) for h in self.hands}
        self.last_tracking_error: float = 0.0
        self.last_slot_errors = np.zeros(0)
        # per-slot flange target POSITIONS (pelvis frame, post-One-Euro — the
        # pose the IK is actually judged against), for the recorder / the
        # MuJoCo replay's "where the policy wanted the hand" marker
        self.last_targets: dict[str, np.ndarray] = {}

    # --- the mode-specific decode, overridden by subclasses -------------------

    def _delta(self, row: np.ndarray, hand: str) -> np.ndarray:
        raise NotImplementedError

    def _hand_block(self, row: np.ndarray, hand: str) -> np.ndarray:
        raise NotImplementedError

    def _row_ok(self, row: np.ndarray) -> bool:
        raise NotImplementedError

    # --- the shared pipeline ---------------------------------------------------

    def convert(self, actions, arm_q14, hand_cmds: dict) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != self.chunk_dim:
            raise ValueError(
                f"{self.mode} mode expects (H, {self.chunk_dim}), got {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise ValueError(f"{self.mode} chunk contains non-finite values")
        for k, row in enumerate(actions):
            if not self._row_ok(row):
                raise ValueError(
                    f"{self.mode} chunk row {k} fails the model-space sanity "
                    "guard (delta metres long, rotation past 2π, or gripper "
                    "far outside its convention) — a mis-normalized or "
                    "corrupted chunk; refusing to make it a pose")

        anchor = self.kin.flange_poses(arm_q14)
        self.kin.ground(arm_q14)

        out = np.empty((len(actions), ROBOT_DIM), dtype=np.float64)
        slot_err = np.zeros(len(actions))
        tgt_pos = {h: np.empty((len(actions), 3)) for h in self.hands}
        for k, row in enumerate(actions):
            targets = {}
            for h in self.hands:
                T = anchor[h] @ self._delta(row, h)
                targets[h] = self._smoother[h].filter(T, self.dt)
                tgt_pos[h][k] = targets[h][:3, 3]
            out[k, ARM] = self.kin.solve(targets)
            slot_err[k] = max(self.kin.tracking_error(targets).values())
            for h in self.hands:
                out[k, HAND[h]] = self._hand_block(row, h)
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


class RelativeEEFChunks(_EEFChunksBase):
    """`relative_eef` mode: (H, 30) anchor-relative vec9 chunks -> (H, 26)
    joints, via `_EEFChunksBase`'s measured pipeline. The decode is
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
        return sanity_check_model_action(row)


class RelativeEEFRotvecChunks(_EEFChunksBase):
    """`relation_eef` mode: (H, 14) anchor-relative rotvec chunks -> (H, 26)
    joints, via `_EEFChunksBase`'s measured pipeline. Deploy-side analogue of
    `ego2g1.train.relation_transforms.RelativeEEFRotvecActions` (the
    training-side transform that built the ground truth this class inverts).
    The two overrides:

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
    chunk_dim = relation_layout.ACTION_DIM
    hands = relation_layout.HANDS

    def __init__(self, kin=None, *, closed_pose: dict[str, np.ndarray] | None = None,
                 **kwargs):
        super().__init__(kin, **kwargs)
        self.closed_pose = (dict(closed_pose) if closed_pose is not None
                            else dict(gripper_calib.BRAINCO_CLOSED_POSE))

    def _delta(self, row, hand):
        return rotvec.vec6_to_se3(row[relation_layout.EEF6[hand]])

    def _hand_block(self, row, hand):
        raw_grip = float(row[relation_layout.GRIP[hand]][0])
        frac = float(np.clip((raw_grip + 1.0) / 2.0, 0.0, 1.0))
        return frac * self.closed_pose[hand]

    def _row_ok(self, row):
        return sanity_check_relation_action(row)


def make_converter(action_mode: str, **kwargs):
    if action_mode == "joint":
        return JointChunks()
    if action_mode == "relative_eef":
        return RelativeEEFChunks(**kwargs)
    if action_mode == "relation_eef":
        return RelativeEEFRotvecChunks(**kwargs)
    raise ValueError(f"unknown action mode {action_mode!r} "
                     "(expected 'joint', 'relative_eef', or 'relation_eef')")
