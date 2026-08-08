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

from ..core import layout, relation_layout, umi_layout

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


# Widest plausible Dex1 gripper command, radians of gear rotation. The training
# data spans 1.20 (fully closed) .. 5.40 (fully open); this is a corruption
# guard with generous slack, NOT a calibration — the model's value is executed
# faithfully within it, never rescaled.
UMI_GRIP_LIMIT_RAD = 8.0


def sanity_check_umi_action(action) -> bool:
    """Guard on a (7,) umi_eef row before it reaches the IK.

    Same shape of check as the relation guard: translation delta <= 1.5 m,
    rotvec magnitude <= 2*pi (any legitimate Rodrigues vector is <= pi; 2*pi
    leaves slack for an unwrapped encoding, beyond that it is garbage). The
    gripper is CONTINUOUS here and in radians, not a {-1,+1} binary, so it is
    bounded by the physical travel rather than by a convention.
    """
    a = np.asarray(action)
    if a.shape != (umi_layout.ACTION_DIM,) or not np.all(np.isfinite(a)):
        return False
    eef = a[umi_layout.EEF6]
    if np.linalg.norm(eef[:3]) > 1.5:
        return False
    if np.linalg.norm(eef[3:]) > 2.0 * np.pi:
        return False
    if abs(float(a[umi_layout.GRIP][0])) > UMI_GRIP_LIMIT_RAD:
        return False
    return True


# --- the boundary ---------------------------------------------------------------
# The converter classes live with their modes now (docs/deploy_refactor_plan
# .md §1: one file per policy family — modes/joint.py, modes/relative_eef.py,
# modes/relation_eef.py, shared machinery in modes/eef.py). Re-exported here
# under their historical names, LAZILY (module __getattr__, PEP 562): an
# eager `from .modes... import ...` at module scope would be circular, since
# the modes files import THIS module for the row-layout constants above.

_MOVED = {
    "JointChunks": ("joint", "JointChunks"),
    "RelativeEEFChunks": ("relative_eef", "RelativeEEFChunks"),
    "RelativeEEFRotvecChunks": ("relation_eef", "RelativeEEFRotvecChunks"),
}


def __getattr__(name):
    if name in _MOVED:
        import importlib

        mod, cls = _MOVED[name]
        return getattr(importlib.import_module(f"{__package__}.modes.{mod}"), cls)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_converter(action_mode: str, **kwargs):
    if action_mode == "joint":
        return __getattr__("JointChunks")()
    if action_mode == "relative_eef":
        return __getattr__("RelativeEEFChunks")(**kwargs)
    if action_mode == "relation_eef":
        return __getattr__("RelativeEEFRotvecChunks")(**kwargs)
    raise ValueError(f"unknown action mode {action_mode!r} "
                     "(expected 'joint', 'relative_eef', or 'relation_eef')")
