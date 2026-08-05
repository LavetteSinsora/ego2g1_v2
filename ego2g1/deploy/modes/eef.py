"""Shared EEF machinery for the two anchor-relative modes
(docs/deploy_refactor_plan.md §2.2): the measured jitter-fix conversion
pipeline (`EEFChunksBase`) and the adapters' shared diagnostics tail
(`convert_with_diagnostics`). The per-mode subclasses live in
relative_eef.py / relation_eef.py — each is the base plus its own pose
decode and hand expansion, nothing more."""

from __future__ import annotations

import numpy as np

from .. import actions as _actions


class EEFChunksBase:
    """The measured jitter-fix pipeline, shared by both EEF modes (these
    were two ~95%-identical classes before the refactor):

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
        from ...kin.filters import OneEuroSE3   # numpy-only

        if kin is None:
            from ..core.kinematics import Kinematics  # mujoco enters here, lazily
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

        out = np.empty((len(actions), _actions.ROBOT_DIM), dtype=np.float64)
        slot_err = np.zeros(len(actions))
        tgt_pos = {h: np.empty((len(actions), 3)) for h in self.hands}
        for k, row in enumerate(actions):
            targets = {}
            for h in self.hands:
                T = anchor[h] @ self._delta(row, h)
                targets[h] = self._smoother[h].filter(T, self.dt)
                tgt_pos[h][k] = targets[h][:3, 3]
            out[k, _actions.ARM] = self.kin.solve(targets)
            slot_err[k] = max(self.kin.tracking_error(targets).values())
            for h in self.hands:
                out[k, _actions.HAND[h]] = self._hand_block(row, h)
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


def convert_with_diagnostics(adapter, out: dict, state, arm_q, hand_cmds) -> dict:
    """The EEF adapters' shared reply tail (was copy-pasted in both): keep the
    raw model-space chunk + the request state for the recorder — without them
    a bad served chunk is undiagnosable from the recording — then convert to
    joint rows and surface the converter's per-slot telemetry."""
    adapter.last_state = np.asarray(state, dtype=np.float64)
    adapter.last_raw_chunk = np.asarray(out["actions"], dtype=np.float64)
    out["actions"] = adapter._converter.convert(out["actions"], arm_q, hand_cmds)
    out["slot_errors_m"] = getattr(adapter._converter, "last_slot_errors", None)
    out["raw_chunk"] = adapter.last_raw_chunk
    out["request_state"] = adapter.last_state
    # per-slot flange target positions (pelvis frame) — replay_mujoco.py's
    # RED "where the policy wanted the hand" marker
    out["flange_targets"] = getattr(adapter._converter, "last_targets", None)
    return out
