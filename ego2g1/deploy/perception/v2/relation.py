"""Snapshot -> 56-dim hand-major relation state (plan §5.4, T2).

This is the deliverable end of the pipeline. Everything upstream exists to
produce, at every policy inference, each object's pose in each hand's flange
frame plus two grasp binaries.

    state = [ left->obj0(9)  left->obj1(9)  left->obj2(9)
              right->obj0(9) right->obj1(9) right->obj2(9)
              grasp_left grasp_right ]

exactly inverting training's
`s2_object_relations/encoding.py::T_left_tcp_object = compose(invert(left[frame]),
object_pose)`. Object ORDER is load-bearing and comes from `task_config.objects`,
already cross-checked against the checkpoint at connect time by
`task_config.validate_against_server_metadata` — a mismatched order feeds the
model one object's geometry under another's slot, which degrades quietly
instead of crashing.

THE ONE RULE: EVERYTHING FROM ONE INSTANT (T2)

    relation = inv( T_pelvis_flange(FK at t_capture) ) @ T_pelvis_object(t_capture)

The flange pose comes from the snapshot, not from fresh FK at send time. An
earlier design composed the stale object pose with fresh FK — free, exact,
continuously available, the standard GPS/IMU pattern. Two things kill it here:

 1. The policy ALSO receives the image, and that image is the frame from
    `t_capture`. Advancing FK but not the image desynchronises two modalities
    that were synchronised in every training sample: the image shows the arm
    where it was, the state vector claims where it is now. Nothing downstream
    can reconcile that.
 2. Fresh-FK composition is only CORRECT if the object is static, and objects
    move while ungrasped (plan §2.3).

What that gives up — the arm's motion between capture and execution — is
corrected on the action side instead, by `d` (T3), which assumes the commanded
chunk was followed. That assumption is sound: the arm is executing our own
chunk.

THREADING
Driven entirely by the CONTROL thread. It calls `on_snapshot` when a new
snapshot appears (detected by `seq`) and `on_control_tick` every tick. The
latch state machines are therefore single-threaded despite spanning two rates
— the perception thread publishes an immutable snapshot and touches nothing
here.
"""

from __future__ import annotations

import logging

import numpy as np

from ....core import relation_layout, se3 as _se3
from .latch import GraspLatch, LatchConfig, LatchResult

logger = logging.getLogger(__name__)

__all__ = ["RelationStateBuilder"]


class RelationStateBuilder:
    """Latch resolution + 56-dim packing, over `PerceptionSnapshot`s.

    Deliberately thin. Detection, depth, filtering and orientation all happen
    inside the perception round and arrive already resolved on the snapshot;
    what is left is the part that must run at control rate (the latch's grasp
    trigger) and the part that must run at send time (the packing).
    """

    def __init__(self, task_config, *, latch_config: LatchConfig | None = None):
        self.task_config = task_config
        self.objects = tuple(task_config.objects)
        self.hands = tuple(task_config.hands)
        self._graspable = {o.instance_id for o in self.objects if o.graspable}
        self._latches = {h: GraspLatch(h, latch_config) for h in self.hands}
        self._last_seq: int | None = None
        self.last_latch_results: dict[str, LatchResult] = {}

        n_expected = relation_layout.N_OBJECTS
        if len(self.objects) != n_expected:
            # The 56-dim layout is fixed by the checkpoint. A roster of a
            # different size cannot be packed into it, and discovering that at
            # the first inference — with the robot live — is too late.
            raise ValueError(
                f"task config has {len(self.objects)} objects but the "
                f"relation state layout is fixed at {n_expected} "
                f"({relation_layout.RELATION_STATE_DIM}-dim). Reconcile the "
                "task config with the connected checkpoint before starting.")

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Discard all latch state. Called once after the startup probe
        inference and at episode boundaries — never on a routine pause, where
        forcing a genuinely-held object back through a full re-confirmation
        would cost a re-latch delay for no benefit."""
        for latch in self._latches.values():
            latch.reset()
        self._last_seq = None
        self.last_latch_results = {}

    @property
    def latches(self) -> dict[str, GraspLatch]:
        return dict(self._latches)

    # -- the two clocks -----------------------------------------------------

    def on_snapshot(self, snapshot) -> bool:
        """Feed a perception round to the latches. Idempotent per `seq`.

        Returns True if this snapshot was new. The control loop calls it every
        tick with whatever the newest snapshot is; only the first call for a
        given round does anything, so the caller does not have to track
        freshness itself.
        """
        if self._last_seq is not None and snapshot.seq <= self._last_seq:
            return False
        if self._last_seq is not None and snapshot.seq > self._last_seq + 1:
            # Not fatal — the latch tolerates gaps — but it means the control
            # loop skipped an entire perception round, which contradicts T3's
            # "a replan waits for the in-flight round" and inflates `d`.
            logger.warning("skipped perception round(s): seq %s -> %s",
                           self._last_seq, snapshot.seq)
        self._last_seq = snapshot.seq
        for latch in self._latches.values():
            latch.on_snapshot(snapshot)
        return True

    def on_control_tick(self, *, t: float, flange_poses: dict[str, np.ndarray],
                        hand_frac: dict[str, float]) -> dict[str, LatchResult]:
        """One 30 Hz tick. `hand_frac` is the last-COMMANDED gripper fraction
        (0 open .. 1 closed), rounded at 0.5 for the latch's closed gate — the
        same convention `modes/relation_eef` decodes and the state vector's
        own grasp bit uses."""
        results: dict[str, LatchResult] = {}
        for hand in self.hands:
            # Cross-hand claiming: an object held by one hand leaves the
            # other's candidate pool, mirroring training's `claimed` set,
            # which is built once per frame from both hands before either
            # hand's nearest-object search runs.
            claimed = {self._latches[o].latched_object
                       for o in self.hands if o != hand}
            claimed.discard(None)
            results[hand] = self._latches[hand].on_control_tick(
                hand_closed=float(hand_frac[hand]) >= 0.5,
                hand_pose=flange_poses[hand],
                t=t,
                eligible_objects=self._graspable - claimed,
            )
        self.last_latch_results = results
        return results

    # -- the deliverable ----------------------------------------------------

    def resolve(self, snapshot) -> dict[str, np.ndarray | None]:
        """Per-object pelvis-frame pose: the rigid prediction for whichever
        hand holds it, else what perception believes.

        The rigid prediction is composed against the SNAPSHOT's flange pose,
        not the current one — §6.7 and T2. Evaluating it at "now" would put
        the object where the arm is now while the image shows where the arm
        was, reintroducing exactly the desynchronisation T2 removes.
        """
        resolved: dict[str, np.ndarray | None] = {}
        for obj in self.objects:
            oid = obj.instance_id
            pose = snapshot.object_pose_pelvis.get(oid)
            for hand in self.hands:
                pose = self._latches[hand].object_pose(
                    oid, pose, snapshot.flange_pelvis[hand])
            resolved[oid] = pose
        return resolved

    def state_for(self, snapshot) -> np.ndarray:
        """The (56,) float32 relation state for this snapshot.

        Raises if any object has never been seen. A zero vec9 is not a pose
        anywhere else in this codebase and it is not one here: the deploy loop
        should retry during a "wait until every object has been seen once"
        warm-up before the policy is ever called, exactly as the runner
        already waits for its first inference chunk before arming the
        starvation watchdog. (Plan Q4 asks what should fill an empty slot
        mid-rollout; until that is answered, failing loud is the only option
        that cannot silently mis-serve.)
        """
        resolved = self.resolve(snapshot)
        missing = [oid for oid, pose in resolved.items() if pose is None]
        if missing:
            raise RuntimeError(
                f"object(s) {missing} have never been detected — no pose to "
                "report. Wait for every roster slot to be seen at least once "
                "before calling the policy.")

        blocks = []
        for hand in self.hands:
            flange_inv = _se3.se3_inv(
                np.asarray(snapshot.flange_pelvis[hand], dtype=np.float64))
            for obj in self.objects:
                blocks.append(_se3.se3_to_vec9(
                    flange_inv @ resolved[obj.instance_id]))
        grasp = np.array([1.0 if snapshot.hand_frac[h] >= 0.5 else 0.0
                          for h in self.hands], dtype=np.float32)
        state = np.concatenate([*blocks, grasp]).astype(np.float32)

        if state.shape != (relation_layout.RELATION_STATE_DIM,):
            raise AssertionError(
                f"packed {state.shape} but the layout is "
                f"({relation_layout.RELATION_STATE_DIM},)")
        return state

    # -- diagnostics --------------------------------------------------------

    def debug_snapshot(self, snapshot) -> dict:
        """JSON-safe view of what produced this state vector.

        One definition of "the perception debug state", shared by the recorder
        and the dashboard, so live and replayed panels cannot diverge. Masks
        are deliberately absent — they are large, they belong to the
        perception round, and the snapshot already carries the areas and
        scores that explain a gating decision.
        """
        resolved = self.resolve(snapshot)
        objects = {}
        for obj in self.objects:
            oid = obj.instance_id
            pose = resolved[oid]
            objects[oid] = {
                "pose_pelvis": None if pose is None else np.asarray(pose).tolist(),
                "det_score": snapshot.det_score.get(oid),
                "tracker_score": snapshot.tracker_score.get(oid),
                "mask_area_px": snapshot.mask_area_px.get(oid),
                "mask_usable": snapshot.mask_usable.get(oid),
                "crop_usable": snapshot.crop_usable.get(oid),
                "depth_m": snapshot.object_depth_m.get(oid),
            }
        hands = {}
        for hand, result in self.last_latch_results.items():
            hands[hand] = {
                "state": result.state.value,
                "latched_object": result.latched_object,
                "candidate_object": result.candidate_object,
                "reason": result.reason,
                "flange_travel_m": result.flange_travel_m,
                "divergence_m": result.divergence_m,
                "usable_observations": result.usable_observations,
                "stale_s": result.stale_s,
            }
        return {
            "seq": snapshot.seq,
            "n_capture": snapshot.n_capture,
            "round_s": snapshot.round_s,
            "objects": objects,
            "hands": hands,
        }
