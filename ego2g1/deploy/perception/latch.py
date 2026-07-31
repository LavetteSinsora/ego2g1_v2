"""Grasp confirmation / kinematic latching, per hand.

docs/relation_deploy_plan.md, §5.4: training's own latch heuristic
(`data_extraction_zh/src/ego_relation/s2_object_relations/encoding.py`'s
`latch_object_poses`) latches purely on "hand closed AND nearest graspable
object within `latch_distance_m`" -- fine offline, where the human
demonstrator's grasp signal is reliable and episodes are curated, but wrong at
deployment: a policy-commanded "closed" hand frequently does NOT mean the
object was actually picked up. Feeding `RelationPerception` (not this module's
concern -- see §5.5) a hallucinated "object rigidly follows the hand" relation
when the object is really still sitting on the table is worse than not
latching at all.

This module adds a confirmation gate on top of that same distance+grasp-bit
test (which still decides *which* object is a latch CANDIDATE): once a hand
closes near an eligible object, freeze the rigid hand->object transform at
that instant and, for a short confirmation window, compare the LIVE-tracked
object pose against what the rigid prediction says it should be. Only if the
two stay in tight agreement for the whole window do we trust the rigid
prediction going forward (LATCHED); if they diverge, or live tracking is lost
without ever converging, we treat it as a MISSED grasp and fall back to
UNLATCHED -- we do not silently pretend success.

Pure numpy, no camera/detector/robot dependency -- unit-testable in complete
isolation with synthetic converging/diverging trajectories (this is
deliberately the first Phase-2 perception task built, per §9 task 9: "write
this test first").
"""

import dataclasses
import enum

import numpy as np

from ego2g1.core.se3 import se3_inv
from ego2g1.core.rotvec import mat_to_rotvec

__all__ = ["LatchState", "LatchConfig", "LatchResult", "GraspLatch"]


class LatchState(enum.Enum):
    """Per-hand latch state (docs/relation_deploy_plan.md §5.4's diagram)."""

    UNLATCHED = "unlatched"
    CANDIDATE = "candidate"
    LATCHED = "latched"


@dataclasses.dataclass
class LatchConfig:
    """Tunables for one hand's `GraspLatch`.

    `latch_distance_m`: candidate-entry distance gate, same threshold CLASS as
    training's `latch_distance_m` (encoding.py's `latch_object_poses`). 0.05 m
    (5 cm) is chosen as a sensible default because it is roughly a BrainCo
    hand's own closed-aperture span at the wrist/TCP reference point used
    elsewhere in this codebase (`TCP_TO_INWARD_PALM`-style convention) -- an
    object further than that from the TCP at the instant the hand closes
    almost certainly was not the one actually contacted, whereas anything
    closer is at least kinematically plausible as a grasp target. Re-tune
    against real touch-calibration data (§6) once available; this is a
    reasonable bring-up default, not a measured constant.

    `confirm_window_ticks`: ~0.4 s at the deploy loop's 30 Hz
    (`ego2g1/deploy/runner.py`), the middle of the plan's quoted "0.3-0.5 s /
    ~10-15 ticks" confirmation window -- long enough that a real rigid grasp's
    tracked pose has visibly started moving WITH the hand (rather than one
    lucky matching frame), short enough not to noticeably delay the relation
    state from reflecting a real, successful grasp.

    `position_tol_m` / `rotation_tol_deg`: how close "converged" means. 2 cm /
    15 deg are deliberately looser than the confirm window is long-lived is
    tight: this must comfortably exceed the live tracker's own noise floor
    (detector+depth-lift jitter, OneEuro lag) while still being far tighter
    than "object visibly still sitting on the table while the hand moves
    away" would ever produce -- a missed grasp diverges by centimeters within
    a few ticks, not millimeters.

    `max_track_loss_ticks`: how many CONSECUTIVE ticks of lost live tracking
    (`tracked_object_poses[id] is None`) are tolerated mid-CANDIDATE before
    giving up. The candidate object is very often briefly occluded by the
    hand's own approach right as it closes, so a short grace period (~0.1 s /
    3 ticks) avoids treating that occlusion itself as a miss; tracking lost
    longer than that is itself evidence something went wrong (dropped object,
    detector confusion) and we fall back to UNLATCHED rather than latching
    blind. See `update`'s docstring for the exact wait-vs-fail semantics.
    """

    latch_distance_m: float = 0.05
    confirm_window_ticks: int = 12
    position_tol_m: float = 0.02
    rotation_tol_deg: float = 15.0
    max_track_loss_ticks: int = 3


@dataclasses.dataclass
class LatchResult:
    """One tick's outcome, returned by `GraspLatch.update`.

    `reason` is set exactly on the tick a CANDIDATE attempt ends without
    reaching LATCHED (`"diverged"`, `"tracking_lost"`,
    `"released_before_confirm"`) -- `None` otherwise (steady UNLATCHED,
    in-progress CANDIDATE, or LATCHED). This is the "diagnostic info ... so a
    caller/recorder can see WHY a latch attempt failed" the plan asks for
    (§3.3's `percept["latch"]`, §5.4).
    """

    state: LatchState
    latched_object: str | None
    candidate_object: str | None
    ticks_in_candidate: int
    tracking_lost_ticks: int
    position_error_m: float | None
    rotation_error_deg: float | None
    reason: str | None


def _rotation_error_deg(R_a, R_b):
    """Angle (deg) of the relative rotation between two (3, 3) matrices."""
    relative = np.swapaxes(R_a, -1, -2) @ R_b
    return float(np.degrees(np.linalg.norm(mat_to_rotvec(relative))))


class GraspLatch:
    """One hand's grasp-confirmation state machine (docs/relation_deploy_plan.md §5.4).

    Call `update(...)` once per tick with this tick's hand-closed bit, hand
    pose, live-tracked object poses, and the set of objects this hand is
    currently allowed to consider (already filtered by the caller to exclude
    objects LATCHED to a *different* hand -- see "Cross-hand claiming" below).
    Call `object_pose(...)` (as many times as needed, any object) to get the
    pose `RelationPerception` should actually feed the policy for that object
    this tick.

    Cross-hand claiming: mirrors training's `claimed` set in
    `encoding.py`'s `latch_object_poses` (an object held by one hand is
    removed from the other hand's candidate pool before its nearest-object
    search runs). This class only tracks ONE hand's state, so it cannot see
    the other hand directly; the caller (owning both hands' `GraspLatch`
    instances, e.g. `RelationPerception`) is responsible for excluding any
    instance_id currently `.latched_object` on the OTHER hand's `GraspLatch`
    from the `eligible_objects` set passed in here, every tick -- exactly the
    same responsibility split as `latch_object_poses` builds `claimed` once
    per frame from both hands' state before either hand's candidate search.
    Within a SINGLE hand, re-claiming a second object while already LATCHED
    is structurally impossible here: candidate search only ever runs from
    UNLATCHED (see `update`), so an already-LATCHED hand simply never looks.
    """

    def __init__(self, config: LatchConfig | None = None):
        self.config = config or LatchConfig()
        self._state = LatchState.UNLATCHED
        self._latched_object: str | None = None
        self._candidate_object: str | None = None
        self._T_hand_object: np.ndarray | None = None
        self._ticks_in_candidate = 0
        self._tracking_lost_ticks = 0
        self._rigid_pose: np.ndarray | None = None   # this tick's frozen-transform prediction

    @property
    def state(self) -> LatchState:
        return self._state

    @property
    def latched_object(self) -> str | None:
        return self._latched_object

    def reset(self):
        """Force UNLATCHED, discarding any in-progress candidate. Not needed
        in normal operation (release already resets cleanly) -- provided for
        callers that need to hard-reset (e.g. after a fault)."""
        self._state = LatchState.UNLATCHED
        self._latched_object = None
        self._candidate_object = None
        self._T_hand_object = None
        self._ticks_in_candidate = 0
        self._tracking_lost_ticks = 0
        self._rigid_pose = None

    def _enter_unlatched(self, reason: str | None):
        self._state = LatchState.UNLATCHED
        self._candidate_object = None
        self._T_hand_object = None
        self._ticks_in_candidate = 0
        self._tracking_lost_ticks = 0
        self._rigid_pose = None
        return reason

    def _result(self, *, position_error=None, rotation_error=None, reason=None) -> LatchResult:
        return LatchResult(
            state=self._state,
            latched_object=self._latched_object,
            candidate_object=self._candidate_object,
            ticks_in_candidate=self._ticks_in_candidate,
            tracking_lost_ticks=self._tracking_lost_ticks,
            position_error_m=position_error,
            rotation_error_deg=rotation_error,
            reason=reason,
        )

    def update(
        self,
        *,
        hand_closed: bool,
        hand_pose: np.ndarray,
        tracked_object_poses: dict,
        eligible_objects: set,
    ) -> LatchResult:
        """One tick.

        hand_pose: (4, 4) this hand's current pose (e.g. flange/TCP in pelvis
            frame -- any consistent frame, matching what `tracked_object_poses`
            is expressed in).
        tracked_object_poses: {instance_id: (4, 4) or None if not currently
            visible to the live tracker}.
        eligible_objects: instance_ids this hand may currently consider --
            graspable objects (per `DeployTaskConfig`) MINUS whatever is
            already latched to the OTHER hand (caller's responsibility, see
            class docstring).

        Timeout semantics for tracking loss mid-CANDIDATE (edge case required
        by the task): each tick the candidate object's tracked pose is `None`
        increments a consecutive-loss counter; while that counter is at or
        below `config.max_track_loss_ticks` we WAIT (the confirmation window
        does not advance, but we do not fail either -- brief occlusion, e.g.
        by the closing hand itself, is expected). A live pose seen again
        resets the counter to zero. Exceeding the tolerance falls back to
        UNLATCHED with `reason="tracking_lost"` -- we never latch onto an
        object we cannot currently see agreeing with the prediction.

        Divergence semantics: any single tick where the live-tracked pose and
        the rigid prediction disagree by more than `position_tol_m` /
        `rotation_tol_deg` ends the candidate attempt immediately
        (`reason="diverged"`) rather than resetting a running counter --
        deliberately conservative, matching "do NOT silently keep reporting an
        object pose that pretends success". The hand may re-enter CANDIDATE
        on a later tick (fresh frozen transform) if it is still closed and an
        eligible object is still within `latch_distance_m`.
        """
        hand_pose = np.asarray(hand_pose, dtype=np.float64)

        if self._state == LatchState.UNLATCHED:
            self._rigid_pose = None
            if hand_closed:
                best_id, best_pose, best_dist = None, None, np.inf
                for oid in eligible_objects:
                    pose = tracked_object_poses.get(oid)
                    if pose is None:
                        continue
                    dist = float(np.linalg.norm(np.asarray(pose)[:3, 3] - hand_pose[:3, 3]))
                    # nearest-by-translation-distance tie-break, matching
                    # training's `np.argsort(distances)` selection.
                    if dist < best_dist:
                        best_id, best_pose, best_dist = oid, np.asarray(pose, dtype=np.float64), dist
                if best_id is not None and best_dist <= self.config.latch_distance_m:
                    self._state = LatchState.CANDIDATE
                    self._candidate_object = best_id
                    self._T_hand_object = se3_inv(hand_pose) @ best_pose
                    self._ticks_in_candidate = 0
                    self._tracking_lost_ticks = 0
            return self._result()

        if self._state == LatchState.CANDIDATE:
            if not hand_closed:
                # Hand released before the window ever confirmed: this is a
                # miss (an early open is not "convergence"), not a silent
                # success -- go back to UNLATCHED and say why.
                self._enter_unlatched(None)
                return self._result(reason="released_before_confirm")

            predicted = hand_pose @ self._T_hand_object
            self._rigid_pose = predicted
            tracked = tracked_object_poses.get(self._candidate_object)

            if tracked is None:
                self._tracking_lost_ticks += 1
                if self._tracking_lost_ticks > self.config.max_track_loss_ticks:
                    self._enter_unlatched(None)
                    return self._result(reason="tracking_lost")
                # brief loss: wait, do not advance or fail the window
                return self._result()

            self._tracking_lost_ticks = 0
            tracked = np.asarray(tracked, dtype=np.float64)
            pos_err = float(np.linalg.norm(predicted[:3, 3] - tracked[:3, 3]))
            rot_err = _rotation_error_deg(predicted[:3, :3], tracked[:3, :3])

            if pos_err > self.config.position_tol_m or rot_err > self.config.rotation_tol_deg:
                self._enter_unlatched(None)
                return self._result(position_error=pos_err, rotation_error=rot_err, reason="diverged")

            self._ticks_in_candidate += 1
            if self._ticks_in_candidate >= self.config.confirm_window_ticks:
                self._latched_object = self._candidate_object
                self._state = LatchState.LATCHED
                self._candidate_object = None
                return self._result(position_error=pos_err, rotation_error=rot_err)
            return self._result(position_error=pos_err, rotation_error=rot_err)

        # LATCHED
        if not hand_closed:
            self._latched_object = None
            self._enter_unlatched(None)
            return self._result()

        self._rigid_pose = hand_pose @ self._T_hand_object
        return self._result()

    def object_pose(self, instance_id: str, tracked_pose: np.ndarray | None) -> np.ndarray | None:
        """The pose `RelationPerception` should use for `instance_id` this
        tick: the rigid-predicted pose if this hand is LATCHED to it, else
        `tracked_pose` passed straight through (including `None` if
        untracked). Must be called after `update()` for this tick."""
        if self._state == LatchState.LATCHED and instance_id == self._latched_object:
            return self._rigid_pose
        return tracked_pose
