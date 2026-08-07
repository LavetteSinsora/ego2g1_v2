"""Kinematic latching, per hand (plan §6).

WHY
When the gripper closes, perception degrades and prediction becomes exact at
the same moment:

  * the visible mask's centroid is pulled toward whatever is still uncovered,
    and that bias MOVES AS THE HAND ROTATES — a signal that looks exactly like
    object motion and is not;
  * the median depth may sample the gripper;
  * orientation has no partial-credit answer from an occluded crop. This is
    the strongest single reason to latch: a wrong rotation is unbounded where
    a biased centroid is not;
  * meanwhile the object now moves rigidly with the hand, so FK gives its pose
    exactly and for free.

So the full pose is latched, position and rotation together — mixing a
predicted position with a measured rotation would make them inconsistent with
each other, which is worse than either alone.

TWO CLOCKS
    control loop   30 Hz      grasp command, FK, hand displacement.
                              Instant and exact.
    perception     2-4.5 Hz   object poses and visibility. Sparse and
                              sometimes degraded.

Anything about the HAND is available immediately; anything about the OBJECT
arrives a few times a second. That asymmetry is why this class has two entry
points rather than one `update()` — collapsing them would force either the
grasp trigger to wait for a snapshot (losing the exact closure instant that
§6.4 depends on) or the divergence test to run on stale object data every tick.

    on_control_tick(...)   30 Hz: entry, release, timeout, rigid prediction
    on_snapshot(...)       per round: confirmation and divergence

STATES
    UNLATCHED --grasp closes near an eligible object--> CANDIDATE
        ^                                                   |
        |                                     hand travels 3-5 cm
        |                                     and the object followed
        |                                                   v
        +---- gripper opens, or sustained ------------- LATCHED
              divergence, or candidate timeout
"""

from __future__ import annotations

import dataclasses
import enum
import logging

import numpy as np

from ....core.se3 import se3_inv

logger = logging.getLogger(__name__)

__all__ = ["LatchState", "LatchConfig", "LatchResult", "GraspLatch"]


class LatchState(enum.Enum):
    UNLATCHED = "unlatched"
    CANDIDATE = "candidate"
    LATCHED = "latched"


@dataclasses.dataclass(frozen=True)
class LatchConfig:
    """Tunables for one hand's latch. Plan Q12 — bring-up defaults, not
    measured constants.

    `latch_distance_m`
        Candidate-entry gate, roughly a BrainCo hand's closed-aperture span at
        the TCP reference point: further than this at the instant of closure
        and it almost certainly was not the object contacted.

    `confirm_displacement_m`
        How far the flange must travel before the confirmation verdict is
        taken. A DISPLACEMENT, not a time window, because the question is "did
        the object move with the hand?" — and if the hand has barely moved, a
        failed grasp is indistinguishable from a good one no matter how many
        samples you collect. At 2 Hz this may be a single observation, but one
        observation after 4 cm of travel is conclusive where three after 5 mm
        are not.

    `position_tol_m`
        Divergence threshold on RELATIVE motion (see `_divergence`).
        Deliberately generous: the test turns a 2-vs-3 cm judgement into a
        0-vs-5 cm one, so it can afford slack, and a spuriously-dropped latch
        is worse than a slightly stale one.

    `divergence_sustain`
        Consecutive diverging observations before the latch is dropped. One
        agreement confirms; N disagreements reject. The asymmetry is
        deliberate — after `confirm_displacement_m` of travel, agreement is
        physically conclusive (stationary objects do not follow hands), while
        a lone disagreement is exactly what one bad depth sample looks like.

    `candidate_timeout_s`
        Backstop for a hand that closes and then does not move: without it,
        CANDIDATE would persist indefinitely waiting for travel that never
        comes.

    `max_stale_s`
        Refuse to freeze a transform against an object observation older than
        this. The §6.4 asymmetry licenses a STALE object pose, but only
        because the object has not been touched — an observation from ten
        seconds ago carries no such guarantee.

    `min_check_travel_m`
        While LATCHED, skip the divergence check when the hand has barely
        moved between two observations: with no relative motion the test has
        no signal and would compare noise against a threshold.

    `divergence_gate`
        Which visibility gate admits an observation to the confirmation and
        divergence tests. `"crop"` (the plan's rule) demands a re-detected,
        substantially complete mask — no evidence beats bad evidence.
        `"mask"` accepts a memory-propagated mask too.

        THIS IS THE KNOB MOST LIKELY TO NEED FLIPPING DURING BRING-UP. Under
        `"crop"`, if the hand occludes the object badly enough that it is
        never re-detected during a grasp, no candidate ever confirms and the
        latch is dead weight — visible as `reason="candidate_timeout"` with
        `usable_observations == 0`. Under `"mask"`, a memory-propagated mask
        follows the tracker's belief, which under a grasp follows the hand,
        so it can spuriously AGREE and confirm a failed grasp. The failure
        modes point in opposite directions, and only hardware says which one
        actually happens.
    """

    latch_distance_m: float = 0.05
    confirm_displacement_m: float = 0.04
    position_tol_m: float = 0.03
    divergence_sustain: int = 2
    candidate_timeout_s: float = 3.0
    max_stale_s: float = 2.0
    min_check_travel_m: float = 0.01
    divergence_gate: str = "crop"

    def __post_init__(self):
        if self.divergence_gate not in ("crop", "mask"):
            raise ValueError(
                f"divergence_gate must be 'crop' or 'mask', got "
                f"{self.divergence_gate!r}")


@dataclasses.dataclass(frozen=True)
class LatchResult:
    """One control tick's outcome.

    `reason` is set exactly on the tick an attempt ends without reaching (or
    while leaving) LATCHED, and is None otherwise. Every diagnostic field is
    here because a latch that silently fails to engage and a latch that
    silently drops look identical from the state vector — the object pose just
    stops agreeing with reality — so the machine has to say what it decided
    and on what evidence.
    """

    state: LatchState
    latched_object: str | None
    candidate_object: str | None
    reason: str | None = None
    flange_travel_m: float | None = None      # since the divergence baseline
    divergence_m: float | None = None
    usable_observations: int = 0              # admitted since entering CANDIDATE
    stale_s: float | None = None              # age of the object pose at freeze
    time_in_candidate_s: float = 0.0


def _pos(T) -> np.ndarray:
    return np.asarray(T, dtype=np.float64)[:3, 3]


class GraspLatch:
    """One hand's latch state machine. Pure numpy — no camera, detector or
    robot dependency, so the whole thing is testable against synthetic
    converging and diverging trajectories."""

    def __init__(self, hand: str, config: LatchConfig | None = None):
        self.hand = str(hand)
        self.config = config or LatchConfig()
        # Last observation of each object we are allowed to trust, kept so the
        # freeze in §6.4 can reach back past the occlusion the closing hand
        # itself creates: {instance_id: (t, (4, 4) pose)}. Survives `reset()`
        # of the state machine only if the caller wants it to — it does not,
        # because a reset means the world is no longer known.
        self._last_usable: dict[str, tuple[float, np.ndarray]] = {}
        self._now: float = 0.0
        self._clear()

    def _clear(self) -> None:
        """Back to UNLATCHED with nothing in flight. One definition of empty,
        shared by construction, release and `reset()`."""
        self._state = LatchState.UNLATCHED
        self._latched: str | None = None
        self._candidate: str | None = None
        self._T_flange_object: np.ndarray | None = None
        # Divergence baseline: (flange position, object position) from ONE
        # instant, so both displacements are measured from a common origin.
        # See _set_baseline for why this cannot be the freeze instant.
        self._baseline: tuple[np.ndarray, np.ndarray] | None = None
        self._freeze_t: float = 0.0
        self._freeze_stale_s: float | None = None
        self._usable_count = 0
        self._sustain = 0
        self._travel_m: float | None = None
        self._divergence_m: float | None = None
        self._pending_exit: str | None = None

    # -- read-only ----------------------------------------------------------

    @property
    def state(self) -> LatchState:
        return self._state

    @property
    def latched_object(self) -> str | None:
        return self._latched

    @property
    def candidate_object(self) -> str | None:
        return self._candidate

    @property
    def transform(self) -> np.ndarray | None:
        """The frozen hand->object transform, or None outside CANDIDATE and
        LATCHED."""
        return None if self._T_flange_object is None else self._T_flange_object.copy()

    def reset(self) -> None:
        """Hard reset for an episode boundary or a fault. Also forgets the
        remembered object observations: after a reset nothing about the world
        is still known to be true, and freezing a transform against a
        pre-reset pose would carry the old episode into the new one."""
        self._last_usable.clear()
        self._clear()

    # -- perception rate ----------------------------------------------------

    def on_snapshot(self, snapshot) -> None:
        """Fold one perception round in: remember usable observations, and run
        confirmation (CANDIDATE) or divergence (LATCHED).

        Everything read here comes from the snapshot's own instant — its
        `flange_pelvis` is the FK at `n_capture`, not fresh FK — so the two
        displacements compared below are genuinely simultaneous. Using a fresh
        flange pose against a stale object pose is precisely the error the
        relative test exists to avoid.
        """
        gate = (snapshot.crop_usable if self.config.divergence_gate == "crop"
                else snapshot.mask_usable)

        for oid, pose in snapshot.object_pose_pelvis.items():
            if pose is not None and gate.get(oid):
                self._last_usable[oid] = (snapshot.t_capture,
                                          np.asarray(pose, dtype=np.float64).copy())

        target = self._candidate or self._latched
        if target is None or self._state is LatchState.UNLATCHED:
            return
        pose = snapshot.object_pose_pelvis.get(target)
        if pose is None or not gate.get(target):
            return                                  # suspend; no evidence
        flange = snapshot.flange_pelvis.get(self.hand)
        if flange is None:
            raise KeyError(
                f"snapshot has no flange pose for hand {self.hand!r} "
                f"(has {sorted(snapshot.flange_pelvis)}). The latch compares "
                "object motion against ITS OWN hand; silently skipping would "
                "suspend the divergence test for the whole rollout.")

        self._usable_count += 1
        obj_p, fl_p = _pos(pose), _pos(flange)

        if self._baseline is None:
            self._set_baseline(fl_p, obj_p)
            return

        base_fl, base_obj = self._baseline
        d_flange = fl_p - base_fl
        d_object = obj_p - base_obj
        travel = float(np.linalg.norm(d_flange))
        divergence = float(np.linalg.norm(d_object - d_flange))
        self._travel_m, self._divergence_m = travel, divergence

        if self._state is LatchState.CANDIDATE:
            self._confirm(travel, divergence, fl_p, obj_p)
        else:
            self._check_latched(travel, divergence, fl_p, obj_p)

    def _set_baseline(self, flange_p: np.ndarray, object_p: np.ndarray) -> None:
        """Anchor both displacements to ONE observed instant.

        Deliberately NOT the freeze instant. The frozen transform is built
        from the flange at closure and the object as last seen BEFORE closure
        (§6.4) — two different times, correctly so. Measuring displacement
        from those same two times would compare a hand that has been reaching
        and closing against an object that has not moved at all, and read the
        result as divergence: a successful grasp would fail its own
        confirmation. The baseline therefore comes from the first admitted
        observation after closure, where both terms are simultaneous.

        This is a refinement of §6.4's "record the flange position at freeze";
        see docs/perception_v2_notes.md.
        """
        self._baseline = (flange_p.copy(), object_p.copy())
        self._travel_m = 0.0
        self._divergence_m = 0.0

    def _confirm(self, travel: float, divergence: float,
                 flange_p: np.ndarray, object_p: np.ndarray) -> None:
        if travel < self.config.confirm_displacement_m:
            return                                  # not enough signal yet
        if divergence <= self.config.position_tol_m:
            # The object moved with the hand over a distance a stationary
            # object could not have. One such observation is conclusive.
            self._latched, self._candidate = self._candidate, None
            self._state = LatchState.LATCHED
            self._sustain = 0
            # LATCHED differences consecutive observations rather than working
            # from a fixed origin, and THIS observation is a perfectly good
            # first term. Clearing the baseline instead would spend the next
            # round re-establishing what is already in hand, delaying every
            # drop detection by one round for no gain.
            self._baseline = (flange_p.copy(), object_p.copy())
            return
        self._sustain += 1
        if self._sustain >= self.config.divergence_sustain:
            self._pending_exit = "confirm_diverged"

    def _check_latched(self, travel: float, divergence: float,
                       flange_p: np.ndarray, object_p: np.ndarray) -> None:
        """While LATCHED the baseline slides: each admitted observation is
        compared against the previous one, not against a fixed origin.

        The occlusion bias cancels only while it is common-mode, i.e. roughly
        constant across the window being differenced. Over a long carry the
        bias drifts as the hand rotates, so a fixed origin would accumulate
        that drift and eventually cross the threshold on a perfectly good
        latch. A one-observation window keeps the differencing interval short
        and the cancellation valid.
        """
        if travel >= self.config.min_check_travel_m:
            if divergence > self.config.position_tol_m:
                self._sustain += 1
                if self._sustain >= self.config.divergence_sustain:
                    self._pending_exit = "diverged"
            else:
                self._sustain = 0
        self._baseline = (flange_p.copy(), object_p.copy())

    # -- control rate -------------------------------------------------------

    def on_control_tick(self, *, hand_closed: bool, hand_pose: np.ndarray,
                        t: float, eligible_objects) -> LatchResult:
        """One 30 Hz tick.

        `hand_pose`  (4, 4) this hand's flange pose in the pelvis frame, this
                     tick. Exact, from FK on measured joints.
        `t`          `time.monotonic()`, the same clock the snapshots stamp.
        `eligible_objects`  instance_ids this hand may consider: graspable,
                     minus whatever is latched to the OTHER hand. Excluding
                     the other hand's claim is the caller's job — this class
                     sees one hand and cannot do it (mirrors training's
                     `claimed` set in `encoding.py::latch_object_poses`).
        """
        hand_pose = np.asarray(hand_pose, dtype=np.float64)
        self._now = float(t)

        # A divergence verdict reached at perception rate takes effect here,
        # at control rate, so that state transitions happen at exactly one
        # place and a caller reading `state` between the two never sees a
        # decision that has not been applied.
        if self._pending_exit is not None:
            reason, self._pending_exit = self._pending_exit, None
            return self._exit(reason)

        if self._state is LatchState.UNLATCHED:
            if hand_closed:
                self._try_enter(hand_pose, eligible_objects)
            return self._result()

        if not hand_closed:
            # An early open is a miss, not a confirmation; a late one is a
            # clean release. Both leave by the same door.
            reason = (None if self._state is LatchState.LATCHED
                      else "released_before_confirm")
            return self._exit(reason)

        if (self._state is LatchState.CANDIDATE
                and self._now - self._freeze_t > self.config.candidate_timeout_s):
            return self._exit("candidate_timeout")

        return self._result()

    def _try_enter(self, hand_pose: np.ndarray, eligible_objects) -> None:
        """Freeze the transform, taking each term from where it is valid.

            T_flange_object = inv( T_flange(t_grasp) ) @ T_object(t_stale)
                                    ^ exact FK at closure  ^ last usable look

        Between the last clean look and closure the HAND moves a great deal —
        it is reaching and closing — while the OBJECT does not move at all,
        because nobody has touched it. So the hand term must be fresh and the
        object term may be stale, position and rotation together.

        Two failure modes this avoids:
          * both from t_stale  -> encodes "the object sits 5 cm from my palm"
                                  and predicts that forever;
          * both at t_grasp    -> derives the object pose from an already-
                                  occluded mask, which is the measurement the
                                  latch exists to stop trusting.

        This does not contradict T2: there is no image to stay consistent
        with (it is an internal transform at a physical event), and the object
        is specifically at rest, so the staticity assumption that fails
        globally holds locally.
        """
        best_id, best_pose, best_dist, best_age = None, None, np.inf, None
        hand_p = _pos(hand_pose)
        for oid in eligible_objects:
            seen = self._last_usable.get(oid)
            if seen is None:
                continue
            t_stale, pose = seen
            age = self._now - t_stale
            if age > self.config.max_stale_s:
                continue
            dist = float(np.linalg.norm(_pos(pose) - hand_p))
            if dist < best_dist:
                best_id, best_pose, best_dist, best_age = oid, pose, dist, age

        if best_id is None or best_dist > self.config.latch_distance_m:
            return

        self._state = LatchState.CANDIDATE
        self._candidate = best_id
        self._T_flange_object = se3_inv(hand_pose) @ best_pose
        self._freeze_t = self._now
        self._freeze_stale_s = best_age
        self._baseline = None
        self._usable_count = 0
        self._sustain = 0
        self._travel_m = self._divergence_m = None
        # How stale the object term was is the first thing to look at when
        # latches start failing: it separates a bad transform from a bad grasp.
        logger.info("latch candidate %s at %.1f cm, object pose %.2f s stale",
                    best_id, best_dist * 100, best_age)

    def _exit(self, reason: str | None) -> LatchResult:
        """Leave for UNLATCHED, reporting the state we are ENTERING but the
        evidence of the attempt that just ended. Those come from opposite
        sides of the clear, so they are captured either side of it: a result
        that said CANDIDATE would be a lie by the time the caller read it, and
        one with zeroed diagnostics would throw away the only record of why
        the attempt failed."""
        ended = (self._travel_m, self._divergence_m, self._usable_count,
                 self._freeze_stale_s)
        self._clear()
        travel, divergence, usable, stale = ended
        return LatchResult(
            state=LatchState.UNLATCHED,
            latched_object=None,
            candidate_object=None,
            reason=reason,
            flange_travel_m=travel,
            divergence_m=divergence,
            usable_observations=usable,
            stale_s=stale,
        )

    def _result(self, *, reason: str | None = None) -> LatchResult:
        return LatchResult(
            state=self._state,
            latched_object=self._latched,
            candidate_object=self._candidate,
            reason=reason,
            flange_travel_m=self._travel_m,
            divergence_m=self._divergence_m,
            usable_observations=self._usable_count,
            stale_s=self._freeze_stale_s,
            time_in_candidate_s=(self._now - self._freeze_t
                                 if self._state is LatchState.CANDIDATE else 0.0),
        )

    # -- resolution ---------------------------------------------------------

    def object_pose(self, instance_id: str, tracked_pose, flange_pose):
        """The pose to report for `instance_id`: the rigid prediction if this
        hand is LATCHED to it, else the tracked pose untouched.

        `flange_pose` is the flange at the SAME instant the reported state
        describes — the snapshot's own `flange_pelvis`, never fresh FK. That
        is T2: the object pose and the arm pose in one state vector must come
        from the same moment, or the image the policy also receives shows the
        arm somewhere the state vector denies.

        Note CANDIDATE deliberately reports the TRACKED pose, not the frozen
        prediction. A candidate is an unconfirmed hypothesis; reporting its
        prediction would feed the policy a hallucinated "the object is in my
        hand" relation during exactly the window where that is most likely to
        be false.
        """
        if self._state is LatchState.LATCHED and instance_id == self._latched:
            if self._T_flange_object is None:
                return tracked_pose
            return np.asarray(flange_pose, dtype=np.float64) @ self._T_flange_object
        return tracked_pose
