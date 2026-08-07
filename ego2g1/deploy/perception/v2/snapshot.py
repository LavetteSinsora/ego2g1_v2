"""The two clocks, and the one object that crosses between them.

docs/perception_v2_pipeline.md §4.2 and T2/T4.

The deploy process runs two loops at unrelated rates:

    control loop    30 Hz    grasp command, FK, hand displacement — instant
                             and exact, because it is the robot's own state
    perception      2-4.5 Hz object poses and visibility — a few times a
                             second, and sometimes degraded

`PerceptionSnapshot` is the only thing that travels from the slow loop to the
fast one. It is frozen, published by rebinding a single attribute, and it
carries EVERYTHING the policy is told about the world at one instant — the
image included. That last part is T2 and it is the whole reason the class
exists rather than a handful of separately-updated attributes: a consumer that
can observe a half-updated perception state gets torn reads that look exactly
like perception noise, and pairing the state vector with a DIFFERENT frame
than the one it was measured from desynchronises two modalities that were
synchronised in every training sample.

`ControlTickLog` is the other direction: the control loop drops its per-tick
state here, and a camera frame arriving at an arbitrary instant binds to the
nearest tick (T4). The perception thread therefore never computes FK itself —
it reads the FK the control loop already computed, at the tick nearest capture.
Quantisation error is at most half a tick (16.7 ms, ~5 mm at arm speed), well
under every other term in the budget, and in exchange action slots land on
real control ticks by construction, so nothing downstream has to interpolate
a chunk.
"""

from __future__ import annotations

import dataclasses
import threading

import numpy as np

__all__ = ["ControlTick", "ControlTickLog", "PerceptionSnapshot"]


@dataclasses.dataclass(frozen=True)
class ControlTick:
    """One 30 Hz control tick's robot state, as the control loop saw it.

    `n` is the tick index the whole timing scheme counts in: action slot *k*
    of a chunk observed at this tick means control tick `n + k`, and `d` is a
    difference of two of these (T4). It must be monotonic and must not reset
    mid-rollout.

    `hand_frac` is the LAST-COMMANDED gripper fraction (0 open .. 1 closed),
    the same convention `modes/relation_eef.RelativeEEFRotvecChunks` decodes.
    It lives here, not on the snapshot's own initiative, because T2 requires
    the grasp binaries in the state vector to come from the capture instant
    like everything else — so they have to be latched when the frame is bound,
    not read at send time.
    """

    n: int
    t: float                                  # time.monotonic() at the tick
    flange_pelvis: dict[str, np.ndarray]      # {hand: (4, 4)} pelvis frame
    hand_frac: dict[str, float]               # {hand: 0..1}


class ControlTickLog:
    """Thread-safe ring buffer of recent `ControlTick`s.

    Written by the control thread every tick, read by the perception thread
    once per round. The lock is held for the length of an append or a scan of
    at most `maxlen` entries — microseconds against a 33 ms tick — and it is a
    real requirement, not defensive habit: a bare `deque` may be appended to
    atomically under the GIL, but ITERATING one while another thread appends
    raises `RuntimeError: deque mutated during iteration`, which would surface
    as a rare mid-rollout crash in the perception thread.

    `maxlen` defaults to 2 s of ticks. It only has to cover the worst case
    binding lag: the time between a frame being captured and the perception
    thread getting around to binding it. That is one camera-read plus
    scheduling — milliseconds — so 2 s is already two orders of magnitude of
    margin.
    """

    def __init__(self, *, maxlen: int = 60):
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._maxlen = int(maxlen)
        self._entries: list[ControlTick] = []
        self._lock = threading.Lock()

    def record(self, n: int, t: float, flange_pelvis: dict[str, np.ndarray],
               hand_frac: dict[str, float]) -> None:
        """Publish one control tick. Copies the arrays: the caller's FK dict is
        typically reused/overwritten next tick, and a snapshot that aliases it
        would silently mutate after publication."""
        tick = ControlTick(
            n=int(n),
            t=float(t),
            flange_pelvis={h: np.asarray(T, dtype=np.float64).copy()
                           for h, T in flange_pelvis.items()},
            hand_frac={h: float(v) for h, v in hand_frac.items()},
        )
        with self._lock:
            self._entries.append(tick)
            if len(self._entries) > self._maxlen:
                del self._entries[:-self._maxlen]

    def nearest(self, t: float) -> ControlTick | None:
        """The tick whose timestamp is closest to `t`, or None if nothing has
        been recorded yet.

        Nearest, not latest-at-or-before: a frame captured 2 ms before a tick
        belongs to that tick, and rounding it backwards would bias every
        binding by up to a full tick in one direction. Nearest keeps the error
        zero-mean and bounded by half a tick.
        """
        with self._lock:
            if not self._entries:
                return None
            return min(self._entries, key=lambda e: abs(e.t - float(t)))

    def latest(self) -> ControlTick | None:
        with self._lock:
            return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


@dataclasses.dataclass(frozen=True)
class PerceptionSnapshot:
    """One perception round's complete, immutable view of the world.

    Every field is measured at, or latched to, the SAME instant (T2):

        t_capture / n_capture   when the stereo pair was read, and the control
                                tick it bound to (T4)
        rgb_left                the exact frame the policy is sent. Not "a
                                recent frame" — this one.
        flange_pelvis           FK at n_capture, from the control loop
        hand_frac               commanded gripper fraction at n_capture
        object_pose_pelvis      what PERCEPTION believes, filtered but not
                                latch-resolved. The latch's rigid prediction
                                is applied later, when the state vector is
                                built (`relation.RelationStateBuilder`), and
                                deliberately not here: the latch consumes
                                these poses to decide whether it is still
                                holding the object, so a snapshot that already
                                contained the latch's own prediction would
                                make the divergence test compare a value with
                                itself and always agree.

    and the S1 signals that say how much to believe each object:

        det_score               did the DETECTOR re-find it this frame?
                                None means no — the mask is memory
                                propagation, i.e. a guess.
        tracker_score           memory-propagation confidence
        mask_area_px            for the area-collapse test
        mask_usable             position/depth may be measured (permissive)
        crop_usable             orientation and the latch divergence test may
                                run (strict). Implies `mask_usable`.

    Two gates rather than the plan's one: position survives occlusion and
    orientation does not, so they cannot share a threshold. See
    `sam3_source.VisibilityGate`.

    `seq` is a monotonic round counter. It exists so "did this replan consume
    the same snapshot as the last one?" is a comparison rather than an
    identity check on a possibly-recycled object — that condition means
    perception is slower than the policy period and the design assumption has
    inverted, so it has to be detectable rather than inferred.

    `round_s` is how long the round that produced this took, so staleness is
    observed rather than assumed. The free-running loop has no fixed rate (T1);
    any constant you write down for it is wrong the moment the scene changes.
    """

    seq: int
    t_capture: float
    n_capture: int
    rgb_left: np.ndarray
    flange_pelvis: dict[str, np.ndarray]
    hand_frac: dict[str, float]
    object_pose_pelvis: dict[str, np.ndarray | None]
    det_score: dict[str, float | None]
    tracker_score: dict[str, float]
    mask_area_px: dict[str, int]
    mask_usable: dict[str, bool]
    crop_usable: dict[str, bool]
    object_depth_m: dict[str, float | None]
    round_s: float

    def __post_init__(self):
        # Fail at construction, not three modules downstream. A snapshot whose
        # per-object dicts disagree on their key sets would make "object 2 is
        # missing" ambiguous between "not detected" and "never asked for".
        keys = set(self.object_pose_pelvis)
        for name in ("det_score", "tracker_score", "mask_area_px",
                     "mask_usable", "crop_usable", "object_depth_m"):
            other = set(getattr(self, name))
            if other != keys:
                raise ValueError(
                    f"PerceptionSnapshot.{name} covers {sorted(other)} but "
                    f"object_pose_pelvis covers {sorted(keys)} — every "
                    "per-object dict must carry the full roster, using None/"
                    "False for 'no measurement'.")
        bad = [k for k in keys if self.crop_usable[k] and not self.mask_usable[k]]
        if bad:
            raise ValueError(
                f"crop_usable without mask_usable for {sorted(bad)} — "
                "crop_usable is the strictly stronger gate and must imply the "
                "weaker one (sam3_source.VisibilityGate).")
        if set(self.hand_frac) != set(self.flange_pelvis):
            raise ValueError(
                f"hand_frac covers {sorted(self.hand_frac)} but flange_pelvis "
                f"covers {sorted(self.flange_pelvis)}")

    def age_s(self, now: float) -> float:
        """Seconds since capture. `now` is a `time.monotonic()` value."""
        return float(now) - self.t_capture

    def usable_objects(self) -> tuple[str, ...]:
        """Roster slots whose crop is trustworthy this round (S1)."""
        return tuple(oid for oid, ok in self.crop_usable.items() if ok)

    def missing_objects(self) -> tuple[str, ...]:
        """Roster slots with no pose at all. Not the same as "not usable": a
        pose can be held from an earlier round and still be the right thing to
        report, whereas a missing one has nothing to report at all."""
        return tuple(oid for oid, p in self.object_pose_pelvis.items()
                     if p is None)
