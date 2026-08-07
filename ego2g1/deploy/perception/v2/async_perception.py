"""The free-running perception thread (plan T1, T3, R3, §4.1).

Rounds run back to back: one finishes, the next starts. There is no cadence to
tune, no reseed to schedule and no tracker Hz to pick — whatever rate results
IS the rate, and it is reported on every snapshot (`round_s`) rather than
assumed. That matters because every downstream window is expressed in seconds
and converted using the measured rate; a constant written in samples would be
wrong the moment the scene changed.

ONE ROUND

    1. read the stereo pair once, stamp it, bind it to the nearest control
       tick and take THAT tick's FK (T4). The pair is read ONCE and handed to
       both arms below — reading separately in each arm serialises them behind
       the camera's lock, which measures lock contention instead of overlap.
    2. SAM 3 (GPU) || StereoSGBM (CPU). The only genuine parallelism in the
       pipeline: without MPS, kernels from two processes are time-sliced
       rather than co-scheduled, so GPU work is additive and the only real
       concurrency is GPU against CPU. Worth ~12 ms of a ~221 ms round.
    3. prune the memory bank (R1).
    4. join masks and depth to camera-frame 3D points.
    5. orientation on usable crops only (S1), skipping latched objects (R2).
    6. filter, lift to the pelvis frame, publish.

WAITING, AND WHY IT IS ALSO THE GPU ARBITRATION

`wait_for_current_round` is the replan primitive (T3). A replan does NOT grab
the newest completed snapshot and fire immediately; it lets the running round
finish and sends THAT round's snapshot:

    newest completed, send now   t_send - t_capture = P..2P   d varies 10-17
    wait for the in-flight round t_send - t_capture = exactly P  d ~= 10, constant

`d` is sent WITH the request and the guidance mask depends on it, so
determinism beats marginal freshness — the same argument `DelayBudget`'s own
docstring makes. It also decides whether the chunk arithmetic closes: a
constant d=10.4 ticks leaves 39.6 usable slots (1.32 s) against a 1 s replan,
while a varying d up to 17 leaves 33 (1.10 s), which is at the edge once slip
is added.

And because a replan already waits for a round boundary, the policy naturally
runs in the gap BETWEEN rounds. That is the whole GPU arbitration (R3) — no
lock, no mid-iteration yield, no torn snapshot. The timing rule is the
arbitration.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..orientation import OrientationRefiner, identity_symmetry_group
from .object_tracker import ObjectTracker
from .sam3_source import join_to_camera
from .snapshot import PerceptionSnapshot

logger = logging.getLogger(__name__)

__all__ = ["AsyncPerception", "PerceptionRound"]


class PerceptionRound:
    """One round's work, with no thread and no clock of its own.

    Split out from `AsyncPerception` so the pipeline can be stepped
    deterministically — from a recording, from a test, or synchronously during
    bring-up — without starting a thread. Everything that decides what the
    policy sees lives here; the thread below only decides when.
    """

    def __init__(self, *, read_stereo, tick_log, sam3, depth_source, calib,
                 T_pelvis_camera, objects, orientation=None,
                 latched_objects=None, tracker_kwargs=None,
                 symmetry_groups=None, nominal_rotations=None,
                 anchor_id=None, clock=time.monotonic):
        self._read_stereo = read_stereo
        self._tick_log = tick_log
        self._sam3 = sam3
        self._depth = depth_source
        self._calib = calib
        self._T_pelvis_camera = np.asarray(T_pelvis_camera, dtype=np.float64)
        self._objects = tuple(objects)
        self._orientation = orientation
        # Training's anchor is `obj_keys[0]` — the first object in the roster
        # (CamTriangulator.py:197). Only it gets the model's own rotation;
        # every other slot's is constructed relative to it. Defaulting to
        # objects[0] rather than requiring the caller to say so keeps deploy
        # and extraction agreeing by construction.
        self._anchor_id = (anchor_id if anchor_id is not None
                           else (self._objects[0].instance_id
                                 if self._objects else None))
        # Reading the control thread's latch state is a plain frozenset read;
        # a stale answer costs one wasted crop, never a wrong pose.
        self._latched_objects = latched_objects or (lambda: frozenset())
        self._tracker_kwargs = dict(tracker_kwargs or {})
        self._symmetry_groups = dict(symmetry_groups or {})
        self._nominal_rotations = dict(nominal_rotations or {})
        self._clock = clock

        self._trackers: dict[str, ObjectTracker] = {}
        self._refiners: dict[str, OrientationRefiner] = {}
        self._seq = 0
        self._prev_capture: float | None = None
        # One worker, created once. A fresh ThreadPoolExecutor per round would
        # spawn and join a thread every ~220 ms for the life of the rollout —
        # cheap individually, but it is pure churn on the hot path and shows
        # up in the round's own jitter, which is the number `d` is sized from.
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sgbm")

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def reset(self) -> None:
        self._trackers.clear()
        self._refiners.clear()
        self._prev_capture = None

    def _nominal_rotation(self, oid: str) -> np.ndarray:
        return np.asarray(self._nominal_rotations.get(oid, np.eye(3)),
                          dtype=np.float64)

    def step(self) -> PerceptionSnapshot | None:
        """Run one round. Returns the snapshot, or None if it could not be
        bound to a control tick (the control loop has not started yet, or has
        stopped publishing — either way there is no FK to pair the image with,
        and a snapshot without one would violate T2)."""
        t0 = self._clock()
        left, right = self._read_stereo()
        t_capture = self._clock()

        tick = self._tick_log.nearest(t_capture)
        if tick is None:
            logger.warning("no control tick to bind a frame to; dropping the "
                           "round. Is the control loop running?")
            return None

        # Genuine overlap, not a GIL illusion: OpenCV's SGBM and torch's CUDA
        # launches both release the GIL. This is the pipeline's only real
        # parallelism — everything else on the GPU is time-sliced and
        # therefore additive.
        depth_future = self._pool.submit(self._depth.estimate, left, right)
        observations = self._sam3.step(left)
        depth = depth_future.result()
        visibility = self._sam3.visibility(observations)
        crop_usable = {oid: v.crop_usable for oid, v in visibility.items()}
        mask_usable = {oid: v.mask_usable for oid, v in visibility.items()}

        points = join_to_camera(observations, depth, self._calib.K_left,
                                visibility=visibility)

        rotations: dict[str, np.ndarray | None] = {}
        if self._orientation is not None:
            # Camera-frame translations, before the lift to pelvis: the
            # relational construction is done in the camera frame to match
            # training term for term (it is frame-covariant, so the lift
            # afterwards carries it correctly).
            points_cam = {oid: (p[0] if p is not None else None)
                          for oid, p in points.items()}
            rotations = self._orientation.estimate(
                left, observations, crop_usable,
                skip=frozenset(self._latched_objects()),
                anchor_id=self._anchor_id, points_cam=points_cam)

        dt = (t_capture - self._prev_capture
              if self._prev_capture is not None else 0.0)
        self._prev_capture = t_capture

        poses: dict[str, np.ndarray | None] = {}
        depths: dict[str, float | None] = {}
        for obj in self._objects:
            oid = obj.instance_id
            measured = points.get(oid)
            depths[oid] = None if measured is None else measured[1]
            poses[oid] = self._fold(oid, measured, rotations.get(oid), dt)

        self._seq += 1
        return PerceptionSnapshot(
            seq=self._seq,
            t_capture=t_capture,
            n_capture=tick.n,
            rgb_left=left,
            flange_pelvis=tick.flange_pelvis,
            hand_frac=tick.hand_frac,
            object_pose_pelvis=poses,
            det_score={oid: o.det_score for oid, o in observations.items()},
            tracker_score={oid: o.tracker_score for oid, o in observations.items()},
            mask_area_px={oid: o.mask_area_px for oid, o in observations.items()},
            mask_usable=mask_usable,
            crop_usable=crop_usable,
            object_depth_m=depths,
            round_s=self._clock() - t0,
        )

    def _fold(self, oid: str, measured, rotation, dt: float
              ) -> np.ndarray | None:
        """Filter one object's measurement into its running estimate.

        Position and rotation are updated INDEPENDENTLY and on different
        gates, which is S1's governing asymmetry made concrete: position
        survives occlusion (a sliver centroid is biased but bounded by the
        object's extent) and orientation does not (the same sliver can be
        wrong by 180 degrees). So a round can update position while holding
        rotation, and that is the common case during a reach.
        """
        point_pelvis = None
        if measured is not None:
            point_camera = measured[0]
            point_pelvis = (self._T_pelvis_camera[:3, :3] @ point_camera
                            + self._T_pelvis_camera[:3, 3])

        tracker = self._trackers.get(oid)
        if tracker is None:
            if point_pelvis is None:
                return None                       # never seen; nothing to say
            initial = np.eye(4)
            initial[:3, :3] = self._nominal_rotation(oid)
            initial[:3, 3] = point_pelvis
            tracker = ObjectTracker(initial, **self._tracker_kwargs)
            self._trackers[oid] = tracker
            self._refiners[oid] = OrientationRefiner(
                self._symmetry_groups.get(oid) or identity_symmetry_group(),
                initial_rotation=initial[:3, :3])
        elif point_pelvis is not None:
            tracker.update(point_pelvis, dt)
        else:
            tracker.hold(dt)

        if rotation is not None:
            # The raw estimate never reaches the state vector directly: the
            # symmetry snap picks whichever equivalent representation is
            # nearest the previous one. That is a GAUGE CHOICE, not a
            # quantisation — `measured @ S` for S in the symmetry group
            # describes the identical physical pose — and it is what stops a
            # stationary object's rotation jumping between equivalent
            # matrices. With identity_symmetry_group() it is an exact
            # pass-through.
            tracker.set_orientation(self._refiners[oid].refresh(rotation))

        return tracker.pose


class AsyncPerception:
    """Runs `PerceptionRound` back to back on one thread and publishes the
    result by rebinding a single attribute."""

    def __init__(self, round_: PerceptionRound, *, name: str = "perception"):
        self._round = round_
        self._name = name
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._latest: PerceptionSnapshot | None = None
        self._published = 0
        self._dropped = 0
        self._errors = 0
        self._last_consumed_seq: int | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("already started")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=self._name,
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                # Daemon, so it will not block exit — but a round wedged in a
                # blocking camera read is worth saying out loud rather than
                # leaving as a silent thread leak.
                logger.error("perception thread did not stop within %.1f s; "
                             "it is probably blocked in a camera read.",
                             timeout_s)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = self._round.step()
            except Exception:                                  # noqa: BLE001
                # One bad frame must not kill perception for the rest of the
                # rollout: the control loop keeps consuming the last good
                # snapshot, which grows stale visibly (`age_s`) rather than
                # vanishing. A persistent fault shows as a climbing error
                # count in `stats()`, which the caller can act on.
                self._errors += 1
                logger.exception("perception round failed (%d total)",
                                 self._errors)
                continue
            if snapshot is None:
                self._dropped += 1
                continue
            with self._cv:
                self._latest = snapshot
                self._published += 1
                self._cv.notify_all()

    # -- reads --------------------------------------------------------------

    def latest(self) -> PerceptionSnapshot | None:
        """The newest completed snapshot, or None before the first round.

        Atomic: the snapshot is frozen and published by one rebinding, so a
        reader either sees the whole previous round or the whole new one,
        never a mixture. A torn read here would look exactly like perception
        noise and would never be diagnosed as a race.
        """
        with self._cv:
            return self._latest

    def wait_for_current_round(self, timeout_s: float
                               ) -> PerceptionSnapshot | None:
        """Block until the in-flight round publishes, then return it (T3).

        This is the replan primitive. On timeout it falls back to the last
        completed snapshot and returns that instead of raising: starving the
        control loop is worse than a larger `d` for one call, and the caller
        can tell the two apart because a fallback snapshot's `age_s` exceeds
        its `round_s`.

        Returns None only if no round has EVER completed.
        """
        deadline = time.monotonic() + float(timeout_s)
        with self._cv:
            target = self._published + 1
            while self._published < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "perception round did not complete within %.0f ms; "
                        "falling back to the last snapshot and accepting a "
                        "larger d for this call.", timeout_s * 1000)
                    break
                self._cv.wait(remaining)
            snapshot = self._latest

        if snapshot is not None:
            if self._last_consumed_seq == snapshot.seq:
                # Perception is slower than the policy period: the design
                # assumption (a round per replan) has inverted, `d` is no
                # longer P + L, and state age is silently doubling.
                logger.error(
                    "replan consumed snapshot seq=%d twice — perception "
                    "(%.0f ms/round) is slower than the replan period. `d` is "
                    "no longer constant and state age is understated.",
                    snapshot.seq, snapshot.round_s * 1000)
            self._last_consumed_seq = snapshot.seq
        return snapshot

    def stats(self) -> dict:
        with self._cv:
            latest = self._latest
            return {
                "published": self._published,
                "dropped": self._dropped,
                "errors": self._errors,
                "round_s": None if latest is None else latest.round_s,
                "seq": None if latest is None else latest.seq,
            }
