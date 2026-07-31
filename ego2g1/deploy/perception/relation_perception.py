"""The integration glue: wires task_config/depth/detector/tracker/orientation/
latch into one 56-dim relation-state vector per tick (docs/relation_deploy_plan.md
§5.5, task 10).

`RelationPerception.observe(...)` is what `ego2g1.deploy.policy_adapter
.RelationPolicyAdapter` is documented to expect in place of its current
pass-through contract (`request["relation_state"]` supplied by the caller) --
see that class's docstring. Wiring this INTO the adapter is a small, separate
step; this module focuses on producing a correct state vector from the four
independently-built Phase 2 pieces.

Per-tick pipeline, per object (in `task_config.objects` order -- the SAME
fixed order the connected checkpoint's `train_config.objects` uses, already
enforced once at connect time via `task_config.validate_against_server_metadata`):

  1. `detector.detect(rgb_left, task_config.objects)` AND `depth_source
     .estimate(...)` -- but only every `detector_period_ticks` ticks
     (default ~2 Hz, per §5.3's tiered cascade), NOT every tick. Both calls
     are genuinely expensive on real hardware (GroundingDINO+SAM2 on a GPU,
     StereoSGBM on the CPU) -- on the ticks in between, neither is called
     at all, and every object falls through to step 3's "not detected this
     tick" branch, which is exactly the fast (~20-30 Hz) tracker tier: no
     separate code path needed, just skipping these two calls. `ObjectSpec`
     duck-types `detector.ObjectQuery` (both are plain `instance_id`/
     `detector_prompt` attributes), so `task_config.objects` is passed
     straight through, no adapter class needed.
  2. A found detection's pixel centroid (mask-median if a mask is present,
     else box center) is looked up in `depth_source.estimate(...)`'s depth
     map and back-projected to a camera-frame 3D point via the calibration's
     `K_left` (pinhole back-projection: `X = (u-cx)*Z/fx`, `Y = (v-cy)*Z/fy`).
  3. The point is placed in the PELVIS frame via `T_pelvis_camera` (§6's
     touch-calibration output) and fed to that object's `ObjectTracker`
     (created lazily on first detection) -- `.update(...)` if detected this
     tick, `.predict(...)` (hold/extrapolate) if missed.
  4. `GraspLatch` (one per hand) decides whether each hand's rigid-latch
     prediction should override the live-tracked pose for objects it holds
     -- see `latch.py`'s docstring for the full state machine.
  5. The resolved per-object pose is expressed in each hand's own flange
     frame (`inv(flange_pose) @ object_pose`, exactly inverting
     `s2_object_relations/encoding.py`'s `T_left_tcp_object = compose(
     invert(left[frame]), object_pose)`) and packed into the 56-dim
     hand-major vector `RelationPrompt` expects.

Orientation -- an HONEST, FLAGGED GAP, not silently papered over: none of
`detector.py` (2D mask/box only), `depth.py` (depth map only), or
`orientation.py` (stabilizes/symmetry-snaps an ALREADY-MEASURED rotation,
produces nothing from pixels itself) actually estimates a 3D orientation
from an image. The training-time reference pipeline used a VLM for this,
explicitly flagged elsewhere in this plan as not real-time. Rather than
invent and ship an unvalidated geometric heuristic (e.g. PCA on a masked
point cloud) dressed up as a real estimator, this module takes an optional
`orientation_estimator` hook (`(rgb, Detection, depth_map, StereoCalibration)
-> (3,3) rotation | None`) and defaults it to `None`, meaning: every
object's orientation is held at its `nominal_rotations` entry (default
identity) FOREVER -- never updated. This is the deliberately safe default:
wrong-but-static beats a fabricated, unvalidated per-frame estimate. Plug in
a real estimator here once one exists and is validated; `OrientationRefiner`
(already built, already tested) is ready to consume its output the moment
one is.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ...core import se3 as _se3
from .depth import StereoCalibration
from .detector import Detection, ObjectDetector
from .latch import GraspLatch, LatchConfig, LatchResult
from .orientation import OrientationRefiner, SymmetryGroup, identity_symmetry_group
from .task_config import DeployTaskConfig
from .tracker import ObjectTracker

OrientationEstimator = Callable[
    [np.ndarray, Detection, np.ndarray, StereoCalibration], "np.ndarray | None"
]


def pixel_depth_to_camera_point(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """(pixel u, pixel v, depth in metres) -> (3,) camera-frame point.

    Standard pinhole back-projection: `X = (u - cx) * Z / fx`,
    `Y = (v - cy) * Z / fy`, `Z = Z`. `K` is the (3, 3) intrinsic matrix
    (`StereoCalibration.K_left`'s own convention).
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = float(depth_m)
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def _sample_depth(detection: Detection, depth_map: np.ndarray) -> float | None:
    """Median depth over `detection.mask` if present (robust to mask-boundary
    noise/holes), else the single pixel at the box center. Returns None if no
    valid (> 0, see `depth.py`'s "0.0 == invalid" convention) depth is found."""
    depth_map = np.asarray(depth_map, dtype=np.float64)
    if detection.mask is not None:
        mask = np.asarray(detection.mask, dtype=bool)
        values = depth_map[mask & (depth_map > 0)]
        if values.size == 0:
            return None
        return float(np.median(values))
    u, v = detection.centroid_uv()
    iu, iv = int(round(u)), int(round(v))
    if not (0 <= iv < depth_map.shape[0] and 0 <= iu < depth_map.shape[1]):
        return None
    z = float(depth_map[iv, iu])
    return z if z > 0 else None


def _pose_from(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


@dataclasses.dataclass
class ObjectDebug:
    """Per-object diagnostics for one `observe()` call -- surfaced so a
    recorder can reconstruct why the state vector looked the way it did
    (same "an unconfirmed miss must be visible, not silently swallowed"
    principle `latch.py`'s `LatchResult` follows)."""

    detected_this_tick: bool
    tracked: bool          # has a live ObjectTracker at all (seen at least once)
    pose_pelvis: np.ndarray | None
    depth_m: float | None


class RelationPerception:
    """Wires task_config/depth/detector/tracker/orientation/latch into the
    56-dim hand-major relation state `RelationPolicyAdapter` needs every tick.

    An object with NO tracker yet (never successfully detected since this
    instance was created) has no pose to report at all -- `observe()` raises
    `RuntimeError` naming which object(s), rather than silently zero-filling
    (a zero vec9 is not a pose anywhere else in this codebase either, see
    `serve/policy.py`'s RTC-prefix padding comment for the same principle).
    The caller (the deploy loop, not built here) should retry `observe()`
    each tick during a "wait for all objects visible" warm-up before ever
    calling the policy, exactly like the runner already waits for the first
    inference chunk before arming the starvation watchdog.
    """

    def __init__(
        self,
        task_config: DeployTaskConfig,
        detector: ObjectDetector,
        depth_source,  # DepthSource-duck-typed: .estimate(rgb_left, rgb_right)
        calib: StereoCalibration,
        T_pelvis_camera: np.ndarray,
        *,
        fps: int = 30,
        detector_period_ticks: int | None = None,  # None -> round(fps / 2), ~2 Hz (§5.3)
        orientation_period_ticks: int = 6,  # ~0.2 Hz at fps=30 (§5.3's cadence)
        orientation_estimator: OrientationEstimator | None = None,
        nominal_rotations: dict[str, np.ndarray] | None = None,
        symmetry_groups: dict[str, SymmetryGroup] | None = None,
        latch_config: LatchConfig | None = None,
        tracker_kwargs: dict | None = None,
    ):
        self.task_config = task_config
        self.detector = detector
        self.depth_source = depth_source
        self.calib = calib
        self.T_pelvis_camera = np.asarray(T_pelvis_camera, dtype=np.float64)
        self.dt = 1.0 / float(fps)
        # The detector (GroundingDINO+SAM2 on real hardware) and the depth
        # estimate that feeds a NEW detection's 3D lift are both genuinely
        # expensive -- unlike a FakeDetector/synthetic depth map in tests,
        # a real GPU call takes real time, and §5.3's whole point is that
        # this stage runs at ~2 Hz while the tracker (predict/extrapolate,
        # cheap) carries every tick in between. Gating BOTH calls behind the
        # same cadence (not just orientation) is the fix for an oversight in
        # the first version of this class, which called the detector every
        # tick -- fine with a FakeDetector, would have blown the timing
        # budget badly against real weights.
        self._detector_period = max(1, int(detector_period_ticks or round(fps / 2.0)))
        self._orientation_period = max(1, int(orientation_period_ticks))
        self._orientation_estimator = orientation_estimator
        self._nominal_rotations = nominal_rotations or {}
        self._symmetry_groups = symmetry_groups or {}
        self._tracker_kwargs = tracker_kwargs or {}

        self._trackers: dict[str, ObjectTracker] = {}
        self._orientation: dict[str, OrientationRefiner] = {}
        self._latches: dict[str, GraspLatch] = {
            h: GraspLatch(latch_config) for h in task_config.hands
        }
        self._tick = 0

    def _nominal_rotation(self, instance_id: str) -> np.ndarray:
        return np.asarray(
            self._nominal_rotations.get(instance_id, np.eye(3, dtype=np.float64)),
            dtype=np.float64,
        )

    def _symmetry_group(self, instance_id: str) -> SymmetryGroup:
        return self._symmetry_groups.get(instance_id) or identity_symmetry_group()

    def _ensure_tracker(self, instance_id: str, initial_pose: np.ndarray) -> ObjectTracker:
        if instance_id not in self._trackers:
            self._trackers[instance_id] = ObjectTracker(initial_pose, **self._tracker_kwargs)
            self._orientation[instance_id] = OrientationRefiner(
                self._symmetry_group(instance_id),
                initial_rotation=initial_pose[:3, :3],
            )
        return self._trackers[instance_id]

    def observe(
        self,
        rgb_left: np.ndarray,
        rgb_right: np.ndarray,
        flange_poses: dict[str, np.ndarray],
        hand_cmds_last: dict[str, float],
    ) -> dict:
        """One tick.

        rgb_left/rgb_right: (H, W, 3) uint8, the stereo pair `depth_source`
            needs (NOT `camera.py`'s concern -- see module docstring).
        flange_poses: {hand: (4, 4)} PELVIS frame, from
            `Kinematics.flange_poses(arm_q)` -- the SAME anchor
            `RelativeEEFRotvecChunks` composes model deltas onto.
        hand_cmds_last: {hand: float in [0, 1]} last-commanded gripper
            fraction (0 == open, 1 == closed, the same `frac` convention
            `RelativeEEFRotvecChunks` decodes -- see `actions.py`); rounded
            to a bool at `>= 0.5` for both the latch's `hand_closed` gate
            and the state vector's own grasp-word bit.

        Returns {"state": (56,) float32, "objects": {instance_id:
        ObjectDebug}, "latch": {hand: LatchResult}}.
        """
        run_detector_this_tick = self._tick % self._detector_period == 0
        if run_detector_this_tick:
            depth_map = self.depth_source.estimate(rgb_left, rgb_right)
            detections = self.detector.detect(rgb_left, self.task_config.objects)
        else:
            # Between detector refreshes: no new 2D/depth measurement at
            # all, for ANY object -- every object falls through to the
            # existing "not detected this tick" branch below, which already
            # means "let the tracker predict/extrapolate". This IS the fast
            # (~20-30 Hz) tracker tier the plan describes; it needs no
            # separate code path, only that the detector/depth calls above
            # are skipped.
            depth_map = None
            detections = {}

        object_debug: dict[str, ObjectDebug] = {}
        tracked_poses_pelvis: dict[str, np.ndarray | None] = {}

        for obj in self.task_config.objects:
            detection = detections.get(obj.instance_id)
            depth_m = None
            if detection is not None:
                depth_m = _sample_depth(detection, depth_map)

            if detection is not None and depth_m is not None:
                u, v = detection.centroid_uv()
                point_camera = pixel_depth_to_camera_point(u, v, depth_m, self.calib.K_left)
                point_pelvis = (
                    self.T_pelvis_camera[:3, :3] @ point_camera + self.T_pelvis_camera[:3, 3]
                )
                if obj.instance_id not in self._trackers:
                    initial_pose = _pose_from(point_pelvis, self._nominal_rotation(obj.instance_id))
                    self._ensure_tracker(obj.instance_id, initial_pose)
                    pose = self._trackers[obj.instance_id].pose
                else:
                    pose, _accepted = self._trackers[obj.instance_id].update(point_pelvis, self.dt)
                detected_this_tick = True
            elif obj.instance_id in self._trackers:
                pose = self._trackers[obj.instance_id].predict(self.dt)
                detected_this_tick = False
            else:
                pose = None
                detected_this_tick = False

            if pose is not None and self._tick % self._orientation_period == 0:
                measured_R = None
                if self._orientation_estimator is not None and detection is not None:
                    measured_R = self._orientation_estimator(rgb_left, detection, depth_map, self.calib)
                if measured_R is not None:
                    refined = self._orientation[obj.instance_id].refresh(measured_R)
                    self._trackers[obj.instance_id].set_orientation(refined)
                    pose = self._trackers[obj.instance_id].pose
                # measured_R is None (no estimator, or nothing detected this
                # cadence tick): orientation intentionally NOT touched, see
                # module docstring -- held at whatever it already was.

            tracked_poses_pelvis[obj.instance_id] = pose
            object_debug[obj.instance_id] = ObjectDebug(
                detected_this_tick=detected_this_tick,
                tracked=obj.instance_id in self._trackers,
                pose_pelvis=pose,
                depth_m=depth_m,
            )

        graspable_ids = {o.instance_id for o in self.task_config.objects if o.graspable}
        hand_closed = {h: hand_cmds_last[h] >= 0.5 for h in self.task_config.hands}

        latch_results: dict[str, LatchResult] = {}
        for h in self.task_config.hands:
            other_latched = {
                self._latches[other].latched_object
                for other in self.task_config.hands
                if other != h
            }
            other_latched.discard(None)
            eligible = graspable_ids - other_latched
            latch_results[h] = self._latches[h].update(
                hand_closed=hand_closed[h],
                hand_pose=flange_poses[h],
                tracked_object_poses=tracked_poses_pelvis,
                eligible_objects=eligible,
            )

        final_poses: dict[str, np.ndarray | None] = {}
        for obj in self.task_config.objects:
            pose = tracked_poses_pelvis[obj.instance_id]
            for h in self.task_config.hands:
                pose = self._latches[h].object_pose(obj.instance_id, pose)
            final_poses[obj.instance_id] = pose

        missing = [oid for oid, pose in final_poses.items() if pose is None]
        if missing:
            raise RuntimeError(
                f"RelationPerception.observe: object(s) {missing} have never been "
                "detected -- no pose to report (a zero vec9 is not a pose, per "
                "this codebase's convention elsewhere). The deploy loop should "
                "retry observe() during a warm-up period until every object has "
                "been seen at least once, before ever calling the policy."
            )

        blocks = []
        for h in self.task_config.hands:
            flange_inv = _se3.se3_inv(np.asarray(flange_poses[h], dtype=np.float64))
            for obj in self.task_config.objects:
                relative = flange_inv @ final_poses[obj.instance_id]
                blocks.append(_se3.se3_to_vec9(relative))
        grasp = np.array(
            [1.0 if hand_closed[h] else 0.0 for h in self.task_config.hands],
            dtype=np.float32,
        )
        state = np.concatenate([*blocks, grasp]).astype(np.float32)

        self._tick += 1
        return {"state": state, "objects": object_debug, "latch": latch_results}
