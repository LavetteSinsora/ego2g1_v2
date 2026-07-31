"""Integration tests for `RelationPerception.observe` (docs/relation_deploy_plan.md
§5.5). Fully synthetic: a fake detector + a fake depth source + a simple
identity-ish camera/pelvis setup, no real image/model/robot needed -- same
discipline as the four Phase 2 modules' own test suites.
"""

import numpy as np
import pytest

from ego2g1.deploy.perception.depth import StereoCalibration
from ego2g1.deploy.perception.detector import Detection, FakeDetector
from ego2g1.deploy.perception.latch import LatchConfig, LatchState
from ego2g1.deploy.perception.relation_perception import (
    RelationPerception,
    pixel_depth_to_camera_point,
)
from ego2g1.deploy.perception.task_config import DeployTaskConfig, ObjectSpec

IMG_H, IMG_W = 64, 64
FX = FY = 50.0
CX = CY = 32.0


def _calib() -> StereoCalibration:
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    return StereoCalibration(
        K_left=K,
        K_right=K.copy(),
        dist_left=np.zeros(5),
        dist_right=np.zeros(5),
        R=np.eye(3),
        T=np.array([0.06, 0.0, 0.0]),
        image_size=(IMG_W, IMG_H),
    )


def _task_config(hands=("left", "right")) -> DeployTaskConfig:
    return DeployTaskConfig(
        objects=(
            ObjectSpec("obj0", "pen holder", "a pen holder .", graspable=False),
            ObjectSpec("obj1", "red cube", "a red cube .", graspable=True),
            ObjectSpec("obj2", "yellow cube", "a yellow cube .", graspable=True),
        ),
        hands=hands,
    )


def _box_detection(instance_id: str, u: float, v: float, half: float = 2.0) -> Detection:
    return Detection(
        instance_id=instance_id,
        confidence=0.9,
        box_xyxy=np.array([u - half, v - half, u + half, v + half]),
    )


def _depth_map(fill: float = 1.0) -> np.ndarray:
    return np.full((IMG_H, IMG_W), fill, dtype=np.float32)


def _perception(task_config=None, **kwargs) -> tuple[RelationPerception, FakeDetector]:
    detector = FakeDetector()
    depth_source = _StaticDepth(_depth_map())
    perception = RelationPerception(
        task_config or _task_config(),
        detector,
        depth_source,
        _calib(),
        T_pelvis_camera=np.eye(4),
        fps=30,
        **kwargs,
    )
    return perception, detector


class _StaticDepth:
    """Minimal DepthSource double: ignores the stereo pair, returns a fixed map."""

    def __init__(self, depth_map: np.ndarray):
        self._map = depth_map

    def estimate(self, rgb_left, rgb_right) -> np.ndarray:
        return self._map


def _rgb() -> np.ndarray:
    return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)


class TestPixelBackprojection:
    def test_center_pixel_gives_pure_depth_along_z(self):
        K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
        p = pixel_depth_to_camera_point(CX, CY, 2.0, K)
        np.testing.assert_allclose(p, [0.0, 0.0, 2.0], atol=1e-9)

    def test_off_center_pixel_gives_expected_xy(self):
        K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
        p = pixel_depth_to_camera_point(CX + FX, CY, 1.0, K)  # one focal-length right
        np.testing.assert_allclose(p, [1.0, 0.0, 1.0], atol=1e-9)


class TestObserveBasics:
    def test_state_shape_matches_hands_times_objects_plus_grasp(self):
        perception, detector = _perception()
        for obj in perception.task_config.objects:
            detector.set_detection(obj.instance_id, _box_detection(obj.instance_id, CX, CY))
        flange = {"left": np.eye(4), "right": np.eye(4)}
        out = perception.observe(_rgb(), _rgb(), flange, {"left": 0.0, "right": 0.0})
        n_hands, n_obj = 2, 3
        assert out["state"].shape == (n_hands * n_obj * 9 + n_hands,)
        assert out["state"].dtype == np.float32

    def test_object_directly_in_front_at_known_depth_gives_expected_vec9(self):
        perception, detector = _perception(hands=("left",)) if False else _perception(
            task_config=_task_config(hands=("left",))
        )
        obj = perception.task_config.objects[0]
        detector.set_detection(obj.instance_id, _box_detection(obj.instance_id, CX, CY))
        for other in perception.task_config.objects[1:]:
            detector.set_detection(other.instance_id, _box_detection(other.instance_id, CX, CY))
        flange = {"left": np.eye(4)}
        out = perception.observe(_rgb(), _rgb(), flange, {"left": 0.0})
        first_block = out["state"][:9]
        # camera == pelvis == flange (all identity), object 1m along +Z,
        # nominal rotation identity -> vec9 = [0,0,1, 1,0,0, 0,1,0]
        np.testing.assert_allclose(first_block, [0, 0, 1, 1, 0, 0, 0, 1, 0], atol=1e-6)

    def test_grasp_bit_reflects_hand_cmds_threshold(self):
        perception, detector = _perception()
        for obj in perception.task_config.objects:
            detector.set_detection(obj.instance_id, _box_detection(obj.instance_id, CX, CY))
        flange = {"left": np.eye(4), "right": np.eye(4)}
        out = perception.observe(_rgb(), _rgb(), flange, {"left": 1.0, "right": 0.2})
        # tail 2 dims: [grasp_left, grasp_right] per RelationPrompt's layout
        assert out["state"][-2] == 1.0
        assert out["state"][-1] == 0.0

    def test_never_detected_object_raises(self):
        perception, detector = _perception()
        # only seed 2 of 3 objects
        objs = perception.task_config.objects
        detector.set_detection(objs[0].instance_id, _box_detection(objs[0].instance_id, CX, CY))
        detector.set_detection(objs[1].instance_id, _box_detection(objs[1].instance_id, CX, CY))
        flange = {"left": np.eye(4), "right": np.eye(4)}
        with pytest.raises(RuntimeError, match=objs[2].instance_id):
            perception.observe(_rgb(), _rgb(), flange, {"left": 0.0, "right": 0.0})

    def test_missed_detection_after_seeding_holds_last_pose_not_raise(self):
        perception, detector = _perception()
        for obj in perception.task_config.objects:
            detector.set_detection(obj.instance_id, _box_detection(obj.instance_id, CX, CY))
        flange = {"left": np.eye(4), "right": np.eye(4)}
        perception.observe(_rgb(), _rgb(), flange, {"left": 0.0, "right": 0.0})

        target = perception.task_config.objects[1].instance_id
        detector.clear_detection(target)
        out = perception.observe(_rgb(), _rgb(), flange, {"left": 0.0, "right": 0.0})
        assert out["objects"][target].detected_this_tick is False
        assert out["objects"][target].tracked is True
        assert out["objects"][target].pose_pelvis is not None
        assert np.all(np.isfinite(out["state"]))


class TestLatchIntegration:
    def test_hand_closing_on_a_consistently_moving_object_reaches_latched(self):
        """Full end-to-end grasp confirmation: hand closes near an object whose
        tracked position keeps agreeing with the rigid hand-follows prediction
        for the whole confirmation window -> LATCHED, mirroring latch.py's own
        "converging trajectory" unit test but driven through the real
        detector/depth/tracker stack instead of calling GraspLatch directly."""
        cfg = _task_config(hands=("left",))
        perception, detector = _perception(
            task_config=cfg, latch_config=LatchConfig(confirm_window_ticks=5, max_track_loss_ticks=1)
        )
        target = cfg.objects[1].instance_id  # graspable
        for obj in cfg.objects:
            detector.set_detection(obj.instance_id, _box_detection(obj.instance_id, CX, CY))

        flange = {"left": np.eye(4)}
        # Seed all trackers first (hand open, nothing near yet).
        perception.observe(_rgb(), _rgb(), flange, {"left": 0.0})

        # Close the hand right at the target object's location (pixel = image
        # center => camera point (0,0,1) == pelvis point, same as flange
        # translation below) and keep both the hand and the object moving
        # together for the whole confirmation window.
        for k in range(1, 8):
            hand_t = np.array([0.0, 0.0, 1.0]) + np.array([0.001 * k, 0.0, 0.0])
            flange_k = {"left": np.eye(4)}
            flange_k["left"][:3, 3] = hand_t
            # Move the detected pixel so the object's DEPTH point tracks the
            # same drift as the hand (both move together => convergence).
            shift_u = CX + FX * (0.001 * k) / 1.0
            detector.set_detection(target, _box_detection(target, shift_u, CY))
            out = perception.observe(_rgb(), _rgb(), flange_k, {"left": 1.0})

        assert out["latch"]["left"].state == LatchState.LATCHED
        assert out["latch"]["left"].latched_object == target

    def test_hand_closing_near_nothing_stays_unlatched(self):
        cfg = _task_config(hands=("left",))
        perception, detector = _perception(task_config=cfg)
        for obj in cfg.objects:
            # place every object far from the origin in pixel space AND depth,
            # so nothing is within latch_distance_m of the hand at the origin
            detector.set_detection(obj.instance_id, _box_detection(obj.instance_id, CX, CY, half=1.0))
        # push all objects far away in depth so nearest distance exceeds the
        # default 0.05 m latch_distance_m
        perception.depth_source = _StaticDepth(_depth_map(fill=5.0))

        flange = {"left": np.eye(4)}
        perception.observe(_rgb(), _rgb(), flange, {"left": 0.0})
        out = perception.observe(_rgb(), _rgb(), flange, {"left": 1.0})
        assert out["latch"]["left"].state == LatchState.UNLATCHED
