"""`runner.py`'s `relation_eef` wiring (docs/relation_deploy_plan.md §3/§5):

  1. `resolve_action_mode`'s (deploy/modes) "auto" branch selects `relation_eef` when the
     connected server advertises `control_mode == "relation_eef"`, alongside
     the existing joint/relative_eef cases (regression-checked too).
  2. `RelationEEFMode.build_adapter` fails loud, naming exactly which CLI flag is
     missing, if any of `--task-config`/`--stereo-calib`/`--camera-extrinsic`
     is absent when relation_eef mode is selected.
  3. An end-to-end `DeployRunner` rollout in relation_eef mode: real
     `RelationPolicyAdapter` + real `RelativeEEFRotvecChunks` + real
     `RelationPerception`, but `FakeDetector`/a static depth double/a fake
     policy client/`MockExecutor`/`StaticCamera` standing in for anything
     that would otherwise need hardware, a GPU, or a live server — mirrors
     tests/test_deploy_runner.py's own MockExecutor-based conventions.

Mirrors tests/deploy/test_relation_conversion.py's and
tests/deploy/perception/test_relation_perception.py's fixture/fake patterns
rather than inventing new ones.
"""

import numpy as np
import pytest

from ego2g1.core import layout, relation_layout
from ego2g1.deploy import actions as _actions
from ego2g1.deploy.camera import StaticCamera
from ego2g1.deploy.executor import MockExecutor
from ego2g1.deploy.perception.depth import StereoCalibration
from ego2g1.deploy.perception.detector import Detection, FakeDetector
from ego2g1.deploy.perception.relation_perception import RelationPerception
from ego2g1.deploy.perception.task_config import DeployTaskConfig, ObjectSpec
from ego2g1.deploy.policy_adapter import RelationPolicyAdapter
from ego2g1.deploy.modes import get as get_mode, resolve_action_mode
from ego2g1.deploy.runner import DeployRunner
from ego2g1.deploy.strategies import SynchronousStrategy

FPS = 30
NO_WAIT = lambda t_end, **kw: None  # noqa: E731


# --------------------------------------------------------------------------
# 1. auto action-mode selection
# --------------------------------------------------------------------------


def test_resolve_action_mode_selects_relation_eef_from_server_control_mode():
    assert resolve_action_mode("auto", "relation_eef") == "relation_eef"


def test_resolve_action_mode_still_selects_joint_and_relative_eef():
    assert resolve_action_mode("auto", "joint") == "joint"
    assert resolve_action_mode("auto", "relative_eef") == "relative_eef"
    # anything else (a future/unknown control_mode) falls back to
    # relative_eef, exactly the original two-way ternary's behavior
    assert resolve_action_mode("auto", "something_未来") == "relative_eef"


def test_resolve_action_mode_explicit_override_passes_through():
    # an explicit override is for deliberately testing a mismatched pairing
    # -- it must NOT be overridden by the server's advertised control_mode
    assert resolve_action_mode("joint", "relation_eef") == "joint"
    assert resolve_action_mode("relative_eef", "relation_eef") == "relative_eef"


# --------------------------------------------------------------------------
# 2. missing-required-args fail loudly, independently
# --------------------------------------------------------------------------


class _ArgsStub:
    """Only the fields `RelationEEFMode.build_adapter` actually reads -- avoids
    depending on tyro/the full `Args` dataclass's unrelated defaults."""

    def __init__(self, *, task_config=None, stereo_calib=None, camera_extrinsic=None):
        self.task_config = task_config
        self.stereo_calib = stereo_calib
        self.camera_extrinsic = camera_extrinsic
        self.prompt = "task"
        self.ik_iters = 25
        self.posture_cost = 0.05
        self.collision_min_dist = 0.005


def test_missing_task_config_fails_loud(tmp_path):
    args = _ArgsStub(stereo_calib=str(tmp_path / "c.npz"),
                     camera_extrinsic=str(tmp_path / "e.npz"))
    with pytest.raises(ValueError, match=r"--task-config"):
        get_mode("relation_eef").build_adapter(client=object(), args=args, fps=FPS)


def test_missing_stereo_calib_fails_loud(tmp_path):
    args = _ArgsStub(task_config=str(tmp_path / "t.yaml"),
                     camera_extrinsic=str(tmp_path / "e.npz"))
    with pytest.raises(ValueError, match=r"--stereo-calib"):
        get_mode("relation_eef").build_adapter(client=object(), args=args, fps=FPS)


def test_missing_camera_extrinsic_fails_loud(tmp_path):
    args = _ArgsStub(task_config=str(tmp_path / "t.yaml"),
                     stereo_calib=str(tmp_path / "c.npz"))
    with pytest.raises(ValueError, match=r"--camera-extrinsic"):
        get_mode("relation_eef").build_adapter(client=object(), args=args, fps=FPS)


def test_all_three_missing_names_all_three_flags():
    args = _ArgsStub()
    with pytest.raises(ValueError) as exc_info:
        get_mode("relation_eef").build_adapter(client=object(), args=args, fps=FPS)
    msg = str(exc_info.value)
    assert "--task-config" in msg
    assert "--stereo-calib" in msg
    assert "--camera-extrinsic" in msg


# --------------------------------------------------------------------------
# 3. end-to-end DeployRunner rollout in relation_eef mode
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kin_smooth():
    from ego2g1.deploy.kinematics import Kinematics
    return Kinematics(ik_iters=25, fps=FPS, posture_cost=0.05)


def _nominal_q0():
    """Same comfortable, reachable synthetic anchor
    tests/deploy/test_relation_conversion.py uses (shoulders clear of the
    torso, elbows bent) -- avoids IK/collision edge cases a bare zero pose
    can hit, for a test that's about the runner's plumbing, not IK itself."""
    q0 = np.zeros(layout.ARM_DOF)
    q0[1], q0[8] = 0.15, -0.15
    q0[3], q0[10] = 0.3, 0.3
    return q0


def _task_config():
    return DeployTaskConfig(objects=(
        ObjectSpec("obj0", "pen holder", "a pen holder .", graspable=False),
        ObjectSpec("obj1", "red cube", "a red cube .", graspable=True),
        ObjectSpec("obj2", "yellow cube", "a yellow cube .", graspable=True),
    ))


def _calib():
    return StereoCalibration(
        K_left=np.array([[50.0, 0, 32.0], [0, 50.0, 32.0], [0, 0, 1]]),
        K_right=np.eye(3), dist_left=np.zeros(5), dist_right=np.zeros(5),
        R=np.eye(3), T=np.array([0.06, 0.0, 0.0]), image_size=(64, 64))


class _StaticDepth:
    """Minimal DepthSource double (same pattern as
    test_relation_conversion.py/test_relation_perception.py): ignores the
    stereo pair, returns a fixed metric depth map."""

    def estimate(self, rgb_left, rgb_right):
        return np.full((64, 64), 1.0, dtype=np.float32)


def _fake_detector(task_config):
    detector = FakeDetector()
    for obj in task_config.objects:
        detector.set_detection(obj.instance_id, Detection(
            instance_id=obj.instance_id, confidence=0.9,
            box_xyxy=np.array([30.0, 30.0, 34.0, 34.0])))
    return detector


class _FakeRelationClient:
    """Minimal PolicyClient stand-in (mirrors test_relation_conversion.py's
    `_FakeRelationClient`): fixed (H, 14) chunk, identity EEF deltas (stays
    at anchor), grippers CLOSED (+1) on both hands so the executed hand
    command is checkable precisely against the default
    `gripper_calib.BRAINCO_CLOSED_POSE` (all-ones)."""

    def __init__(self, horizon=4):
        self.action_horizon = horizon
        self.fps = FPS
        self.control_mode = "relation_eef"
        self.calls = 0

    def infer(self, image, state, prompt, **kw):
        self.calls += 1
        chunk = np.zeros((self.action_horizon, relation_layout.ACTION_DIM), dtype=np.float32)
        chunk[:, relation_layout.GRIP["left"]] = 1.0
        chunk[:, relation_layout.GRIP["right"]] = 1.0
        return {"actions": chunk}


def test_deploy_runner_relation_eef_end_to_end(kin_smooth):
    task_config = _task_config()
    detector = _fake_detector(task_config)
    perception = RelationPerception(
        task_config, detector, _StaticDepth(), _calib(),
        T_pelvis_camera=np.eye(4), fps=FPS)

    client = _FakeRelationClient(horizon=4)
    adapter = RelationPolicyAdapter(client, "task", kin=kin_smooth, perception=perception)

    q0 = _nominal_q0()
    executor = MockExecutor(fps=FPS, initial_q=q0)
    executor.connect()
    cam = StaticCamera(shape=(64, 64, 3))
    cam.connect()

    strategy = SynchronousStrategy(adapter, chunk_size=client.action_horizon)
    runner = DeployRunner(adapter=adapter, strategy=strategy, executor=executor,
                          camera=cam, fps=FPS, wait=NO_WAIT,
                          mode="relation_eef", max_steps=8)

    runner.run()

    assert not runner.watchdog.tripped
    assert runner.steps_executed == 8
    assert len(executor.sent) == 8
    assert not executor.damped
    assert client.calls >= 2   # 8 steps / horizon 4

    for _t_target, row in executor.sent:
        assert row.shape == (_actions.ROBOT_DIM,)
        assert np.all(np.isfinite(row))
        for h in layout.HANDS:
            hand_cmd = row[_actions.HAND[h]]
            assert hand_cmd.min() >= -1e-6
            assert hand_cmd.max() <= 1.0 + 1e-6

    # identity-delta chunk -> the arm barely moves off the reachable anchor
    fk_last = kin_smooth.flange_poses(executor.sent[-1][1][_actions.ARM])
    anchor = kin_smooth.flange_poses(q0)
    for h in layout.HANDS:
        assert np.linalg.norm(fk_last[h][:3, 3] - anchor[h][:3, 3]) < 0.05

    # gripper CLOSED (+1) every chunk -> executed hand command converges to
    # the default BRAINCO_CLOSED_POSE (all-ones)
    last_row = executor.sent[-1][1]
    for h in layout.HANDS:
        np.testing.assert_allclose(last_row[_actions.HAND[h]], 1.0, atol=1e-3)

    # relation_mode's last_hands bookkeeping: a SCALAR fraction per hand
    # (not the old modes' (6,)-vector), tracking the executed command
    for h in relation_layout.HANDS:
        assert isinstance(runner.last_hands[h], float)
        assert runner.last_hands[h] == pytest.approx(1.0, abs=1e-3)


def test_deploy_runner_relation_eef_last_hands_starts_open_scalar():
    """Before any step executes, relation_mode's last_hands is a scalar 0.0
    per hand (open) -- not the old modes' np.zeros(HAND_DIM) vector."""
    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = DeployRunner(
        adapter=object(), strategy=object(), executor=executor,
        fps=FPS, wait=NO_WAIT, mode="relation_eef")
    assert runner.last_hands == {"left": 0.0, "right": 0.0}
    for h in relation_layout.HANDS:
        assert isinstance(runner.last_hands[h], float)


def test_telemetry_relation_field_after_relation_eef_run(kin_smooth):
    """`DeployRunner.telemetry()`'s new `"relation"` key (the dashboard's
    detector/tracker/latch overlay data, docs/relation_deploy_plan.md's
    dashboard-overlay plan): populated + JSON-serializable after a real
    relation_eef rollout, with per-object detection/tracking info and
    per-hand grasp/latch state matching what `RelationPerception.observe`
    actually computed."""
    import json

    task_config = _task_config()
    detector = _fake_detector(task_config)
    perception = RelationPerception(
        task_config, detector, _StaticDepth(), _calib(),
        T_pelvis_camera=np.eye(4), fps=FPS)

    client = _FakeRelationClient(horizon=4)
    adapter = RelationPolicyAdapter(client, "task", kin=kin_smooth, perception=perception)

    q0 = _nominal_q0()
    executor = MockExecutor(fps=FPS, initial_q=q0)
    executor.connect()
    cam = StaticCamera(shape=(64, 64, 3))
    cam.connect()

    strategy = SynchronousStrategy(adapter, chunk_size=client.action_horizon)
    runner = DeployRunner(adapter=adapter, strategy=strategy, executor=executor,
                          camera=cam, fps=FPS, wait=NO_WAIT,
                          mode="relation_eef", max_steps=8)
    runner.run()

    t = runner.telemetry()
    json.dumps(t)   # must be JSON-serializable, same as every other field
    rel = t["relation"]
    assert rel is not None

    obj_ids = {o["instance_id"] for o in rel["objects"]}
    assert obj_ids == {o.instance_id for o in task_config.objects}
    for o in rel["objects"]:
        assert o["tracked"] is True
        assert o["confidence"] == pytest.approx(0.9)
        assert o["box_xyxy"] is not None
        assert o["position_pelvis"] is not None

    hands_by_name = {h["hand"]: h for h in rel["hands"]}
    assert set(hands_by_name) == set(task_config.hands)
    for h in task_config.hands:
        # client always commands grippers CLOSED (+1) -- see
        # _FakeRelationClient -- so both hands read as closed
        assert hands_by_name[h]["hand_closed"] is True
        assert hands_by_name[h]["state"] in ("unlatched", "candidate", "latched")

    assert isinstance(rel["events"], list)


def test_telemetry_relation_is_none_outside_relation_mode():
    """joint/relative_eef deploys have no perception stack at all -- the
    dashboard's overlay panels must see a clean `None`, not a crash."""
    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = DeployRunner(
        adapter=object(), strategy=object(), executor=executor,
        fps=FPS, wait=NO_WAIT, mode="joint")
    assert runner.telemetry()["relation"] is None


def test_reset_to_episode_refuses_in_relation_mode():
    executor = MockExecutor(fps=FPS)
    executor.connect()
    runner = DeployRunner(
        adapter=object(), strategy=object(), executor=executor,
        fps=FPS, wait=NO_WAIT, mode="relation_eef", dataset="fake://dataset",
        gated=True)
    with pytest.raises(NotImplementedError, match="relation_eef"):
        runner.reset_to_episode(0)
