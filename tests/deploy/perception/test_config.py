"""Phase-6 refactor pins (docs/deploy_refactor_plan.md §6): every perception
tuning knob has one YAML-loadable owner, unknown keys fail loud, and the
calibration artifacts carry provenance."""

import numpy as np
import pytest

from ego2g1.deploy.perception.config import RelationPerceptionConfig
from ego2g1.deploy.perception.latch import LatchConfig


def test_defaults_match_constructor_defaults():
    cfg = RelationPerceptionConfig.load(None)
    assert cfg.detector_period_ticks is None      # -> ~2 Hz from fps
    assert cfg.orientation_period_ticks == 6
    assert cfg.latch_config() == LatchConfig()
    assert cfg.tracker == {} and cfg.sgbm == {}


def test_yaml_round_trip_with_overrides(tmp_path):
    y = tmp_path / "p.yaml"
    y.write_text(
        "detector_period_ticks: 15\n"
        "latch:\n  latch_distance_m: 0.08\n  confirm_window_ticks: 20\n"
        "tracker:\n  min_residual_m: 0.02\n"
        "sgbm:\n  num_disparities: 64\n")
    cfg = RelationPerceptionConfig.load(y)
    assert cfg.detector_period_ticks == 15
    lc = cfg.latch_config()
    assert lc.latch_distance_m == pytest.approx(0.08)
    assert lc.confirm_window_ticks == 20
    assert lc.position_tol_m == LatchConfig().position_tol_m   # untouched default
    assert cfg.tracker == {"min_residual_m": 0.02}
    assert cfg.sgbm == {"num_disparities": 64}
    # as_dict is JSON-safe and embeds into meta.json via build_meta
    import json

    from ego2g1.deploy.record.schema import build_meta

    meta = build_meta(mode="sync", action_mode="relation_eef", fps=30,
                      horizon=8, source="test",
                      strategy_params={"inference_hz": 0, "exp_weight_m": 0,
                                       "max_latency_steps": 0,
                                       "min_smooth_steps": 0},
                      perception_config=cfg.as_dict())
    json.dumps(meta)
    assert meta["perception_config"]["latch"]["latch_distance_m"] == 0.08


def test_unknown_keys_fail_loud(tmp_path):
    y = tmp_path / "p.yaml"
    y.write_text("detektor_period_ticks: 15\n")
    with pytest.raises(ValueError, match="unknown perception-config key"):
        RelationPerceptionConfig.load(y)
    y.write_text("latch:\n  latch_distanse_m: 0.08\n")
    with pytest.raises(ValueError, match="unknown latch key"):
        RelationPerceptionConfig.load(y)


def test_config_reaches_relation_perception(tmp_path):
    """The loaded latch override actually lands on the GraspLatch instances
    — the 'retuning no longer needs a source edit' claim, executable."""
    from ego2g1.deploy.perception.depth import StereoCalibration
    from ego2g1.deploy.perception.detector import FakeDetector
    from ego2g1.deploy.perception.relation_perception import RelationPerception
    from ego2g1.deploy.perception.task_config import DeployTaskConfig, ObjectSpec

    y = tmp_path / "p.yaml"
    y.write_text("detector_period_ticks: 7\nlatch:\n  latch_distance_m: 0.11\n")
    cfg = RelationPerceptionConfig.load(y)

    calib = StereoCalibration(
        K_left=np.eye(3), K_right=np.eye(3), dist_left=np.zeros(5),
        dist_right=np.zeros(5), R=np.eye(3), T=np.array([0.06, 0, 0]),
        image_size=(64, 64))
    rp = RelationPerception(
        DeployTaskConfig(objects=(ObjectSpec("o", "cube", "a cube ."),)),
        FakeDetector(), None, calib, np.eye(4), fps=30,
        detector_period_ticks=cfg.detector_period_ticks,
        orientation_period_ticks=cfg.orientation_period_ticks,
        latch_config=cfg.latch_config(), tracker_kwargs=cfg.tracker)
    assert rp._detector_period == 7
    for latch in rp.latches.values():
        assert latch.config.latch_distance_m == pytest.approx(0.11)


def test_touch_calib_writes_provenance(tmp_path):
    from ego2g1.deploy.perception import touch_calib

    rng = np.random.default_rng(0)
    pts_pelvis = rng.uniform(-0.3, 0.3, size=(6, 3))
    R = np.eye(3)
    t = np.array([0.1, 0.0, 0.2])
    pts_camera = (pts_pelvis - t) @ R          # exact correspondence, R=I
    np.save(tmp_path / "cam.npy", pts_camera)
    np.save(tmp_path / "pel.npy", pts_pelvis)
    out = tmp_path / "camera_calib.npz"
    touch_calib._cli_solve(str(tmp_path / "cam.npy"), str(tmp_path / "pel.npy"),
                           str(out))
    data = np.load(out)
    assert data["method"].item() == "touch_calib"
    assert "solved_iso" in data.files
    assert "T_pelvis_camera" in data.files
