"""Phase-4 refactor pins (docs/deploy_refactor_plan.md §5): the dashboard's
data shape is ONE declared dataclass, and the relation panel + perception
overlay consume the RECORDED `percept` shape — so the live page, the preview
tool, the demo loop, and the replay dashboard cannot drift apart.
"""

import dataclasses
import json

import numpy as np
import pytest

from ego2g1.deploy.ui.overlay import draw_perception_overlay, project_to_pixel
from ego2g1.deploy.ui.telemetry import (
    TelemetrySnapshot,
    executor_row_groups,
    relation_panel,
)


def test_snapshot_is_json_serializable_with_defaults():
    d = TelemetrySnapshot(now=1.0).to_json()
    json.dumps(d)
    assert d["groups"] == executor_row_groups()
    assert d["relation"] is None and d["replay"] is None


def test_snapshot_rejects_unknown_keys():
    with pytest.raises(TypeError):
        TelemetrySnapshot(now=1.0, wall_sloot=3)   # the typo class this kills


def test_all_producers_emit_identical_key_sets():
    """The four producers all construct TelemetrySnapshot, so their key sets
    are identical BY CONSTRUCTION — this pins that no producer bypasses it."""
    from ego2g1.deploy.dashboard import _DemoLoop

    declared = {f.name for f in dataclasses.fields(TelemetrySnapshot)}
    assert set(_DemoLoop().telemetry()) == declared


_SNAPSHOT = {
    "objects": {
        "obj0": {"confidence": 0.9, "box_xyxy": [1.0, 2.0, 8.0, 9.0],
                 "tracked_pose": np.eye(4).tolist(), "last_accepted": True,
                 "detected_this_tick": True, "tracked": True, "depth_m": 0.5},
        "obj1": {"detected_this_tick": False, "tracked": False,
                 "depth_m": None},
    },
    "hands": {
        "left": {"state": "latched", "latched_object": "obj0",
                 "rigid_pose": np.eye(4).tolist(), "hand_closed": True,
                 "candidate_object": None, "ticks_in_candidate": 0,
                 "reason": None},
        "right": {"state": "unlatched", "latched_object": None,
                  "rigid_pose": None},   # old recording: no hand_closed field
    },
}


def test_relation_panel_from_percept_shape():
    panel = relation_panel(_SNAPSHOT, events=[{"t": 1.0, "kind": "hand"}])
    json.dumps(panel)
    by_id = {o["instance_id"]: o for o in panel["objects"]}
    assert by_id["obj0"]["confidence"] == pytest.approx(0.9)
    assert by_id["obj0"]["position_pelvis"] == [0.0, 0.0, 0.0]
    assert by_id["obj1"]["position_pelvis"] is None
    hands = {h["hand"]: h for h in panel["hands"]}
    assert hands["left"]["hand_closed"] is True
    # pre-field recording fallback: unlatched -> open
    assert hands["right"]["hand_closed"] is False
    assert panel["events"] == [{"t": 1.0, "kind": "hand"}]
    assert relation_panel(None, []) is None


def test_overlay_draws_from_percept_shape():
    cv2 = pytest.importorskip("cv2")  # noqa: F841
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    K = np.array([[20.0, 0, 16.0], [0, 20.0, 16.0], [0, 0, 1.0]])
    T = np.eye(4)
    flange = {"left": np.diag([1.0, 1.0, 1.0, 1.0])}
    flange["left"][:3, 3] = [0.0, 0.0, 0.5]   # 0.5 m in front of the camera
    snapshot = {
        "objects": {"obj0": {"confidence": 0.9, "box_xyxy": [2, 2, 10, 10],
                             "tracked_pose": flange["left"].tolist()}},
        "hands": {"left": {"state": "latched",
                           "rigid_pose": flange["left"].tolist()}},
    }
    out = draw_perception_overlay(rgb, snapshot, K, T, flange_poses=flange)
    assert out.shape == rgb.shape
    assert out.any()                 # something was drawn
    assert not rgb.any()             # input not mutated


def test_project_to_pixel_behind_camera_is_none():
    K = np.eye(3)
    assert project_to_pixel(np.array([0.0, 0.0, -1.0]), np.eye(4), K) is None
