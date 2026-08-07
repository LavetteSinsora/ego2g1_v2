"""Shared builders for the perception-v2 tests.

Every test here runs on any machine: no GPU, no SAM 3 weights, no camera. The
modules that need those (`sam3_source.Sam3Source`, `orientation_v2
.OrientAnythingV2`) are deliberately thin wrappers around code that is tested
directly, so what is left untested here is exactly the model call itself —
see docs/perception_v2_notes.md for the hardware checks that cover it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ego2g1.deploy.perception.v2.snapshot import PerceptionSnapshot

HANDS = ("left", "right")
OBJECTS = ("obj0", "obj1", "obj2")


def pose(xyz, R=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    return T


class Spec:
    """Minimal stand-in for `task_config.ObjectSpec` (duck-typed everywhere)."""

    def __init__(self, instance_id, prompt=None, graspable=True):
        self.instance_id = instance_id
        self.category = instance_id
        self.detector_prompt = prompt or f"a {instance_id}"
        self.graspable = graspable


class TaskConfig:
    def __init__(self, objects, hands=HANDS):
        self.objects = tuple(objects)
        self.hands = tuple(hands)


def make_snapshot(*, seq=1, t=0.0, n=0, objects=None, flange=None,
                  hand_frac=None, usable=True, mask_usable=None,
                  round_s=0.22) -> PerceptionSnapshot:
    """A snapshot with sensible defaults, overridable per field.

    `objects` maps instance_id -> (4,4) pose or None. `usable` may be a bool
    (applied to every object) or a per-object dict; `mask_usable` defaults to
    `crop_usable` OR-ed with "has a pose", preserving the invariant that
    crop_usable implies mask_usable.
    """
    objects = objects if objects is not None else {
        oid: pose([0.5, 0.0, 0.0]) for oid in OBJECTS}
    flange = flange if flange is not None else {
        h: pose([0.0, 0.0, 0.0]) for h in HANDS}
    hand_frac = hand_frac if hand_frac is not None else {h: 0.0 for h in flange}
    crop = ({oid: bool(usable) for oid in objects}
            if isinstance(usable, bool) else dict(usable))
    mask = (dict(mask_usable) if mask_usable is not None
            else {oid: crop[oid] or objects[oid] is not None for oid in objects})
    return PerceptionSnapshot(
        seq=seq, t_capture=t, n_capture=n,
        rgb_left=np.zeros((4, 4, 3), dtype=np.uint8),
        flange_pelvis=flange, hand_frac=hand_frac,
        object_pose_pelvis=objects,
        det_score={oid: (0.9 if crop[oid] else None) for oid in objects},
        tracker_score={oid: 0.9 for oid in objects},
        mask_area_px={oid: 1000 for oid in objects},
        mask_usable=mask, crop_usable=crop,
        object_depth_m={oid: 0.5 for oid in objects},
        round_s=round_s)


@pytest.fixture
def task_config():
    return TaskConfig([Spec(oid) for oid in OBJECTS])
