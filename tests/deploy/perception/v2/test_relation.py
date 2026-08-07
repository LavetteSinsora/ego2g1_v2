"""Snapshot -> 56-dim relation state (§5.4, T2).

The layout assertions here are the ones that matter most: a mispacked state
vector does not crash, it serves a plausible-looking wrong policy.
"""

from __future__ import annotations

import numpy as np
import pytest

from ego2g1.core import relation_layout, se3
from ego2g1.deploy.perception.v2.latch import LatchConfig, LatchState
from ego2g1.deploy.perception.v2.relation import RelationStateBuilder

from .conftest import OBJECTS, Spec, TaskConfig, make_snapshot, pose

CFG = LatchConfig(latch_distance_m=0.05, confirm_displacement_m=0.04,
                  position_tol_m=0.03, divergence_sustain=2)


def _builder(task_config):
    return RelationStateBuilder(task_config, latch_config=CFG)


# --- layout -----------------------------------------------------------------

def test_state_is_hand_major_and_inverts_the_training_encoding(task_config):
    builder = _builder(task_config)
    flange = {"left": pose([0.1, 0.2, 0.3]), "right": pose([0.4, 0.5, 0.6])}
    objects = {oid: pose([i * 0.1, 0.0, 0.7]) for i, oid in enumerate(OBJECTS)}
    snap = make_snapshot(objects=objects, flange=flange)

    state = builder.state_for(snap)
    assert state.shape == (relation_layout.RELATION_STATE_DIM,)
    assert state.dtype == np.float32

    # Hand-major: three left blocks, then three right, then the grasp bits.
    k = 0
    for hand in ("left", "right"):
        inv = se3.se3_inv(flange[hand])
        for oid in OBJECTS:
            want = se3.se3_to_vec9(inv @ objects[oid])
            np.testing.assert_allclose(state[k * 9:(k + 1) * 9], want,
                                       rtol=1e-6, atol=1e-6)
            k += 1


def test_grasp_bits_come_from_the_snapshot_not_from_now(task_config):
    builder = _builder(task_config)
    snap = make_snapshot(hand_frac={"left": 0.9, "right": 0.1})
    # A control tick with the OPPOSITE command must not change the vector:
    # T2 says the grasp binaries are latched at capture like everything else.
    builder.on_control_tick(t=0.0, flange_poses=snap.flange_pelvis,
                            hand_frac={"left": 0.0, "right": 1.0})
    state = builder.state_for(snap)
    assert state[-2] == 1.0 and state[-1] == 0.0


def test_a_roster_of_the_wrong_size_is_refused_at_construction():
    with pytest.raises(ValueError, match="relation state layout is fixed"):
        RelationStateBuilder(TaskConfig([Spec("only_one")]))


def test_never_seen_object_fails_loud(task_config):
    builder = _builder(task_config)
    objects = {oid: pose([0.5, 0, 0]) for oid in OBJECTS}
    objects["obj1"] = None
    with pytest.raises(RuntimeError, match="obj1"):
        builder.state_for(make_snapshot(objects=objects))


# --- T2 ---------------------------------------------------------------------

def test_the_flange_comes_from_the_snapshot_never_from_fresh_fk(task_config):
    """The load-bearing property of the whole design. The policy also receives
    the snapshot's image; advancing FK but not the image would desynchronise
    two modalities that were synchronised in every training sample."""
    builder = _builder(task_config)
    captured = {"left": pose([0.1, 0, 0]), "right": pose([0.2, 0, 0])}
    snap = make_snapshot(flange=captured)

    builder.on_control_tick(t=1.0,
                            flange_poses={"left": pose([9.0, 9.0, 9.0]),
                                          "right": pose([9.0, 9.0, 9.0])},
                            hand_frac={"left": 0.0, "right": 0.0})
    state = builder.state_for(snap)

    inv = se3.se3_inv(captured["left"])
    want = se3.se3_to_vec9(inv @ snap.object_pose_pelvis["obj0"])
    np.testing.assert_allclose(state[:9], want, rtol=1e-6, atol=1e-6)


# --- latch integration ------------------------------------------------------

def _grasp(builder, hand="left"):
    """Drive one hand through a successful grasp of obj0."""
    obj = pose([0.50, 0, 0])
    near = {"left": pose([0.49, 0, 0]), "right": pose([-1.0, 0, 0])}
    objects = {oid: pose([0.5 + i, 0, 0]) for i, oid in enumerate(OBJECTS)}
    objects["obj0"] = obj

    builder.on_snapshot(make_snapshot(seq=1, t=0.0, objects=objects, flange=near))
    closed = {h: (1.0 if h == hand else 0.0) for h in ("left", "right")}
    builder.on_control_tick(t=0.1, flange_poses=near, hand_frac=closed)

    builder.on_snapshot(make_snapshot(seq=2, t=0.2, objects=objects, flange=near))
    lifted = dict(near)
    lifted[hand] = pose([0.49, 0, 0.05])
    objects_up = dict(objects)
    objects_up["obj0"] = pose([0.50, 0, 0.05])
    builder.on_snapshot(make_snapshot(seq=3, t=0.4, objects=objects_up,
                                      flange=lifted))
    results = builder.on_control_tick(t=0.5, flange_poses=lifted,
                                      hand_frac=closed)
    return results, lifted, objects_up


def test_a_latched_object_is_reported_rigidly(task_config):
    builder = _builder(task_config)
    results, lifted, objects_up = _grasp(builder)
    assert results["left"].state is LatchState.LATCHED

    snap = make_snapshot(seq=4, t=0.6, objects=objects_up, flange=lifted,
                         hand_frac={"left": 1.0, "right": 0.0})
    resolved = builder.resolve(snap)
    expected = lifted["left"] @ builder.latches["left"].transform
    np.testing.assert_allclose(resolved["obj0"], expected, atol=1e-12)
    # Untouched objects still come straight from perception.
    np.testing.assert_allclose(resolved["obj1"], objects_up["obj1"])


def test_one_hand_cannot_claim_the_other_hands_object(task_config):
    """Mirrors training's `claimed` set: an object held by one hand leaves the
    other's candidate pool before its nearest-object search runs."""
    builder = _builder(task_config)
    _grasp(builder, hand="left")
    assert builder.latches["left"].latched_object == "obj0"

    both = {"left": pose([0.49, 0, 0.05]), "right": pose([0.49, 0, 0.05])}
    builder.on_control_tick(t=1.0, flange_poses=both,
                            hand_frac={"left": 1.0, "right": 1.0})
    assert builder.latches["right"].state is LatchState.UNLATCHED


def test_snapshots_are_folded_in_exactly_once(task_config):
    builder = _builder(task_config)
    snap = make_snapshot(seq=7)
    assert builder.on_snapshot(snap) is True
    assert builder.on_snapshot(snap) is False, (
        "the control loop calls this every tick with whatever is newest; "
        "re-folding a round would double-count its evidence")


def test_reset_clears_the_latches(task_config):
    builder = _builder(task_config)
    _grasp(builder)
    builder.reset()
    assert builder.latches["left"].state is LatchState.UNLATCHED
    assert builder.on_snapshot(make_snapshot(seq=1)) is True


def test_debug_snapshot_is_json_safe(task_config):
    import json

    builder = _builder(task_config)
    results, lifted, objects_up = _grasp(builder)
    snap = make_snapshot(seq=4, objects=objects_up, flange=lifted)
    json.dumps(builder.debug_snapshot(snap))
