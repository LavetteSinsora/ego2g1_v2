"""Unit tests for `ego2g1.deploy.perception.latch.GraspLatch`.

Pure synthetic numpy trajectories -- no camera, detector, or robot needed
(docs/relation_deploy_plan.md §9, Phase-2 task 9: "unit-testable in complete
isolation ... write this test first").

Tick-counting convention used throughout: the tick that transitions
UNLATCHED -> CANDIDATE (freezing `T_hand_object`) is the "entry tick" and
does not itself count toward `confirm_window_ticks` (comparing a
just-frozen transform against the pose it was frozen from is trivially
exact, so it is not a real confirmation sample). Reaching LATCHED therefore
takes `confirm_window_ticks + 1` total `update()` calls: one entry tick plus
`confirm_window_ticks` converging ticks. `_latch_onto` below encodes this so
each test doesn't have to re-derive it.
"""

import numpy as np
import pytest

from ego2g1.deploy.perception.latch import GraspLatch, LatchConfig, LatchState


def _pose(t, R=None):
    """Build a (4, 4) from a translation (3,) and an optional rotation (3, 3)."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def _hand_trajectory(n, start=(0.0, 0.0, 0.0), step=(0.01, 0.0, 0.0)):
    """n hand poses translating linearly, identity rotation."""
    start = np.asarray(start, dtype=np.float64)
    step = np.asarray(step, dtype=np.float64)
    return [_pose(start + i * step) for i in range(n)]


def _default_config():
    # small, fast-to-test window; same shape of contract as the documented
    # defaults, just fewer ticks so tests run instantly.
    return LatchConfig(
        latch_distance_m=0.05,
        confirm_window_ticks=5,
        position_tol_m=0.02,
        rotation_tol_deg=15.0,
        max_track_loss_ticks=3,
    )


def _latch_onto(latch, hands, start_t, obj_id, T_hand_object, *, extra_tracks=None):
    """Drive `latch` through one full entry-tick + confirm-window of PERFECT
    convergence for `obj_id`, starting at `hands[start_t]`. Returns
    (final_result, next_t) where `next_t` is the index of the first tick NOT
    consumed. `extra_tracks(t)` optionally supplies other objects' tracked
    poses at tick t (e.g. to prove they're correctly ignored)."""
    t = start_t
    result = None
    n_calls = latch.config.confirm_window_ticks + 1   # entry + confirm ticks
    for _ in range(n_calls):
        tracked = {obj_id: hands[t] @ T_hand_object}
        if extra_tracks is not None:
            tracked.update(extra_tracks(t))
        result = latch.update(
            hand_closed=True,
            hand_pose=hands[t],
            tracked_object_poses=tracked,
            eligible_objects=set(tracked.keys()),
        )
        t += 1
    return result, t


# ---------------------------------------------------------------------------
# 1. Converging trajectory -> latches; rigid prediction wins thereafter.
# ---------------------------------------------------------------------------


def test_converging_trajectory_latches_and_rigid_prediction_wins_after():
    cfg = _default_config()
    latch = GraspLatch(cfg)
    hands = _hand_trajectory(cfg.confirm_window_ticks + 5)
    obj_id = "red_cube"

    # Object starts 3 cm from the hand (within latch_distance_m) and then
    # tracks EXACTLY the rigid prediction (hand_pose @ T_hand_object) for the
    # whole window -- a textbook successful grasp.
    obj0 = _pose(hands[0][:3, 3] + np.array([0.03, 0.0, 0.0]))
    T_hand_object = np.linalg.inv(hands[0]) @ obj0

    result, t = _latch_onto(latch, hands, 0, obj_id, T_hand_object)

    assert result.state == LatchState.LATCHED
    assert result.latched_object == obj_id
    assert result.reason is None

    # Post-latch: even if the live tracker later disagrees wildly (e.g. it
    # got confused, or is reporting a stale/wrong detection), object_pose()
    # must still return the rigid-predicted pose, not the disagreeing one.
    bogus_tracked = _pose(hands[t][:3, 3] + np.array([5.0, 5.0, 5.0]))
    result = latch.update(
        hand_closed=True,
        hand_pose=hands[t],
        tracked_object_poses={obj_id: bogus_tracked},
        eligible_objects={obj_id},
    )
    assert result.state == LatchState.LATCHED
    expected_rigid = hands[t] @ T_hand_object
    got = latch.object_pose(obj_id, bogus_tracked)
    np.testing.assert_allclose(got[:3, 3], expected_rigid[:3, 3], atol=1e-9)
    assert not np.allclose(got[:3, 3], bogus_tracked[:3, 3])


# ---------------------------------------------------------------------------
# 2. Diverging trajectory -> stays/returns unlatched, miss visible in diagnostics.
# ---------------------------------------------------------------------------


def test_diverging_trajectory_falls_back_to_unlatched_with_visible_diagnostic():
    cfg = _default_config()
    latch = GraspLatch(cfg)
    hands = _hand_trajectory(cfg.confirm_window_ticks + 5)
    obj_id = "red_cube"

    obj0 = _pose(hands[0][:3, 3] + np.array([0.02, 0.0, 0.0]))
    T_hand_object = np.linalg.inv(hands[0]) @ obj0

    # Entry tick: perfect agreement (T_hand_object is frozen from this very
    # sample). From tick 1 on, the object stays put in the world (a missed
    # grasp -- hand moves away, object doesn't) instead of following the
    # hand -> position error grows every tick until it trips the tolerance.
    t = 0
    last = latch.update(hand_closed=True, hand_pose=hands[t],
                         tracked_object_poses={obj_id: hands[t] @ T_hand_object},
                         eligible_objects={obj_id})
    assert last.state == LatchState.CANDIDATE
    t += 1

    diverged = None
    for _ in range(cfg.confirm_window_ticks + 3):
        last = latch.update(hand_closed=True, hand_pose=hands[t],
                             tracked_object_poses={obj_id: obj0},
                             eligible_objects={obj_id})
        t += 1
        if last.reason == "diverged":
            diverged = last
            break

    assert diverged is not None, "expected divergence before the window completed"
    assert diverged.state == LatchState.UNLATCHED
    assert diverged.reason == "diverged"
    # the miss must be diagnosable, not silently swallowed
    assert diverged.position_error_m is not None
    assert diverged.position_error_m > cfg.position_tol_m
    assert latch.latched_object is None


# ---------------------------------------------------------------------------
# 3. Tracking lost mid-window -> documented wait-then-fail timeout behavior.
# ---------------------------------------------------------------------------


def test_brief_tracking_loss_waits_and_can_still_latch():
    cfg = _default_config()
    latch = GraspLatch(cfg)
    hands = _hand_trajectory(2 * cfg.confirm_window_ticks + cfg.max_track_loss_ticks + 5)
    obj_id = "red_cube"

    obj0 = _pose(hands[0][:3, 3] + np.array([0.01, 0.0, 0.0]))
    T_hand_object = np.linalg.inv(hands[0]) @ obj0

    # tick 0: enter candidate
    r = latch.update(hand_closed=True, hand_pose=hands[0],
                      tracked_object_poses={obj_id: hands[0] @ T_hand_object},
                      eligible_objects={obj_id})
    assert latch.state == LatchState.CANDIDATE
    assert r.ticks_in_candidate == 0

    # brief loss for `max_track_loss_ticks` ticks (at the tolerance boundary,
    # not beyond it) -- must wait, not fail, and must not advance the window.
    t = 1
    for _ in range(cfg.max_track_loss_ticks):
        r = latch.update(hand_closed=True, hand_pose=hands[t],
                          tracked_object_poses={obj_id: None},
                          eligible_objects={obj_id})
        assert r.state == LatchState.CANDIDATE, "brief loss must not fail the candidate"
        assert r.reason is None
        t += 1
    assert r.ticks_in_candidate == 0, "window must not advance while tracking is lost"

    # tracking recovers and converges for the remainder of the window -> latches.
    for _ in range(cfg.confirm_window_ticks):
        tracked = hands[t] @ T_hand_object
        r = latch.update(hand_closed=True, hand_pose=hands[t],
                          tracked_object_poses={obj_id: tracked},
                          eligible_objects={obj_id})
        t += 1
    assert r.state == LatchState.LATCHED
    assert r.latched_object == obj_id


def test_prolonged_tracking_loss_falls_back_to_unlatched():
    cfg = _default_config()
    latch = GraspLatch(cfg)
    hands = _hand_trajectory(cfg.max_track_loss_ticks + 5)
    obj_id = "red_cube"

    obj0 = _pose(hands[0][:3, 3] + np.array([0.01, 0.0, 0.0]))
    T_hand_object = np.linalg.inv(hands[0]) @ obj0

    latch.update(hand_closed=True, hand_pose=hands[0],
                 tracked_object_poses={obj_id: hands[0] @ T_hand_object},
                 eligible_objects={obj_id})
    assert latch.state == LatchState.CANDIDATE

    r = None
    for t in range(1, cfg.max_track_loss_ticks + 2):
        r = latch.update(hand_closed=True, hand_pose=hands[t],
                          tracked_object_poses={obj_id: None},
                          eligible_objects={obj_id})

    assert r.state == LatchState.UNLATCHED
    assert r.reason == "tracking_lost"
    assert latch.latched_object is None


# ---------------------------------------------------------------------------
# 4. Release, then re-latch onto a DIFFERENT object -- clean state reset.
# ---------------------------------------------------------------------------


def test_release_then_relatch_onto_different_object():
    cfg = _default_config()
    latch = GraspLatch(cfg)
    hands = _hand_trajectory(2 * cfg.confirm_window_ticks + 10)
    obj_a, obj_b = "pen_holder", "yellow_cube"

    obj_a0 = _pose(hands[0][:3, 3] + np.array([0.01, 0.0, 0.0]))
    T_a = np.linalg.inv(hands[0]) @ obj_a0

    result, t = _latch_onto(latch, hands, 0, obj_a, T_a)
    assert result.state == LatchState.LATCHED
    assert result.latched_object == obj_a

    # hand opens -> release
    r = latch.update(hand_closed=False, hand_pose=hands[t],
                      tracked_object_poses={obj_a: hands[t] @ T_a},
                      eligible_objects={obj_a, obj_b})
    assert r.state == LatchState.UNLATCHED
    assert latch.latched_object is None
    entry_t = t + 1

    # hand closes again, this time nearest to a DIFFERENT object -> must be
    # able to latch onto obj_b (fresh T_hand_object, no leftover state from A).
    # obj_a, if it were still being considered, sits far behind now (hand
    # moved on) -- included in every tick to prove it is not what gets latched.
    obj_b0 = _pose(hands[entry_t][:3, 3] + np.array([0.015, 0.0, 0.0]))
    T_b = np.linalg.inv(hands[entry_t]) @ obj_b0
    result, t = _latch_onto(
        latch, hands, entry_t, obj_b, T_b,
        extra_tracks=lambda tt: {obj_a: obj_a0},
    )
    assert result.state == LatchState.LATCHED
    assert result.latched_object == obj_b


# ---------------------------------------------------------------------------
# 5. Two eligible objects at candidate-entry -> nearest one chosen.
# ---------------------------------------------------------------------------


def test_nearest_object_chosen_at_candidate_entry():
    cfg = _default_config()
    latch = GraspLatch(cfg)
    hand = _pose((0.0, 0.0, 0.0))

    near_id, far_id = "near_obj", "far_obj"
    near_pose = _pose((0.01, 0.0, 0.0))   # 1 cm away
    far_pose = _pose((0.04, 0.0, 0.0))    # 4 cm away, still within latch_distance_m

    r = latch.update(
        hand_closed=True, hand_pose=hand,
        tracked_object_poses={near_id: near_pose, far_id: far_pose},
        eligible_objects={near_id, far_id},
    )
    assert r.state == LatchState.CANDIDATE
    assert r.candidate_object == near_id


def test_object_outside_latch_distance_is_not_a_candidate():
    cfg = _default_config()
    latch = GraspLatch(cfg)
    hand = _pose((0.0, 0.0, 0.0))
    far_id = "far_obj"
    far_pose = _pose((10.0, 0.0, 0.0))   # far outside latch_distance_m

    r = latch.update(
        hand_closed=True, hand_pose=hand,
        tracked_object_poses={far_id: far_pose},
        eligible_objects={far_id},
    )
    assert r.state == LatchState.UNLATCHED
    assert r.candidate_object is None


# ---------------------------------------------------------------------------
# Extra edge cases called out explicitly in the task: claimed-object logic.
# ---------------------------------------------------------------------------


def test_already_latched_hand_does_not_reconsider_a_second_object():
    """Within ONE hand: while LATCHED, a second eligible object appearing
    close by must not steal the latch -- candidate search only ever runs
    from UNLATCHED."""
    cfg = _default_config()
    latch = GraspLatch(cfg)
    hands = _hand_trajectory(cfg.confirm_window_ticks + 5)
    obj_a, obj_b = "obj_a", "obj_b"

    obj_a0 = _pose(hands[0][:3, 3] + np.array([0.01, 0.0, 0.0]))
    T_a = np.linalg.inv(hands[0]) @ obj_a0

    result, t = _latch_onto(latch, hands, 0, obj_a, T_a)
    assert result.state == LatchState.LATCHED and result.latched_object == obj_a

    # obj_b now appears essentially on top of the hand -- still must not be
    # able to preempt the existing latch.
    obj_b_pose = hands[t]
    r = latch.update(
        hand_closed=True, hand_pose=hands[t],
        tracked_object_poses={obj_a: hands[t] @ T_a, obj_b: obj_b_pose},
        eligible_objects={obj_a, obj_b},
    )
    assert r.state == LatchState.LATCHED
    assert r.latched_object == obj_a


def test_cross_hand_claimed_object_excluded_from_other_hands_candidates():
    """Mirrors training's `claimed` set (encoding.py's `latch_object_poses`):
    an object already latched to one hand must be excluded from the OTHER
    hand's eligible set by the caller (e.g. RelationPerception owning both
    GraspLatch instances), so the second hand cannot also claim it."""
    cfg = _default_config()
    left = GraspLatch(cfg)
    right = GraspLatch(cfg)

    shared_obj = "shared_obj"
    other_obj = "other_obj"
    left_hand_pose = _pose((0.0, 0.0, 0.0))
    right_hand_pose = _pose((0.02, 0.0, 0.0))   # also close to shared_obj

    obj0 = _pose((0.005, 0.0, 0.0))
    T_left = np.linalg.inv(left_hand_pose) @ obj0

    hands_left = [left_hand_pose] * (cfg.confirm_window_ticks + 1)
    result, _ = _latch_onto(left, hands_left, 0, shared_obj, T_left)
    assert result.state == LatchState.LATCHED and result.latched_object == shared_obj

    # right hand tries to close near the same object. The caller must
    # exclude `left.latched_object` from right's eligible set -- verify that
    # when it does, right does NOT enter CANDIDATE on shared_obj.
    other_pose = _pose((5.0, 0.0, 0.0))   # far away, not a plausible candidate
    eligible_for_right = {shared_obj, other_obj} - {left.latched_object}
    r_right = right.update(
        hand_closed=True, hand_pose=right_hand_pose,
        tracked_object_poses={shared_obj: left_hand_pose @ T_left, other_obj: other_pose},
        eligible_objects=eligible_for_right,
    )
    assert r_right.candidate_object != shared_obj
    assert right.latched_object is None


# ---------------------------------------------------------------------------
# Rotation-tolerance divergence (part of the same convergence test, not just
# translation) -- a pure-rotation mismatch must also be caught.
# ---------------------------------------------------------------------------


def test_rotation_divergence_alone_is_caught():
    from ego2g1.core.rotvec import rotvec_to_mat

    cfg = _default_config()
    latch = GraspLatch(cfg)
    hands = _hand_trajectory(cfg.confirm_window_ticks + 5)
    obj_id = "red_cube"

    obj0 = _pose(hands[0][:3, 3] + np.array([0.01, 0.0, 0.0]))
    T_hand_object = np.linalg.inv(hands[0]) @ obj0

    t = 0
    last = latch.update(hand_closed=True, hand_pose=hands[t],
                         tracked_object_poses={obj_id: hands[t] @ T_hand_object},
                         eligible_objects={obj_id})
    assert last.state == LatchState.CANDIDATE
    t += 1

    big_rot = rotvec_to_mat(np.array([0.0, 0.0, np.deg2rad(45.0)]))
    diverged = None
    for _ in range(cfg.confirm_window_ticks + 3):
        predicted = hands[t] @ T_hand_object
        # position tracks perfectly, but orientation drifts by 45 deg --
        # should trip the rotation tolerance even though position agrees.
        tracked = predicted.copy()
        tracked[:3, :3] = predicted[:3, :3] @ big_rot
        last = latch.update(hand_closed=True, hand_pose=hands[t],
                             tracked_object_poses={obj_id: tracked},
                             eligible_objects={obj_id})
        t += 1
        if last.reason == "diverged":
            diverged = last
            break

    assert diverged is not None
    assert diverged.rotation_error_deg is not None
    assert diverged.rotation_error_deg > cfg.rotation_tol_deg
    assert latch.latched_object is None
