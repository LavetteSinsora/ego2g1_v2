"""The §6 latch state machine.

This is the most intricate piece of new logic and the one whose failures are
least visible from outside: a latch that never engages and a latch that
engages wrongly both show up only as an object pose that stops agreeing with
reality. So the tests here assert on the DECISIONS and the evidence behind
them, not just the end state.
"""

from __future__ import annotations

import numpy as np
import pytest

from ego2g1.deploy.perception.v2.latch import (
    GraspLatch, LatchConfig, LatchState,
)

from .conftest import make_snapshot, pose

CFG = LatchConfig(latch_distance_m=0.05, confirm_displacement_m=0.04,
                  position_tol_m=0.03, divergence_sustain=2,
                  candidate_timeout_s=3.0, max_stale_s=2.0)


def _latch(**over):
    return GraspLatch("left", LatchConfig(**{**CFG.__dict__, **over}))


def _snap(t, obj_xyz, flange_xyz, *, seq=1, usable=True):
    return make_snapshot(
        seq=seq, t=t,
        objects={"obj0": None if obj_xyz is None else pose(obj_xyz)},
        flange={"left": pose(flange_xyz)},
        hand_frac={"left": 0.0},
        usable=usable)


# --- entry ------------------------------------------------------------------

def test_no_candidate_until_hand_closes():
    latch = _latch()
    latch.on_snapshot(_snap(0.0, [0.50, 0, 0], [0.48, 0, 0]))
    r = latch.on_control_tick(hand_closed=False, hand_pose=pose([0.48, 0, 0]),
                              t=0.1, eligible_objects={"obj0"})
    assert r.state is LatchState.UNLATCHED


def test_candidate_needs_an_object_within_reach():
    latch = _latch()
    latch.on_snapshot(_snap(0.0, [0.50, 0, 0], [0.0, 0, 0]))
    r = latch.on_control_tick(hand_closed=True, hand_pose=pose([0.0, 0, 0]),
                              t=0.1, eligible_objects={"obj0"})
    assert r.state is LatchState.UNLATCHED, "30 cm away is not a grasp"


def test_freeze_uses_fresh_fk_and_the_stale_object_pose():
    """§6.4's asymmetry, which is the single easiest thing to get wrong.

    Between the last clean look and closure the HAND moves a great deal and
    the OBJECT does not move at all. So the transform must pair the flange at
    closure with the object as last SEEN — and composing it back against that
    same flange must reproduce the object exactly.
    """
    latch = _latch()
    seen = [0.50, 0.0, 0.10]
    latch.on_snapshot(_snap(0.0, seen, [0.10, 0, 0]))     # object seen, hand far

    # The hand travels for a second, then closes right next to the object.
    closure = pose([0.49, 0.0, 0.11])
    latch.on_control_tick(hand_closed=True, hand_pose=closure, t=1.0,
                          eligible_objects={"obj0"})

    assert latch.state is LatchState.CANDIDATE
    predicted = closure @ latch.transform
    np.testing.assert_allclose(predicted[:3, 3], seen, atol=1e-12)


def test_freeze_refuses_a_stale_observation():
    latch = _latch(max_stale_s=0.5)
    latch.on_snapshot(_snap(0.0, [0.50, 0, 0], [0.10, 0, 0]))
    r = latch.on_control_tick(hand_closed=True, hand_pose=pose([0.49, 0, 0]),
                              t=5.0, eligible_objects={"obj0"})
    assert r.state is LatchState.UNLATCHED, (
        "a 5 s old object pose carries no 'nobody touched it' guarantee")


# --- confirmation -----------------------------------------------------------

def _enter_candidate(latch, *, obj=(0.50, 0.0, 0.0), hand=(0.49, 0.0, 0.0)):
    latch.on_snapshot(_snap(0.0, list(obj), list(hand)))
    latch.on_control_tick(hand_closed=True, hand_pose=pose(hand), t=0.1,
                          eligible_objects={"obj0"})
    assert latch.state is LatchState.CANDIDATE
    return latch


def test_successful_grasp_confirms_after_enough_travel():
    latch = _enter_candidate(_latch())
    # First post-closure snapshot sets the divergence baseline.
    latch.on_snapshot(_snap(0.2, [0.50, 0, 0], [0.49, 0, 0], seq=2))
    assert latch.state is LatchState.CANDIDATE, "baseline alone confirms nothing"

    # Hand travels 5 cm; the object travels with it.
    latch.on_snapshot(_snap(0.4, [0.50, 0, 0.05], [0.49, 0, 0.05], seq=3))
    assert latch.state is LatchState.LATCHED

    r = latch.on_control_tick(hand_closed=True, hand_pose=pose([0.49, 0, 0.05]),
                              t=0.5, eligible_objects={"obj0"})
    assert r.latched_object == "obj0"


def test_failed_grasp_diverges_and_does_not_latch():
    latch = _enter_candidate(_latch())
    latch.on_snapshot(_snap(0.2, [0.50, 0, 0], [0.49, 0, 0], seq=2))
    # Hand lifts; the object stays on the table.
    latch.on_snapshot(_snap(0.4, [0.50, 0, 0], [0.49, 0, 0.05], seq=3))
    latch.on_snapshot(_snap(0.6, [0.50, 0, 0], [0.49, 0, 0.10], seq=4))

    r = latch.on_control_tick(hand_closed=True, hand_pose=pose([0.49, 0, 0.10]),
                              t=0.7, eligible_objects={"obj0"})
    assert r.state is LatchState.UNLATCHED
    assert r.reason == "confirm_diverged"
    assert r.divergence_m == pytest.approx(0.10, abs=1e-9)


def test_a_single_bad_sample_does_not_reject_a_good_grasp():
    """One agreement confirms; N disagreements reject. The asymmetry is the
    design: after 4 cm of travel, agreement is physically conclusive, while a
    lone disagreement is exactly what one bad depth sample looks like."""
    latch = _enter_candidate(_latch())
    latch.on_snapshot(_snap(0.2, [0.50, 0, 0], [0.49, 0, 0], seq=2))
    # One outlier: the object appears not to have moved.
    latch.on_snapshot(_snap(0.4, [0.50, 0, 0], [0.49, 0, 0.05], seq=3))
    assert latch.state is LatchState.CANDIDATE, "one sample must not reject"
    # Then it agrees again.
    latch.on_snapshot(_snap(0.6, [0.50, 0, 0.10], [0.49, 0, 0.10], seq=4))
    assert latch.state is LatchState.LATCHED


def test_confirmation_waits_for_travel_not_for_time():
    """A hand that has barely moved cannot distinguish a good grasp from a bad
    one no matter how many samples arrive."""
    latch = _enter_candidate(_latch())
    for i, z in enumerate([0.0, 0.002, 0.004, 0.006, 0.008], start=2):
        # Object stationary — a FAILED grasp — but with sub-threshold travel.
        latch.on_snapshot(_snap(0.2 * i, [0.50, 0, 0], [0.49, 0, z], seq=i))
    assert latch.state is LatchState.CANDIDATE, (
        "with 8 mm of travel there is no signal either way")


def test_candidate_times_out_when_the_hand_never_moves():
    latch = _enter_candidate(_latch(candidate_timeout_s=1.0))
    r = latch.on_control_tick(hand_closed=True, hand_pose=pose([0.49, 0, 0]),
                              t=5.0, eligible_objects={"obj0"})
    assert r.reason == "candidate_timeout"
    assert r.state is LatchState.UNLATCHED


def test_release_before_confirm_is_a_miss_not_a_success():
    latch = _enter_candidate(_latch())
    r = latch.on_control_tick(hand_closed=False, hand_pose=pose([0.49, 0, 0]),
                              t=0.5, eligible_objects={"obj0"})
    assert r.reason == "released_before_confirm"
    assert r.latched_object is None


# --- visibility gating ------------------------------------------------------

def test_unusable_observations_suspend_the_check():
    """§6.6's guard: no evidence beats bad evidence. An unusable round must
    neither confirm nor reject."""
    latch = _enter_candidate(_latch())
    latch.on_snapshot(_snap(0.2, [0.50, 0, 0], [0.49, 0, 0], seq=2))
    before = latch.state
    # A big apparent divergence, but the crop is not trustworthy.
    latch.on_snapshot(_snap(0.4, [0.50, 0, 0], [0.49, 0, 0.10], seq=3,
                            usable=False))
    r = latch.on_control_tick(hand_closed=True, hand_pose=pose([0.49, 0, 0.10]),
                              t=0.5, eligible_objects={"obj0"})
    assert latch.state is before
    assert r.reason is None
    assert r.usable_observations == 1, "the unusable round must not be counted"


def test_mask_gate_admits_what_the_crop_gate_rejects():
    """`divergence_gate` is the knob most likely to be flipped during bring-up,
    so the two settings must actually differ."""
    strict = _enter_candidate(_latch(divergence_gate="crop"))
    loose = _enter_candidate(_latch(divergence_gate="mask"))
    for latch in (strict, loose):
        latch.on_snapshot(_snap(0.2, [0.50, 0, 0], [0.49, 0, 0], seq=2))
    # crop_usable False but mask_usable True (a memory-propagated mask).
    snap = make_snapshot(
        seq=3, t=0.4, objects={"obj0": pose([0.50, 0, 0.05])},
        flange={"left": pose([0.49, 0, 0.05])}, hand_frac={"left": 1.0},
        usable=False, mask_usable={"obj0": True})
    strict.on_snapshot(snap)
    loose.on_snapshot(snap)
    assert strict.state is LatchState.CANDIDATE
    assert loose.state is LatchState.LATCHED


# --- while latched ----------------------------------------------------------

def _latched():
    latch = _enter_candidate(_latch())
    latch.on_snapshot(_snap(0.2, [0.50, 0, 0], [0.49, 0, 0], seq=2))
    latch.on_snapshot(_snap(0.4, [0.50, 0, 0.05], [0.49, 0, 0.05], seq=3))
    assert latch.state is LatchState.LATCHED
    return latch


def test_latched_pose_is_rigid_and_evaluated_at_the_snapshot_flange():
    """§6.7 and T2: the prediction is composed against the flange from the
    SNAPSHOT's instant, never a fresher one."""
    latch = _latched()
    flange_then = pose([0.60, 0.10, 0.20])
    flange_now = pose([0.90, 0.90, 0.90])
    got = latch.object_pose("obj0", pose([0, 0, 0]), flange_then)
    np.testing.assert_allclose(got, flange_then @ latch.transform)
    assert not np.allclose(got, flange_now @ latch.transform)


def test_dropped_object_unlatches_after_sustained_divergence():
    latch = _latched()
    # Object left behind while the hand keeps moving.
    latch.on_snapshot(_snap(0.6, [0.50, 0, 0.05], [0.49, 0, 0.15], seq=4))
    latch.on_snapshot(_snap(0.8, [0.50, 0, 0.05], [0.49, 0, 0.25], seq=5))
    r = latch.on_control_tick(hand_closed=True, hand_pose=pose([0.49, 0, 0.25]),
                              t=0.9, eligible_objects={"obj0"})
    assert r.reason == "diverged"
    assert r.state is LatchState.UNLATCHED


def test_carrying_far_does_not_drift_into_a_false_divergence():
    """The occlusion bias cancels only while it is common-mode. A fixed origin
    would accumulate drift over a long carry; the sliding one-observation
    window keeps the differencing interval short."""
    latch = _latched()
    # Carry 60 cm, with a constant 2 cm centroid bias from partial occlusion.
    for i, z in enumerate(np.arange(0.10, 0.70, 0.05), start=4):
        latch.on_snapshot(_snap(0.2 * i, [0.50, 0.02, z], [0.49, 0, z], seq=i))
    assert latch.state is LatchState.LATCHED


def test_opening_the_gripper_releases_immediately():
    latch = _latched()
    r = latch.on_control_tick(hand_closed=False, hand_pose=pose([0.49, 0, 0.05]),
                              t=1.0, eligible_objects={"obj0"})
    assert r.state is LatchState.UNLATCHED
    assert r.reason is None, "a clean release is not a failure"


def test_candidate_reports_the_tracked_pose_not_its_own_prediction():
    """An unconfirmed hypothesis must not be fed to the policy as fact."""
    latch = _enter_candidate(_latch())
    tracked = pose([1.0, 2.0, 3.0])
    assert latch.object_pose("obj0", tracked, pose([0, 0, 0])) is tracked


def test_reset_forgets_remembered_observations():
    latch = _latch()
    latch.on_snapshot(_snap(0.0, [0.50, 0, 0], [0.49, 0, 0]))
    latch.reset()
    r = latch.on_control_tick(hand_closed=True, hand_pose=pose([0.49, 0, 0]),
                              t=0.1, eligible_objects={"obj0"})
    assert r.state is LatchState.UNLATCHED, (
        "after a reset nothing about the world is still known to be true")


def test_snapshot_missing_this_hand_is_an_error_not_a_silent_skip():
    latch = _enter_candidate(GraspLatch("right", CFG))
    with pytest.raises(KeyError, match="right"):
        latch.on_snapshot(_snap(0.2, [0.50, 0, 0], [0.49, 0, 0], seq=2))


def test_bad_divergence_gate_is_rejected_at_construction():
    with pytest.raises(ValueError, match="divergence_gate"):
        LatchConfig(divergence_gate="whatever")
