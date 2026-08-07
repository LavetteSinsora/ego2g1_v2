"""The round and the thread (T1/T3/T4), with fakes for the two GPU stages.

What is being tested is the WIRING — tick binding, the two visibility gates
reaching the right consumers, orientation being skipped for latched objects,
staleness accounting, and the replan primitive's blocking semantics. The model
calls themselves are hardware-gated; see docs/perception_v2_notes.md.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from ego2g1.deploy.perception.v2.async_perception import (
    AsyncPerception, PerceptionRound,
)
from ego2g1.deploy.perception.v2.sam3_source import (
    SlotObservation, Visibility,
)
from ego2g1.deploy.perception.v2.snapshot import ControlTickLog

from .conftest import OBJECTS, Spec, pose

K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


class FakeCalib:
    K_left = K


class FakeSam3:
    """Returns a fixed mask per slot; `usable` controls the two gates."""

    def __init__(self, usable=True, mask_usable=None):
        self.usable = usable
        self.mask_usable = usable if mask_usable is None else mask_usable
        self.steps = 0
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:22, 10:22] = True                     # 144 px, centred at 15.5
        self._mask = mask

    def step(self, rgb):
        self.steps += 1
        return {oid: SlotObservation(oid, self._mask, None, 0.9, 0.9,
                                     int(self._mask.sum()), False)
                for oid in OBJECTS}

    def visibility(self, observations):
        return {oid: Visibility(self.mask_usable, self.usable)
                for oid in observations}


class FakeDepth:
    def estimate(self, left, right):
        return np.full((32, 32), 1.0)


class FakeOrient:
    def __init__(self):
        self.calls: list[tuple] = []

    def estimate(self, rgb, observations, crop_usable, *, skip=frozenset(),
                 anchor_id=None, points_cam=None):
        wanted = [o for o in observations if crop_usable.get(o) and o not in skip]
        self.calls.append((tuple(wanted), frozenset(skip), anchor_id))
        return {oid: np.eye(3) for oid in wanted}


def build(*, usable=True, mask_usable=None, orient=None, latched=None,
          clock=None):
    log = ControlTickLog()
    for i in range(10):
        log.record(i, i * 0.1, {"left": pose([i * 0.01, 0, 0])}, {"left": 0.3})
    sam3 = FakeSam3(usable, mask_usable)
    times = iter(clock) if clock is not None else None
    round_ = PerceptionRound(
        read_stereo=lambda: (np.zeros((32, 32, 3), np.uint8),
                             np.zeros((32, 32, 3), np.uint8)),
        tick_log=log, sam3=sam3, depth_source=FakeDepth(), calib=FakeCalib(),
        T_pelvis_camera=np.eye(4), objects=[Spec(o) for o in OBJECTS],
        orientation=orient, latched_objects=(lambda: frozenset(latched or ())),
        clock=(lambda: next(times)) if times else (lambda: 0.5))
    return round_, sam3, log


# --- one round --------------------------------------------------------------

def test_a_round_produces_a_complete_snapshot():
    round_, _, _ = build()
    snap = round_.step()
    assert set(snap.object_pose_pelvis) == set(OBJECTS)
    assert all(p is not None for p in snap.object_pose_pelvis.values())
    assert snap.seq == 1
    round_.close()


def test_the_frame_binds_to_the_nearest_control_tick(monkeypatch):
    """T4: the perception thread never computes FK itself, it reads the FK the
    control loop already computed at the tick nearest capture."""
    round_, _, _ = build(clock=[0.0, 0.42, 0.42])      # capture at t=0.42
    snap = round_.step()
    assert snap.n_capture == 4                          # ticks are 0.1 s apart
    np.testing.assert_allclose(snap.flange_pelvis["left"][0, 3], 0.04)
    round_.close()


def test_a_round_without_a_control_tick_is_dropped():
    """A snapshot with no FK to pair the image with would violate T2, so there
    is nothing honest to publish."""
    round_, _, _ = build()
    round_._tick_log = ControlTickLog()                 # noqa: SLF001
    assert round_.step() is None
    round_.close()


def test_grasp_binaries_come_from_the_bound_tick():
    round_, _, _ = build()
    snap = round_.step()
    assert snap.hand_frac == {"left": 0.3}
    round_.close()


def test_round_s_is_measured_not_assumed():
    """T1: whatever rate results IS the rate. Every downstream window converts
    using it, so it has to be observed."""
    round_, _, _ = build(clock=[10.0, 10.0, 10.25])
    assert round_.step().round_s == pytest.approx(0.25)
    round_.close()


# --- the two gates reach the right consumers --------------------------------

def test_an_unusable_crop_still_updates_position():
    """S1's asymmetry end to end: mask_usable feeds position, crop_usable
    gates orientation."""
    orient = FakeOrient()
    round_, _, _ = build(usable=False, mask_usable=True, orient=orient)
    snap = round_.step()
    assert all(p is not None for p in snap.object_pose_pelvis.values())
    assert snap.crop_usable == {o: False for o in OBJECTS}
    assert orient.calls[-1][0] == (), "no orientation from an unusable crop"
    round_.close()


def test_an_unusable_mask_yields_no_pose_at_all():
    round_, _, _ = build(usable=False, mask_usable=False)
    snap = round_.step()
    assert snap.missing_objects() == tuple(OBJECTS)
    round_.close()


def test_a_latched_object_costs_no_orientation_inference():
    """R2's third lever: while an object is rigidly held its pose comes from
    FK, so inferring its orientation is both unnecessary and unreliable."""
    orient = FakeOrient()
    round_, _, _ = build(orient=orient, latched={"obj1"})
    round_.step()
    wanted, skipped, _ = orient.calls[-1]
    assert "obj1" not in wanted and skipped == {"obj1"}
    round_.close()


def test_the_anchor_defaults_to_the_first_roster_entry():
    """Training's anchor is obj_keys[0] (CamTriangulator.py:197). Only it gets
    the model's own rotation; every other slot's is constructed relative to
    it, so getting this wrong restructures 2 of 3 slots."""
    orient = FakeOrient()
    round_, _, _ = build(orient=orient)
    round_.step()
    assert orient.calls[-1][2] == OBJECTS[0]
    round_.close()


def test_position_holds_when_a_measurement_stops_arriving():
    round_, sam3, _ = build()
    first = round_.step()
    sam3.mask_usable = sam3.usable = False
    held = round_.step()
    np.testing.assert_allclose(held.object_pose_pelvis["obj0"],
                               first.object_pose_pelvis["obj0"])
    round_.close()


# --- the thread and the replan primitive ------------------------------------

class SlowRound:
    """A round that publishes on demand, so the blocking semantics of
    `wait_for_current_round` can be tested without racing a real clock."""

    def __init__(self):
        self.release = threading.Event()
        self.seq = 0

    def step(self):
        self.release.wait(2.0)
        self.release.clear()
        self.seq += 1
        from .conftest import make_snapshot
        return make_snapshot(seq=self.seq, t=self.seq * 0.2)


def test_wait_returns_the_round_that_was_in_flight():
    """T3: do NOT grab the newest completed snapshot and send immediately —
    let the running round finish and send ITS snapshot, which is what pins
    t_send - t_capture at exactly P and makes `d` constant."""
    slow = SlowRound()
    perception = AsyncPerception(slow)
    perception.start()
    try:
        slow.release.set()
        first = perception.wait_for_current_round(2.0)
        assert first.seq == 1
        slow.release.set()
        assert perception.wait_for_current_round(2.0).seq == 2
    finally:
        slow.release.set()
        perception.stop(1.0)


def test_wait_falls_back_rather_than_starving_the_control_loop():
    """On timeout, a larger `d` for one call beats a control loop with nothing
    to execute. The caller can tell the two apart: a fallback snapshot's age
    exceeds its round_s."""
    slow = SlowRound()
    perception = AsyncPerception(slow)
    perception.start()
    try:
        slow.release.set()
        perception.wait_for_current_round(2.0)
        stale = perception.wait_for_current_round(0.05)   # nothing in flight
        assert stale.seq == 1
    finally:
        slow.release.set()
        perception.stop(1.0)


def test_wait_returns_none_before_any_round_completes():
    slow = SlowRound()
    perception = AsyncPerception(slow)
    perception.start()
    try:
        assert perception.wait_for_current_round(0.05) is None
    finally:
        slow.release.set()
        perception.stop(1.0)


def test_a_failing_round_does_not_kill_the_thread():
    """One bad frame must not end perception for the rest of the rollout: the
    control loop keeps consuming the last good snapshot, which grows stale
    visibly rather than vanishing."""
    class Flaky:
        def __init__(self):
            self.n = 0

        def step(self):
            self.n += 1
            if self.n < 3:
                raise RuntimeError("bad frame")
            from .conftest import make_snapshot
            threading.Event().wait(0.01)
            return make_snapshot(seq=self.n)

    perception = AsyncPerception(Flaky())
    perception.start()
    try:
        assert perception.wait_for_current_round(2.0) is not None
        assert perception.stats()["errors"] == 2
    finally:
        perception.stop(1.0)


def test_starting_twice_is_an_error():
    perception = AsyncPerception(SlowRound())
    perception.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            perception.start()
    finally:
        perception._round.release.set()                   # noqa: SLF001
        perception.stop(1.0)
