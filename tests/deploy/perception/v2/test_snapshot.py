"""`PerceptionSnapshot` invariants and the T4 tick binding."""

from __future__ import annotations

import numpy as np
import pytest

from ego2g1.deploy.perception.v2.snapshot import ControlTickLog

from .conftest import make_snapshot, pose


# --- ControlTickLog (T4) ----------------------------------------------------

def _log(n=10, dt=1 / 30):
    log = ControlTickLog(maxlen=n)
    for i in range(n):
        log.record(i, i * dt, {"left": pose([i * 0.01, 0, 0])}, {"left": 0.0})
    return log


def test_nearest_rounds_to_the_closest_tick_not_backwards():
    """Rounding backwards would bias every binding by up to a full tick in one
    direction; nearest keeps the error zero-mean and under half a tick."""
    log = _log()
    dt = 1 / 30
    assert log.nearest(5 * dt + 0.001).n == 5
    assert log.nearest(5 * dt - 0.001).n == 5
    assert log.nearest(5 * dt + 0.6 * dt).n == 6


def test_quantisation_error_stays_under_half_a_tick():
    log = _log(n=30)
    dt = 1 / 30
    for t in np.linspace(0.0, 29 * dt, 200):
        assert abs(log.nearest(t).t - t) <= dt / 2 + 1e-12


def test_nearest_is_none_before_the_control_loop_starts():
    assert ControlTickLog().nearest(0.0) is None


def test_recorded_poses_are_copied():
    """The control loop reuses its FK dict every tick; a snapshot that aliased
    it would silently mutate after publication."""
    log = ControlTickLog()
    fk = {"left": pose([1.0, 0, 0])}
    log.record(0, 0.0, fk, {"left": 0.0})
    fk["left"][0, 3] = 99.0
    assert log.nearest(0.0).flange_pelvis["left"][0, 3] == 1.0


def test_the_ring_is_bounded():
    log = ControlTickLog(maxlen=5)
    for i in range(100):
        log.record(i, i * 0.03, {"left": pose([0, 0, 0])}, {"left": 0.0})
    assert len(log) == 5
    assert log.latest().n == 99


def test_concurrent_append_and_scan_do_not_raise():
    """A bare deque may be appended to atomically under the GIL, but iterating
    one while another thread appends raises — which would surface as a rare
    mid-rollout crash in the perception thread."""
    import threading

    log = ControlTickLog(maxlen=64)
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            log.record(i, i * 1e-4, {"left": pose([0, 0, 0])}, {"left": 0.0})
            i += 1

    def reader():
        try:
            while not stop.is_set():
                log.nearest(0.5)
        except Exception as exc:                              # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    threading.Event().wait(0.2)
    stop.set()
    for t in threads:
        t.join(2)
    assert not errors


# --- PerceptionSnapshot -----------------------------------------------------

def test_mismatched_per_object_dicts_are_rejected():
    """Every per-object dict must carry the full roster, so that "object 2 is
    missing" is never ambiguous between "not detected" and "never asked
    for"."""
    from ego2g1.deploy.perception.v2.snapshot import PerceptionSnapshot
    common = dict(seq=1, t_capture=0.0, n_capture=0,
                  rgb_left=np.zeros((2, 2, 3), np.uint8),
                  flange_pelvis={"left": pose([0, 0, 0])},
                  hand_frac={"left": 0.0},
                  object_pose_pelvis={"a": None, "b": None},
                  tracker_score={"a": 0.0, "b": 0.0},
                  mask_area_px={"a": 0, "b": 0},
                  mask_usable={"a": False, "b": False},
                  crop_usable={"a": False, "b": False},
                  object_depth_m={"a": None, "b": None}, round_s=0.2)
    with pytest.raises(ValueError, match="det_score"):
        PerceptionSnapshot(det_score={"a": None}, **common)


def test_crop_usable_must_imply_mask_usable():
    with pytest.raises(ValueError, match="crop_usable without mask_usable"):
        make_snapshot(objects={"obj0": pose([0, 0, 0])},
                      usable={"obj0": True}, mask_usable={"obj0": False})


def test_hand_dicts_must_agree():
    from ego2g1.deploy.perception.v2.snapshot import PerceptionSnapshot
    with pytest.raises(ValueError, match="hand_frac"):
        PerceptionSnapshot(
            seq=1, t_capture=0.0, n_capture=0,
            rgb_left=np.zeros((2, 2, 3), np.uint8),
            flange_pelvis={"left": pose([0, 0, 0]), "right": pose([0, 0, 0])},
            hand_frac={"left": 0.0},
            object_pose_pelvis={}, det_score={}, tracker_score={},
            mask_area_px={}, mask_usable={}, crop_usable={},
            object_depth_m={}, round_s=0.2)


def test_helpers():
    snap = make_snapshot(t=10.0,
                         objects={"a": pose([0, 0, 0]), "b": None},
                         usable={"a": True, "b": False})
    assert snap.usable_objects() == ("a",)
    assert snap.missing_objects() == ("b",)
    assert snap.age_s(10.25) == pytest.approx(0.25)
