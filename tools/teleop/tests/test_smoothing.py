"""One-Euro finger smoother: attenuates jitter, tracks fast motion, holds through dropouts.
Plus --fingers-only, which pins the arm while leaving the fingers live."""

import numpy as np

from tools.teleop.retarget import _OneEuro, SIDES, TeleopRetargeter
from tools.teleop.source import Hdf5Source

EPISODE = "data/put_bottle_in_box_ego/episode_1.hdf5"


def test_constant_input_is_passed_through():
    f = _OneEuro(min_cutoff=1.5, beta=0.05, d_cutoff=1.0)
    x = np.full(6, 0.4, dtype=np.float32)
    out = [f.filter(x, 1 / 60.0) for _ in range(50)]
    assert np.allclose(out[-1], x, atol=1e-4)


def test_single_tick_spike_is_attenuated():
    f = _OneEuro(min_cutoff=1.5, beta=0.05, d_cutoff=1.0)
    dt = 1 / 60.0
    for _ in range(30):
        f.filter(np.zeros(6, dtype=np.float32), dt)   # settle at 0
    spike = f.filter(np.ones(6, dtype=np.float32), dt)  # one-tick jump to 1
    assert (spike < 0.6).all(), "spike not attenuated"


def test_fast_ramp_tracked_with_bounded_lag():
    f = _OneEuro(min_cutoff=1.5, beta=0.05, d_cutoff=1.0)
    dt = 1 / 60.0
    xs = np.linspace(0, 1, 40).astype(np.float32)
    out = np.array([f.filter(np.full(6, x), dt)[0] for x in xs])
    # adaptivity: the moving signal is followed, ending within a small lag of the target
    assert out[-1] > 0.82 and out[-1] < 1.0


def _engaged(arm_follow):
    src = Hdf5Source(EPISODE)
    B = {s: np.eye(4) for s in SIDES}
    rt = TeleopRetargeter(B, rate_limit=False, engage_ramp_s=0.0,
                          arm_follow=arm_follow)
    rt.calibrate({s: src.pose[s] for s in SIDES})
    first = next(i for i in range(src.n) if all(src.at(i).active[s] for s in SIDES))
    anchor = {s: np.eye(4) for s in SIDES}
    rt.set_heading_matrix(np.eye(3))
    rt.engage(src.at(first), anchor, now=0.0)
    return src, rt, first, anchor


def test_fingers_only_pins_the_arm_but_not_the_fingers():
    """The whole point: wrist motion drives nothing, finger motion still drives the hand."""
    src, rt, first, anchor = _engaged(arm_follow=False)
    seen = []
    for i in range(first, min(first + 120, src.n)):
        t, c, _ = rt.step(src.at(i), now=i / 40.0)
        for s in SIDES:
            assert np.allclose(t[s], anchor[s], atol=1e-12), \
                f"{s}: arm moved in fingers-only mode"
        seen.append(np.concatenate([c[s] for s in SIDES]))
    seen = np.stack(seen)
    assert seen.std(axis=0).max() > 1e-3, "fingers never moved — the hand path is dead"


def test_arm_follow_default_does_move_the_arm():
    """Guard the guard: the pin must be the flag's doing, not a broken arm path."""
    src, rt, first, anchor = _engaged(arm_follow=True)
    moved = False
    for i in range(first, min(first + 120, src.n)):
        t, _, _ = rt.step(src.at(i), now=i / 40.0)
        if any(not np.allclose(t[s], anchor[s], atol=1e-6) for s in SIDES):
            moved = True
            break
    assert moved, "arm never moved with arm_follow=True"


def test_reset_seeds_state():
    f = _OneEuro(min_cutoff=1.5, beta=0.05, d_cutoff=1.0)
    f.filter(np.zeros(6, dtype=np.float32), 1 / 60.0)
    held = np.full(6, 0.7, dtype=np.float32)
    f.reset(held)
    # after reset, the next fresh sample eases FROM the held value (no snap through a stale
    # filter state) -- a large-dt first step passes the input through
    assert np.allclose(f.filter(held, 1 / 60.0), held, atol=1e-4)
