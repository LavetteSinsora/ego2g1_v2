"""The two arm-target compositions, each checked against the pipeline's own formula.

s002_action_label/eef_label.py builds the label as  G(t) = pelvis^-1 . S . T_w(t) . B,
which factors as  C . R_w . B_R  (orientation) and  C . p_w + const  (position), with
C = pelvis^-1 . S a heading yaw.

  * ABSOLUTE mode (the default) reproduces G(t) by USING C: orientation is C . R_w . B_R
    and the engage-relative position telescopes to G(t)'s position. Tested with C set to
    the exact pelvis^-1 . S.
  * RELATIVE mode reproduces G(t) by CANCELLING S: the delta G(t0)^-1 G(t) drops S and
    pelvis entirely, so a wildly wrong S must leave the answer untouched.

Both are checked ELEMENTWISE on the rotation matrices, not by geodesic angle: near
identity that metric is arccos((tr-1)/2), whose derivative blows up, so a 1e-16 trace
error reads as ~1e-6 deg -- a property of the metric, not the transform.
"""

import numpy as np
import pytest

from tools.teleop._vendor.de.common import frames
from tools.teleop.retarget import SIDES, TeleopRetargeter
from tools.teleop.source import Hdf5Source

EPISODE = "data/put_bottle_in_box_ego/episode_1.hdf5"


def _rigid(yaw_deg: float, t) -> np.ndarray:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return frames.se3_from_rot(R, np.asarray(t, dtype=np.float64))


@pytest.fixture(scope="module")
def src():
    s = Hdf5Source(EPISODE)
    assert s.n > 100
    return s


def _worst(a, b):
    return float(np.abs(a - b).max())


def test_absolute_reproduces_the_label(src):
    """Absolute mode with the exact heading equals G(t), orientation and position."""
    # S = pelvis at identity here, so C = S; a non-trivial yaw+translation, and a real B.
    S = _rigid(37.0, [0.4, -0.2, 0.9])
    C = S[:3, :3]
    B = {"left": _rigid(61.0, [0, 0, 0]), "right": _rigid(-113.0, [0, 0, 0])}

    def label(side, i):
        return S @ src.at(i).wrist_se3(side) @ B[side]

    # reacquire_gap_s huge: this test steps sparsely (every 7th frame) to check the
    # steady-state composition, which must not be read as dropouts.
    rt = TeleopRetargeter(B, rate_limit=False, orientation="absolute", engage_ramp_s=0.0,
                          reacquire_gap_s=1e9)
    rt.calibrate({s: src.pose[s][:60] for s in SIDES})
    rt.set_heading_matrix(C)

    t0 = 5
    rt.engage(src.at(t0), {s: label(s, t0) for s in SIDES}, now=0.0)

    worst = 0.0
    for i in range(t0, src.n, 7):
        sample = src.at(i)
        if not all(sample.active[s] for s in SIDES):
            continue
        targets, _, _ = rt.step(sample, now=i / 40.0)
        for s in SIDES:
            worst = max(worst, _worst(targets[s], label(s, i)))

    assert worst < 1e-12, f"absolute target differs from G(t) by {worst:.3e}"


def test_relative_cancels_S(src):
    """Relative mode: a wildly wrong S must leave the answer untouched."""
    S = _rigid(137.0, [1.7, -0.9, 0.35])
    pelvis = _rigid(-42.0, [-0.4, 2.2, 0.81])
    pelvis_inv = frames.se3_inv(pelvis)
    B = {"left": _rigid(61.0, [0, 0, 0]), "right": _rigid(-113.0, [0, 0, 0])}

    def label(side, i):
        return pelvis_inv @ S @ src.at(i).wrist_se3(side) @ B[side]

    rt = TeleopRetargeter(B, rate_limit=False, orientation="relative", reacquire_gap_s=1e9)
    rt.calibrate({s: src.pose[s][:60] for s in SIDES})

    t0 = 5
    rt.engage(src.at(t0), {s: label(s, t0) for s in SIDES}, now=0.0)

    worst = 0.0
    for i in range(t0, src.n, 7):
        sample = src.at(i)
        if not all(sample.active[s] for s in SIDES):
            continue
        targets, _, _ = rt.step(sample, now=i / 40.0)
        for s in SIDES:
            worst = max(worst, _worst(targets[s], label(s, i)))

    assert worst < 1e-12, f"relative target drifted {worst:.3e} -- S did NOT cancel"


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_engage_is_a_no_op_at_the_anchor(src, mode):
    """At the instant of engagement the target must BE the anchor: the robot cannot jump
    when the operator takes control. In absolute mode this holds because the orientation
    ramp starts at u=0 (the anchor orientation) and position is identity at engage."""
    B = {"left": _rigid(61.0, [0, 0, 0]), "right": _rigid(-113.0, [0, 0, 0])}
    rt = TeleopRetargeter(B, rate_limit=False, orientation=mode, engage_ramp_s=0.5)
    rt.calibrate({s: src.pose[s][:60] for s in SIDES})
    if mode == "absolute":
        rt.set_heading_matrix(np.eye(3))

    anchor = {"left": _rigid(11.0, [0.2, 0.3, 0.9]),
              "right": _rigid(-7.0, [0.2, -0.3, 0.9])}
    t0 = 12
    rt.engage(src.at(t0), anchor, now=0.0)
    targets, _, _ = rt.step(src.at(t0), now=0.0)     # now == engage time -> u = 0

    for s in SIDES:
        assert np.allclose(targets[s], anchor[s], atol=1e-12), \
            f"{mode}/{s}: engaging moved the arm by " \
            f"{np.linalg.norm(targets[s][:3,3]-anchor[s][:3,3])*1e3:.3f} mm"


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_dropout_holds_pose_and_grip(src, mode):
    """A hand that loses tracking must freeze, not extrapolate and not open."""
    B = {s: np.eye(4) for s in SIDES}
    rt = TeleopRetargeter(B, rate_limit=False, orientation=mode, engage_ramp_s=0.0)
    rt.calibrate({s: src.pose[s][:60] for s in SIDES})
    if mode == "absolute":
        rt.set_heading_matrix(np.eye(3))

    anchor = {s: np.eye(4) for s in SIDES}
    rt.engage(src.at(0), anchor, now=0.0)
    targets, cmds, _ = rt.step(src.at(20), now=0.5)
    held_target = {s: targets[s].copy() for s in SIDES}
    held_cmd = {s: cmds[s].copy() for s in SIDES}

    dead = src.at(40)
    dead.active = {s: False for s in SIDES}
    targets2, cmds2, info = rt.step(dead, now=0.6)

    assert set(info["dropped"]) == set(SIDES)
    for s in SIDES:
        assert np.array_equal(targets2[s], held_target[s]), f"{mode}/{s}: arm moved on dropout"
        assert np.array_equal(cmds2[s], held_cmd[s]), f"{mode}/{s}: grip changed on dropout"


def _engage_and_settle(src, *, reacquire_gap_s=0.12, reacquire_ramp_s=0.3):
    """Engage absolute mode, track ~10 frames, return (rt, held-target-dict, last-time)."""
    B = {"left": _rigid(61.0, [0, 0, 0]), "right": _rigid(-113.0, [0, 0, 0])}
    rt = TeleopRetargeter(B, rate_limit=False, orientation="absolute", engage_ramp_s=0.0,
                          reacquire_ramp_s=reacquire_ramp_s, reacquire_gap_s=reacquire_gap_s)
    rt.calibrate({s: src.pose[s][:60] for s in SIDES})
    rt.set_heading_matrix(np.eye(3))
    anchor = {"left": _rigid(10.0, [0.2, 0.3, 0.9]),
              "right": _rigid(-10.0, [0.2, -0.3, 0.9])}
    rt.engage(src.at(5), anchor, now=0.0)
    targets = None
    for i in range(5, 15):
        targets, _, _ = rt.step(src.at(i), now=(i - 5) / 40.0)
    return rt, {s: targets[s].copy() for s in SIDES}, (14 - 5) / 40.0


def test_reacquire_does_not_jump(src):
    """A hand that drops out, while the operator keeps moving, and returns must pick up
    from where the arm HELD -- not teleport to the hand's new pose."""
    rt, held, t_last = _engage_and_settle(src)

    # dropout for several ticks; the arm must hold exactly where it was
    t = t_last
    for i in range(15, 25):
        t = (i - 5) / 40.0
        dead = src.at(i)
        dead.active = {s: False for s in SIDES}
        td, _, _ = rt.step(dead, now=t)
        for s in SIDES:
            assert np.array_equal(td[s], held[s]), f"{s}: arm moved during dropout"

    # hand returns at a WILDLY different wrist pose (src.at(60)); first target back must
    # equal the held pose to numerical precision -- no jump.
    reacq = src.at(60)
    t_re = t + 0.2
    tr, _, info = rt.step(reacq, now=t_re)
    assert set(info["reacquired"]) == set(SIDES)
    for s in SIDES:
        d = float(np.abs(tr[s] - held[s]).max())
        assert d < 1e-8, f"{s}: re-acquire jumped by {d:.3e} (pos+rot elementwise)"

    # then orientation eases toward the live pose over the ramp (position stays, wrist fixed)
    moved = False
    for k in range(1, 25):
        tk, _, _ = rt.step(src.at(60), now=t_re + k * 0.02)
        for s in SIDES:
            if not np.allclose(tk[s][:3, :3], held[s][:3, :3], atol=1e-4):
                moved = True
    assert moved, "orientation never ramped toward the live pose after re-acquire"


def test_reacquire_after_loop_skipped_ticks(src):
    """When BOTH hands are out the loop skips ticks entirely (nothing is stepped). The
    gap-based re-acquire must still fire on the first step back -- not a 'was active last
    tick' flag, which never saw the gap."""
    rt, held, t_last = _engage_and_settle(src)

    # no step() calls at all for a full second, then a step at a moved wrist
    tr, _, info = rt.step(src.at(60), now=t_last + 1.0)
    assert set(info["reacquired"]) == set(SIDES)
    for s in SIDES:
        assert np.allclose(tr[s], held[s], atol=1e-8), f"{s}: jumped after a skipped-tick gap"


def test_micro_blip_is_not_a_reacquire(src):
    """A gap shorter than reacquire_gap_s is continuous tracking, not a re-acquire -- it
    must NOT re-anchor (which would silently freeze the position reference)."""
    rt, _, t_last = _engage_and_settle(src)
    # one skipped frame: gap ~0.05 s < 0.12 s
    _, _, info = rt.step(src.at(16), now=t_last + 0.05)
    assert info["reacquired"] == [], "a micro-blip was mistaken for a re-acquire"


def test_absolute_orientation_is_engage_independent(src):
    """The point of absolute mode: the SAME hand orientation maps to the SAME flange
    orientation, no matter what pose you engaged in. Two different anchors, same hand
    frame -> identical target orientation (positions differ; orientations must not)."""
    B = {"left": _rigid(61.0, [0, 0, 0]), "right": _rigid(-113.0, [0, 0, 0])}

    def run(anchor):
        rt = TeleopRetargeter(B, rate_limit=False, orientation="absolute", engage_ramp_s=0.0,
                              reacquire_gap_s=1e9)
        rt.calibrate({s: src.pose[s][:60] for s in SIDES})
        rt.set_heading_matrix(np.eye(3))
        rt.engage(src.at(5), anchor, now=0.0)
        return rt.step(src.at(60), now=2.0)[0]

    a1 = {"left": _rigid(20.0, [0.1, 0.2, 0.8]), "right": _rigid(20.0, [0.1, -0.2, 0.8])}
    a2 = {"left": _rigid(-50.0, [0.4, 0.1, 1.1]), "right": _rigid(-50.0, [0.4, -0.1, 1.1])}
    t1, t2 = run(a1), run(a2)

    for s in SIDES:
        assert np.allclose(t1[s][:3, :3], t2[s][:3, :3], atol=1e-12), \
            f"{s}: absolute orientation depended on the engage pose"
        assert not np.allclose(t1[s][:3, 3], t2[s][:3, 3]), \
            f"{s}: positions should differ (they are engage-relative)"
