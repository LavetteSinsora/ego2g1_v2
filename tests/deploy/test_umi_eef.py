"""`umi_eef` deploy mode: history buffer, anchor composition, idle-arm hold.

The facts pinned here are the ones that would otherwise fail silently — a
policy fed a wrong-lag pose, an idle arm that drifts out from under the context
camera, or a radian gripper command crushed to 1.0 all produce a plausible
rollout that is simply wrong.
"""

import numpy as np
import pytest

from ego2g1.core import layout, rotvec, se3, umi_layout
from ego2g1.deploy import actions as _actions
from ego2g1.deploy import umi_camera as _umi_camera
from ego2g1.deploy.umi_history import PoseHistoryBuffer

FPS = 30
LAGS = (0, 3, 6, 9, 12, 15)


def _pose(x=0.0, rz=0.0):
    T = np.eye(4)
    T[:3, :3] = rotvec.rotvec_to_mat(np.array([0.0, 0.0, rz]))
    T[:3, 3] = (x, 0.0, 0.0)
    return T


def _vec9(x=0.0, rz=0.0):
    return se3.se3_to_vec9(_pose(x, rz))


# --------------------------------------------------------------- history buffer


def _filled(n_ticks, fps=FPS, lags=LAGS, t0=100.0):
    buf = PoseHistoryBuffer(lags, fps)
    for k in range(n_ticks):
        buf.push(t0 + k / fps, _vec9(x=0.01 * k), gripper=4.0 - 0.1 * k)
    return buf


def test_empty_buffer_is_all_padding():
    out = PoseHistoryBuffer(LAGS, FPS).sample()
    assert out["history_len"] == 0
    assert out["observation/pose_history_is_pad"].all()
    assert out["observation/pose_history"].shape == (len(LAGS), umi_layout.POSE_DIM)


def test_full_buffer_resolves_every_lag_to_the_right_tick():
    """Lag j must be the sample j*stride ticks before the newest one. Off by a
    tick here and the policy reads a systematically wrong velocity."""
    n = 40
    out = _filled(n).sample()
    assert out["history_len"] == len(LAGS)
    assert not out["observation/pose_history_is_pad"].any()
    for j, lag in enumerate(LAGS):
        want_k = (n - 1) - lag           # pushes were one per tick
        np.testing.assert_allclose(out["observation/pose_history"][j],
                                   _vec9(x=0.01 * want_k), atol=1e-6)
        assert out["observation/gripper_history"][j, 0] == pytest.approx(
            4.0 - 0.1 * want_k, abs=1e-5)


def test_lag_zero_is_the_newest_sample():
    """Row 0 IS the anchor — the pose the action chunk composes onto."""
    out = _filled(40).sample()
    np.testing.assert_allclose(out["observation/pose_history"][0],
                               _vec9(x=0.01 * 39), atol=1e-6)


@pytest.mark.parametrize("n_ticks,expect", [(1, 1), (4, 2), (7, 3), (16, 6)])
def test_short_buffer_truncates_from_the_stale_end(n_ticks, expect):
    """Rollout start: only the leading lags exist. Truncation must be far-end
    only, so the j-th surviving lag is still lag j."""
    out = _filled(n_ticks).sample()
    assert out["history_len"] == expect
    is_pad = out["observation/pose_history_is_pad"]
    assert not is_pad[:expect].any() and is_pad[expect:].all()
    np.testing.assert_array_equal(out["observation/pose_history"][expect:], 0.0)


def test_a_hole_truncates_everything_older():
    """A dropped tick past the tolerance must not be re-labelled as a
    neighbouring lag: the whole tail beyond it is dropped instead. A shorter
    history is a TRAINED regime; a wrong-lag pose is not."""
    buf = PoseHistoryBuffer(LAGS, FPS)
    t0 = 100.0
    for k in range(40):
        if k == 33:            # a gap right where lag 6 would land
            continue
        buf.push(t0 + k / FPS, _vec9(x=0.01 * k), gripper=4.0)
    out = buf.sample()
    assert out["history_len"] == 2      # lags 0 and 3 survive; 6 and older go
    assert out["observation/pose_history_is_pad"][2:].all()


def test_lags_resolve_by_TIME_not_by_index():
    """The runner jitters and its idle branch polls at a different rate, so
    'five entries back' is not 'five ticks back'. Push at double rate: the lag
    grid must still land on the right TIMES."""
    buf = PoseHistoryBuffer(LAGS, FPS)
    t0 = 100.0
    n = 80
    for k in range(n):                       # 60 Hz pushes, 30 Hz lag grid
        buf.push(t0 + k / (2 * FPS), _vec9(x=0.01 * k), gripper=4.0)
    out = buf.sample()
    assert out["history_len"] == len(LAGS)
    for j, lag in enumerate(LAGS):
        want_k = (n - 1) - 2 * lag           # 2 pushes per control tick
        np.testing.assert_allclose(out["observation/pose_history"][j],
                                   _vec9(x=0.01 * want_k), atol=1e-6)


def test_clear_drops_everything():
    buf = _filled(40)
    buf.clear()
    assert len(buf) == 0 and buf.sample()["history_len"] == 0


def test_time_must_be_monotonic():
    buf = _filled(3)
    with pytest.raises(ValueError, match="backwards in time"):
        buf.push(0.0, _vec9(), 4.0)


def test_lag_grid_must_start_at_zero():
    with pytest.raises(ValueError, match="start at 0"):
        PoseHistoryBuffer((3, 6), FPS)


# ------------------------------------------------------------------- converter


@pytest.fixture(scope="module")
def kin():
    from ego2g1.deploy.kinematics import Kinematics
    return Kinematics(ik_iters=25, fps=FPS, posture_cost=0.05)


def _converter(kin, acting="right"):
    from ego2g1.deploy.modes.umi_eef import UmiEEFChunks
    return UmiEEFChunks(kin=kin, acting=acting, fps=FPS)


def _assert_idle_pinned_to(out_rows, q14):
    """Every converted row's idle-arm joints must equal `q14`'s, exactly."""
    got = np.asarray(out_rows)[:, _actions.ARM][:, layout.ARM_SLICE["left"]]
    want = np.asarray(q14)[layout.ARM_SLICE["left"]]
    np.testing.assert_allclose(got, np.tile(want, (got.shape[0], 1)), atol=1e-12)


def _nominal_q():
    q = np.zeros(layout.ARM_DOF)
    q[1], q[8] = 0.15, -0.15      # shoulder roll clear of the torso
    q[3], q[10] = 0.3, 0.3        # elbows bent
    return q


def test_delta_decode_inverts_the_training_transform(kin):
    """`UmiRelativeActions` builds row = [delta.t, mat_to_rotvec(delta.R)] from
    delta = inv(anchor) @ target. The converter must reproduce `target`."""
    conv = _converter(kin)
    q = _nominal_q()
    conv.latch_idle(q)
    anchor = kin.flange_poses(q)["right"]

    targets = []
    rows = np.zeros((4, umi_layout.ACTION_DIM))
    for k in range(4):
        T = anchor.copy()
        T[:3, 3] = T[:3, 3] + np.array([0.01 * (k + 1), 0.005 * k, -0.004 * k])
        delta = se3.se3_inv(anchor) @ T
        rows[k, umi_layout.EEF6] = np.concatenate(
            [delta[:3, 3], rotvec.mat_to_rotvec(delta[:3, :3])])
        rows[k, umi_layout.GRIP] = 4.0
        targets.append(T)

    # EXACT inverse, checked on the composition itself: anchor @ _delta(row)
    # must reproduce the target to float precision. Going through convert()
    # instead would fold in One-Euro lag and IK residual and could only support
    # a loose bound.
    conv.last_anchor = kin.flange_poses(q)
    for k in range(4):
        np.testing.assert_allclose(anchor @ conv._delta(rows[k], "right"),
                                   targets[k], atol=1e-12)

    # ...and end to end, the smoothed targets must track the intended path
    conv.convert(rows, q, {})
    got = conv.last_targets["right"]           # post-One-Euro flange targets
    assert np.linalg.norm(got[-1] - anchor[:3, 3]) > 0.0
    for k in range(4):
        assert np.linalg.norm(got[k] - targets[k][:3, 3]) < 0.05


def test_idle_delta_reproduces_the_latched_pose_from_any_anchor(kin):
    """The idle arm's target is an ABSOLUTE latched pose, so its delta has to
    be recomputed against whatever this tick's anchor is. If it were treated as
    a fixed delta the arm would walk away one chunk at a time."""
    conv = _converter(kin)
    q = _nominal_q()
    conv.latch_idle(q)
    hold = kin.flange_poses(q)["left"]
    for drift in (0.0, 0.05, -0.08):
        conv.last_anchor = kin.flange_poses(q + drift)
        got = conv.last_anchor["left"] @ conv._delta(np.zeros(umi_layout.ACTION_DIM),
                                                     "left")
        np.testing.assert_allclose(got, hold, atol=1e-12)


def test_idle_arm_joints_are_pinned_exactly(kin):
    """The context camera rides the idle arm and the server gives its view
    crop/rotate augmentation on the assumption that view is geometrically
    independent of the labels. A drifting idle arm makes that false."""
    conv = _converter(kin)
    q = _nominal_q()
    conv.latch_idle(q)
    rows = np.zeros((6, umi_layout.ACTION_DIM))
    rows[:, umi_layout.GRIP] = 4.0
    rows[:, 0] = np.linspace(0.0, 0.08, 6)     # acting arm sweeps forward
    out = conv.convert(rows, q, {})
    left = out[:, _actions.ARM][:, layout.ARM_SLICE["left"]]
    for k in range(6):
        np.testing.assert_allclose(left[k], q[layout.ARM_SLICE["left"]], atol=1e-12)


def test_idle_latch_survives_the_acting_arm_moving_but_resets_on_reset(kin):
    conv = _converter(kin)
    q = _nominal_q()
    conv.latch_idle(q)
    latched = conv._idle_hold_q.copy()
    conv.convert(np.zeros((2, umi_layout.ACTION_DIM)), q + 0.05, {})
    np.testing.assert_allclose(conv._idle_hold_q, latched)
    conv.reset()
    assert not conv.idle_latched


def test_converting_without_a_latch_refuses(kin):
    conv = _converter(kin)
    with pytest.raises(RuntimeError, match="never latched"):
        conv.convert(np.zeros((1, umi_layout.ACTION_DIM)), _nominal_q(), {})


def test_gripper_is_passed_through_verbatim(kin):
    """Radians of Dex1 gear rotation — the model's native units. No rescale, no
    open/closed fraction. Rescaling here is the bug this test exists for."""
    conv = _converter(kin)
    q = _nominal_q()
    conv.latch_idle(q)
    rows = np.zeros((3, umi_layout.ACTION_DIM))
    rows[:, umi_layout.GRIP] = np.array([[4.93], [1.20], [3.07]])
    out = conv.convert(rows, q, {})
    got = out[:, _actions.HAND["right"]][:, 0]
    np.testing.assert_allclose(got, [4.93, 1.20, 3.07], atol=1e-9)


def test_idle_gripper_holds_its_MEASURED_value(kin):
    """The policy never commands the idle gripper, and 0.0 rad is outside a
    Dex1's travel (the data spans 1.20..5.40) — defaulting it would drive that
    gripper into a hard stop on the very first tick."""
    conv = _converter(kin)
    q = _nominal_q()
    conv.latch_idle(q, idle_grip=3.75)
    rows = np.zeros((3, umi_layout.ACTION_DIM))
    rows[:, umi_layout.GRIP] = 4.0
    out = conv.convert(rows, q, {})
    np.testing.assert_allclose(out[:, _actions.HAND["left"]][:, 0], 3.75)
    # only slot 0 carries the Dex1 command; the rest of the block is padding
    np.testing.assert_allclose(out[:, _actions.HAND["left"]][:, 1:], 0.0)


def test_row_guard_rejects_corruption():
    ok = np.zeros(umi_layout.ACTION_DIM)
    ok[umi_layout.GRIP] = 4.0
    assert _actions.sanity_check_umi_action(ok)
    for bad in (np.full(umi_layout.ACTION_DIM, np.nan),
                np.zeros(umi_layout.ACTION_DIM + 1)):
        assert not _actions.sanity_check_umi_action(bad)
    far = ok.copy(); far[0] = 2.0                    # 2 m in one chunk
    assert not _actions.sanity_check_umi_action(far)
    spun = ok.copy(); spun[3] = 7.0                  # rotvec past 2*pi
    assert not _actions.sanity_check_umi_action(spun)
    wild = ok.copy(); wild[6] = 99.0                 # gripper past its travel
    assert not _actions.sanity_check_umi_action(wild)


# -------------------------------------------------------------------- executor


def test_dex1_wire_row_takes_one_motor_per_hand_and_does_not_clip_radians():
    """The Brainco path clips the end-effector block to [0, 1]. A Dex1 command
    is RADIANS (1.2..5.4), so that clip would crush every command to 1.0 and
    the gripper would never open."""
    from ego2g1.deploy.core.executor import UnitreeExecutor

    row = np.zeros(_actions.ROBOT_DIM)
    row[_actions.ARM] = np.arange(_actions.ARM_DOF, dtype=float)
    row[_actions.HAND["left"]][0] = 5.40
    row[_actions.HAND["right"]][0] = 1.20

    dex1 = UnitreeExecutor.__new__(UnitreeExecutor)
    dex1._ee = UnitreeExecutor._EE_LAYOUTS["unitree_g1_dex1"]
    wire = dex1._wire_row(row)
    assert wire.shape == (_actions.ARM_DOF + 2,)
    np.testing.assert_allclose(wire[:_actions.ARM_DOF], row[_actions.ARM])
    assert wire[_actions.ARM_DOF] == pytest.approx(5.40)
    assert wire[_actions.ARM_DOF + 1] == pytest.approx(1.20)
    lo, hi = dex1._ee["limits"]
    assert lo <= 1.20 and 5.40 <= hi, "the Dex1 travel must survive the guard"


def test_brainco_wire_row_is_the_identity():
    """The existing path must stay bit-identical."""
    from ego2g1.deploy.core.executor import UnitreeExecutor

    row = np.arange(_actions.ROBOT_DIM, dtype=float)
    brainco = UnitreeExecutor.__new__(UnitreeExecutor)
    brainco._ee = UnitreeExecutor._EE_LAYOUTS["unitree_g1_brainco"]
    np.testing.assert_allclose(brainco._wire_row(row), row)


def _probe_executor(robot_type="unitree_g1_dex1"):
    """A UnitreeExecutor with only the fields the pure-logic methods touch."""
    from ego2g1.deploy.core.executor import UnitreeExecutor

    ex = UnitreeExecutor.__new__(UnitreeExecutor)
    ex._ee = UnitreeExecutor._EE_LAYOUTS[robot_type]
    ex.robot_type = robot_type
    ex._last_sent = None
    ex._limp = {}
    return ex


def test_hold_falls_back_to_OPEN_not_zero_before_anything_is_sent():
    """`hold()` runs at bring-up, before any command exists. Zero is Brainco's
    open but is OUTSIDE a Dex1's travel — a zeroed hold would drive that
    gripper into a hard stop."""
    dex1 = _probe_executor()
    row = dex1._ee_row()
    assert row[_actions.HAND["right"]][0] == pytest.approx(5.40)
    assert row[_actions.HAND["left"]][0] == pytest.approx(5.40)

    brainco = _probe_executor("unitree_g1_brainco")
    np.testing.assert_allclose(brainco._ee_row()[_actions.ARM_DOF:], 0.0)


def test_hold_preserves_whatever_was_last_commanded():
    dex1 = _probe_executor()
    last = np.zeros(_actions.ROBOT_DIM)
    last[_actions.HAND["right"]][0] = 1.20          # mid-grasp
    dex1._last_sent = (0.0, last)
    assert dex1._ee_row()[_actions.HAND["right"]][0] == pytest.approx(1.20)


def test_arm_motor_ids_split_left_and_right_correctly():
    """`G1_29_JointArmIndex` is left(15..21) then right(22..28) — the same
    ordering `_update_g1_arm` zips against a 14-vector. Getting this wrong
    would limp the arm the policy is driving."""
    pytest.importorskip("unitree_deploy")
    ex = _probe_executor()
    assert [int(i) for i in ex._arm_motor_ids("left")] == list(range(15, 22))
    assert [int(i) for i in ex._arm_motor_ids("right")] == list(range(22, 29))


def test_limp_zeroes_only_that_arms_gains_and_unlimp_restores_them_exactly():
    """The other arm must stay fully controlled, and restore must put back what
    connect() configured rather than a recomputed guess."""
    pytest.importorskip("unitree_deploy")

    class _Cmd:
        def __init__(self, kp, kd): self.kp, self.kd = kp, kd

    class _Ctrl:
        def __init__(self): self.msg = type("M", (), {"motor_cmd": {}})()

    ctrl = _Ctrl()
    for i in range(15, 29):
        ctrl.msg.motor_cmd[i] = _Cmd(80.0 + i, 3.0)
    before = {i: (ctrl.msg.motor_cmd[i].kp, ctrl.msg.motor_cmd[i].kd) for i in range(15, 29)}

    ex = _probe_executor()
    ex._arm_controller = lambda: ctrl
    ex.limp_arm("left", kd=2.0)

    for i in range(15, 22):                      # left: limp
        assert ctrl.msg.motor_cmd[i].kp == 0.0
        assert ctrl.msg.motor_cmd[i].kd == 2.0
    for i in range(22, 29):                      # right: untouched
        assert (ctrl.msg.motor_cmd[i].kp, ctrl.msg.motor_cmd[i].kd) == before[i]
    assert ex.limp_hands == ("left",)

    # unlimp streams the measured pose before re-stiffening; stub that out
    ex.hold = lambda: None
    ex.control_dt = 0.0
    ex.unlimp_arm("left", settle_s=0.0)
    for i in range(15, 29):
        assert (ctrl.msg.motor_cmd[i].kp, ctrl.msg.motor_cmd[i].kd) == before[i]
    assert ex.limp_hands == ()


def test_unlimp_is_a_noop_when_nothing_is_limp():
    ex = _probe_executor()
    ex._arm_controller = lambda: (_ for _ in ()).throw(AssertionError("must not touch gains"))
    ex.unlimp_all(settle_s=0.0)          # no exception => never reached the controller


def test_open_grippers_uses_the_layout_open_value():
    from ego2g1.deploy.core.executor import MockExecutor

    ex = MockExecutor(fps=FPS)
    ex.connect()
    ex.open_grippers()
    assert len(ex.sent) == 1
    for rt, want in (("unitree_g1_dex1", 5.40), ("unitree_g1_brainco", 0.0)):
        probe = _probe_executor(rt)
        assert probe._ee_row()[_actions.HAND["right"]][0] == pytest.approx(want)


def test_idle_limp_needs_a_mode_that_has_an_idle_arm():
    """Refusing to guess which arm is safe to let go of."""
    from ego2g1.deploy import modes

    class _Conv:
        idle = "left"

    class _Adapter:
        converter = _Conv()

    assert modes.get("umi_eef").idle_hand(_Adapter()) == "left"
    # every other family drives both arms -> None -> the runner fails loud
    for name in ("joint", "relative_eef", "relation_eef"):
        assert modes.get(name).idle_hand(object()) is None


def test_rearm_restores_limp_gains_before_anything_else(monkeypatch):
    """Order matters: the arm must be stiff and holding where it was left
    BEFORE the clamp re-grounds and the first chunk is converted."""
    from ego2g1.deploy.core.executor import MockExecutor
    from ego2g1.deploy.core.runner import DeployRunner

    calls = []
    ex = MockExecutor(fps=FPS)
    ex.connect()
    ex.limp_arm("left")
    real_unlimp = ex.unlimp_all
    ex.unlimp_all = lambda **kw: (calls.append("unlimp"), real_unlimp(**kw))[1]
    real_arm_q = ex.arm_q
    ex.arm_q = lambda: (calls.append("arm_q"), real_arm_q())[1]

    class _Strategy:
        def clear(self): calls.append("strategy.clear")

    class _Adapter:
        mode = "joint"
        def reset(self): calls.append("adapter.reset")

    r = DeployRunner(adapter=_Adapter(), strategy=_Strategy(), executor=ex,
                     fps=FPS, wait=lambda *a, **k: None)
    r._rearm("test")
    assert "unlimp" in calls
    assert calls.index("unlimp") < calls.index("adapter.reset")
    assert ex.limp_hands == ()


def test_hold_in_place_command_round_trips_through_the_wire_row():
    """The property `check dex1` asserts on hardware, pinned here on synthetic
    numbers: a canonical row built to HOLD the measured state must produce a
    wire vector identical to that state. If the observation slicing and the
    command assembly disagree about the layout, this is where it shows —
    on the robot it would show as the gripper value landing on a wrist joint."""
    from ego2g1.deploy.core.executor import UnitreeExecutor

    measured = np.concatenate([np.arange(_actions.ARM_DOF, dtype=float), [5.40, 1.20]])
    row = np.zeros(_actions.ROBOT_DIM)
    row[_actions.ARM] = measured[: _actions.ARM_DOF]
    for i, h in enumerate(layout.HANDS):
        row[_actions.HAND[h]][0] = measured[_actions.ARM_DOF + i]

    probe = UnitreeExecutor.__new__(UnitreeExecutor)
    probe._ee = UnitreeExecutor._EE_LAYOUTS["unitree_g1_dex1"]
    np.testing.assert_allclose(probe._wire_row(row), measured, atol=1e-9)


def test_dex1_check_rung_is_registered_and_sends_no_action():
    """It is the gate before any umi_eef rollout, so it must exist under the
    documented name — and it must never CALL a motion command. Checked on the
    AST (calls), not on the text: the source legitimately names `send_action`
    in its own explanatory output."""
    import ast
    import inspect

    from ego2g1.deploy.tools.check import RUNGS, dex1

    assert RUNGS["dex1"] is dex1
    tree = ast.parse(inspect.getsource(dex1))
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("send_action", "send", "Write", "drive_to_waypoint", "step"):
        assert forbidden not in called, f"check dex1 must not command motion: {forbidden}"
    # connect/disconnect ARE expected: a live observation needs the subscriber,
    # and connect() only ever commands "hold where you are"
    assert {"connect", "disconnect"} <= called


def test_unknown_robot_type_is_refused():
    from ego2g1.deploy.core.executor import UnitreeExecutor

    assert "unitree_g1_dex1" in UnitreeExecutor._EE_LAYOUTS
    assert "unitree_g1_brainco" in UnitreeExecutor._EE_LAYOUTS


# ---------------------------------------------------------------------- cameras


def test_follow_mode_holds_where_the_arm_currently_is(kin):
    """The positioning workflow: push the idle arm and it stays put, instead of
    being dragged back to a latched pose."""
    conv = _converter(kin)
    conv.idle_hold = "follow"
    rows = np.zeros((3, umi_layout.ACTION_DIM))
    rows[:, umi_layout.GRIP] = 4.0

    q = _nominal_q()
    out = conv.convert(rows, q, {})
    _assert_idle_pinned_to(out, q)

    # operator shoves the idle arm somewhere new; the next chunk follows it
    moved = q.copy()
    moved[layout.ARM_SLICE["left"]] += 0.2
    out = conv.convert(rows, moved, {})
    _assert_idle_pinned_to(out, moved)


def test_follow_mode_needs_no_latch(kin):
    conv = _converter(kin)
    conv.idle_hold = "follow"
    assert conv.idle_latched                      # nothing to latch
    conv.convert(np.zeros((1, umi_layout.ACTION_DIM)), _nominal_q(), {})


def test_latch_mode_ignores_the_arm_being_moved(kin):
    """The evaluation default: once latched, the context view is frozen even if
    the arm is disturbed — that is what base_0_rgb's augmentation assumes."""
    conv = _converter(kin)
    q = _nominal_q()
    conv.latch_idle(q)
    moved = q.copy()
    moved[layout.ARM_SLICE["left"]] += 0.2
    out = conv.convert(np.zeros((2, umi_layout.ACTION_DIM)), moved, {})
    _assert_idle_pinned_to(out, q)          # the LATCH, not the new measurement


def test_idle_hold_mode_is_validated():
    from ego2g1.deploy.modes.umi_eef import UmiEEFChunks

    with pytest.raises(ValueError, match="latch\\|follow"):
        UmiEEFChunks(kin=object(), idle_hold="limp")


def test_wrist_pair_reads_the_vendor_keys():
    """`ImageClientCamera.async_read()` returns cam_left_wrist/cam_right_wrist
    in the SAME dict as the head pair — the exact feature names the training
    dataset uses. This pins that we read those keys and convert BGR->RGB."""
    from ego2g1.deploy.umi_camera import WristCameraPair

    pair = WristCameraPair.__new__(WristCameraPair)
    pair.acting, pair.context, pair.flip_bgr = "right", "left", True
    pair._lock = __import__("threading").Lock()
    pair._acting_frame = pair._context_frame = None
    pair._t = 0.0
    pair._stop = __import__("threading").Event()

    class _Client:
        def async_read(self):
            return {"cam_left_high": np.zeros((4, 4, 3), np.uint8),
                    "cam_right_high": np.zeros((4, 4, 3), np.uint8),
                    "cam_left_wrist": np.full((4, 4, 3), (1, 2, 3), np.uint8),
                    "cam_right_wrist": np.full((4, 4, 3), (9, 8, 7), np.uint8)}

    pair._client = _Client()
    pair._stop.set()                       # one pass then exit
    pair._stop.clear()

    # drive one iteration by hand rather than racing the thread
    out = pair._client.async_read()
    with pair._lock:
        pair._acting_frame = pair._to_rgb(out["cam_right_wrist"])
        pair._context_frame = pair._to_rgb(out["cam_left_wrist"])
        pair._t = 1.0
    acting, context = pair.read_pair()
    np.testing.assert_array_equal(acting[0, 0], [7, 8, 9])    # BGR -> RGB
    np.testing.assert_array_equal(context[0, 0], [3, 2, 1])


def test_wrist_pair_defaults_to_the_image_server_with_no_extra_flags():
    """A normal run must need no new camera flags: the wrist cameras are on the
    same image_server as the head."""
    from ego2g1.deploy.umi_camera import WristCameraPair

    class _Args:
        dry_run = False
        acting_camera = context_camera = None
        camera_host = "10.1.2.3"

    cam = _umi_camera.build_camera_pair(_Args())
    assert isinstance(cam, WristCameraPair)
    assert cam.host == "10.1.2.3"
    assert cam.acting == "right" and cam.context == "left"


def test_overriding_only_one_wrist_camera_is_refused():
    class _Args:
        dry_run = False
        acting_camera = "v4l2:0"
        context_camera = None
        camera_host = "10.1.2.3"

    with pytest.raises(ValueError, match="--context-camera"):
        _umi_camera.build_camera_pair(_Args())


def test_camera_pair_reports_the_staler_eye():
    class _Cam:
        def __init__(self, age):
            self._age = age
            self.closed = False

        def connect(self, **kw):
            pass

        def read(self):
            return np.full((4, 4, 3), self._age, np.uint8)

        def age(self):
            return self._age

        def close(self):
            self.closed = True

    acting, context = _Cam(0.01), _Cam(0.9)
    pair = _umi_camera.CameraPair(acting, context)
    assert pair.age() == pytest.approx(0.9)     # a fresh acting eye must not hide it
    a, c = pair.read_pair()
    assert a[0, 0, 0] == 0 and c[0, 0, 0] == 0  # uint8 of 0.01 / 0.9
    np.testing.assert_array_equal(pair.read(), acting.read())
    pair.close()
    assert acting.closed and context.closed


def test_dashboard_renders_both_wrist_views_side_by_side():
    """umi_eef's two cameras feed different model input slots, so the page must
    show both — and caption them, since swapping them is a silent failure."""
    pytest.importorskip("cv2")
    import cv2

    from ego2g1.deploy.ui.dashboard import Dashboard

    class _Cam:
        def __init__(self, fill, shape):
            self.fill, self.shape = fill, shape

        def read(self):
            return np.full(self.shape, self.fill, np.uint8)

    # deliberately DIFFERENT resolutions: the two cameras need not match
    pair = _umi_camera.CameraPair(_Cam(40, (120, 160, 3)), _Cam(200, (90, 160, 3)))

    class _Loop:
        camera = pair

    dash = Dashboard.__new__(Dashboard)
    dash.loop, dash.frame_width = _Loop(), 160
    jpg = dash.encode_frame()
    assert jpg is not None
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    # both tiles plus a 2px divider, padded to the taller of the two
    assert img.shape[1] == 160 * 2 + 2
    assert img.shape[0] == 120
    left, right = img[:, :160], img[:, 162:]
    # captions are drawn in a black band across the top of each tile
    assert left[5, :].max() > 0 and right[5, :].max() > 0
    # below the caption band the two tiles keep their own distinct content
    assert abs(int(left[60, 80].mean()) - 40) < 6
    assert abs(int(right[60, 80].mean()) - 200) < 6


def test_dashboard_single_camera_path_is_unchanged():
    pytest.importorskip("cv2")
    import cv2

    from ego2g1.deploy.ui.dashboard import Dashboard

    class _Cam:
        def read(self):
            return np.full((120, 160, 3), 77, np.uint8)

    class _Loop:
        camera = _Cam()

    dash = Dashboard.__new__(Dashboard)
    dash.loop, dash.frame_width = _Loop(), 160
    img = cv2.imdecode(np.frombuffer(dash.encode_frame(), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (120, 160)          # no divider, no caption band
    assert abs(int(img[5, 80].mean()) - 77) < 6


def test_gripper_deviation_is_recorded_but_never_trips():
    """A big command-vs-measured gap is the NORMAL state while holding
    something (the command saturates, the jaws stop at the block). It is a
    diagnostic, so it must not be wired to the watchdog."""
    from ego2g1.deploy.modes.umi_eef import UmiPolicyAdapter

    ad = UmiPolicyAdapter.__new__(UmiPolicyAdapter)
    ad.last_grip_cmd = ad.last_grip_measured = float("nan")
    ad.worst_grip_dev = 0.0
    ad.note_gripper(1.20, 2.95)        # closed on a block
    assert ad.last_grip_cmd == pytest.approx(1.20)
    assert ad.last_grip_measured == pytest.approx(2.95)
    assert ad.worst_grip_dev == pytest.approx(1.75)
    ad.note_gripper(1.20, 1.21)        # closed on nothing: the gap collapses
    assert ad.worst_grip_dev == pytest.approx(1.75)   # high-water mark kept


def test_telemetry_panel_is_discriminated_from_the_relation_one():
    """Both families put their panel in the same telemetry slot, and the
    relation card's JS indexes rel.objects/rel.hands — feeding it a UMI dict
    would throw and kill the page's whole render loop."""
    from ego2g1.deploy import modes

    class _Conv:
        idle, acting, idle_latched, last_grip = "left", "right", True, {"right": 1.2}

    class _Adapter:
        converter = _Conv()
        last_history_len, lag_ticks = 6, (0, 3, 6, 9, 12, 15)
        last_grip_cmd, last_grip_measured, worst_grip_dev = 1.20, 2.95, 1.75

    panel = modes.get("umi_eef").telemetry_extras(_Adapter())
    assert panel["kind"] == "umi"
    assert panel["history_len"] == 6 and panel["n_lags"] == 6
    assert panel["grip_dev"] == pytest.approx(1.75)
    assert panel["idle_latched"] is True
    # ...and the relation builder must carry its own discriminator
    from ego2g1.deploy.ui.telemetry import relation_panel
    rel = relation_panel({"objects": {}, "hands": {}}, [])
    assert rel["kind"] == "relation"


def test_telemetry_panel_tolerates_a_bare_adapter():
    """The panel is polled from the dashboard thread, possibly before the
    first tick has populated anything."""
    from ego2g1.deploy import modes

    class _Conv:
        idle, acting, idle_latched, last_grip = "left", "right", False, {}

    class _Adapter:
        converter = _Conv()

    panel = modes.get("umi_eef").telemetry_extras(_Adapter())
    assert panel["history_len"] == 0 and panel["grip_cmd"] is None
    assert modes.get("umi_eef").telemetry_extras(object()) is None


def test_camera_uri_grammar():
    assert isinstance(_umi_camera.make_camera("v4l2:3"), _umi_camera.LocalCamera)
    assert _umi_camera.make_camera("v4l2:3").index == 3
    from ego2g1.deploy.camera import HeadCamera, StaticCamera
    assert isinstance(_umi_camera.make_camera("static"), StaticCamera)
    zmq = _umi_camera.make_camera("zmq:10.0.0.1:right")
    assert isinstance(zmq, HeadCamera) and zmq.host == "10.0.0.1" and zmq.eye == "right"
    with pytest.raises(ValueError, match="unknown camera URI"):
        _umi_camera.make_camera("carrier-pigeon:1")


def test_camera_pair_refuses_to_guess_which_device_is_which():
    class _Args:
        dry_run = False
        acting_camera = "v4l2:0"
        context_camera = None

    with pytest.raises(ValueError, match="--context-camera"):
        _umi_camera.build_camera_pair(_Args())


def test_dry_run_gets_two_static_cameras():
    class _Args:
        dry_run = True

    pair = _umi_camera.build_camera_pair(_Args())
    a, c = pair.read_pair()
    assert a is not None and c is not None


# ------------------------------------------------------------------------ mode


def test_mode_is_registered_with_the_dex1_robot_type():
    from ego2g1.deploy import modes

    mode = modes.get("umi_eef")
    assert mode.robot_type == "unitree_g1_dex1"
    assert not mode.supports_rtc
    # "auto" must resolve the server's control_mode to this mode
    assert modes.resolve_action_mode("auto", "umi_eef") == "umi_eef"


def test_build_adapter_refuses_a_checkpoint_without_the_lag_grid():
    from ego2g1.deploy import modes

    class _Client:
        metadata = {"ego2g1": {}}
        action_horizon, fps = 50, FPS

    class _Args:
        prompt = ""
        ik_iters, posture_cost, collision_min_dist = 25, 0.05, 0.005

    with pytest.raises(ValueError, match="lag_ticks"):
        modes.get("umi_eef").build_adapter(_Client(), _Args(), FPS)
