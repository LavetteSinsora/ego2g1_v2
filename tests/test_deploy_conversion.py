"""The EEF->joint conversion: the measured jitter fix, asserted.

docs/jitter_root_cause.md: Pico-grade noise on EEF targets passes through a
fully-converged IK into 26+ rad/s² joint zig-zag; the fix (OneEuroSE3 before,
posture-tracks-last @ 0.05 inside, JointFilter after) cut worst-joint accel
RMS to 4-7. This test rebuilds that experiment synthetically and pins the
ratio: the smooth pipeline must beat a raw fully-converged IK by >3x on
worst-joint accel RMS while keeping EEF position error bounded.
"""

import numpy as np
import pytest

from ego2g1.core import layout, se3

FPS = 30
H = 90


@pytest.fixture(scope="module")
def kin_smooth():
    from ego2g1.deploy.kinematics import Kinematics
    return Kinematics(ik_iters=25, fps=FPS, posture_cost=0.05)


def _smooth_reference_trajectory(kin):
    """A slow, feasible dual-arm sweep: joint-space sinusoids around a nominal
    posture, FK'd to pelvis-frame flange poses per tick."""
    q0 = np.zeros(layout.ARM_DOF)
    q0[1], q0[8] = 0.15, -0.15         # shoulder roll clear of the torso
    q0[3], q0[10] = 0.3, 0.3           # elbows slightly bent
    t = np.arange(H + 1) / FPS
    Q = np.tile(q0, (H + 1, 1))
    Q[:, 0] += 0.25 * np.sin(2 * np.pi * 0.2 * t)        # L shoulder pitch
    Q[:, 3] += 0.20 * np.sin(2 * np.pi * 0.15 * t + 1)   # L elbow
    Q[:, 7] += 0.25 * np.sin(2 * np.pi * 0.2 * t + 2)    # R shoulder pitch
    Q[:, 12] += 0.15 * np.sin(2 * np.pi * 0.25 * t)      # R wrist pitch
    poses = [kin.flange_poses(Q[k]) for k in range(H + 1)]
    return Q, poses


def _add_pico_noise(poses, rng, rot_deg=1.5, pos_m=0.003):
    """White rotational (the killer — the Jacobian amplifies it into the
    wrist) + translational noise, per tick per hand."""
    noisy = []
    for p in poses:
        out = {}
        for h in layout.HANDS:
            T = p[h].copy()
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            ang = np.deg2rad(rot_deg) * rng.normal()
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
            T[:3, :3] = T[:3, :3] @ R
            T[:3, 3] = T[:3, 3] + rng.normal(scale=pos_m, size=3)
            out[h] = T
        noisy.append(out)
    return noisy


def _chunk_from_poses(poses):
    """Anchor-relative (H, 30) chunk: row k = anchor^-1 @ pose_{k+1}, hands 0."""
    chunk = np.zeros((H, layout.DIM))
    for h in layout.HANDS:
        inv0 = se3.se3_inv(poses[0][h])
        for k in range(H):
            chunk[k, layout.EEF[h]] = se3.se3_to_vec9(inv0 @ poses[k + 1][h])
    return chunk


def _accel_rms_worst(q, fps=FPS):
    a = np.diff(np.asarray(q), n=2, axis=0) * fps * fps
    return float(np.sqrt((a ** 2).mean(axis=0)).max())


def test_smooth_pipeline_beats_raw_ik_by_3x(kin_smooth):
    from ego2g1.deploy.actions import ARM, RelativeEEFChunks
    from ego2g1.deploy.kinematics import Kinematics

    rng = np.random.default_rng(0)
    Q, poses_true = _smooth_reference_trajectory(kin_smooth)
    poses_noisy = _add_pico_noise(poses_true, rng)
    # the anchor is the (noise-free) measured pose at the obs tick
    poses_noisy[0] = poses_true[0]
    chunk = _chunk_from_poses(poses_noisy)
    hand0 = {h: np.zeros(layout.HAND_DIM) for h in layout.HANDS}

    # --- raw baseline: fully-converged IK, nominal posture, no smoothing.
    # This is what s003 did, and what measured 26 rad/s².
    kin_raw = Kinematics(ik_iters=25, fps=FPS, posture_cost=1e-3,
                         filter_weights=())
    kin_raw.ground(Q[0])
    q_raw = []
    for k in range(H):
        targets = {h: se3.compose(poses_true[0][h], chunk[k, layout.EEF[h]])
                   for h in layout.HANDS}
        q_raw.append(kin_raw.solve(targets))
    rms_raw = _accel_rms_worst(q_raw)

    # --- the deploy pipeline: OneEuroSE3 -> posture-tracks-last -> JointFilter
    conv = RelativeEEFChunks(kin=kin_smooth, fps=FPS)
    out = conv.convert(chunk, Q[0], hand0)
    rms_smooth = _accel_rms_worst(out[:, ARM])

    assert rms_raw > 3.0, f"raw baseline suspiciously smooth ({rms_raw:.1f}) — noise too weak"
    ratio = rms_raw / max(rms_smooth, 1e-9)
    assert ratio > 3.0, (
        f"smoothing pipeline only improved accel RMS {ratio:.1f}x "
        f"(raw {rms_raw:.1f}, smooth {rms_smooth:.1f} rad/s²)")

    # --- EEF error stays bounded: FK the smoothed joints against the TRUE
    # (noise-free) targets. Budget from the measured table: <= ~2 cm mean.
    errs = []
    for k in range(H):
        fk = kin_smooth.flange_poses(out[k, ARM])
        for h in layout.HANDS:
            errs.append(np.linalg.norm(
                fk[h][:3, 3] - poses_true[k + 1][h][:3, 3]))
    errs = np.array(errs)
    assert errs.mean() < 0.02, f"mean EEF error {errs.mean()*1000:.1f} mm"
    assert errs.max() < 0.05, f"max EEF error {errs.max()*1000:.1f} mm"


def test_identity_chunk_stays_at_anchor(kin_smooth):
    from ego2g1.deploy.actions import ARM, RelativeEEFChunks

    conv = RelativeEEFChunks(kin=kin_smooth, fps=FPS)
    conv.reset()
    q0 = np.zeros(layout.ARM_DOF)
    q0[1], q0[8] = 0.15, -0.15
    chunk = np.zeros((10, layout.DIM))
    for h in layout.HANDS:
        chunk[:, layout.EEF[h]] = se3.se3_to_vec9(np.eye(4))
    out = conv.convert(chunk, q0, {h: np.zeros(6) for h in layout.HANDS})
    anchor = kin_smooth.flange_poses(q0)
    fk = kin_smooth.flange_poses(out[-1, ARM])
    for h in layout.HANDS:
        assert np.linalg.norm(fk[h][:3, 3] - anchor[h][:3, 3]) < 0.01


def test_joint_mode_passthrough_and_padding():
    from ego2g1.deploy.actions import ARM, HAND, JointChunks, ROBOT_DIM

    conv = JointChunks()
    hands = {"left": np.full(6, 0.25), "right": np.full(6, 0.5)}
    # (H, 14): hands padded with the held command
    a14 = np.random.default_rng(1).normal(size=(5, 14)) * 0.1
    out = conv.convert(a14, np.zeros(14), hands)
    assert out.shape == (5, ROBOT_DIM)
    np.testing.assert_allclose(out[:, ARM], a14)
    np.testing.assert_allclose(out[:, HAND["left"]], 0.25)
    np.testing.assert_allclose(out[:, HAND["right"]], 0.5)
    # (H, 26): hand dims clipped to [0, 1]
    a26 = np.zeros((3, ROBOT_DIM))
    a26[:, HAND["left"]] = 2.0
    a26[:, HAND["right"]] = -1.0
    out = conv.convert(a26, np.zeros(14), hands)
    np.testing.assert_allclose(out[:, HAND["left"]], 1.0)
    np.testing.assert_allclose(out[:, HAND["right"]], 0.0)
    # garbage refused
    with pytest.raises(ValueError):
        conv.convert(np.full((2, 14), np.nan), np.zeros(14), hands)
    with pytest.raises(ValueError):
        conv.convert(np.zeros((2, 17)), np.zeros(14), hands)


class _FakeClient:
    """Minimal PolicyClient stand-in for adapter tests."""

    def __init__(self, horizon=6, control_mode="relative_eef"):
        self.action_horizon = horizon
        self.fps = FPS
        self.control_mode = control_mode
        self.last_request = None

    def infer(self, image, state, prompt, *, prev_chunk=None, d=0, n_prefix=None):
        self.last_request = {"image": image, "state": np.asarray(state),
                             "prompt": prompt, "prev_chunk": prev_chunk,
                             "d": d, "n_prefix": n_prefix}
        chunk = np.zeros((self.action_horizon, layout.DIM), dtype=np.float32)
        for h in layout.HANDS:
            chunk[:, layout.EEF[h]] = se3.se3_to_vec9(np.eye(4))
        return {"actions": chunk}


def test_relative_eef_adapter_state_and_rtc_reanchor(kin_smooth):
    from ego2g1.deploy import actions as _actions
    from ego2g1.deploy.actions import RelativeEEFChunks
    from ego2g1.deploy.policy_adapter import RelativeEEFPolicyAdapter

    client = _FakeClient()
    adapter = RelativeEEFPolicyAdapter(
        client, "task", converter=RelativeEEFChunks(kin=kin_smooth, fps=FPS))
    q = np.zeros(layout.ARM_DOF)
    q[1], q[8] = 0.15, -0.15
    hands = {h: np.full(6, 0.3) for h in layout.HANDS}

    # prev plan = "hold exactly where we are": deltas vs the new anchor must be
    # identity vec9 rows.
    prev_rows = np.zeros((3, _actions.ROBOT_DIM))
    prev_rows[:, _actions.ARM] = q
    out = adapter.infer({"arm_q": q, "hand_cmds": hands, "image": None,
                         "prompt": "task", "enable_rtc": True,
                         "inference_delay": 2, "prev_action_chunk": prev_rows})
    assert out["actions"].shape == (client.action_horizon, _actions.ROBOT_DIM)

    req = client.last_request
    # the model state is the FK state, not raw joints
    expected_state = kin_smooth.state(q, hands)
    np.testing.assert_allclose(req["state"], expected_state, atol=1e-6)
    assert req["d"] == 2 and req["n_prefix"] == 3
    ident = se3.se3_to_vec9(np.eye(4))
    for i in range(3):
        for h in layout.HANDS:
            np.testing.assert_allclose(
                req["prev_chunk"][i, layout.EEF[h]], ident, atol=1e-6)
    # padding rows stay zero (n_prefix marks the real ones)
    assert np.all(req["prev_chunk"][3:] == 0)
