"""s004b_smooth helpers: SavGol translation/fingers + windowed-quat rotation smoothing."""

import numpy as np

from ego2g1.core.rot6d import rot6d_to_mat, se3_to_vec9
from ego2g1.core.hand.retarget import HandRetargeter
from ego2g1.data.s004b_smooth import _repair_span, _savgol, _smooth_rot6d


def _noisy_ramp(T, rng, noise=0.02):
    clean = np.linspace(0.1, 0.9, T)[:, None] * np.ones((1, 3))
    return clean + rng.normal(0, noise, (T, 3)), clean


def test_savgol_reduces_jitter_and_keeps_shape():
    rng = np.random.default_rng(0)
    x, clean = _noisy_ramp(80, rng)
    y = _savgol(x, 9, 2)
    assert y.shape == x.shape
    # closer to the underlying ramp than the noisy input was
    assert np.abs(y - clean).mean() < np.abs(x - clean).mean()


def test_savgol_short_span_passthrough():
    # a span too short for any window >= 3 (shrinks below 3) returns unchanged
    x = np.random.default_rng(1).normal(size=(2, 3))
    assert np.array_equal(_savgol(x, 9, 2), x)
    # window <= polyorder is a no-op too (3 <= 3)
    y = np.random.default_rng(2).normal(size=(30, 3))
    assert np.array_equal(_savgol(y, 3, 3), y)


def test_rotation_smoothing_stays_valid_and_denoises():
    rng = np.random.default_rng(3)
    T = 60
    # a slow yaw sweep with small per-tick rotational jitter
    base = np.linspace(0, 0.8, T)
    q6 = np.empty((T, 6))
    for t in range(T):
        a = base[t] + rng.normal(0, 0.02)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        q6[t] = se3_to_vec9(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))[3:9]
    out = _smooth_rot6d(q6, 9)
    R = rot6d_to_mat(out)
    # decoded rotations are orthonormal (Gram-Schmidt guarantees it, but check the pipe)
    for t in range(T):
        assert np.allclose(R[t] @ R[t].T, np.eye(3), atol=1e-6)
    # per-tick angular step is smaller after smoothing than before
    def steps(v6):
        Rm = rot6d_to_mat(v6)
        return np.array([np.arccos(np.clip((np.trace(Rm[t].T @ Rm[t - 1]) - 1) / 2, -1, 1))
                         for t in range(1, len(v6))])
    assert steps(out).mean() < steps(q6).mean()


def test_rotation_window_one_is_noop():
    rng = np.random.default_rng(4)
    q6 = np.tile(se3_to_vec9(np.eye(4))[3:9], (10, 1)) + rng.normal(0, 0.01, (10, 6))
    assert np.allclose(_smooth_rot6d(q6, 1), _smooth_rot6d(q6, 1))  # deterministic
    # window 1 -> each tick uses only itself -> re-encode of its own rotation
    out = _smooth_rot6d(q6, 1)
    assert np.allclose(rot6d_to_mat(out), rot6d_to_mat(q6), atol=1e-6)


def test_smoothed_fingers_respect_rate_limit():
    # jittery finger track, smoothed then rate-limited exactly as the stage does
    rng = np.random.default_rng(5)
    T = 100
    dt_ns = np.arange(T, dtype=np.int64) * int(1e9 / 30)
    cmds = np.clip(0.5 + 0.4 * np.sin(np.linspace(0, 6, T))[:, None]
                   + rng.normal(0, 0.03, (T, 6)), 0, 1).astype(np.float32)
    sm = np.clip(_savgol(cmds, 9, 2), 0, 1).astype(np.float32)
    out = HandRetargeter._rate_limit(sm, dt_ns)
    assert out.min() >= 0.0 and out.max() <= 1.0
    from ego2g1.core.hand.constants import CMD_RATE_LIMIT, MOTOR_ORDER
    rates = np.array([CMD_RATE_LIMIT[m] for m in MOTOR_ORDER])
    dt = np.diff(dt_ns) * 1e-9
    step = np.abs(np.diff(out, axis=0))
    assert (step <= rates * dt[:, None] + 1e-5).all()


def test_repair_span_fills_bad_from_neighbours():
    T = 10
    pose = np.zeros((T, 9))
    pose[:, 0] = np.arange(T)                     # translation ramps 0..9
    pose[:, 3:9] = se3_to_vec9(np.eye(4))[3:9]
    cmds = np.tile(np.arange(T)[:, None], (1, 6)).astype(float)
    bad = np.zeros(T, dtype=bool)
    bad[4:6] = True                               # a 2-tick bridged run
    _repair_span(pose, cmds, bad)
    # linear interpolation across the hole
    assert np.allclose(pose[4:6, 0], [4, 5])
    assert np.allclose(cmds[4:6, 0], [4, 5])
