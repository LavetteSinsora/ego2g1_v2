"""Synthetic corner-case tests for the pure-math pipeline pieces (numpy only;
no mujoco/lerobot/real recordings needed).

Covers the failure modes a resampling + filter/split pipeline is most prone
to: off-by-one at run boundaries, quaternion hemisphere/convention slips,
degenerate inputs (empty runs, single samples, all-bad episodes), and the
loader's boundary indexing.

Run: .venv/bin/python -m pytest tests/test_corner_cases.py -q
"""

import numpy as np
import pytest

from ego2g1.core import frames
from ego2g1.core.episode import (build_control_grid, make_control_grid_ticks,
                              spike_mask, Episode, _resample_side)
from ego2g1.core.rot6d import mat_to_6d, rot6d_to_mat, se3_to_vec9, vec9_to_se3
from ego2g1.data.config import PipelineConfig
from ego2g1.core.boundary import BoundaryAwareIndices
from ego2g1.core.relative_actions import RelativeChunkActions
from ego2g1.data.config import load_config
from ego2g1.data.s004_filter_split import (_bridge_short_runs, _residual_bad,
                                 _split_runs, _sustained)


def _rand_rot(rng):
    q = frames.quat_normalize(rng.standard_normal(4))
    return frames.mat_from_quat(q)


# --------------------------------------------------------------- quaternions

def test_quat_mat_roundtrip_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = _rand_rot(rng)
        R2 = frames.mat_from_quat(frames.quat_from_mat(R))
        assert np.abs(R2 - R).max() < 1e-9


def test_quat_from_mat_near_180_deg_branches():
    # 180-degree rotations exercise every branch of Shepperd's method
    for axis in (np.eye(3)):
        R = 2 * np.outer(axis, axis) - np.eye(3)   # rot by pi about axis
        R2 = frames.mat_from_quat(frames.quat_from_mat(R))
        assert np.abs(R2 - R).max() < 1e-9


def test_slerp_sign_correction():
    qa = np.array([1.0, 0, 0, 0])
    qb_pos = frames.quat_normalize(np.array([0.9, 0.1, 0.2, 0.3]))
    qm_pos = frames.quat_slerp(qa, qb_pos, 0.5)
    qm_neg = frames.quat_slerp(qa, -qb_pos, 0.5)   # same rotation, flipped sign
    Ra = frames.mat_from_quat(qm_pos)
    Rb = frames.mat_from_quat(qm_neg)
    assert np.abs(Ra - Rb).max() < 1e-9
    # midpoint angle is half the endpoint angle
    half = frames.rot_geodesic_deg(np.eye(3), Ra)
    full = frames.rot_geodesic_deg(np.eye(3), frames.mat_from_quat(qb_pos))
    assert abs(2 * half - full) < 1e-6


def test_slerp_endpoints_and_parallel():
    qa = frames.quat_normalize(np.array([0.7, 0.1, -0.3, 0.2]))
    assert np.abs(frames.quat_slerp(qa, qa, 0.37) - qa).max() < 1e-12
    qb = frames.quat_normalize(qa + 1e-6)
    q = frames.quat_slerp(qa, qb, 0.5)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9


# --------------------------------------------------------------------- rot6d

def test_rot6d_roundtrip_and_gram_schmidt():
    rng = np.random.default_rng(1)
    for _ in range(100):
        R = _rand_rot(rng)
        R2 = rot6d_to_mat(mat_to_6d(R))
        assert np.abs(R2 - R).max() < 1e-9
    # a noisy (non-orthonormal) 6d still decodes to a proper rotation
    d6 = mat_to_6d(_rand_rot(rng)) + rng.normal(0, 0.1, 6)
    R = rot6d_to_mat(d6)
    assert np.abs(R @ R.T - np.eye(3)).max() < 1e-9
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_vec9_se3_roundtrip():
    rng = np.random.default_rng(2)
    T = np.eye(4)
    T[:3, :3] = _rand_rot(rng)
    T[:3, 3] = rng.standard_normal(3)
    assert np.abs(vec9_to_se3(se3_to_vec9(T)) - T).max() < 1e-9


# ------------------------------------------------------- s004 split / bridge

def test_split_runs_boundaries():
    # good run at the very start, middle, and end; min_len enforcement
    good = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=bool)
    starts, ends, real_end = _split_runs(good, min_len=2, T=len(good))
    assert starts.tolist() == [0, 4, 7]
    assert ends.tolist() == [3, 6, 10]
    assert real_end.tolist() == [False, False, True]   # only the run ending at T


def test_split_runs_short_runs_dropped_and_all_bad():
    good = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
    starts, ends, _ = _split_runs(good, min_len=2, T=len(good))
    assert starts.tolist() == [2] and ends.tolist() == [4]
    starts, ends, real_end = _split_runs(np.zeros(5, dtype=bool), 1, 5)
    assert len(starts) == 0 and len(ends) == 0 and len(real_end) == 0


def test_split_runs_trailing_bad_is_not_real_end():
    good = np.array([1, 1, 1, 1, 0], dtype=bool)
    _, ends, real_end = _split_runs(good, min_len=2, T=len(good))
    assert ends.tolist() == [4] and real_end.tolist() == [False]


def test_bridge_interior_only():
    #                 0  1  2  3  4  5  6  7  8  9
    bad = np.array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=bool)
    out, bridged = _bridge_short_runs(bad, max_ticks=2)
    # leading run (touches 0) and trailing run (touches T-1) must NOT bridge
    assert out.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert bridged.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_bridge_run_longer_than_max_not_bridged():
    bad = np.array([0, 1, 1, 1, 0], dtype=bool)
    out, bridged = _bridge_short_runs(bad, max_ticks=2)
    assert out.tolist() == bad.tolist() and not bridged.any()
    out, bridged = _bridge_short_runs(bad, max_ticks=3)
    assert not out.any() and bridged.sum() == 3


def test_bridge_disabled_and_all_bad():
    bad = np.array([0, 1, 0], dtype=bool)
    out, bridged = _bridge_short_runs(bad, max_ticks=0)
    assert out.tolist() == bad.tolist() and not bridged.any()
    allbad = np.ones(4, dtype=bool)
    out, bridged = _bridge_short_runs(allbad, max_ticks=10)
    assert out.all() and not bridged.any()   # touches both edges


def test_sustained_runs():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert _sustained(m, 3).tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0]
    assert _sustained(m, 1).tolist() == m.tolist()
    assert not _sustained(np.zeros(5, dtype=bool), 2).any()
    # run reaching the last tick
    tail = np.array([0, 0, 1, 1], dtype=bool)
    assert _sustained(tail, 2).tolist() == [0, 0, 1, 1]


def test_residual_bad_and_empty_finger_selection():
    res = np.zeros((6, 5))
    res[2, 1] = 0.030                             # 30 mm on the index finger
    res[4, 4] = 0.030                             # 30 mm on the pinky
    hand_valid = np.ones(6, dtype=bool)
    hand_valid[3] = False
    out = _residual_bad(res, [0, 1, 2], max_mm=25.0, hand_valid=hand_valid)
    assert out.tolist() == [0, 0, 1, 0, 0, 0]     # pinky (col 4) not gated
    # invalid ticks never flag, even over threshold
    res[3, 0] = 0.030
    assert not _residual_bad(res, [0], 25.0, hand_valid)[3]
    # hand_residual_fingers=() disables the filter instead of crashing
    assert not _residual_bad(res, [], 25.0, hand_valid).any()


# --------------------------------------------------- resampling corner cases

def _mk_episode(track_ns, pose_l, active_l, cam_ns):
    n = len(track_ns)
    pose7 = np.zeros((n, 7))
    pose7[:, :3] = pose_l
    pose7[:, 6] = 1.0                              # identity xyzw
    return Episode(path="", name="synthetic", attrs={},
                   track_ns=np.asarray(track_ns, dtype=np.int64),
                   lw7=pose7, rw7=pose7.copy(),
                   l_active=np.asarray(active_l, dtype=bool),
                   r_active=np.asarray(active_l, dtype=bool),
                   cam_ns=np.asarray(cam_ns, dtype=np.int64), jpegs=None)


def test_resample_exact_hits_and_out_of_range():
    track = np.array([0, 100, 200], dtype=np.int64) * 1_000_000   # ms -> ns
    pose7 = np.zeros((3, 7))
    pose7[:, 0] = [0.0, 1.0, 2.0]
    pose7[:, 6] = 1.0
    active = np.ones(3, dtype=bool)
    ticks = np.array([-50, 0, 50, 100, 200, 250], dtype=np.int64) * 1_000_000
    pos, quat, valid = _resample_side(track, pose7, active, ticks, max_gap_ms=150)
    assert valid.tolist() == [False, True, True, True, True, False]
    # exact hits and midpoint lerp
    assert pos[1, 0] == 0.0 and pos[3, 0] == 1.0 and pos[4, 0] == 2.0
    assert abs(pos[2, 0] - 0.5) < 1e-12
    # out-of-range ticks are filled from the nearest valid tick
    assert pos[0, 0] == pos[1, 0] and pos[5, 0] == pos[4, 0]
    for q in quat:
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9


def test_resample_gap_and_inactive_invalidate():
    track = np.array([0, 100, 400, 500], dtype=np.int64) * 1_000_000
    pose7 = np.zeros((4, 7)); pose7[:, 6] = 1.0
    active = np.array([True, True, True, False])
    ticks = np.array([50, 250, 450], dtype=np.int64) * 1_000_000
    _, _, valid = _resample_side(track, pose7, active, ticks, max_gap_ms=150)
    assert valid.tolist() == [True, False, False]  # gap 300ms; inactive bracket


def test_grid_ticks_and_cam_match_nearest():
    hz = 10.0                                     # 100 ms ticks
    track = (np.arange(11) * 100_000_000).astype(np.int64)      # 0..1s
    pose = np.zeros((11, 3)); pose[:, 1] = 1.0     # y-up plausible height
    cam = np.array([0, 130, 480, 950], dtype=np.int64) * 1_000_000
    ep = _mk_episode(track, pose, np.ones(11, bool), cam)
    grid = build_control_grid(ep, hz=hz, max_gap_ms=150)
    ticks = make_control_grid_ticks(ep, hz)
    assert ticks[0] == cam[0] and len(ticks) == len(grid.ticks_ns)
    # tick spacing exactly 100 ms
    assert set(np.diff(ticks).tolist()) == {100_000_000}
    # nearest camera frame, ties/ends included
    t_ms = ticks / 1e6
    expect = [int(np.argmin(np.abs(cam / 1e6 - t))) for t in t_ms]
    assert grid.cam_match.tolist() == expect


def test_spike_mask_flags_teleport_endpoints_only():
    track = (np.arange(5) * 10_000_000).astype(np.int64)         # 100 Hz
    pose7 = np.zeros((5, 7)); pose7[:, 6] = 1.0
    pose7[3:, 0] = 0.5                              # 50 cm jump between 2 and 3
    bad = spike_mask(track, pose7, np.ones(5, bool),
                     max_speed_m_s=2.0, min_step_m=0.03)
    assert bad.tolist() == [False, False, True, True, False]
    # below the displacement floor: fast but tiny steps stay unflagged
    pose7 = np.zeros((5, 7)); pose7[:, 6] = 1.0
    pose7[3:, 0] = 0.02                             # 2 cm < 3 cm floor
    bad = spike_mask(track, pose7, np.ones(5, bool), 2.0, 0.03)
    assert not bad.any()


# ------------------------------------------------------------ loader pieces

def test_boundary_indices_edges():
    H = 5
    # ep0: len 8, not real end -> anchors 0..2 ; ep1: len 8, real end -> 0..7
    idx = BoundaryAwareIndices([8, 8], [False, True], H,
                               allow_terminal_padding=True)
    assert idx.total_frames == 16
    assert list(idx.indices) == [0, 1, 2] + [8 + t for t in range(8)]
    # padding off: both episodes truncate
    idx = BoundaryAwareIndices([8, 8], [False, True], H, False)
    assert list(idx.indices) == [0, 1, 2, 8, 9, 10]
    # episode shorter than H contributes nothing (no negative counts)
    idx = BoundaryAwareIndices([3], [False], H, True)
    assert len(idx) == 0 and idx.total_frames == 3
    # anchor_bad frames are excluded as anchors
    idx = BoundaryAwareIndices([8, 8], [False, True], H, True,
                               anchor_bad=[[1], [0, 7]])
    assert list(idx.indices) == [0, 2] + [8 + t for t in range(1, 7)]


def test_relative_actions_identity_and_composition():
    rng = np.random.default_rng(4)
    H = 4
    T_seq = []
    for _ in range(H + 1):
        T = np.eye(4)
        T[:3, :3] = _rand_rot(rng)
        T[:3, 3] = rng.standard_normal(3)
        T_seq.append(T)
    pose = np.stack([se3_to_vec9(t) for t in T_seq]).astype(np.float32)
    hand = rng.random((H, 6)).astype(np.float32)
    sample = {"pose.left": pose, "hand.left": hand,
              "pose.right": pose.copy(), "hand.right": hand.copy(),
              "task": "t"}
    out = RelativeChunkActions(("left", "right"))(dict(sample))
    acts = out["actions"]
    assert acts.shape == (H, 30) and "pose.left" not in out and out["task"] == "t"
    # deltas compose back: anchor @ delta_k == pose_k
    anchor = T_seq[0]
    for k in range(H):
        T_tgt = anchor @ vec9_to_se3(acts[k, :9].astype(np.float64))
        assert np.abs(T_tgt - T_seq[k + 1]).max() < 1e-5
        assert np.abs(acts[k, 9:15] - hand[k]).max() == 0
    # static chunk -> identity deltas (zero translation, 6d of identity)
    still = np.repeat(pose[:1], H + 1, axis=0)
    out = RelativeChunkActions(("left", "right"))(
        {"pose.left": still, "hand.left": hand,
         "pose.right": still, "hand.right": hand})
    d = out["actions"][:, :9]
    assert np.abs(d[:, :3]).max() < 1e-6
    assert np.abs(d[:, 3:9] - np.array([1, 0, 0, 0, 1, 0])).max() < 1e-6


def test_relative_actions_shape_validation():
    H = 3
    pose = np.zeros((H + 1, 9), dtype=np.float32); pose[:, 3] = 1; pose[:, 7] = 1
    with pytest.raises(ValueError):
        RelativeChunkActions(("left",))({"pose.left": pose,
                                         "hand.left": np.zeros((H + 1, 6))})
    with pytest.raises(ValueError):
        RelativeChunkActions(("left",))({"pose.left": pose[:, :8],
                                         "hand.left": np.zeros((H, 6))})


# ------------------------------------------------------------- hand retarget

def test_rate_limit_and_invalid_fill():
    from ego2g1.core.hand.retarget import HandRetargeter
    # _rate_limit clamps per-tick steps to rate * dt
    cmds = np.array([[0.0] * 6, [1.0] * 6, [1.0] * 6], dtype=np.float32)
    ts = np.array([0, 33_333_333, 66_666_666], dtype=np.int64)
    out = HandRetargeter._rate_limit(cmds.copy(), ts)
    from ego2g1.core.hand.constants import CMD_RATE_LIMIT, MOTOR_ORDER
    rates = np.array([CMD_RATE_LIMIT[m] for m in MOTOR_ORDER])
    step = rates * (33_333_333 * 1e-9)
    assert np.allclose(out[1], np.minimum(1.0, step), atol=1e-6)
    assert np.all(out[2] >= out[1]) and np.all(out[2] <= out[1] + step + 1e-6)
    assert HandRetargeter._rate_limit(cmds, None) is cmds

    # leading-invalid fill: searchsorted remap takes the FIRST valid command
    valid = np.array([False, False, True, True, False])
    valid_idx = np.flatnonzero(valid)
    fill_from = valid_idx[np.clip(
        np.searchsorted(valid_idx, np.arange(5), side="right") - 1, 0, None)]
    assert fill_from.tolist() == [2, 2, 2, 3, 3]


def test_blocked_mask_requires_sustained_static_error():
    from ego2g1.core.hand.screen import BLOCK_FRAMES, ERR_HARD, blocked_mask
    T = BLOCK_FRAMES + 4
    cmds = np.full((T, 6), 0.5, dtype=np.float32)      # static command
    err = np.zeros((T, 6), dtype=np.float32)
    err[2:, 0] = ERR_HARD + 0.01
    out = blocked_mask(err, cmds)
    assert not out[: 2 + BLOCK_FRAMES - 1].any()
    assert out[2 + BLOCK_FRAMES - 1:].all()
    # same error while the command is moving: chase lag, never "blocked"
    moving = np.cumsum(np.full((T, 6), 0.01, dtype=np.float32), axis=0)
    assert not blocked_mask(err, moving).any()


# ------------------------------------------------------------------- config

def test_config_hash_sensitivity():
    a = PipelineConfig()
    assert a.stage_hash("s001") != PipelineConfig(control_hz=15).stage_hash("s001")
    # downstream stages inherit upstream fields through the dep closure
    assert a.stage_hash("s005") != PipelineConfig(control_hz=15).stage_hash("s005")
    # path-only fields never touch any stage hash
    assert a.config_hash == PipelineConfig(work_dir="/elsewhere").config_hash
    # s005-only fields don't invalidate s004
    assert a.stage_hash("s004") == PipelineConfig(task_prompt="x").stage_hash("s004")
    # smoothing fields flow into s005 (via the s004b_smooth dep) but must NOT invalidate
    # the s004 filters that run before smoothing
    assert a.stage_hash("s005") != PipelineConfig(smooth_window=7).stage_hash("s005")
    assert a.stage_hash("s004") == PipelineConfig(smooth_window=7).stage_hash("s004")


def test_min_subepisode_ticks_is_anchor_plus_chunk():
    cfg = PipelineConfig(action_horizon=50)
    assert cfg.min_subepisode_ticks == 51


def test_load_config_coerces_numeric_overrides():
    # int override of a float field hashes like the equivalent default
    cfg = load_config(overrides={"control_hz": 30})
    assert isinstance(cfg.control_hz, float)
    assert cfg.config_hash == PipelineConfig().config_hash
    # float-tuple fields coerce element-wise; int-tuple fields stay ints
    cfg = load_config(overrides={"revo2_mount_rpy_deg": [0, 0, 0],
                                 "image_size": [224, 224]})
    assert cfg.revo2_mount_rpy_deg == (0.0, 0.0, 0.0)
    assert all(isinstance(x, float) for x in cfg.revo2_mount_rpy_deg)
    assert cfg.stage_hash("b_calib") == PipelineConfig().stage_hash("b_calib")
    assert cfg.image_size == (224, 224)
    assert all(isinstance(x, int) for x in cfg.image_size)


def test_s005_rejects_unimplemented_state_content():
    from ego2g1.data import s005_write_lerobot
    with pytest.raises(SystemExit, match="state_content"):
        s005_write_lerobot.run_global(PipelineConfig(state_content="eef"), [])
