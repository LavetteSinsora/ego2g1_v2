"""UmiTrainConfig: transforms, per-lag normalization, prompt, config invariants.

The load-bearing facts these pin, each of which would otherwise train to a
plausible-looking loss:
  - the SHARED ANCHOR: the action chunk and the state history are both expressed
    in the frame of pose_history[0];
  - the DATASET OFF-BY-ONE: action[t] is the state at t+1, so the pose at tick t
    is action[t-1] and the delta_timestamps offsets differ by one between the
    pose and the gripper;
  - FAR-END truncation only, which is what makes "j-th token == lag j" exact and
    is forced by relying on RoPE for lag identity.
"""

import numpy as np
import pytest

# tests/train/conftest.py already skips this directory without the train group;
# repeated here so the module is safe to import on its own.
pytest.importorskip("openpi", reason="train group not installed — run: uv sync --group train")

from ego2g1.core import rot6d, rotvec  # noqa: E402
from ego2g1.train import config as _config  # noqa: E402
from ego2g1.train import norm as _norm  # noqa: E402
from ego2g1.train import umi_transforms as _ut  # noqa: E402


def _pose(t=(0.0, 0.0, 0.0), rv=(0.0, 0.0, 0.0)):
    """vec9 of the SE(3) pose with translation t and rotation exp(rv)."""
    T = np.eye(4)
    T[:3, :3] = rotvec.rotvec_to_mat(np.asarray(rv, dtype=np.float64))
    T[:3, 3] = t
    return rot6d.se3_to_vec9(T)


def _history(n_lags=3, grip=None):
    """n_lags poses walking backward in +x, plus grippers."""
    poses = np.stack([_pose((-0.01 * j, 0.0, 0.0)) for j in range(n_lags)])
    g = np.arange(n_lags, dtype=np.float64)[:, None] if grip is None else np.asarray(grip).reshape(-1, 1)
    return poses, g


# ------------------------------------------------------------------ lag grid


def test_lag_ticks_start_at_zero_and_step_by_stride():
    assert _ut.lag_ticks(5, 3) == (0, 3, 6, 9, 12, 15)
    assert _ut.lag_ticks(0, 3) == (0,)


@pytest.mark.parametrize("bad", [(-1, 3), (5, 0)])
def test_lag_ticks_rejects_degenerate_grids(bad):
    with pytest.raises(ValueError):
        _ut.lag_ticks(*bad)


def test_delta_timestamps_gather_only_action():
    """ONE key, both directions. `observation.state` must NOT be gathered: it is
    the same gripper numbers at a one-tick-different alignment, so a second
    gather would mean a second offset list and a second pad mask to obtain
    bit-identical values — two things to keep in sync instead of none."""
    lags = (0, 3, 6)
    dt = _ut.make_delta_timestamps(action_horizon=4, fps=30.0, lags=lags)
    assert set(dt) == {"action"}
    # backward half: lag l is action[t-1-l], because action[t] describes tick t+1
    for off, l in zip(dt["action"][: len(lags)], lags, strict=True):
        assert off == pytest.approx(-(1 + l) / 30.0)
    # forward half: slot k covers the target at t+1+k
    assert dt["action"][len(lags):] == [k / 30.0 for k in range(4)]


# ------------------------------------------------------------------ layout


def test_split_gathered_partitions_the_combined_action_gather():
    """Pose AND gripper come out of the same rows — that is what makes it
    structurally impossible for them to describe different ticks."""
    n_lags, h = 3, 4
    gathered = np.arange((n_lags + h) * 10, dtype=np.float64).reshape(n_lags + h, 10)
    is_pad = np.array([False] * (n_lags + h))
    out = _ut.UmiSplitGathered(n_lags=n_lags)({"action": gathered, "action_is_pad": is_pad})
    np.testing.assert_array_equal(out["observation/pose_history"], gathered[:n_lags, :9])
    np.testing.assert_array_equal(out["observation/gripper_history"], gathered[:n_lags, 9:])
    np.testing.assert_array_equal(out["observation/targets"], gathered[n_lags:])
    assert out["observation/pose_history_is_pad"].shape == (n_lags,)
    assert "action" not in out


def test_split_gathered_rejects_a_layout_mismatch():
    with pytest.raises(ValueError, match="n_lags"):
        _ut.UmiSplitGathered(n_lags=8)({"action": np.zeros((4, 10))})


# ------------------------------------------------------------------ actions


def test_relative_actions_are_anchored_at_history_row_zero():
    """The shared-anchor invariant, from the action side: slot k is
    inv(T_anchor) @ T_target_k with T_anchor == pose_history[0]."""
    poses, _ = _history(3)
    targets = np.concatenate(
        [np.stack([_pose((0.1 * k, 0.0, 0.0), (0.0, 0.0, 0.02 * k)) for k in range(5)]),
         np.full((5, 1), 3.0)], axis=-1)
    out = _ut.UmiRelativeActions()(
        {"observation/pose_history": poses, "observation/targets": targets})
    a = out["actions"]
    assert a.shape == (5, 7)
    t_anchor = rot6d.vec9_to_se3(poses[0])
    for k in range(5):
        want = np.linalg.inv(t_anchor) @ rot6d.vec9_to_se3(targets[k, :9])
        np.testing.assert_allclose(a[k, :3], want[:3, 3], atol=1e-6)
        np.testing.assert_allclose(a[k, 3:6], rotvec.mat_to_rotvec(want[:3, :3]), atol=1e-6)
    # gripper passes through raw — it is continuous and gets normalized later
    np.testing.assert_allclose(a[:, 6], 3.0)


def test_relative_actions_at_the_anchor_are_zero():
    poses, _ = _history(2)
    targets = np.concatenate([poses[:1], [[0.0]]], axis=-1)
    out = _ut.UmiRelativeActions()(
        {"observation/pose_history": poses, "observation/targets": targets})
    np.testing.assert_allclose(out["actions"][0, :6], 0.0, atol=1e-9)


def test_relative_actions_pass_through_without_the_pose_keys():
    data = {"state": np.zeros(1)}
    assert _ut.UmiRelativeActions()(data) is data


# ------------------------------------------------------------------ history


def test_history_row_zero_pose_is_structurally_zero():
    """The shared-anchor invariant, from the history side. This is also the fact
    that lets lag 0 be a token at all: its pose part carries nothing, its
    gripper dim carries the CURRENT aperture."""
    poses, grip = _history(4, grip=[5.0, 4.0, 3.0, 2.0])
    out = _ut.UmiStateHistory()(
        {"observation/pose_history": poses, "observation/gripper_history": grip})
    h = out["history"]
    assert h.shape == (4, 7)
    np.testing.assert_allclose(h[0, :6], 0.0, atol=1e-9)
    assert h[0, 6] == pytest.approx(5.0)
    # lag j walks back in -x by 0.01 per lag
    for j in range(4):
        assert h[j, 0] == pytest.approx(-0.01 * j, abs=1e-6)


def test_history_carries_current_gripper_as_state():
    poses, grip = _history(3, grip=[4.2, 1.0, 1.0])
    out = _ut.UmiStateHistory()(
        {"observation/pose_history": poses, "observation/gripper_history": grip})
    np.testing.assert_allclose(out["state"], [4.2])


@pytest.mark.parametrize("n_avail", [1, 2, 3])
def test_history_truncates_from_the_stale_end_on_padding(n_avail):
    """Near an episode start LeRobot clamps the gather and flags the invented
    rows. Those rows must be dropped from the END, so the j-th surviving token
    is still lag j."""
    n = 4
    poses, grip = _history(n, grip=[9.0, 8.0, 7.0, 6.0])
    is_pad = np.array([j >= n_avail for j in range(n)])
    out = _ut.UmiStateHistory()({
        "observation/pose_history": poses,
        "observation/gripper_history": grip,
        "observation/pose_history_is_pad": is_pad,
    })
    assert out["history_len"] == n_avail
    # surviving rows keep their lag identity; the rest are zeroed
    for j in range(n_avail):
        assert out["history"][j, 6] == pytest.approx(9.0 - j)
    np.testing.assert_array_equal(out["history"][n_avail:], 0.0)


def test_history_length_draw_is_capped_by_availability():
    """A draw of 4 cannot conjure rows the episode does not have."""
    poses, grip = _history(4)
    is_pad = np.array([False, False, True, True])
    probs = (0.0, 0.0, 0.0, 0.0, 1.0)   # always draw 4
    out = _ut.UmiStateHistory(length_probs=probs)({
        "observation/pose_history": poses,
        "observation/gripper_history": grip,
        "observation/pose_history_is_pad": is_pad,
    })
    assert out["history_len"] == 2


def test_history_length_draw_follows_the_distribution():
    poses, grip = _history(4)
    probs = (0.0, 1.0, 0.0, 0.0, 0.0)
    tf = _ut.UmiStateHistory(length_probs=probs)
    for _ in range(20):
        out = tf({"observation/pose_history": poses, "observation/gripper_history": grip})
        assert out["history_len"] == 1
        np.testing.assert_array_equal(out["history"][1:], 0.0)


def test_history_lag_zero_padding_is_refused():
    """Lag 0 is the anchor's own tick. If it is padding the sample has no real
    anchor at all — tick 0 of every episode — and must be excluded upstream via
    anchor_bad rather than trained on."""
    poses, grip = _history(3)
    with pytest.raises(ValueError, match="anchor_bad"):
        _ut.UmiStateHistory()({
            "observation/pose_history": poses,
            "observation/gripper_history": grip,
            "observation/pose_history_is_pad": np.array([True, True, True]),
        })


def test_history_permute_reorders_only_surviving_rows():
    poses, grip = _history(5, grip=[1.0, 2.0, 3.0, 4.0, 5.0])
    tf = _ut.UmiStateHistory(fixed_len=3, permute=True)
    seen = set()
    for _ in range(50):
        out = tf({"observation/pose_history": poses, "observation/gripper_history": grip})
        live = out["history"][:3, 6]
        assert sorted(live.tolist()) == [1.0, 2.0, 3.0]   # same rows, new order
        assert live.tolist() != [1.0, 2.0, 3.0]           # a no-op measures nothing
        np.testing.assert_array_equal(out["history"][3:], 0.0)
        seen.add(tuple(live.tolist()))
    assert len(seen) > 1


def test_history_pool_replaces_the_block_wholesale():
    poses, grip = _history(3)
    pool = np.full((1, 3, 7), 7.0)
    out = _ut.UmiStateHistory(pool=pool, fixed_len=3)(
        {"observation/pose_history": poses, "observation/gripper_history": grip})
    np.testing.assert_allclose(out["history"], 7.0)


def test_history_rejects_a_multi_dim_gripper():
    poses, _ = _history(3)
    with pytest.raises(ValueError, match="continuous gripper"):
        _ut.UmiStateHistory()({
            "observation/pose_history": poses,
            "observation/gripper_history": np.zeros((3, 2)),
        })


# ------------------------------------------------------------------ normalize


def test_normalize_history_is_per_lag():
    """Per-lag, not pooled: lag 5's displacement is far larger than lag 1's, so
    pooled stats would leave the near lags at a fraction of unit scale. This
    deliberately inverts the relational config's pooled-across-objects rule."""
    mean = np.stack([np.full(7, float(j)) for j in range(3)])
    std = np.stack([np.full(7, float(j + 1)) for j in range(3)])
    rel = np.stack([np.full(7, 2.0 * j) for j in range(3)])
    out = _ut.NormalizeHistory(mean=mean, std=std, clip=10.0)({"relations": rel})
    for j in range(3):
        np.testing.assert_allclose(out["relations"][j], (2.0 * j - j) / (j + 1), atol=1e-6)


def test_normalize_history_clips():
    mean = np.zeros((2, 7))
    std = np.ones((2, 7))
    out = _ut.NormalizeHistory(mean=mean, std=std, clip=2.0)({"relations": np.full((2, 7), 99.0)})
    np.testing.assert_allclose(out["relations"], 2.0)


def test_normalize_history_shape_mismatch_raises():
    with pytest.raises(ValueError, match="per-lag stats"):
        _ut.NormalizeHistory(mean=np.zeros((3, 7)), std=np.ones((3, 7)))(
            {"relations": np.zeros((2, 7))})


# ------------------------------------------------------------------ prompt


def test_prompt_task_comes_from_the_sample():
    pb = _ut.UmiPrompt(task="fallback")
    r = pb({"prompt": "Place the red block on top of the yellow block.", "history_len": 2})
    assert r["prompt"].startswith("Task: Place the red block on top of the yellow block. ")
    assert "fallback" not in r["prompt"]


def test_prompt_emits_one_sentinel_per_surviving_lag():
    pb = _ut.UmiPrompt(task="t")
    for n in range(0, 5):
        p = pb({"history_len": n})["prompt"]
        assert p.count(_ut.HISTORY_SENTINEL) == n
        assert (_ut.HISTORY_SEGMENT in p) == (n > 0)
        assert p.endswith("Action: ")
        assert "<<<control_mode>>> end effector <<<control_mode>>>" in p


def test_prompt_has_no_gripper_word():
    """The gripper is continuous, so any word would need a threshold — the same
    ambiguity that made a binary grasp head not worth having. It travels in dim
    6 of every history token instead."""
    p = _ut.UmiPrompt(task="t")({"history_len": 3})["prompt"]
    assert "closed" not in p and "open" not in p and "Gripper" not in p


def test_prompt_requires_a_task():
    with pytest.raises(ValueError, match="task string"):
        _ut.UmiPrompt()({"history_len": 1})


# ------------------------------------------------------------------ inputs


def _inputs_sample(n_lags=3):
    return {
        "observation/image_wrist": np.zeros((8, 8, 3), np.uint8),
        "observation/image_context": np.ones((8, 8, 3), np.uint8),
        "state": np.zeros(1, np.float32),
        "history": np.zeros((n_lags, 7), np.float32),
        "actions": np.zeros((4, 7), np.float32),
        "prompt": "x",
    }


def test_inputs_put_the_acting_camera_in_a_wrist_slot_and_context_in_base():
    """openpi gates crop/rotate augmentation on the substring "wrist": the
    ACTING camera must not receive it (its geometry is coupled to the
    anchor-relative labels), the static context view should."""
    from openpi.models import model as _m

    out = _ut.UmiInputs(model_type=_m.ModelType.PI0)(_inputs_sample())
    assert out["image_mask"]["base_0_rgb"] and out["image_mask"]["right_wrist_0_rgb"]
    assert not out["image_mask"]["left_wrist_0_rgb"]
    np.testing.assert_array_equal(out["image"]["base_0_rgb"], 1)      # context
    np.testing.assert_array_equal(out["image"]["right_wrist_0_rgb"], 0)  # acting
    # the history reaches the model through Observation's `relations` field
    assert out["relations"].shape == (3, 7)


def test_inputs_reject_a_non_wrist_acting_slot():
    from openpi.models import model as _m

    with pytest.raises(ValueError, match="wrist slot"):
        _ut.UmiInputs(model_type=_m.ModelType.PI0, acting_slot="base_0_rgb")


def test_inputs_reject_a_moving_context_camera():
    from openpi.models import model as _m

    with pytest.raises(NotImplementedError, match="context_is_static"):
        _ut.UmiInputs(model_type=_m.ModelType.PI0, context_is_static=False)


def test_outputs_trim_the_model_padding():
    out = _ut.UmiOutputs(action_dim=7)({"actions": np.zeros((4, 32))})
    assert out["actions"].shape == (4, 7)


# ------------------------------------------------------------------ config


def test_config_defaults_are_consistent():
    c = _config.UmiTrainConfig()
    assert c.n_lags == c.history_lags + 1
    assert len(c.history_len_probs) == c.n_lags + 1
    assert c.lag_ticks == _ut.lag_ticks(c.history_lags, c.history_stride)
    assert c.history_dim == _ut.HISTORY_DIM
    assert c.gripper_dims == (c.action_dim_actual - 1,)


def test_mode_a_is_the_same_config_with_a_different_length_distribution():
    """The two intended operating modes are ONE knob, not two code paths: mode A
    is history length 1 (lag 0 only = current gripper, no motion history)."""
    c = _config.UmiTrainConfig(history_len_probs=_config.UmiTrainConfig.MODE_A_LENGTH_PROBS)
    assert c.history_len_probs[1] == 1.0
    assert sum(c.history_len_probs) == pytest.approx(1.0)


@pytest.mark.parametrize("probs", [
    (0.5, 0.5),                                  # wrong length
    (0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5),         # does not sum to 1
    (0.0, -0.1, 0.2, 0.3, 0.2, 0.2, 0.2),        # negative
])
def test_config_rejects_a_malformed_length_distribution(probs):
    with pytest.raises(ValueError, match="history_len_probs"):
        _config.UmiTrainConfig(history_len_probs=probs)


def test_config_model_config_wires_the_injection_and_drops_the_grasp_head():
    c = _config.UmiTrainConfig()
    mc = c.model_config()
    assert mc.n_objects == c.n_lags          # "n injected tokens"
    assert mc.relation_dim == c.history_dim
    assert mc.grasp_head is False
    assert mc.inject_ordered is True
    assert mc.action_dim_actual == 7


def test_feature_flags_require_the_serving_critical_ones():
    flags = _config.UmiTrainConfig().feature_flags()
    for key in ("state_history", "continuous_gripper", "wrist_cameras",
                "relative_eef_rotvec_actions", "action_norm_scheme"):
        assert flags[key]["required"] is True, key
    assert flags["state_history"]["pose_source"] == "measured"


def test_feature_flags_are_all_declared_supported():
    from ego2g1.train import stamp as _stamp

    flags = _config.UmiTrainConfig().feature_flags()
    required = {k for k, v in flags.items() if isinstance(v, dict) and v.get("required")}
    assert required <= _stamp.SUPPORTED_FEATURES


def test_deploy_layout_agrees_with_the_training_config():
    """`ego2g1.core.umi_layout` is what the DEPLOY side slices chunks with. It
    must be derived from the same numbers the config trains on, not eyeballed —
    a disagreement here decodes the gripper as a rotation component and still
    produces a plausible-looking pose."""
    from ego2g1.core import umi_layout

    c = _config.UmiTrainConfig()
    assert umi_layout.ACTION_DIM == c.action_dim_actual
    assert umi_layout.HISTORY_DIM == c.history_dim
    assert umi_layout.POSE_DIM == _ut.POSE_DIM
    assert list(range(umi_layout.ACTION_DIM))[umi_layout.GRIP] == list(c.gripper_dims)
    assert umi_layout.ACTING_HAND == c.hand
    assert umi_layout.IDLE_HAND != c.hand
    assert umi_layout.default_lag_ticks() == c.lag_ticks


# ------------------------------------------------------- state_mode variants


def test_gripper_token_mode_gathers_only_the_anchor():
    """No pose history — but lag 0 still comes back, because it is the chunk's
    ANCHOR, not just a history token."""
    c = _config.UmiTrainConfig(state_mode="gripper_token")
    assert c.lag_ticks == (0,)
    assert c.n_lags == 1
    assert c.injects_tokens is False


def test_gripper_token_mode_builds_a_STOCK_param_tree():
    """n_objects=0 means ego2g1.train.model never constructs RelationEncoder,
    so the checkpoint has the same parameters as plain pi05."""
    mc = _config.UmiTrainConfig(state_mode="gripper_token").model_config()
    assert mc.n_objects == 0
    assert mc.inject_ordered is False
    assert mc.grasp_head is False
    assert mc.action_dim_actual == 7        # the action space is unchanged


def test_history_mode_is_unchanged_by_the_new_field():
    c = _config.UmiTrainConfig()
    assert c.state_mode == "history"
    assert c.lag_ticks == (0, 3, 6, 9, 12, 15) and c.n_lags == 6
    assert c.injects_tokens is True
    assert c.model_config().n_objects == 6


def test_the_two_modes_require_mutually_exclusive_serving_features():
    hist = _config.UmiTrainConfig().feature_flags()
    tok = _config.UmiTrainConfig(state_mode="gripper_token").feature_flags()
    assert hist["state_history"]["required"] is True and "gripper_token" not in hist
    assert tok["gripper_token"]["required"] is True and "state_history" not in tok
    assert tok["gripper_token"]["bins"] == 256
    # both must still be declared servable
    from ego2g1.train import stamp as _stamp
    for flags in (hist, tok):
        req = {k for k, v in flags.items() if isinstance(v, dict) and v.get("required")}
        assert req <= _stamp.SUPPORTED_FEATURES


def test_gripper_token_mode_skips_the_history_length_validation():
    """history_len_probs is sized for the history grid; it is meaningless when
    there is no history, so it must not be validated against n_lags=1."""
    c = _config.UmiTrainConfig(state_mode="gripper_token")
    assert len(c.history_len_probs) == 7 and c.n_lags == 1   # would fail in history mode


@pytest.mark.parametrize("bad", [{"state_mode": "elsewhere"},
                                 {"state_mode": "gripper_token", "gripper_bins": 1}])
def test_bad_state_mode_settings_are_refused(bad):
    with pytest.raises(ValueError):
        _config.UmiTrainConfig(**bad)


def test_gripper_digitization_matches_the_pi05_scheme():
    """Quantile-normalize to [-1,1], then pi05's own 256-bin digitize. Reusing
    that arithmetic is what keeps the digits ordinary number tokens."""
    q01, q99 = 1.2, 5.4
    assert _ut.digitize_gripper(q01, q01, q99, 256) == 0
    assert _ut.digitize_gripper(q99, q01, q99, 256) == 255
    mid = _ut.digitize_gripper((q01 + q99) / 2, q01, q99, 256)
    assert 126 <= mid <= 129
    # monotone, and clipped rather than wrapped outside the quantile range
    assert _ut.digitize_gripper(q01 - 10, q01, q99, 256) == 0
    assert _ut.digitize_gripper(q99 + 10, q01, q99, 256) == 255
    seq = [_ut.digitize_gripper(v, q01, q99, 256) for v in np.linspace(q01, q99, 50)]
    assert seq == sorted(seq)


def test_gripper_digitization_refuses_a_dead_column():
    with pytest.raises(ValueError, match="never moves"):
        _ut.digitize_gripper(1.0, 2.0, 2.0, 256)


def test_gripper_prompt_is_the_requested_shape():
    p = _ut.UmiGripperPrompt(q01=1.2, q99=5.4, bins=256, task="fallback")
    out = p({"prompt": "put the bottle in the box", "state": np.array([4.93])})
    assert out["prompt"] == (
        "Task: put the bottle in the box "
        "<<<control_mode>>> end effector <<<control_mode>>> "
        f"Gripper: {out['gripper_bin']} Action: ")
    assert "State history:" not in out["prompt"]
    assert _ut.HISTORY_SENTINEL not in out["prompt"]


def test_omit_gripper_drops_the_whole_segment():
    """val_no_gripper: asks whether the CHANNEL is used, not whether the value
    is read. A wrong bin is still a plausible number in a familiar slot."""
    p = _ut.UmiGripperPrompt(q01=1.2, q99=5.4, bins=256, task="t",
                             include_gripper=False)
    out = p({"prompt": "put the bottle in the box", "state": np.array([4.93])})
    assert "Gripper:" not in out["prompt"]
    assert out["prompt"] == (
        "Task: put the bottle in the box "
        "<<<control_mode>>> end effector <<<control_mode>>> Action: ")
    # the bin is still computed and reported, it just does not reach the prompt
    assert 0 <= out["gripper_bin"] < 256


def test_the_two_modes_no_proprioception_prompts_are_identical():
    """`val_no_gripper` and history mode's `val_nohist` must produce the SAME
    string, or the two experiments' no-proprioception numbers are not
    comparable."""
    task = "put the bottle in the box"
    tok = _ut.UmiGripperPrompt(q01=1.2, q99=5.4, bins=256,
                               include_gripper=False).build_prompt(task, 0)
    hist = _ut.UmiPrompt().build_prompt(task, 0)
    assert tok == hist


def test_gripper_prompt_shuffle_ablation_changes_only_the_bin():
    p = _ut.UmiGripperPrompt(q01=1.2, q99=5.4, bins=256, task="t", shuffle=True)
    seen = {p({"state": np.array([4.93])})["gripper_bin"] for _ in range(60)}
    assert len(seen) > 1, "the shuffled pool must actually randomize the bin"


def test_inputs_omit_relations_when_nothing_is_injected():
    """inputs_spec does not declare `relations` at n_objects=0, so emitting it
    would shape-mismatch on the first real batch."""
    from openpi.models import model as _m

    sample = _inputs_sample()
    out = _ut.UmiInputs(model_type=_m.ModelType.PI0, inject=False)(sample)
    assert "relations" not in out
    assert out["image_mask"]["base_0_rgb"] and out["image_mask"]["right_wrist_0_rgb"]
    out = _ut.UmiInputs(model_type=_m.ModelType.PI0, inject=True)(sample)
    assert out["relations"].shape == (3, 7)


@pytest.mark.parametrize("mode,pools", [
    ("history", [{"history_fixed_len": 6},
                 {"history_fixed_len": 6, "permute_history": True},
                 {"history_fixed_len": 0}]),
    ("gripper_token", [{}, {"shuffle_gripper": True}, {"omit_gripper": True}]),
])
def test_every_val_pool_kwarg_builds_a_data_config(mode, pools):
    """main_umi passes these straight through as **kwargs; a name that
    create_umi_data_config does not accept only explodes at val-pool
    construction time, minutes into a run."""
    from ego2g1.train import data_config as _dc

    c = _config.UmiTrainConfig(state_mode=mode)
    mc = c.model_config()
    for kwargs in pools:
        cfg = _dc.create_umi_data_config(c, mc, stats_dir="/nonexistent",
                                         skip_norm_stats=True, **kwargs)
        assert cfg.repo_id == c.repo_id
        # the injection field must be present iff this mode has an encoder
        names = [type(t).__name__ for t in cfg.data_transforms.inputs]
        assert ("UmiPrompt" in names) == c.injects_tokens
        assert ("UmiGripperPrompt" in names) != c.injects_tokens


def test_relation_config_is_untouched_by_inject_ordered():
    """The UMI config must not have changed the relational one's behaviour."""
    mc = _config.EgoRelationTrainConfig().model_config()
    assert mc.inject_ordered is False
    assert "inject_ordered" not in mc.feature_flags()


# ------------------------------------------------------------------ norm stats


def _stats(n_lags=3, h=4):
    q01 = np.tile(np.linspace(-1, -0.1, 7), (h, 1))
    q99 = np.tile(np.linspace(0.1, 1, 7), (h, 1))
    mean = np.zeros((n_lags, 7))
    std = np.ones((n_lags, 7))
    std[0, :6] = 0.0    # structural: lag 0's pose in its own frame
    return _norm.UmiNormStats(
        action_q01=q01, action_q99=q99, history_mean=mean, history_std=std,
        gripper_dims=(6,), provenance={},
    )


def test_stats_sanity_exempts_only_lag_zero_pose_dims():
    assert _norm.check_umi_stats_sanity(_stats()) == []


def test_stats_sanity_flags_a_dead_lag_dim():
    s = _stats()
    s.history_std[1, 2] = 0.0
    problems = _norm.check_umi_stats_sanity(s)
    assert any("lag 1 dim 2" in p for p in problems)


def test_stats_sanity_flags_a_dead_gripper_column():
    s = _stats()
    s.history_std[0, 6] = 0.0     # lag 0's GRIPPER is not exempt, only its pose
    problems = _norm.check_umi_stats_sanity(s)
    assert any("lag 0 dim 6" in p for p in problems)


def test_stats_sanity_flags_a_broken_shared_anchor():
    s = _stats()
    s.history_mean[0, 0] = 0.5
    assert _norm.lag_zero_pose_is_nonzero(s)
    assert any("come apart" in p for p in _norm.check_umi_stats_sanity(s))


def test_stats_roundtrip(tmp_path):
    s = _stats()
    _norm.save_umi(tmp_path, s)
    back = _norm.load_umi(tmp_path)
    np.testing.assert_allclose(back.action_q01, s.action_q01)
    np.testing.assert_allclose(back.history_std, s.history_std)
    assert back.gripper_dims == (6,)


def test_load_umi_missing_points_at_the_command():
    with pytest.raises(FileNotFoundError, match="--umi"):
        _norm.load_umi("/nonexistent")


# ------------------------------------------------------------------ probe


def test_diagnostics_history_segment_matches_the_transform():
    """The probe duplicates the segment string rather than importing it; this is
    the test that keeps the two in sync."""
    from ego2g1.train import diagnostics as _diagnostics

    assert _diagnostics._HISTORY_SEGMENT == _ut.HISTORY_SEGMENT
