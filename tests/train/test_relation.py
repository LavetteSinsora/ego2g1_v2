"""Gates for the relational config (EgoRelationTrainConfig).

The invariants here are the ones whose violation would still train to a
plausible-looking loss: a mispaired object, a dropped param, a silently
non-invertible normalization, an encoder that starts by injecting noise into a
pretrained circuit.
"""

import dataclasses
import json
from unittest import mock

import flax.traverse_util as tu
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.model as _model

from ego2g1.core import rot6d, rotvec
from ego2g1.train import config as _config
from ego2g1.train import diagnostics as _diagnostics
from ego2g1.train import model as ego_model
from ego2g1.train import norm as _norm
from ego2g1.train import observation_patch as _obs_patch
from ego2g1.train import relation_transforms as _rt
from ego2g1.train import weight_loader as _wl


def _state(n_obj, n_hands=2, grasp=(0.0, 1.0)):
    """Synthetic observation.state: distinct relation values + VALID binary grasp
    flags. arange alone would put 54.0 in a grasp slot, which RelationPrompt
    rightly refuses."""
    rel = np.arange(9 * n_hands * n_obj, dtype=np.float64)
    return np.concatenate([rel, np.asarray(grasp[:n_hands], dtype=np.float64)])

_KW = dict(paligemma_variant="dummy", action_expert_variant="dummy", pi05=True,
           action_horizon=8, action_dim=16, max_token_len=32, dtype="float32")


def _cfg(**kw):
    return ego_model.Ego2G1Pi0Config(
        **_KW, action_dim_actual=14, n_objects=3, relation_dim=18,
        relation_hidden=32, grasp_head=True, state_dim=54, **kw
    )


def _obs_with_sentinels(cfg, batch=2, slots=(5, 9, 13)):
    obs = cfg.fake_obs(batch)
    tp = np.asarray(obs.tokenized_prompt).copy()
    for s in slots:
        tp[:, s] = cfg.relation_sentinel_id
    return dataclasses.replace(obs, tokenized_prompt=jnp.asarray(tp))


# ---------------------------------------------------------------- rotation math


def test_rotvec_round_trip_and_edges():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 3))
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    for theta in (0.0, 1e-9, 1e-4, 0.5, 1.9, np.pi - 1e-3):
        w = v * theta
        np.testing.assert_allclose(rotvec.mat_to_rotvec(rotvec.rotvec_to_mat(w)), w, atol=1e-9)
        R = rotvec.rotvec_to_mat(w)
        np.testing.assert_allclose(R @ np.swapaxes(R, -1, -2), np.broadcast_to(np.eye(3), R.shape), atol=1e-9)


def test_rotvec_matches_scipy():
    scipy_rot = pytest.importorskip("scipy.spatial.transform").Rotation
    R = scipy_rot.random(500, random_state=1).as_matrix()
    # away from theta=pi the log map is well conditioned; near pi it is not, for
    # anyone's implementation, so compare as ROTATIONS there
    got = rotvec.mat_to_rotvec(R)
    back = rotvec.rotvec_to_mat(got)
    err = np.linalg.norm(scipy_rot.from_matrix(back @ np.swapaxes(R, -1, -2)).as_rotvec(), axis=-1)
    assert err.max() < 1e-8


# --------------------------------------------------------------------- actions


def test_relative_action_round_trip():
    """Rebuilding absolute targets from the relative chunk must recover them."""
    rng = np.random.default_rng(0)
    scipy_rot = pytest.importorskip("scipy.spatial.transform").Rotation
    h = 8

    def vec9(t, R):
        return np.concatenate([t, R[:, 0], R[:, 1]])

    ref, tgt = [], []
    for hand in range(2):
        ref.append(vec9(rng.normal(size=3), scipy_rot.random(random_state=hand).as_matrix()))
    ref = np.concatenate(ref)
    rows = []
    for k in range(h):
        row = [vec9(rng.normal(size=3), scipy_rot.random(random_state=100 + 2 * k + j).as_matrix())
               for j in range(2)]
        rows.append(np.concatenate(row + [np.array([k % 2, (k + 1) % 2], float)]))
    tgt = np.stack(rows)

    out = _rt.RelativeEEFRotvecActions()(
        {"action": tgt, "observation/action_reference_tcp": ref}
    )["actions"]
    assert out.shape == (h, 14)

    for hand in range(2):
        t_cur = rot6d.vec9_to_se3(ref[hand * 9:(hand + 1) * 9])
        for k in range(h):
            d = np.eye(4)
            d[:3, 3] = out[k, hand * 6:hand * 6 + 3]
            d[:3, :3] = rotvec.rotvec_to_mat(out[k, hand * 6 + 3:hand * 6 + 6])
            np.testing.assert_allclose(t_cur @ d,
                                       rot6d.vec9_to_se3(tgt[k, hand * 9:(hand + 1) * 9]),
                                       atol=1e-6)
    # gripper: {0,1} -> {-1,+1}, at the TAIL
    np.testing.assert_allclose(out[:, 12:14], tgt[:, 18:20] * 2 - 1, atol=1e-6)


def test_relative_actions_pass_through_without_poses():
    data = {"state": np.zeros(2)}
    assert _rt.RelativeEEFRotvecActions()(data) is data


# ------------------------------------------------------------- prompt / relations


@pytest.mark.parametrize("shuffle", [False, True])
def test_prompt_and_relations_stay_paired(shuffle):
    """THE shuffle invariant: whatever order the names appear in, each name must
    carry its own object's geometry."""
    names = ("pen holder", "red cube", "yellow cube")
    state = _state(3)
    canon = _rt.RelationPrompt(object_prompt_names=names, task="t")({"observation/state": state})
    truth = {names[i]: canon["relations"][i] for i in range(3)}

    pb = _rt.RelationPrompt(object_prompt_names=names, shuffle=shuffle, task="t")
    for _ in range(200):
        r = pb({"observation/state": state})
        order = [o.replace(f" {_rt.RELATION_SENTINEL}", "")
                 for o in r["prompt"].split("Objects: ")[1].replace(" Action: ", "").split(", ")]
        assert sorted(order) == sorted(names)
        for slot, nm in enumerate(order):
            np.testing.assert_array_equal(r["relations"][slot], truth[nm])


def test_relation_interleave_is_hand_major():
    """Row k = object k in the LEFT tcp frame, then in the RIGHT tcp frame."""
    state = _state(3)
    r = _rt.RelationPrompt(object_prompt_names=("a", "b", "c"), task="t")(
        {"observation/state": state}
    )["relations"]
    for k in range(3):
        np.testing.assert_array_equal(
            r[k], np.concatenate([state[9 * k:9 * k + 9], state[27 + 9 * k:27 + 9 * k + 9]])
        )


def test_swap_relations_keeps_prompt_but_permutes_rows():
    names = ("pen holder", "red cube", "yellow cube")
    state = _state(3)
    base = _rt.RelationPrompt(object_prompt_names=names, task="t")({"observation/state": state})
    pb = _rt.RelationPrompt(object_prompt_names=names, swap_relations=True, task="t")
    for _ in range(50):
        r = pb({"observation/state": state})
        assert r["prompt"] == base["prompt"]
        assert not np.array_equal(r["relations"], base["relations"])


def test_no_relations_pool_drops_the_object_segment():
    r = _rt.RelationPrompt(object_prompt_names=("a", "b"), include_objects=False, task="t")(
        {"observation/state": _state(2)}
    )
    assert "Objects:" not in r["prompt"] and _rt.RELATION_SENTINEL not in r["prompt"]
    assert "Left hand:" in r["prompt"] and "Action:" in r["prompt"]


def test_grasp_words_follow_the_binary():
    names = ("a",)
    for g, word in ((0.0, "open"), (1.0, "closed")):
        state = np.concatenate([np.zeros(18), [g, g]])
        r = _rt.RelationPrompt(object_prompt_names=names, task="t")({"observation/state": state})
        assert f"Left hand: {word}" in r["prompt"] and f"Right hand: {word}" in r["prompt"]


def test_state_width_mismatch_raises():
    with pytest.raises(ValueError, match="observation/state"):
        _rt.RelationPrompt(object_prompt_names=("a", "b", "c"), task="t")(
            {"observation/state": np.zeros(10)}
        )


def test_sentinel_is_one_token_and_matches_the_config_default():
    """The model hard-codes the sentinel id so building a config never downloads a
    tokenizer; this is the test that keeps the two in sync."""
    pytest.importorskip("sentencepiece")
    assert _rt.sentinel_token_id() == ego_model.Ego2G1Pi0Config(**_KW).relation_sentinel_id


def test_tokenize_refuses_to_truncate():
    tok = _rt.RelationTokenizePrompt(max_token_len=8)
    with pytest.raises(ValueError, match="max_token_len"):
        tok({"prompt": "Task: " + "a very long prompt " * 20 + " Action: "})


# ------------------------------------------------------- action normalization


def _grid(h=8, d=14):
    rng = np.random.default_rng(0)
    q01 = -np.abs(rng.normal(size=(h, d))) - 0.5
    q99 = np.abs(rng.normal(size=(h, d))) + 0.5
    return q01, q99


def test_per_slot_quantile_inverse_is_exact():
    q01, q99 = _grid()
    fwd = _rt.PerSlotQuantizeActions(q01=q01, q99=q99, gripper_dims=(12, 13), clamp=None)
    inv = _rt.PerSlotQuantizeActionsInverse(q01=q01, q99=q99, gripper_dims=(12, 13))
    rng = np.random.default_rng(1)
    actions = rng.uniform(q01, q99)                       # inside the quantile band
    actions[..., 12:14] = np.sign(rng.normal(size=(8, 2)))
    got = inv(fwd({"actions": actions}))["actions"]
    np.testing.assert_allclose(got[..., :14], actions, atol=1e-5)


def test_per_slot_quantile_leaves_gripper_dims_alone():
    q01, q99 = _grid()
    actions = np.zeros((8, 14))
    actions[..., 12:14] = [1.0, -1.0]
    out = _rt.PerSlotQuantizeActions(q01=q01, q99=q99, gripper_dims=(12, 13))(
        {"actions": actions}
    )["actions"]
    np.testing.assert_allclose(out[..., 12:14], actions[..., 12:14], atol=0)
    # non-gripper dims DID move (0 is not the middle of a random band)
    assert not np.allclose(out[..., :12], 0.0)


def test_per_slot_quantile_inverse_ignores_pad_dims():
    q01, q99 = _grid()
    inv = _rt.PerSlotQuantizeActionsInverse(q01=q01, q99=q99, gripper_dims=(12, 13))
    padded = np.zeros((8, 16))
    padded[..., 14:] = 7.0
    out = inv({"actions": padded})["actions"]
    np.testing.assert_allclose(out[..., 14:], 7.0)


def test_relation_z_score_clips():
    mean = np.zeros(18)
    std = np.ones(18)
    rel = np.full((3, 18), 100.0)
    out = _rt.NormalizeRelations(mean=mean, std=std, clip=5.0)({"relations": rel})["relations"]
    assert out.max() == pytest.approx(5.0)


# ------------------------------------------------------------ observation patch


def test_observation_patch_fingerprints_hold():
    _obs_patch.verify_fingerprints()


def test_observation_patch_fingerprint_guard_raises_on_drift():
    with mock.patch.object(_obs_patch, "_STOCK_FINGERPRINTS",
                           {"Observation": "dead", "preprocess_observation": "beef"}):
        with pytest.raises(_obs_patch.StockSourceChangedError):
            _obs_patch.verify_fingerprints()


def test_relations_survive_preprocess_observation():
    """The regression test for the subtle half of the patch: stock
    preprocess_observation rebuilds Observation from an explicit field list and
    would drop `relations` before embed_prefix ever sees it."""
    _obs_patch.apply()
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    img = jnp.zeros((2, 224, 224, 3), jnp.float32)
    rel = jnp.arange(2 * 3 * 18, dtype=jnp.float32).reshape(2, 3, 18)
    obs = _model.Observation.from_dict({
        "image": {k: img for k in keys},
        "image_mask": {k: jnp.ones((2,), bool) for k in keys},
        "state": jnp.zeros((2, 16), jnp.float32),
        "tokenized_prompt": jnp.ones((2, 8), jnp.int32),
        "tokenized_prompt_mask": jnp.ones((2, 8), bool),
        "relations": rel,
    })
    assert obs.relations is not None
    for train in (False, True):
        post = _model.preprocess_observation(jax.random.key(0), obs, train=train)
        np.testing.assert_array_equal(np.asarray(post.relations), np.asarray(rel))
    # and a sample without relations still behaves exactly like stock
    obs2 = dataclasses.replace(obs, relations=None)
    assert _model.preprocess_observation(None, obs2, train=False).relations is None


def test_relations_is_a_pytree_leaf():
    _obs_patch.apply()
    obs = _model.Observation(images={}, image_masks={}, state=jnp.zeros((2, 4)),
                             relations=jnp.zeros((2, 3, 18)))
    assert jax.tree.map(lambda x: x[:1], obs).relations.shape == (1, 3, 18)


# ---------------------------------------------------------------- weight loader


def test_weight_loader_keeps_ego2g1_modules():
    """Stock drops any reference param not matching '.*lora.*'; ours must keep
    the relation encoder and grasp head."""
    ref = {
        "PaliGemma": {"llm": {"w": np.zeros((2, 2))}},
        "relation_encoder": {"gate": {"kernel": np.ones((18, 4))}},
        "grasp_head": {"query": {"kernel": np.ones((4, 1))}},
    }
    loaded = {"PaliGemma": {"llm": {"w": np.full((2, 2), 7.0)}}}
    merged = _wl._merge_params(loaded, ref, missing_regex=_wl.MISSING_REGEX)
    assert set(tu.flatten_dict(merged, sep="/")) == set(tu.flatten_dict(ref, sep="/"))
    assert merged["PaliGemma"]["llm"]["w"][0, 0] == 7.0
    _wl._assert_no_dropped_params(ref, merged)

    # stock really would have dropped them (this is what the class exists for)
    stock = _wl._merge_params(loaded, ref, missing_regex=".*lora.*")
    dropped = set(tu.flatten_dict(ref, sep="/")) - set(tu.flatten_dict(stock, sep="/"))
    assert dropped == {"relation_encoder/gate/kernel", "grasp_head/query/kernel"}


def test_weight_loader_guard_catches_an_unnamed_module():
    ref = {"brand_new_head": {"kernel": np.ones((3, 3))}}
    merged = _wl._merge_params({}, ref, missing_regex=_wl.MISSING_REGEX)
    with pytest.raises(ValueError, match="dropped by the weight loader"):
        _wl._assert_no_dropped_params(ref, merged)


# ------------------------------------------------------------------------ model


def test_inputs_spec_declares_relations():
    obs_spec, _ = _cfg().inputs_spec(batch_size=2)
    assert obs_spec.relations.shape == (2, 3, 18)


def test_zero_init_makes_step_zero_a_true_no_op():
    """Safeguard 3: at step 0 the prefix must be EXACTLY the plain-text prefix.

    Residual injection is what makes this possible -- overwriting the sentinel's
    embedding would put a zero vector there instead, which is just as
    out-of-distribution for the pretrained VLM as an oversized one."""
    cfg = _cfg()
    m = cfg.create(jax.random.key(0))
    obs = _obs_with_sentinels(cfg)
    with_rel, mask_a, ar_a = m.embed_prefix(obs)
    without, mask_b, ar_b = m.embed_prefix(dataclasses.replace(obs, relations=None))
    np.testing.assert_array_equal(np.asarray(with_rel), np.asarray(without))
    np.testing.assert_array_equal(np.asarray(mask_a), np.asarray(mask_b))
    np.testing.assert_array_equal(np.asarray(ar_a), np.asarray(ar_b))


def test_injection_touches_exactly_the_sentinel_slots():
    cfg = _cfg()
    m = cfg.create(jax.random.key(0))
    obs = _obs_with_sentinels(cfg, slots=(5, 9, 13))
    base, _, _ = m.embed_prefix(dataclasses.replace(obs, relations=None))
    m.relation_encoder.mlp.out.kernel.value = jnp.ones_like(m.relation_encoder.mlp.out.kernel.value) * 0.05
    got, _, _ = m.embed_prefix(obs)

    changed = np.flatnonzero(np.asarray(jnp.abs(got - base).sum(-1) > 1e-6)[0])
    text_start = base.shape[1] - obs.tokenized_prompt.shape[1]
    np.testing.assert_array_equal(changed - text_start, np.array([5, 9, 13]))
    # residual, not overwrite: the delta equals the encoder rows, evaluated at the
    # same live base norm embed_prefix uses
    np.testing.assert_allclose(
        np.asarray((got - base)[0, changed, :]),
        np.asarray(m.relation_encoder(obs.relations, m.sentinel_base_norm(base, obs))[0]),
        atol=1e-5,
    )


def test_prefix_length_unchanged_by_injection():
    cfg = _cfg()
    m = cfg.create(jax.random.key(0))
    obs = _obs_with_sentinels(cfg)
    a, _, _ = m.embed_prefix(obs)
    b, _, _ = m.embed_prefix(dataclasses.replace(obs, relations=None))
    assert a.shape == b.shape


def test_loss_dim_weights_reduce_to_mean_when_uniform():
    cfg = _cfg()
    m = cfg.create(jax.random.key(0))
    sq = jnp.asarray(np.random.default_rng(0).normal(size=(2, 8, 14)) ** 2)
    np.testing.assert_allclose(np.asarray(m._reduce_dims(sq)),
                               np.asarray(jnp.mean(sq, axis=-1)), rtol=1e-6)


def test_loss_dim_weights_upweight_the_gripper():
    w = tuple([0.5] * 12 + [4.0, 4.0])
    m = _cfg(loss_dim_weights=w).create(jax.random.key(0))
    sq = jnp.ones((1, 8, 14))
    got = float(m._reduce_dims(sq)[0, 0])
    assert got == pytest.approx(float(np.mean(w)))


def test_loss_dim_weights_length_is_validated():
    with pytest.raises(ValueError, match="loss_dim_weights"):
        _cfg(loss_dim_weights=(1.0, 2.0))


def test_compute_loss_with_aux_produces_grasp_terms_and_gradients():
    import flax.nnx as nnx

    cfg = _cfg()
    m = cfg.create(jax.random.key(0))
    obs, act = _obs_with_sentinels(cfg), cfg.fake_act(2)
    chunked, aux = m.compute_loss_with_aux(jax.random.key(1), obs, act, gripper_dims=(12, 13))
    assert chunked.shape == (2, 8)
    assert {"grasp_bce", "grasp_logits", "text_norm", "base_norm", "delta_norm",
            "alpha", "rotation_deg", "token_sep_deg", "delta_sep_deg"} <= set(aux)
    assert aux["grasp_logits"].shape == (2, 8, 2)
    assert float(aux["delta_norm"]) == 0.0             # zero-init
    assert float(aux["rotation_deg"]) == 0.0           # ... so nothing is rotated
    assert float(aux["token_sep_deg"]) == 0.0          # ... and the objects coincide
    assert float(aux["base_norm"]) > 0.0               # but the base is real
    assert float(aux["text_norm"]) > 0.0
    assert float(aux["alpha"]) == 1.0

    def loss_fn(mm):
        l, a = mm.compute_loss_with_aux(jax.random.key(1), obs, act, gripper_dims=(12, 13))
        return jnp.mean(l) + 0.2 * a["grasp_bce"]

    grads = nnx.grad(loss_fn)(m).to_pure_dict()
    for name in ("relation_encoder", "grasp_head"):
        n = sum(float(jnp.sum(jnp.square(x))) for x in jax.tree.leaves(grads[name]))
        assert n > 0.0, f"no gradient reached {name}"


def test_30_dim_config_has_no_relational_modules():
    """The legacy path's param tree must stay stock-shaped."""
    m = ego_model.Ego2G1Pi0Config(**_KW, action_dim_actual=14).create(jax.random.key(0))
    assert m.relation_encoder is None and m.grasp_head is None


def test_grasp_head_requires_objects():
    with pytest.raises(ValueError, match="grasp_head"):
        ego_model.Ego2G1Pi0Config(**_KW, grasp_head=True)


# --------------------------------------------------------------------- auc / bce


def test_grasp_auc_is_rank_based_and_handles_single_class():
    from ego2g1.train import relation as _relation

    logits = np.array([[-5.0, 5.0], [-4.0, 4.0]])
    targets = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    assert _relation.grasp_auc(logits, targets) == pytest.approx(1.0)
    assert _relation.grasp_auc(-logits, targets) == pytest.approx(0.0)
    assert np.isnan(_relation.grasp_auc(logits, np.ones_like(targets)))


def test_grasp_bce_is_minimized_by_correct_logits():
    from ego2g1.train import relation as _relation

    targets = jnp.asarray(np.random.default_rng(0).choice([-1.0, 1.0], size=(2, 8, 2)))
    good = float(_relation.grasp_bce_loss(targets * 20.0, targets))
    bad = float(_relation.grasp_bce_loss(-targets * 20.0, targets))
    assert good < 1e-6 < bad


# ---------------------------------------------------------------- stats artifact


def test_relation_stats_round_trip(tmp_path):
    stats = _norm.RelationNormStats(
        action_q01=np.zeros((8, 14)), action_q99=np.ones((8, 14)),
        relation_mean=np.zeros(18), relation_std=np.ones(18),
        gripper_dims=(12, 13), provenance={"model_space_variance": [0.1] * 14},
    )
    _norm.save_relation(tmp_path, stats)
    got = _norm.load_relation(tmp_path)
    np.testing.assert_array_equal(got.action_q99, stats.action_q99)
    assert got.gripper_dims == (12, 13)
    assert got.provenance["model_space_variance"] == [0.1] * 14


def test_relation_stats_sanity_flags_dead_dims():
    stats = _norm.RelationNormStats(
        action_q01=np.zeros((8, 14)), action_q99=np.ones((8, 14)),
        relation_mean=np.zeros(18), relation_std=np.ones(18),
        gripper_dims=(12, 13), provenance={},
    )
    stats.action_q99[:, 3] = 0.0          # dim 3 never moves
    problems = _norm.check_relation_stats_sanity(stats)
    assert any("dim 3" in p for p in problems)


def test_loss_dim_weights_split_scale_from_importance():
    """w_gripper must mean 'worth N EEF dims' AFTER variance normalization, so the
    gripper's share of the weighted MSE lands on the intended fraction."""
    from ego2g1.train import data_config as _dc

    var = [0.13] * 12 + [0.78, 0.76]
    stats = _norm.RelationNormStats(
        action_q01=np.zeros((8, 14)), action_q99=np.ones((8, 14)),
        relation_mean=np.zeros(18), relation_std=np.ones(18),
        gripper_dims=(12, 13), provenance={"model_space_variance": var},
    )
    for w_gripper, expected in ((1.0, 2 / 14), (3.0, 6 / 18)):
        w = _dc.loss_dim_weights(stats, 14, (12, 13), w_gripper)
        assert np.mean(w) == pytest.approx(1.0)
        share = sum(w[d] * (1 + var[d]) for d in (12, 13)) / sum(
            wi * (1 + v) for wi, v in zip(w, var, strict=True)
        )
        assert share == pytest.approx(expected, abs=1e-6)


def test_loss_dim_weights_refuses_stats_without_variance():
    from ego2g1.train import data_config as _dc

    stats = _norm.RelationNormStats(
        action_q01=np.zeros((8, 14)), action_q99=np.ones((8, 14)),
        relation_mean=np.zeros(18), relation_std=np.ones(18),
        gripper_dims=(12, 13), provenance={},
    )
    with pytest.raises(ValueError, match="model_space_variance"):
        _dc.loss_dim_weights(stats, 14, (12, 13), 3.0)


# ------------------------------------------------------------------- diagnostics


def _segment_obs(names, **kw):
    pytest.importorskip("sentencepiece")
    _obs_patch.apply()
    tok = _rt.RelationTokenizePrompt(max_token_len=96)
    state = _state(len(names))
    pb = _rt.RelationPrompt(object_prompt_names=names, task="put the cube away", **kw)
    rows = [tok(pb({"observation/state": state})) for _ in range(2)]
    ids = jnp.asarray(np.stack([r["tokenized_prompt"] for r in rows]))
    msk = jnp.asarray(np.stack([r["tokenized_prompt_mask"] for r in rows]))
    return _model.Observation(images={}, image_masks={}, state=jnp.zeros((2, 2)),
                              tokenized_prompt=ids, tokenized_prompt_mask=msk)


def test_segment_masks_partition_the_prompt():
    names = ("pen holder", "red cube", "yellow cube")
    obs = _segment_obs(names, shuffle=True)
    masks = _diagnostics.relation_segment_masks(
        obs, object_prompt_names=names, sentinel_id=7
    )
    coarse = [k for k in masks if not k.startswith("text/obj_vec/")]
    for i in range(2):
        total = sum((masks[k][i] > 0).astype(int) for k in coarse)
        assert total.max() <= 1, "segments must not overlap"
        assert (masks["text/obj_vectors"][i] > 0).sum() == len(names)
        per_obj = sum(int((masks[k][i] > 0).sum()) for k in masks if k.startswith("text/obj_vec/"))
        assert per_obj == len(names), "every sentinel must be attributed to exactly one object"
        assert (masks["text/hand_left"][i] > 0).sum() > 0
        assert (masks["text/hand_right"][i] > 0).sum() > 0


def test_segment_masks_handle_the_no_relations_pool():
    names = ("pen holder", "red cube")
    obs = _segment_obs(names, include_objects=False)
    masks = _diagnostics.relation_segment_masks(obs, object_prompt_names=names, sentinel_id=7)
    assert (masks["text/obj_vectors"] > 0).sum() == 0
    assert (masks["text/obj_names"] > 0).sum() == 0
    assert (masks["text/hand_left"] > 0).sum() > 0


def test_segment_masks_raise_on_a_foreign_prompt():
    _obs_patch.apply()
    obs = _model.Observation(images={}, image_masks={},
                             state=jnp.zeros((1, 2)),
                             tokenized_prompt=jnp.ones((1, 8), jnp.int32),
                             tokenized_prompt_mask=jnp.ones((1, 8), bool))
    with pytest.raises(ValueError, match="segment keyword"):
        _diagnostics.relation_segment_masks(obs, object_prompt_names=("a",), sentinel_id=7)


# ------------------------------------------------------------------------ config


def test_relation_config_derived_fields():
    c = _config.EgoRelationTrainConfig()
    assert c.n_objects == 3
    assert c.relation_dim == 18
    assert c.state_dim == 54
    assert c.gripper_dims == (12, 13)
    assert c.action_dim_actual == 7 * len(c.hands)


def test_relation_config_feature_flags_are_all_supported():
    from ego2g1.train import stamp as _stamp

    flags = _config.EgoRelationTrainConfig().feature_flags()
    required = {k for k, v in flags.items() if isinstance(v, dict) and v.get("required")}
    assert required <= _stamp.SUPPORTED_FEATURES, required - _stamp.SUPPORTED_FEATURES


def test_relation_config_validates_shapes():
    with pytest.raises(ValueError, match="action_dim_actual"):
        _config.EgoRelationTrainConfig(action_dim_actual=30)
    with pytest.raises(ValueError, match="prompt names"):
        _config.EgoRelationTrainConfig(object_prompt_names=("only one",))
    with pytest.raises(ValueError, match="w_aux"):
        _config.EgoRelationTrainConfig(grasp_head=False, w_aux=0.2)


def test_graphdef_is_stable_across_constructions():
    """Two constructions of the same config must produce EQUAL nnx graphdefs.

    init_train_state builds the model twice -- once under jax.eval_shape for the
    shape tree, once under jit -- and nnx requires the graphdefs to match. They
    are compared including STATIC fields, and `kernel_init` is one, so passing a
    freshly built initializer (nnx.initializers.lecun_normal() returns a NEW
    closure per call) makes two identical models compare unequal. The failure is
    a multi-thousand-line graphdef diff whose only delta is a function's id, so
    it is worth catching here rather than on a training box.
    """
    import flax.nnx as nnx

    cfg = _cfg()
    assert nnx.graphdef(cfg.create(jax.random.key(0))) == nnx.graphdef(
        cfg.create(jax.random.key(0))
    )


def _wake_encoder(enc, seed=1, scale=0.1):
    """Push the zero-initialized out-projection off zero so the delta is nonzero."""
    import flax.nnx as nnx  # noqa: F401  (kernel is an nnx.Param)

    shape = enc.mlp.out.kernel.value.shape
    enc.mlp.out.kernel.value = jax.random.normal(jax.random.key(seed), shape) * scale
    return enc


def test_injection_is_sized_relative_to_the_base():
    """||delta|| == alpha * ||base||, for whatever base it is handed.

    THE regression test for relation_v1. Safeguard 2 used to be an ABSOLUTE
    target measured offline from the checkpoint; it came back 1.63 against a
    sentinel embedding of 487.76, so the injection rotated the token 0.2 deg and
    the three objects reached attention at cosine 0.99999 -- indistinguishable.
    Training could not correct it either: growing a scale 284x is a multiplicative
    fix driven by additive SGD steps whose gradient is tiny precisely because the
    injection is tiny (the checkpoint's per-channel mean moved 0.6% in 10k steps).
    Sizing relative to the live base makes the whole class of error impossible.
    """
    enc = _wake_encoder(_cfg(relation_alpha=0.5).create(jax.random.key(0)).relation_encoder)
    rel = jax.random.normal(jax.random.key(2), (2, 3, 18))
    for base_norm in (1.0, 34.77, 487.76):
        got = np.linalg.norm(np.asarray(enc(rel, base_norm)), axis=-1)
        np.testing.assert_allclose(got, 0.5 * base_norm, rtol=1e-4)


def test_zero_init_survives_relative_scaling():
    """Safeguard 3: the delta is EXACTLY zero at step 0, for any base norm.

    The relative scaling multiplies by alpha*base/sqrt(width), so a large base
    must not turn the zero-init into a nonzero perturbation.
    """
    enc = _cfg().create(jax.random.key(0)).relation_encoder
    rel = jax.random.normal(jax.random.key(1), (2, 3, 18))
    np.testing.assert_array_equal(np.asarray(enc(rel, 487.76)), 0.0)


def test_embed_prefix_rotates_the_sentinel_by_arctan_alpha():
    """End-to-end: the injected token is rotated arctan(alpha) off the base.

    Rotation is the whole effect -- gemma.RMSNorm discards per-token magnitude
    and keeps only direction -- so this is the quantity the canary logs, and the
    one that read 0.2 deg for all of relation_v1.
    """
    alpha = 1.0
    cfg = _cfg(relation_alpha=alpha)
    model = cfg.create(jax.random.key(0))
    _wake_encoder(model.relation_encoder)
    obs = _obs_with_sentinels(cfg)
    obs = dataclasses.replace(obs, relations=jnp.asarray(
        jax.random.normal(jax.random.key(3), (obs.state.shape[0], 3, 18))))

    with_inj, _, _ = model.embed_prefix(obs)
    base, _, _ = model.embed_prefix(dataclasses.replace(obs, relations=None))

    slot = np.asarray(obs.tokenized_prompt == cfg.relation_sentinel_id)
    length = obs.tokenized_prompt.shape[1]
    b, t = np.nonzero(slot)
    a = np.asarray(base[:, -length:, :])[b, t]
    c = np.asarray(with_inj[:, -length:, :])[b, t]
    cos = np.sum(a * c, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(c, axis=-1))
    np.testing.assert_allclose(
        np.degrees(np.arccos(np.clip(cos, -1, 1))).mean(), np.degrees(np.arctan(alpha)), rtol=0.02
    )
    # and non-sentinel positions must be untouched
    keep = ~slot
    np.testing.assert_array_equal(
        np.asarray(with_inj[:, -length:, :])[keep], np.asarray(base[:, -length:, :])[keep]
    )


def test_legacy_config_hash_is_pinned():
    """Extracting the shared base must not change any existing checkpoint's
    norm-stats identity."""
    assert _config.Ego2G1TrainConfig().config_hash() == "89bad451eff962b4"
