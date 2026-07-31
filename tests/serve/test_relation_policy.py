"""ego2g1.serve.policy support for EgoRelationTrainConfig checkpoints.

Two things are pinned:
- config_from_stamp dispatches on the stamp's new "config_class" key, for both
  config families (round-trip), without breaking the pre-existing
  Ego2G1TrainConfig path.
- create_policy() builds a servable Ego2G1Policy end-to-end for a relational
  checkpoint: real (tiny) params on disk, a real stamp, a real
  relation_stats.npz, loaded through the actual resolution/dispatch code, then
  a real .infer() call.

The tiny model uses the same "dummy" paligemma/action_expert variant pattern
tests/train/test_serve.py and tests/train/test_relation.py already use to run
model construction on CPU without downloading real pi05_base weights.
EgoRelationTrainConfig.model_config() itself has no variant-override hook, so
the fixture patches it (mock.patch.object) to return the dummy-sized
Ego2G1Pi0Config while keeping every other train_config field (and therefore
every dispatch/resolution code path this test exists to exercise) real.
"""

import dataclasses
from unittest import mock

import numpy as np
import orbax.checkpoint as ocp
import pytest

pytest.importorskip("sentencepiece")

import flax.nnx as nnx
import jax

from ego2g1.serve import policy as _policy
from ego2g1.train import config as _config
from ego2g1.train import model as ego_model
from ego2g1.train import norm as _norm
from ego2g1.train import stamp as _stamp

# --------------------------------------------------------------------------
# stamp round-trip (config_from_stamp dispatch on "config_class")
# --------------------------------------------------------------------------


def test_config_from_stamp_roundtrip_relation(tmp_path):
    cfg = _config.EgoRelationTrainConfig(
        objects=("pen holder", "red cube"),
        object_prompt_names=("holder", "red"),
        val_source_episodes=("ds/episode_1", "ds/episode_2"),
    )
    _stamp.write_stamp(tmp_path, cfg, "cafebabe00000000")
    stamp = _stamp.read_stamp(tmp_path)
    assert stamp["config_class"] == "EgoRelationTrainConfig"

    rebuilt = _policy.config_from_stamp(stamp)
    assert isinstance(rebuilt, _config.EgoRelationTrainConfig)
    assert dataclasses.asdict(rebuilt) == dataclasses.asdict(cfg)
    assert rebuilt.config_hash() == cfg.config_hash()


def test_config_from_stamp_roundtrip_legacy_still_works(tmp_path):
    """The pre-existing Ego2G1TrainConfig path must be unaffected by the new
    config_class dispatch (byte-identical reconstruction)."""
    cfg = _config.Ego2G1TrainConfig(
        val_real_episodes=("t/e1", "t/e2"), expected_config_hash="cafebabe12345678",
        per_slot_floor_c=0.2,
    )
    _stamp.write_stamp(tmp_path, cfg, "cafebabe12345678")
    stamp = _stamp.read_stamp(tmp_path)
    assert stamp["config_class"] == "Ego2G1TrainConfig"

    rebuilt = _policy.config_from_stamp(stamp)
    assert isinstance(rebuilt, _config.Ego2G1TrainConfig)
    assert dataclasses.asdict(rebuilt) == dataclasses.asdict(cfg)
    assert rebuilt.config_hash() == cfg.config_hash()


def test_config_from_stamp_defaults_missing_config_class_to_legacy(tmp_path):
    """A checkpoint stamped before "config_class" existed carries no such key —
    it necessarily predates EgoRelationTrainConfig, so it must still resolve to
    Ego2G1TrainConfig."""
    cfg = _config.Ego2G1TrainConfig()
    _stamp.write_stamp(tmp_path, cfg, "cafebabe00000000")
    stamp = _stamp.read_stamp(tmp_path)
    del stamp["config_class"]  # simulate a pre-existing checkpoint

    rebuilt = _policy.config_from_stamp(stamp)
    assert isinstance(rebuilt, _config.Ego2G1TrainConfig)
    assert rebuilt == cfg


# --------------------------------------------------------------------------
# end-to-end: a real (tiny) relational checkpoint through create_policy()
# --------------------------------------------------------------------------

_DUMMY_MODEL_KW = dict(paligemma_variant="dummy", action_expert_variant="dummy", max_token_len=64)


def _dummy_model_config(train_config: _config.EgoRelationTrainConfig) -> ego_model.Ego2G1Pi0Config:
    """Same shape contract as EgoRelationTrainConfig.model_config() (n_objects,
    relation_dim, grasp_head, state_dim, action_dim_actual all real), but with
    the dummy/tiny gemma variants so model construction + a forward pass run
    fast on CPU."""
    return ego_model.Ego2G1Pi0Config(
        pi05=True,
        action_dim=train_config.action_dim,
        action_horizon=train_config.action_horizon,
        action_dim_actual=train_config.action_dim_actual,
        n_objects=train_config.n_objects,
        relation_dim=train_config.relation_dim,
        relation_hidden=32,
        grasp_head=train_config.grasp_head,
        state_dim=train_config.state_dim,
        **_DUMMY_MODEL_KW,
    )


def _write_fixture_checkpoint(tmp_path, train_config):
    """A minimal but REAL on-disk checkpoint: params/, ego2g1_stamp.json, and
    assets_ego2g1/relation_stats.npz, laid out exactly as train.py leaves them
    (mirrors _fake_run_dir in tests/train/test_corner_cases.py)."""
    run_dir = tmp_path / "checkpoints" / train_config.name / train_config.exp_name
    step_dir = run_dir / "29999"

    model_config = _dummy_model_config(train_config)
    model = model_config.create(jax.random.key(0))
    params = nnx.state(model).to_pure_dict()
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(step_dir / "params", {"params": params})

    _stamp.write_stamp(run_dir, train_config, "cafebabe12345678")

    h, d_real = train_config.action_horizon, train_config.action_dim_actual
    stats = _norm.RelationNormStats(
        action_q01=np.full((h, d_real), -1.0),
        action_q99=np.full((h, d_real), 1.0),
        relation_mean=np.zeros(train_config.relation_dim),
        relation_std=np.ones(train_config.relation_dim),
        gripper_dims=train_config.gripper_dims,
        provenance={"model_space_variance": [1.0] * d_real},
    )
    _norm.save_relation(run_dir / "assets_ego2g1", stats)

    return run_dir, step_dir


def _random_relation_obs(train_config, rng):
    """A hand-built inference observation: HWC uint8 image + the flattened
    hand-major relation state (RelationPrompt's contract — see
    ego2g1.train.relation_transforms), with VALID binary grasp flags at the
    tail (RelationPrompt refuses anything else)."""
    n_hands = len(train_config.hands)
    n_obj = train_config.n_objects
    relation_part = rng.normal(size=9 * n_hands * n_obj).astype(np.float32)
    grasp_part = rng.choice([0.0, 1.0], size=n_hands).astype(np.float32)
    state = np.concatenate([relation_part, grasp_part])
    assert state.shape == (56,)  # 9*2*3 + 2, the EgoRelationTrainConfig defaults
    return {
        "observation/image": rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": state,
        "prompt": "test task",
    }


def test_create_policy_relation_end_to_end(tmp_path):
    train_config = _config.EgoRelationTrainConfig()  # every default: 3 objects, 2 hands, action_dim_actual=14

    with mock.patch.object(_config.EgoRelationTrainConfig, "model_config", _dummy_model_config):
        run_dir, step_dir = _write_fixture_checkpoint(tmp_path, train_config)
        policy = _policy.create_policy(step_dir)

        meta = policy.metadata["ego2g1"]
        assert meta["control_mode"] == "relation_eef"
        assert meta["objects"] == list(train_config.objects)
        assert meta["object_prompt_names"] == list(train_config.object_prompt_names)
        assert meta["n_objects"] == train_config.n_objects == 3

        rng = np.random.default_rng(0)
        out = policy.infer(_random_relation_obs(train_config, rng))

        assert out["actions"].shape == (train_config.action_horizon, train_config.action_dim_actual)
        assert np.isfinite(out["actions"]).all()
        assert out["rtc"]["sampler"] == "plain"


def test_create_policy_relation_rtc_prefix_fails_loud(tmp_path):
    """RTC-prefix continuation is out of scope for relation checkpoints (see
    ego2g1/serve/policy.py's Ego2G1Policy docstring) — a client that sends
    prev_chunk against one must get a loud, unambiguous error, not a silently
    wrong sample."""
    train_config = _config.EgoRelationTrainConfig()

    with mock.patch.object(_config.EgoRelationTrainConfig, "model_config", _dummy_model_config):
        _run_dir, step_dir = _write_fixture_checkpoint(tmp_path, train_config)
        policy = _policy.create_policy(step_dir)

        rng = np.random.default_rng(1)
        obs = _random_relation_obs(train_config, rng)
        obs["prev_chunk"] = np.zeros((train_config.action_horizon, train_config.action_dim_actual), np.float32)
        obs["d"] = 2

        with pytest.raises(NotImplementedError, match="RTC prefix"):
            policy.infer(obs)
