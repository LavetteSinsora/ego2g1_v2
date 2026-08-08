"""Serving-side policy construction. The ONLY supported way to serve an
ego2g1 checkpoint — it enforces the stamp guard and runs the exact
training-time transform stack (TRAINING_PLAN.md §4): the per-slot gain grid's
inverse MUST run before pooled Unnormalize (else early slots inflate up to
1/c in real units).

Norm-stats resolution (see `resolve_norm_assets`), in order:
1. an explicit `assets_dir=` (caller knows best);
2. the checkpoint's OWN copies — `<step>/assets/<repo_id>/norm_stats.json` +
   `<run>/assets_ego2g1/per_slot_stats.npz`. Preferred: they travel with the
   checkpoint and are provably the ones it trained with;
3. the training-time assets dir `<assets_base_dir>/<name>/<repo_id>` that
   compute_norm_stats writes and train.py reads (resolved from the CWD, so run
   from the openpi root). Used only when the checkpoint carries no copies —
   WARNS, because a live assets dir may have been recomputed since training.

RTC (see ego2g1.serve.rtc): the request may carry `prev_chunk` (the previous
chunk's leftover, RE-ANCHORED by the client, in raw robot action space) and `d`
(inference delay in ticks). When it does, `prev_chunk` is fed through the SAME
input transform chain as a training action array — so Normalize + PerSlotRescale
land each row at its DESTINATION slot's constants, which is the only correct way
to build the guidance target. Reusing the previous chunk's model-space tensor
would be wrong by the ratio of the two slots' sigmas (up to 10x near slot 0).

With no `prev_chunk`, this is bit-identical to the plain openpi Policy.
"""

import dataclasses
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.training.checkpoints as _checkpoints
from openpi.shared import nnx_utils

from ego2g1.train import config as _config
from ego2g1.train import data_config as _data_config
from ego2g1.train import stamp as _stamp
from ego2g1.serve import rtc as _rtc


def resolve_run_dir(checkpoint_dir: str | pathlib.Path) -> pathlib.Path:
    """Locate the run-level artifacts for a checkpoint.

    Orbax saves each step under <run>/<step>/{params,assets,...}, while
    train.py writes the stamp and per-slot artifact once at the run root.
    `create_policy` takes the STEP dir (it holds params); the stamp lives one
    level up. Accept either level so a flat copied-out checkpoint also works.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    if (checkpoint_dir / _stamp.STAMP_FILENAME).exists():
        return checkpoint_dir
    if (checkpoint_dir.parent / _stamp.STAMP_FILENAME).exists():
        return checkpoint_dir.parent
    return checkpoint_dir  # let read_stamp raise its diagnostic


def config_from_stamp(
    stamp: dict,
) -> _config.Ego2G1TrainConfig | _config.EgoRelationTrainConfig | _config.UmiTrainConfig:
    """Rebuild the training config from a stamp (single source of truth).
    optimizer/lr_schedule are dropped (not needed at serving); JSON lists are
    restored to the tuples the dataclass expects.

    Dispatches on stamp["config_class"] (ego2g1.train.stamp.write_stamp) to
    build either Ego2G1TrainConfig (30-dim absolute) or EgoRelationTrainConfig
    (relational state + 14-dim relative-EEF-rotvec actions). Checkpoints
    stamped before "config_class" existed carry no such key — they predate
    EgoRelationTrainConfig entirely, so they default to Ego2G1TrainConfig."""
    cfg_dict = dict(stamp["ego2g1_config"])
    for k in ("optimizer", "lr_schedule"):
        cfg_dict.pop(k, None)

    config_class = stamp.get("config_class", "Ego2G1TrainConfig")
    if config_class == "UmiTrainConfig":
        for k in ("val_source_episodes", "history_len_probs"):
            if k in cfg_dict and isinstance(cfg_dict[k], list):
                cfg_dict[k] = tuple(cfg_dict[k])
        # No legacy-default backfill: this config family has carried every field
        # its __post_init__ relies on since its introduction, so there is no
        # pre-UmiTrainConfig checkpoint missing any of them.
        return _config.UmiTrainConfig(**cfg_dict)

    if config_class == "EgoRelationTrainConfig":
        for k in ("hands", "val_source_episodes", "objects", "object_prompt_names"):
            if k in cfg_dict and isinstance(cfg_dict[k], list):
                cfg_dict[k] = tuple(cfg_dict[k])
        # No legacy-default backfill here: every field EgoRelationTrainConfig's
        # __post_init__ relies on (per_slot_center, model_space_clamp, ...) has
        # carried a real default since this config family's introduction —
        # there is no pre-EgoRelationTrainConfig checkpoint missing them.
        return _config.EgoRelationTrainConfig(**cfg_dict)

    for k in ("hands", "val_real_episodes", "degenerate_dim_allowlist"):
        if k in cfg_dict and isinstance(cfg_dict[k], list):
            cfg_dict[k] = tuple(cfg_dict[k])
    # Checkpoints stamped before a feature existed were trained WITHOUT it —
    # missing keys must resolve to the legacy behavior, never to a new default.
    cfg_dict.setdefault("per_slot_center", False)
    cfg_dict.setdefault("model_space_clamp", None)
    return _config.Ego2G1TrainConfig(**cfg_dict)


NORM_STATS_FILENAME = "norm_stats.json"


def resolve_norm_assets(checkpoint_dir, run_dir, train_config, assets_dir=None
                        ) -> tuple[pathlib.Path, pathlib.Path]:
    """-> (pooled_dir with norm_stats.json, per_slot_dir with per_slot_stats.npz).

    The two artifacts may live in different directories (the checkpoint keeps
    pooled stats per step but the per-slot grid once at the run root), so they
    are resolved independently and NOTHING is written into the checkpoint.
    Resolution order documented in the module docstring."""
    from ego2g1.train import norm as _norm

    pooled_ck = checkpoint_dir / "assets" / train_config.repo_id
    per_slot_ck_run = run_dir / "assets_ego2g1"
    train_assets = train_config.assets_dirs / train_config.repo_id
    searched = []

    def pick(filename, candidates):
        for d in candidates:
            searched.append(d / filename)
            if (d / filename).exists():
                return d
        return None

    if assets_dir is not None:
        d = pathlib.Path(assets_dir)
        missing = [f for f in (NORM_STATS_FILENAME, _norm.PER_SLOT_FILENAME) if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"--assets-dir {d} is missing {missing}")
        return d, d

    pooled = pick(NORM_STATS_FILENAME, [pooled_ck, train_assets])
    per_slot = pick(_norm.PER_SLOT_FILENAME, [pooled_ck, per_slot_ck_run, train_assets])
    if pooled is None or per_slot is None:
        raise FileNotFoundError(
            "norm assets not found. Searched:\n  " + "\n  ".join(str(p) for p in searched) +
            "\nRun `python -m ego2g1.train.compute_norm_stats` from the openpi root, or pass --assets-dir."
        )
    if pooled == train_assets or per_slot == train_assets:
        print(f"WARNING: falling back to the training assets dir {train_assets} (the checkpoint does not "
              "carry its own copies). Confirm these stats are the ones this checkpoint trained with — "
              "they are not pinned to it.")
    return pooled, per_slot


def resolve_relation_norm_assets(checkpoint_dir, run_dir, train_config, assets_dir=None) -> pathlib.Path:
    """-> directory holding relation_stats.npz (ego2g1.train.norm.RELATION_FILENAME).

    EgoRelationTrainConfig needs exactly one stats artifact (unlike the 30-dim
    config's pooled/per-slot pair), so this returns a single directory. Mirrors
    resolve_norm_assets' resolution order EXACTLY:
    1. an explicit `assets_dir=` (caller knows best);
    2. the checkpoint's OWN copy — train.py writes relation_stats.npz to
       `<run>/assets_ego2g1/` (train.py:643, the same place save_per_slot writes
       the 30-dim grid), so that is checked at both the step dir (in case a flat
       copied-out checkpoint carries its own) and the run dir;
    3. the training-time assets dir `<assets_base_dir>/<name>/<repo_id>` that
       `compute_norm_stats --config relation` writes and train.py reads.
       Used only when the checkpoint carries no copy — WARNS, because a live
       assets dir may have been recomputed since training.
    """
    from ego2g1.train import norm as _norm

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    run_dir = pathlib.Path(run_dir)

    if assets_dir is not None:
        d = pathlib.Path(assets_dir)
        if not (d / _norm.RELATION_FILENAME).exists():
            raise FileNotFoundError(f"--assets-dir {d} is missing {_norm.RELATION_FILENAME}")
        return d

    own_ck = checkpoint_dir / "assets_ego2g1"
    own_run = run_dir / "assets_ego2g1"
    train_assets = train_config.assets_dirs / train_config.repo_id
    searched = []

    for d in (own_ck, own_run, train_assets):
        searched.append(d / _norm.RELATION_FILENAME)
        if (d / _norm.RELATION_FILENAME).exists():
            if d == train_assets:
                print(f"WARNING: falling back to the training assets dir {train_assets} (the checkpoint "
                      "does not carry its own copy). Confirm these stats are the ones this checkpoint "
                      "trained with — they are not pinned to it.")
            return d

    raise FileNotFoundError(
        f"{_norm.RELATION_FILENAME} not found. Searched:\n  " + "\n  ".join(str(p) for p in searched) +
        "\nRun `python -m ego2g1.train.compute_norm_stats --config relation` from the openpi root, "
        "or pass --assets-dir."
    )


def resolve_umi_norm_assets(checkpoint_dir, run_dir, train_config, assets_dir=None) -> pathlib.Path:
    """-> directory holding umi_stats.npz (ego2g1.train.norm.UMI_FILENAME).

    Same single-artifact shape and the same resolution order as
    `resolve_relation_norm_assets`; see that docstring for why each step exists.
    """
    from ego2g1.train import norm as _norm

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    run_dir = pathlib.Path(run_dir)

    if assets_dir is not None:
        d = pathlib.Path(assets_dir)
        if not (d / _norm.UMI_FILENAME).exists():
            raise FileNotFoundError(f"--assets-dir {d} is missing {_norm.UMI_FILENAME}")
        return d

    own_ck = checkpoint_dir / "assets_ego2g1"
    own_run = run_dir / "assets_ego2g1"
    train_assets = train_config.assets_dirs / train_config.repo_id
    searched = []

    for d in (own_ck, own_run, train_assets):
        searched.append(d / _norm.UMI_FILENAME)
        if (d / _norm.UMI_FILENAME).exists():
            if d == train_assets:
                print(f"WARNING: falling back to the training assets dir {train_assets} (the checkpoint "
                      "does not carry its own copy). Confirm these stats are the ones this checkpoint "
                      "trained with — they are not pinned to it.")
            return d

    raise FileNotFoundError(
        f"{_norm.UMI_FILENAME} not found. Searched:\n  " + "\n  ".join(str(p) for p in searched) +
        "\nRun `python -m ego2g1.train.compute_norm_stats --umi` from the openpi root, "
        "or pass --assets-dir."
    )


class Ego2G1Policy(_policy.Policy):
    """openpi Policy + RTC. Accepts optional `prev_chunk` / `d` in the request.

    Which sampler runs is decided by the CHECKPOINT (its stamp's rtc_training
    flag), never by a user flag — so the robot client is identical whether the
    checkpoint wants guided or pinned RTC, and a future rtc_training retrain is
    a server-side swap the client never sees.

    `relation_mode=True` (an EgoRelationTrainConfig checkpoint) disables the
    RTC-prefix path with a loud NotImplementedError instead of silently running
    it: the prefix's ride through the relational input-transform chain
    (PerSlotQuantizeActions instead of PerSlotRescale, a differently-shaped
    action space, gripper dims exempted differently) has never been verified
    correct, and this codebase's whole stamping mechanism exists specifically
    so an unverified serving path fails loud rather than silently wrong (see
    module docstring). RTC-prefix support for relation checkpoints is out of
    scope for this change (docs/relation_deploy_plan.md §8); only the plain
    (no-prefix) path is supported here. The plain path itself is untouched —
    `infer()`'s `sampler is PLAIN` branch calls `super().infer()` generically,
    with no 30-dim-specific assumptions.
    """

    def __init__(self, model, *, rtc: _rtc.RTCConfig, rtc_training: bool,
                 action_horizon: int, relation_mode: bool = False,
                 no_prefix_reason: str | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self._rtc = rtc
        self._rtc_training = rtc_training
        self._action_horizon = action_horizon
        self._relation_mode = relation_mode
        # `relation_mode` keeps its historical meaning and message; UmiTrainConfig
        # checkpoints pass their own reason. Both say the same thing: the prefix's
        # ride through a non-30-dim input-transform chain has never been verified,
        # and this codebase's stamping mechanism exists so an unverified serving
        # path fails loud rather than silently wrong.
        self._no_prefix_reason = no_prefix_reason or (
            "RTC prefix continuation is not implemented for relational "
            "(EgoRelationTrainConfig) checkpoints — only the plain (no-prefix) "
            "sampler is supported. See docs/relation_deploy_plan.md §8."
            if relation_mode else None
        )

        self._sample_guided = nnx_utils.module_jit(
            model.sample_actions_guided,
            static_argnames=("num_steps", "max_guidance_weight", "use_vjp"),
        )
        # Only meaningful on an rtc_training checkpoint; building it is free.
        self._sample_pinned = nnx_utils.module_jit(
            model.sample_actions_rtc, static_argnames=("num_steps",)
        )

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        inputs = dict(obs)
        prev_chunk = inputs.pop("prev_chunk", None)
        d = int(inputs.pop("d", 0) or 0)
        # How many rows of prev_chunk are real; the rest is zero padding. A zero
        # vec9 is NOT a pose (rot6d_to_mat(zeros) has det 0), so neither the mask
        # nor the pinned sampler may ever touch it.
        n_real = int(inputs.pop("n_prefix", 0) or 0) or self._action_horizon
        n_real = max(0, min(n_real, self._action_horizon))
        # Cap the committed prefix at what we actually have. Without this the
        # pinned sampler would freeze padding as clean ground truth (F14).
        d = int(np.clip(d, 0, max(0, n_real - 1)))

        has_prefix = prev_chunk is not None and n_real > 0
        if has_prefix and self._no_prefix_reason is not None:
            raise NotImplementedError(self._no_prefix_reason)
        sampler = _rtc.select_sampler(
            rtc_training=self._rtc_training,
            has_prefix=has_prefix,
            rtc_enabled=self._rtc.enabled,
        )

        if sampler is _rtc.Sampler.PLAIN:
            # Stock path, untouched — this is what --blocking and the first chunk
            # of every episode take, and it must stay bit-identical to openpi.
            out = super().infer(inputs, noise=noise)
            out["rtc"] = {"sampler": sampler.value, "d": 0}
            return out

        # RTC: the prefix rides the normal input chain as if it were a training
        # action array, so it gets normalized + per-slot-rescaled at the correct
        # destination slots.
        inputs["actions"] = np.asarray(prev_chunk, dtype=np.float32)
        inputs = self._input_transform(inputs)
        prefix = inputs.pop("actions")  # (H, action_dim) in MODEL space

        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        prefix = jnp.asarray(prefix)[np.newaxis, ...]
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = {}
        if noise is not None:
            n = jnp.asarray(noise)
            sample_kwargs["noise"] = n[None, ...] if n.ndim == 2 else n

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if sampler is _rtc.Sampler.GUIDED:
            weights = _rtc.prefix_weights(
                d, self._rtc.overlap, self._action_horizon, n_real,
                self._rtc.prefix_attention_schedule,
            )
            actions = self._sample_guided(
                sample_rng, observation, prefix, jnp.asarray(weights),
                num_steps=self._rtc.num_steps,
                max_guidance_weight=self._rtc.max_guidance_weight,
                use_vjp=self._rtc.use_vjp,
                **sample_kwargs,
            )
        else:  # PINNED — train-time RTC checkpoint
            actions = self._sample_pinned(
                sample_rng, observation, prefix, jnp.asarray(d, dtype=jnp.int32),
                num_steps=self._rtc.num_steps, **sample_kwargs,
            )
        model_time = time.monotonic() - start_time

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        outputs["rtc"] = {"sampler": sampler.value, "d": d, "n_prefix": n_real}
        return outputs


# Metadata sentinel for an EgoRelationTrainConfig checkpoint's control_mode. The
# 30-dim config's control_mode metadata mirrors train_config.control_mode
# verbatim (CONTROL_MODE_EEF="end effector" or CONTROL_MODE_JOINT="joint" —
# ego2g1.train.transforms), so this must not collide with either: the relation
# config's own control_mode field holds the SAME "end effector"/"joint" string
# (it is embedded in the prompt marker), but the metadata key must instead
# signal the 14-dim relational action space to the client, distinctly from the
# 30-dim scheme's two existing values.
RELATION_CONTROL_MODE_METADATA = "relation_eef"

# Same idea for UmiTrainConfig: the metadata control_mode signals the ACTION
# SPACE to the client (7-dim single-arm anchor-relative rotvec + continuous
# gripper), which is distinct from both the 30-dim scheme's "end effector" /
# "joint" and from the relational config's 14-dim space.
UMI_CONTROL_MODE_METADATA = "umi_eef"


def create_policy(checkpoint_dir: str | pathlib.Path, *, default_prompt: str | None = None,
                  assets_dir: str | pathlib.Path | None = None,
                  rtc: _rtc.RTCConfig | None = None) -> Ego2G1Policy:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    run_dir = resolve_run_dir(checkpoint_dir)
    stamp = _stamp.check_supported(run_dir)

    train_config = config_from_stamp(stamp)
    model_config = train_config.model_config()

    model = model_config.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    import openpi.transforms as transforms

    rtc = rtc if rtc is not None else _rtc.RTCConfig()
    is_relation = isinstance(train_config, _config.EgoRelationTrainConfig)
    is_umi = isinstance(train_config, _config.UmiTrainConfig)
    no_prefix_reason = None

    if is_umi:
        stats_dir = resolve_umi_norm_assets(checkpoint_dir, run_dir, train_config, assets_dir)
        data_cfg = _data_config.create_umi_data_config(
            train_config, model_config, stats_dir=stats_dir,
            # Serving must be deterministic and must use everything it has: no
            # length draw, full history. Truncation at deploy is the RUNTIME's
            # job — it happens only when the pose ring buffer genuinely has not
            # filled yet (episode start, or after a re-arm cleared it), and it is
            # expressed by sending fewer rows, not by sampling here.
            history_fixed_len=train_config.n_lags,
        )
        # create_umi_data_config returns norm_stats={} BY DESIGN (see its
        # docstring): the real normalization (NormalizeHistory,
        # PerSlotQuantizeActions/Inverse) already lives inside
        # model_transforms.inputs/outputs, so Normalize/Unnormalize must NOT be
        # inserted here.
        policy_transforms = [
            transforms.InjectDefaultPrompt(default_prompt),
            *data_cfg.data_transforms.inputs,
            *data_cfg.model_transforms.inputs,
        ]
        policy_output_transforms = [
            *data_cfg.model_transforms.outputs,
            *data_cfg.data_transforms.outputs,
        ]
        no_prefix_reason = (
            "RTC prefix continuation is not implemented for UMI (UmiTrainConfig) "
            "checkpoints — only the plain (no-prefix) sampler is supported. The "
            "reanchor-prefix math for rotvec deltas under per-slot-quantized "
            "actions is a separate design, the same gap the relational config has."
        )
        ego_metadata = {
            "hands": [train_config.hand],
            "action_horizon": train_config.action_horizon,
            "action_dim": train_config.action_dim_actual,
            "fps": train_config.fps,
            "control_mode": UMI_CONTROL_MODE_METADATA,
            "rtc_training": bool(train_config.rtc_training),
            "rtc": {
                "enabled": rtc.enabled,
                "overlap": rtc.overlap,
                "max_guidance_weight": rtc.max_guidance_weight,
                "schedule": rtc.prefix_attention_schedule.value,
                "use_vjp": rtc.use_vjp,
                "num_steps": rtc.num_steps,
            },
            # Everything the robot client needs to build the request without
            # importing ego2g1.train (which pulls in JAX).
            "hand": train_config.hand,
            "n_lags": train_config.n_lags,
            "lag_ticks": list(train_config.lag_ticks),
            "history_dim": train_config.history_dim,
            # P(history length = j) as TRAINED. Advertised so deploy can warn
            # when --history-len asks for a length this checkpoint never saw —
            # index 0 is 0.0 by default, i.e. a prompt with no `State history:`
            # segment at all is out of distribution.
            "history_len_probs": list(train_config.history_len_probs),
            "acting_slot": train_config.acting_slot,
            "gripper_dims": list(train_config.gripper_dims),
            # MEASURED poses, never commanded — the training data has no commanded
            # pose at all, so feeding commanded ones at deploy shifts the history
            # distribution by the arm's tracking error.
            "pose_source": "measured",
        }
    elif is_relation:
        stats_dir = resolve_relation_norm_assets(checkpoint_dir, run_dir, train_config, assets_dir)
        data_cfg = _data_config.create_relation_data_config(
            train_config, model_config, stats_dir=stats_dir,
            # serving must be deterministic — only TRAINING shuffles object order.
            shuffle_objects=False,
        )
        # create_relation_data_config returns norm_stats={} BY DESIGN (see its
        # docstring): the real normalization (NormalizeRelations,
        # PerSlotQuantizeActions/Inverse) already lives inside
        # model_transforms.inputs/outputs, so Normalize/Unnormalize must NOT be
        # inserted here — they would be dead weight at best (norm_stats={} makes
        # them a no-op) and a silent double-normalization bug at worst if that
        # docstring's contract ever changes without this file being updated.
        policy_transforms = [
            transforms.InjectDefaultPrompt(default_prompt),
            *data_cfg.data_transforms.inputs,
            *data_cfg.model_transforms.inputs,
        ]
        policy_output_transforms = [
            *data_cfg.model_transforms.outputs,
            *data_cfg.data_transforms.outputs,
        ]
        ego_metadata = {
            "hands": list(train_config.hands),
            "action_horizon": train_config.action_horizon,
            "action_dim": train_config.action_dim_actual,
            "fps": train_config.fps,
            "control_mode": RELATION_CONTROL_MODE_METADATA,
            "rtc_training": bool(train_config.rtc_training),
            "rtc": {
                "enabled": rtc.enabled,
                "overlap": rtc.overlap,
                "max_guidance_weight": rtc.max_guidance_weight,
                "schedule": rtc.prefix_attention_schedule.value,
                "use_vjp": rtc.use_vjp,
                "num_steps": rtc.num_steps,
            },
            "objects": list(train_config.objects),
            "object_prompt_names": list(train_config.object_prompt_names),
            "n_objects": train_config.n_objects,
        }
    else:
        pooled_dir, per_slot_dir = resolve_norm_assets(checkpoint_dir, run_dir, train_config, assets_dir)
        data_cfg = _data_config.create_data_config(
            train_config, model_config, norm_assets_dir=pooled_dir, per_slot_dir=per_slot_dir,
            # a blind checkpoint (state_dropout_p >= 1.0) never saw a state digit and
            # must not be shown one; a dropout checkpoint is served WITH the real state.
            state_mode=_data_config.serve_state_mode(train_config),
        )
        policy_transforms = [
            transforms.InjectDefaultPrompt(default_prompt),
            *data_cfg.data_transforms.inputs,
            transforms.Normalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.model_transforms.inputs,
        ]
        policy_output_transforms = [
            *data_cfg.model_transforms.outputs,
            transforms.Unnormalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm),
            *data_cfg.data_transforms.outputs,
        ]
        ego_metadata = {
            "hands": list(train_config.hands),
            "action_horizon": train_config.action_horizon,
            "action_dim": train_config.action_dim_actual,
            "fps": train_config.fps,
            "control_mode": train_config.control_mode,
            "rtc_training": bool(train_config.rtc_training),
            "rtc": {
                "enabled": rtc.enabled,
                "overlap": rtc.overlap,
                "max_guidance_weight": rtc.max_guidance_weight,
                "schedule": rtc.prefix_attention_schedule.value,
                "use_vjp": rtc.use_vjp,
                "num_steps": rtc.num_steps,
            },
        }

    return Ego2G1Policy(
        model,
        rtc=rtc,
        rtc_training=bool(train_config.rtc_training),
        action_horizon=train_config.action_horizon,
        relation_mode=is_relation,
        no_prefix_reason=no_prefix_reason,
        transforms=policy_transforms,
        output_transforms=policy_output_transforms,
        metadata={
            "ego2g1_stamp": {k: stamp[k] for k in ("feature_flags", "ego2g1_config_hash",
                                                    "extraction_config_hash", "openpi_commit")},
            # The client reads its layout from here instead of importing ego2g1.config
            # (which pulls in JAX). Keeps the robot PC config-free.
            "ego2g1": ego_metadata,
        },
    )
