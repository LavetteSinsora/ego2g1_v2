"""Ego2G1 training entrypoint. Run from the openpi root:
    uv run python -m ego2g1.train.train --exp-name my_run

Reuses scripts/train.py wholesale (init_logging, init_wandb,
init_train_state, sharding, checkpointing). Differences: our config
dataclass, our dataset/DataConfig construction, a train_step that also logs
the E001 per-slot loss decomposition, an in-loop validation eval on held-out
real episodes, and checkpoint stamping (feature flags + both stats artifacts).
"""

import dataclasses
import functools
import importlib.util
import logging
import math
import pathlib
import sys

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
import wandb
from flax.training import common_utils

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _openpi_config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

from ego2g1.train import config as _config
from ego2g1.train import data_config as _data_config
from ego2g1.train import dataset as _dataset
from ego2g1.train import stamp as _stamp

# per-slot loss buckets logged to wandb (E001 eval item 3): early slots are
# the mechanism check for the floored rescale, late slots the control.
SLOT_BUCKETS = {"slots_00_04": (0, 5), "slots_05_24": (5, 25), "slots_25_49": (25, 50)}


def _slot_bucket_means(chunked_loss: jnp.ndarray, action_horizon: int) -> dict[str, jnp.ndarray]:
    out = {}
    for name, (lo, hi) in SLOT_BUCKETS.items():
        hi = min(hi, action_horizon)
        if lo < action_horizon:
            out[name] = jnp.mean(chunked_loss[..., lo:hi])
    return out


@at.typecheck
def train_step(
    config: _openpi_config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Stock scripts/train.py train_step + per-slot loss decomposition in info."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), chunked_loss

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, chunked_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # gradient decomposition by module: which part of the network training is
    # actually moving (SigLIP vision encoder vs 2B prefix expert vs 300M action
    # expert vs the small projection/time-MLP heads). Expert-1 params carry the
    # "_1" name suffix (cf. Pi0Config.get_freeze_filter).
    _img = nnx_utils.PathRegex(".*img.*")
    _llm = nnx_utils.PathRegex(".*llm.*")
    _expert1 = nnx_utils.PathRegex(".*llm.*_1.*")
    grad_groups = {
        "grad_norm/siglip": _img,
        "grad_norm/prefix_expert": nnx.All(_llm, nnx.Not(_expert1)),
        "grad_norm/action_expert": _expert1,
        "grad_norm/heads": nnx.All(nnx.Not(_img), nnx.Not(_llm)),
    }

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **{k: optax.global_norm(grads.filter(f)) for k, f in grad_groups.items()},
        **{f"loss/{k}": v for k, v in _slot_bucket_means(chunked_loss, config.model.action_horizon).items()},
    }
    return new_state, info


@at.typecheck
def eval_step(
    config: _openpi_config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """compute_loss on a val batch, eval mode, FIXED rng: the same (t, noise)
    draws every eval, so the curve is comparable across steps. Uses EMA params
    when available (those are what get served)."""
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, actions = batch
    chunked_loss = model.compute_loss(rng, observation, actions, train=False)
    return {
        "val/loss": jnp.mean(chunked_loss),
        **{f"val/{k}": v for k, v in _slot_bucket_means(chunked_loss, config.model.action_horizon).items()},
    }


def _attention_probe(config: _config.Ego2G1TrainConfig, state: training_utils.TrainState, val_batch) -> dict:
    """Attention allocation of action tokens on a small fixed probe batch
    (ego2g1.diagnostics), reduced to wandb scalars: mass per token group,
    overall and at the first/last layer, plus mean attention entropy.
    Eager (un-jitted); uses EMA params like eval_step.

    Always run on the REAL-state val pool: on a blind batch there is no state to
    attend to and attn/text_state is 0 by construction."""
    from ego2g1.train import diagnostics as _diagnostics

    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    obs, actions = jax.tree.map(lambda x: x[: config.probe_batch_size], val_batch)
    out = _diagnostics.attention_allocation(
        model, obs, actions, state_token_ids=_diagnostics.digit_token_ids()
    )
    per_layer = out["per_layer"]  # (L, G)
    payload = {}
    means = {}
    for g, name in enumerate(out["group_names"]):
        key = name.replace("/", "_")
        means[name] = float(per_layer[:, g].mean())
        payload[f"attn/{key}"] = means[name]
        payload[f"attn_first_layer/{key}"] = float(per_layer[0, g])
        payload[f"attn_last_layer/{key}"] = float(per_layer[-1, g])
    payload["attn/entropy"] = float(out["entropy_per_layer"].mean())
    # The attention-side counterpart of val/state_gap: of the attention the action
    # tokens spend on CONDITIONING (state digits vs task words vs pixels), what
    # share goes to the state? Climbing here + a widening val/state_gap is the
    # signature of a policy solving the task from proprioception.
    cond = {k: v for k, v in means.items() if k.startswith(("text/", "img/"))}
    denom = sum(cond.values())
    if denom > 0 and "text/state" in cond:
        payload["attn/percentage_state-attn_prefix_attn"] = cond["text/state"] / denom
    return payload


def _save_best(
    manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    val_loss: float,
) -> None:
    """Save params + assets ONLY (no train_state) to the best-checkpoint manager.

    The best checkpoint is for evaluation and serving, never for resuming, and
    ego2g1.serve.create_policy reads exactly <step>/params + the assets. Dropping
    the optimizer state (AdamW moments, ~2x the params) makes it about a third
    the size of a full checkpoint, which is what makes re-crowning it at every
    eval affordable. _split_params hands back the EMA params when EMA is on —
    the same weights eval_step scored, so the metric and the artifact agree."""

    def save_assets(directory: epath.Path):
        data_config = data_loader.data_config()
        if data_config.norm_stats is not None and data_config.asset_id is not None:
            import openpi.shared.normalize as _normalize_mod

            _normalize_mod.save(directory / data_config.asset_id, data_config.norm_stats)

    with at.disable_typechecking():
        _, params = _checkpoints._split_params(state)  # noqa: SLF001
    manager.save(
        step,
        {"assets": save_assets, "params": {"params": params}},
        metrics={"val_loss": float(val_loss)},
    )


def _load_stock_train_module():
    """Import scripts/train.py (not a package) for its init functions."""
    path = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "third_party"
        / "openpi"
        / "scripts"
        / "train.py"
    )
    spec = importlib.util.spec_from_file_location("openpi_scripts_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _to_openpi_train_config(config: _config.Ego2G1TrainConfig, data_cfg) -> _openpi_config.TrainConfig:
    """Adapter so stock init_train_state/checkpointing code can be reused.
    The `data` factory just returns our prebuilt DataConfig."""

    @dataclasses.dataclass(frozen=True)
    class _Fixed(_openpi_config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return data_cfg

    return _openpi_config.TrainConfig(
        name=config.name,
        exp_name=config.exp_name,
        model=config.model_config(),
        weight_loader=config.weight_loader(),
        data=_Fixed(repo_id=config.repo_id),
        optimizer=config.optimizer,
        lr_schedule=config.lr_schedule(),
        batch_size=config.batch_size,
        num_train_steps=config.num_train_steps,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
        keep_period=config.keep_period,
        num_workers=config.num_workers,
        seed=config.seed,
        ema_decay=config.ema_decay,
        checkpoint_base_dir=config.checkpoint_base_dir,
        assets_base_dir=config.assets_base_dir,
        fsdp_devices=config.fsdp_devices,
        wandb_enabled=config.wandb_enabled,
        project_name=config.wandb_project,
        resume=config.resume,
        overwrite=config.overwrite,
    )


def main(config: _config.Ego2G1TrainConfig):
    stock = _load_stock_train_module()
    stock.init_logging()

    model_config = config.model_config()
    meta = _dataset.assert_dataset_compatible(
        config.dataset_root, config.expected_config_hash, model_config.action_horizon, config.fps
    )
    norm_assets_dir = config.assets_dirs / config.repo_id
    # TRAIN stack: "dropout" mode covers all three regimes — p=0.0 never masks
    # (byte-identical to the stock prompt), p=1.0 always masks, 0<p<1 flips a coin.
    data_cfg = _data_config.create_data_config(
        config, model_config, norm_assets_dir=norm_assets_dir,
        state_mode="dropout", dropout_p=config.state_dropout_p,
    )
    train_config = _to_openpi_train_config(config, data_cfg)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size {config.batch_size} % devices {jax.device_count()} != 0")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        train_config.checkpoint_dir, keep_period=config.keep_period,
        overwrite=config.overwrite, resume=config.resume,
    )
    # Tier 3 of checkpoint retention. The other two come from the manager above:
    # max_to_keep=1 (latest, resumable) and keep_period (permanent archive). A
    # single manager cannot also track the best, because orbax's best_fn retargets
    # max_to_keep at the best checkpoints and the guaranteed-latest is lost — hence
    # a second manager in its own directory.
    # NOTE on --resume: this manager starts cold, so it can re-crown a checkpoint
    # worse than a pre-crash best. Acceptable; the archive still has the periodic ones.
    best_manager = ocp.CheckpointManager(
        epath.Path(train_config.checkpoint_dir) / "best",
        item_handlers={"assets": _checkpoints.CallbackHandler(), "params": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            best_fn=lambda metrics: float(metrics["val_loss"]),
            best_mode="min",
            keep_checkpoints_without_metrics=False,
            create=True,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )
    stock.init_wandb(train_config, resuming=resuming, enabled=config.wandb_enabled)

    # stamp before training starts so even a crashed run is identifiable;
    # stock save_assets writes pooled norm stats per checkpoint step, the
    # per-slot artifact is copied next to the stamp once (it is step-invariant)
    _stamp.write_stamp(train_config.checkpoint_dir, config, meta["config_hash"])
    from ego2g1.train import norm as _norm
    per_slot = _norm.load_per_slot(norm_assets_dir)
    _norm.save_per_slot(train_config.checkpoint_dir / "assets_ego2g1", per_slot)
    # best/ is its own run dir as far as serving is concerned: resolve_run_dir
    # looks in the checkpoint dir and then its PARENT, so best/<step> resolves to
    # best/ — which needs its own stamp + per-slot stats or create_policy refuses it.
    _stamp.write_stamp(train_config.checkpoint_dir / "best", config, meta["config_hash"])
    _norm.save_per_slot(train_config.checkpoint_dir / "best" / "assets_ego2g1", per_slot)

    torch_dataset = _dataset.create_dataset(config, model_config, split="train")
    transformed = _data_loader.transform_dataset(torch_dataset, data_cfg)
    torch_loader = _data_loader.TorchDataLoader(
        transformed,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    data_loader = _data_loader.DataLoaderImpl(data_cfg, torch_loader)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Three fixed val pools, loaded once (the val split is small). All three draw
    # the SAME ticks in the same order (shuffle with the same seed), so the only
    # difference between the curves is what the prompt says about the state:
    #   val          real state digits          — the deployed condition
    #   val_blind    "State: unknown"           — how well it does with NO state
    #   val_shuffled a random other tick's state — how badly a WRONG state hurts
    # eval uses a fixed rng, so every curve is comparable across steps too.
    val_pools: dict[str, list] = {}
    if config.eval_interval > 0 and config.val_real_episodes:
        val_dataset = _dataset.create_dataset(config, model_config, split="val")
        pool_kwargs = {
            "val": {"state_mode": "real"},
            "val_blind": {"state_mode": "blind"},
            "val_shuffled": {
                "state_mode": "real",
                "shuffle_state_pool": _dataset.raw_state_pool(config, split="val"),
            },
        }
        for prefix, kwargs in pool_kwargs.items():
            pool_cfg = _data_config.create_data_config(
                config, model_config, norm_assets_dir=norm_assets_dir, **kwargs
            )
            val_loader = _data_loader.TorchDataLoader(
                _data_loader.transform_dataset(val_dataset, pool_cfg),
                local_batch_size=config.batch_size // jax.process_count(),
                sharding=data_sharding,
                shuffle=True,
                num_batches=config.eval_num_batches,
                num_workers=0,
                seed=config.seed,
            )
            val_pools[prefix] = list(iter(_data_loader.DataLoaderImpl(pool_cfg, val_loader)))
        logging.info(f"Loaded {config.eval_num_batches} fixed val batches x {len(val_pools)} pools "
                     f"({len(val_dataset)} datapoints from {len(config.val_real_episodes)} real episodes)")
    elif config.eval_interval > 0:
        logging.warning("eval_interval > 0 but val_real_episodes is empty — validation disabled")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = stock.init_train_state(train_config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, train_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        functools.partial(eval_step, train_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    eval_rng = jax.random.key(config.seed + 1)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step,
                     total=config.num_train_steps, dynamic_ncols=True)

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)
        if val_pools and config.eval_interval > 0 and (step % config.eval_interval == 0 or step == config.num_train_steps - 1):
            val_reduced = {}
            for prefix, batches in val_pools.items():
                with sharding.set_mesh(mesh):
                    # same fold_in sequence for every pool: identical (t, noise)
                    # draws, so the pools differ ONLY in the prompt's state segment
                    val_infos = [peval_step(jax.random.fold_in(eval_rng, i), train_state, vb)
                                 for i, vb in enumerate(batches)]
                reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(val_infos)))
                val_reduced.update({k.replace("val/", f"{prefix}/", 1): v for k, v in reduced.items()})
            # state_gap: how much the policy LEANS on proprioception (big = it is
            # solving the task from state, which is what we are trying to break).
            # state_trust: how much a WRONG state hurts beyond having none at all
            # (big = it does not merely use the state, it is enslaved to it).
            if {"val/loss", "val_blind/loss", "val_shuffled/loss"} <= val_reduced.keys():
                val_reduced["val/state_gap"] = val_reduced["val_blind/loss"] - val_reduced["val/loss"]
                val_reduced["val/state_trust"] = val_reduced["val_shuffled/loss"] - val_reduced["val_blind/loss"]
            pbar.write(f"Step {step} [val]: " + ", ".join(f"{k}={v:.4f}" for k, v in val_reduced.items()))
            wandb.log(val_reduced, step=step)
            _save_best(best_manager, train_state, data_loader, step, float(val_reduced["val/loss"]))
        if val_pools and config.probe_interval > 0 and (step % config.probe_interval == 0 or step == config.num_train_steps - 1):
            wandb.log(_attention_probe(config, train_state, val_pools["val"][0]), step=step)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint managers to finish")
    checkpoint_manager.wait_until_finished()
    best_manager.wait_until_finished()


# ---------------------------------------------------------------------------
# EgoRelationTrainConfig: relational state + 14-dim rotvec actions
# ---------------------------------------------------------------------------

# Action dim groups logged separately. The gripper is 2 of 14 dims but carries a
# third of the weighted loss, and its error mode (a sign flip at the wrong moment
# = dropped object) is nothing like an EEF dim's, so a single loss number hides
# the thing most likely to go wrong.
def _dim_group_means(sq_loss, gripper_dims: tuple[int, ...]) -> dict:
    """sq_loss (b, ah, d_real) -> per-group scalars."""
    d_real = sq_loss.shape[-1]
    eef = [d for d in range(d_real) if d not in gripper_dims]
    return {
        "loss/dims_eef": jnp.mean(sq_loss[..., eef]),
        "loss/dims_gripper": jnp.mean(sq_loss[..., list(gripper_dims)]),
    }


@at.typecheck
def relation_train_step(
    config: _openpi_config.TrainConfig,
    ego_config: _config.EgoRelationTrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Stock train_step + the weighted flow loss, the aux grasp BCE, and the
    per-dim-group / per-slot decompositions."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    grip = ego_config.gripper_dims
    w_aux = ego_config.w_aux

    @at.typecheck
    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation, actions):
        chunked_loss, aux = model.compute_loss_with_aux(
            rng, observation, actions, train=True, gripper_dims=grip
        )
        total = jnp.mean(chunked_loss)
        if "grasp_bce" in aux:
            total = total + w_aux * aux["grasp_bce"]
        return total, (chunked_loss, aux)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, (chunked_loss, aux)), grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True
    )(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    _img = nnx_utils.PathRegex(".*img.*")
    _llm = nnx_utils.PathRegex(".*llm.*")
    _expert1 = nnx_utils.PathRegex(".*llm.*_1.*")
    grad_groups = {
        "grad_norm/siglip": _img,
        "grad_norm/prefix_expert": nnx.All(_llm, nnx.Not(_expert1)),
        "grad_norm/action_expert": _expert1,
        # the two ego2g1-only modules, split out: if the encoder's gradient is
        # orders below the rest, the injected tokens are not being learned
        "grad_norm/relation_encoder": nnx_utils.PathRegex(".*relation_encoder.*"),
        "grad_norm/grasp_head": nnx_utils.PathRegex(".*grasp_head.*"),
    }

    info = {
        "loss": loss,
        "loss/flow": jnp.mean(chunked_loss),
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **{k: optax.global_norm(grads.filter(f)) for k, f in grad_groups.items()},
        **{f"loss/{k}": v for k, v in _slot_bucket_means(chunked_loss, config.model.action_horizon).items()},
    }
    if "grasp_bce" in aux:
        info["loss/grasp_bce"] = aux["grasp_bce"]
    if "delta_norm" in aux:
        # The injection canaries. rotation_deg is THE one to watch: gemma.RMSNorm
        # keeps only direction, so rotation is the injection's entire effect.
        # Healthy = arctan(alpha) ~ 45 deg at alpha=1. relation_v1 sat at 0.2 deg
        # for 10k steps and every downstream metric was measured on a policy that
        # could not see object geometry.
        info["relation/rotation_deg"] = aux["rotation_deg"]
        info["relation/token_sep_deg"] = aux["token_sep_deg"]   # objects distinguishable?
        info["relation/delta_sep_deg"] = aux["delta_sep_deg"]   # encoder collapsed?
        info["relation/alpha"] = aux["alpha"]
        info["relation/base_norm"] = aux["base_norm"]
        info["relation/delta_norm"] = aux["delta_norm"]
        info["relation/text_norm"] = aux["text_norm"]
    return new_state, info


def main_relation(config: _config.EgoRelationTrainConfig):
    stock = _load_stock_train_module()
    stock.init_logging()

    from ego2g1.train import dataset as _rel_dataset
    from ego2g1.train import norm as _norm
    from ego2g1.train import relation as _relation

    model_config_base = config.model_config()
    meta = _rel_dataset.assert_relation_dataset_compatible(
        config.dataset_root, config.expected_config_hash, model_config_base.action_horizon,
        config.fps, config.n_objects, config.hands,
    )
    stats_dir = config.assets_dirs / config.repo_id
    stats = _norm.load_relation(stats_dir)

    # The loss weights are DERIVED from the stats artifact, never hard-coded, so
    # they track the data: variance-normalize every dim, then apply w_gripper.
    weights = _data_config.loss_dim_weights(
        stats, config.action_dim_actual, config.gripper_dims, config.w_gripper
    )
    model_config = dataclasses.replace(model_config_base, loss_dim_weights=weights)
    logging.info(f"loss_dim_weights (mean 1): {[round(w, 4) for w in weights]}")

    # Safeguard 2 needs no resolution step any more: the injection is sized
    # relative to the sentinel embedding, read live inside embed_prefix. The
    # offline measurement this replaces is what broke relation_v1.
    if config.n_objects:
        logging.info(f"relation alpha (injection size as a fraction of the base): "
                     f"{config.relation_alpha} -> {math.degrees(math.atan(config.relation_alpha)):.1f} deg rotation")

    data_cfg = _data_config.create_relation_data_config(
        config, model_config, stats_dir=stats_dir, shuffle_objects=config.shuffle_object_order,
    )
    train_config = _to_openpi_train_config(config, data_cfg)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size {config.batch_size} % devices {jax.device_count()} != 0")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        train_config.checkpoint_dir, keep_period=config.keep_period,
        overwrite=config.overwrite, resume=config.resume,
    )
    best_manager = ocp.CheckpointManager(
        epath.Path(train_config.checkpoint_dir) / "best",
        item_handlers={"assets": _checkpoints.CallbackHandler(), "params": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            best_fn=lambda metrics: float(metrics["val_loss"]),
            best_mode="min",
            keep_checkpoints_without_metrics=False,
            create=True,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )
    stock.init_wandb(train_config, resuming=resuming, enabled=config.wandb_enabled)

    # Stamp before training so even a crashed run is identifiable. The relation
    # stats artifact is step-invariant, so it is copied next to the stamp once --
    # and into best/, which resolve_run_dir treats as its own run dir.
    _stamp.write_stamp(train_config.checkpoint_dir, config, meta["config_hash"])
    _norm.save_relation(train_config.checkpoint_dir / "assets_ego2g1", stats)
    _stamp.write_stamp(train_config.checkpoint_dir / "best", config, meta["config_hash"])
    _norm.save_relation(train_config.checkpoint_dir / "best" / "assets_ego2g1", stats)

    torch_dataset = _rel_dataset.create_relation_dataset(config, model_config, split="train")
    transformed = _data_loader.transform_dataset(torch_dataset, data_cfg)
    torch_loader = _data_loader.TorchDataLoader(
        transformed,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    data_loader = _data_loader.DataLoaderImpl(data_cfg, torch_loader)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Validation pools. All four draw the SAME ticks in the same order with the
    # same fixed rng, so the only difference between the curves is what the prompt
    # says about the objects:
    #   val                 real prompt, fixed object order  — the deployed condition
    #   val_swapped         names kept, relation ROWS permuted — referential binding
    #   val_shuffled_order  object order shuffled             — permutation invariance
    #   val_no_relations    Objects: segment removed          — is the channel used at all
    val_pools: dict[str, list] = {}
    if config.eval_interval > 0 and config.val_source_episodes:
        val_dataset = _rel_dataset.create_relation_dataset(config, model_config, split="val")
        pool_kwargs = {
            "val": {"shuffle_objects": False},
            "val_swapped": {"shuffle_objects": False, "swap_relations": True},
            "val_shuffled_order": {"shuffle_objects": True},
            "val_no_relations": {"shuffle_objects": False, "include_objects": False},
        }
        for prefix, kwargs in pool_kwargs.items():
            pool_cfg = _data_config.create_relation_data_config(
                config, model_config, stats_dir=stats_dir, **kwargs
            )
            val_loader = _data_loader.TorchDataLoader(
                _data_loader.transform_dataset(val_dataset, pool_cfg),
                local_batch_size=config.batch_size // jax.process_count(),
                sharding=data_sharding,
                shuffle=True,
                num_batches=config.eval_num_batches,
                num_workers=0,
                seed=config.seed,
            )
            val_pools[prefix] = list(iter(_data_loader.DataLoaderImpl(pool_cfg, val_loader)))
        logging.info(f"Loaded {config.eval_num_batches} fixed val batches x {len(val_pools)} pools "
                     f"({len(val_dataset)} datapoints from {len(config.val_source_episodes)} episodes)")
    elif config.eval_interval > 0:
        logging.warning("eval_interval > 0 but val_source_episodes is empty — validation disabled")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = stock.init_train_state(train_config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(relation_train_step, train_config, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    grip = config.gripper_dims

    def _eval_step(rng, state, batch):
        params = state.ema_params if state.ema_params is not None else state.params
        model = nnx.merge(state.model_def, params)
        model.eval()
        observation, actions = batch
        chunked_loss, aux = model.compute_loss_with_aux(
            rng, observation, actions, train=False, gripper_dims=grip
        )
        out = {"val/loss": jnp.mean(chunked_loss),
               **{f"val/{k}": v for k, v in _slot_bucket_means(chunked_loss, config.action_horizon).items()}}
        if "grasp_bce" in aux:
            out["val/grasp_bce"] = aux["grasp_bce"]
            out["val/grasp_logits"] = aux["grasp_logits"]
            out["val/grasp_targets"] = aux["grasp_targets"]
        return out

    peval_step = jax.jit(
        _eval_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    eval_rng = jax.random.key(config.seed + 1)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step,
                     total=config.num_train_steps, dynamic_ncols=True)

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)
        if val_pools and config.eval_interval > 0 and (step % config.eval_interval == 0 or step == config.num_train_steps - 1):
            val_reduced = {}
            for prefix, batches in val_pools.items():
                with sharding.set_mesh(mesh):
                    val_infos = [peval_step(jax.random.fold_in(eval_rng, i), train_state, vb)
                                 for i, vb in enumerate(batches)]
                # AUC is rank-based: it must be computed over the POOLED logits,
                # not averaged across per-batch AUCs.
                auc = float("nan")
                if "val/grasp_logits" in val_infos[0]:
                    logits = np.concatenate([np.asarray(v["val/grasp_logits"]) for v in val_infos])
                    targets = np.concatenate([np.asarray(v["val/grasp_targets"]) for v in val_infos])
                    auc = _relation.grasp_auc(logits, targets)
                scalars = [{k: v for k, v in v.items() if not k.startswith("val/grasp_l")
                            and k != "val/grasp_targets"} for v in val_infos]
                reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(scalars)))
                val_reduced.update({k.replace("val/", f"{prefix}/", 1): v for k, v in reduced.items()})
                if auc == auc:  # not nan
                    val_reduced[f"{prefix}/grasp_auc"] = auc
            pbar.write(f"Step {step} [val]: " + ", ".join(f"{k}={v:.4f}" for k, v in val_reduced.items()))
            wandb.log(val_reduced, step=step)
            _save_best(best_manager, train_state, data_loader, step, float(val_reduced["val/loss"]))
        if val_pools and config.probe_interval > 0 and (step % config.probe_interval == 0 or step == config.num_train_steps - 1):
            wandb.log(_relation_attention_probe(config, train_state, val_pools["val"][0]), step=step)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint managers to finish")
    checkpoint_manager.wait_until_finished()
    best_manager.wait_until_finished()


def _relation_attention_probe(config: _config.EgoRelationTrainConfig, state, val_batch) -> dict:
    """Per-prompt-segment attention allocation of the action tokens.

    Segments are recovered from the tokenized prompt itself (the injection
    sentinel plus a piece-level scan), so nothing has to be threaded through the
    Observation. Eager, on a 2-sample probe batch, using EMA params like eval."""
    from ego2g1.train import diagnostics as _diagnostics

    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    obs, actions = jax.tree.map(lambda x: x[: config.probe_batch_size], val_batch)
    out = _diagnostics.attention_allocation(
        model, obs, actions,
        segment_masks=_diagnostics.relation_segment_masks(
            obs, object_prompt_names=tuple(config.object_prompt_names),
            sentinel_id=config.model_config().relation_sentinel_id,
        ),
    )
    per_layer = out["per_layer"]
    payload = {}
    for g, name in enumerate(out["group_names"]):
        key = name.replace("/", "_")
        payload[f"attn/{key}"] = float(per_layer[:, g].mean())
        payload[f"attn_first_layer/{key}"] = float(per_layer[0, g])
        payload[f"attn_last_layer/{key}"] = float(per_layer[-1, g])
    payload["attn/entropy"] = float(out["entropy_per_layer"].mean())
    return payload


# ---------------------------------------------------------------------------
# UmiTrainConfig: state-history tokens + 7-dim rotvec actions, one arm
# ---------------------------------------------------------------------------


@at.typecheck
def umi_train_step(
    config: _openpi_config.TrainConfig,
    ego_config: _config.UmiTrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Stock train_step + the weighted flow loss and the decompositions.

    No auxiliary loss term: this config has no grasp head (the gripper is
    continuous, so there is no binary to calibrate), which makes the total loss
    exactly the weighted flow loss.
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    grip = ego_config.gripper_dims

    @at.typecheck
    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation, actions):
        chunked_loss, aux = model.compute_loss_with_aux(
            rng, observation, actions, train=True, gripper_dims=grip
        )
        return jnp.mean(chunked_loss), (chunked_loss, aux)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, (chunked_loss, aux)), grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True
    )(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    _img = nnx_utils.PathRegex(".*img.*")
    _llm = nnx_utils.PathRegex(".*llm.*")
    _expert1 = nnx_utils.PathRegex(".*llm.*_1.*")
    grad_groups = {
        "grad_norm/siglip": _img,
        "grad_norm/prefix_expert": nnx.All(_llm, nnx.Not(_expert1)),
        "grad_norm/action_expert": _expert1,
        # the history encoder (`relation_encoder` by name, see umi_transforms):
        # if its gradient is orders below the rest, the injected tokens are not
        # being learned
        "grad_norm/history_encoder": nnx_utils.PathRegex(".*relation_encoder.*"),
    }

    info = {
        "loss": loss,
        "loss/flow": jnp.mean(chunked_loss),
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **{k: optax.global_norm(grads.filter(f)) for k, f in grad_groups.items()},
        **{f"loss/{k}": v for k, v in _slot_bucket_means(chunked_loss, config.model.action_horizon).items()},
    }
    if "delta_norm" in aux:
        # rotation_deg is THE canary: gemma.RMSNorm keeps only direction, so
        # rotating the sentinel is the injection's entire effect. Healthy =
        # arctan(alpha) ~ 45 deg at alpha=1.
        info["history/rotation_deg"] = aux["rotation_deg"]
        # most-recent vs most-stale lag. Near 0 means the encoder emits the same
        # token whatever the lag, i.e. the window carries no temporal information
        # at all -- which the relational config's mean-pairwise metric could not
        # distinguish from a healthy smooth sequence.
        info["history/endpoint_sep_deg"] = aux["endpoint_sep_deg"]
        info["history/alpha"] = aux["alpha"]
        info["history/base_norm"] = aux["base_norm"]
        info["history/delta_norm"] = aux["delta_norm"]
        info["history/text_norm"] = aux["text_norm"]
    return new_state, info


def main_umi(config: _config.UmiTrainConfig):
    stock = _load_stock_train_module()
    stock.init_logging()

    from ego2g1.train import dataset as _umi_dataset
    from ego2g1.train import norm as _norm

    model_config_base = config.model_config()
    meta = _umi_dataset.assert_umi_dataset_compatible(
        config.dataset_root, config.expected_config_hash, model_config_base.action_horizon,
        config.fps, config.hand, config.n_lags, max(config.lag_ticks),
    )
    stats_dir = config.assets_dirs / config.repo_id
    stats = _norm.load_umi(stats_dir)

    # The shared-anchor invariant, checked as a number rather than trusted: both
    # the action chunk and the state history are expressed in the frame of
    # pose_history[0], so lag 0's pose dims must be identically zero. If they are
    # not, the two inputs are in different frames and the model would still train
    # to a plausible-looking loss.
    if _norm.lag_zero_pose_is_nonzero(stats):
        raise ValueError(
            "lag 0's pose stats are not identically zero: the action frame and the "
            "history frame have come apart. Recompute stats with this code version "
            "(`python -m ego2g1.train.compute_norm_stats --umi`)."
        )

    # Loss weights are DERIVED from the stats artifact, never hard-coded, so they
    # track the data: variance-normalize every dim, then apply w_gripper. Unlike
    # the relational config the gripper is normalized like everything else here,
    # so w_gripper means what it reads.
    weights = _data_config.loss_dim_weights(
        stats, config.action_dim_actual, config.gripper_dims, config.w_gripper
    )
    model_config = dataclasses.replace(model_config_base, loss_dim_weights=weights)
    logging.info(f"loss_dim_weights (mean 1): {[round(w, 4) for w in weights]}")
    logging.info(f"history: {config.n_lags} tokens at ticks {list(config.lag_ticks)} "
                 f"(t-0 .. t-{max(config.lag_ticks) / config.fps:.2f}s), "
                 f"length probs {list(config.history_len_probs)}")
    logging.info(f"history alpha (injection size as a fraction of the base): "
                 f"{config.history_alpha} -> {math.degrees(math.atan(config.history_alpha)):.1f} deg rotation")

    data_cfg = _data_config.create_umi_data_config(
        config, model_config, stats_dir=stats_dir,
        history_length_probs=config.history_len_probs,
    )
    train_config = _to_openpi_train_config(config, data_cfg)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size {config.batch_size} % devices {jax.device_count()} != 0")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        train_config.checkpoint_dir, keep_period=config.keep_period,
        overwrite=config.overwrite, resume=config.resume,
    )
    best_manager = ocp.CheckpointManager(
        epath.Path(train_config.checkpoint_dir) / "best",
        item_handlers={"assets": _checkpoints.CallbackHandler(), "params": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            best_fn=lambda metrics: float(metrics["val_loss"]),
            best_mode="min",
            keep_checkpoints_without_metrics=False,
            create=True,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )
    stock.init_wandb(train_config, resuming=resuming, enabled=config.wandb_enabled)

    _stamp.write_stamp(train_config.checkpoint_dir, config, meta["config_hash"])
    _norm.save_umi(train_config.checkpoint_dir / "assets_ego2g1", stats)
    _stamp.write_stamp(train_config.checkpoint_dir / "best", config, meta["config_hash"])
    _norm.save_umi(train_config.checkpoint_dir / "best" / "assets_ego2g1", stats)

    torch_dataset = _umi_dataset.create_umi_dataset(config, model_config, split="train")
    transformed = _data_loader.transform_dataset(torch_dataset, data_cfg)
    torch_loader = _data_loader.TorchDataLoader(
        transformed,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    data_loader = _data_loader.DataLoaderImpl(data_cfg, torch_loader)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Validation pools. All four draw the SAME ticks in the same order with the
    # same fixed rng, and all PIN the history length to full: they are asking
    # about the history's CONTENT, and letting the length vary too would
    # confound two questions.
    #   val            full history, in order         — the deployed condition
    #   val_permuted   lag order shuffled             — is lag ORDER read at all?
    #   val_random     another frame's history        — dead reckoning detector
    #   val_nohist     the segment removed entirely   — is the channel used at all?
    #
    # val_permuted is a GATE, not a diagnostic: lag identity is carried only by
    # prompt position (RoPE), with no per-lag text labels. If permuting barely
    # moves the loss, that decision was wrong and the labels are required.
    val_pools: dict[str, list] = {}
    if config.eval_interval > 0 and config.val_source_episodes:
        val_dataset = _umi_dataset.create_umi_dataset(config, model_config, split="val")
        history_pool = _umi_dataset.umi_raw_history(config, split="val", full_only=True)
        full = config.n_lags
        pool_kwargs = {
            "val": {"history_fixed_len": full},
            "val_permuted": {"history_fixed_len": full, "permute_history": True},
            "val_random": {"history_fixed_len": full, "history_pool": history_pool},
            "val_nohist": {"history_fixed_len": 0},
        }
        for prefix, kwargs in pool_kwargs.items():
            pool_cfg = _data_config.create_umi_data_config(
                config, model_config, stats_dir=stats_dir, **kwargs
            )
            val_loader = _data_loader.TorchDataLoader(
                _data_loader.transform_dataset(val_dataset, pool_cfg),
                local_batch_size=config.batch_size // jax.process_count(),
                sharding=data_sharding,
                shuffle=True,
                num_batches=config.eval_num_batches,
                num_workers=0,
                seed=config.seed,
            )
            val_pools[prefix] = list(iter(_data_loader.DataLoaderImpl(pool_cfg, val_loader)))
        logging.info(f"Loaded {config.eval_num_batches} fixed val batches x {len(val_pools)} pools "
                     f"({len(val_dataset)} datapoints from {len(config.val_source_episodes)} episodes; "
                     f"random-history pool {history_pool.shape})")
    elif config.eval_interval > 0:
        logging.warning("eval_interval > 0 but val_source_episodes is empty — validation disabled")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = stock.init_train_state(train_config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(umi_train_step, train_config, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    grip = config.gripper_dims

    def _eval_step(rng, state, batch):
        params = state.ema_params if state.ema_params is not None else state.params
        model = nnx.merge(state.model_def, params)
        model.eval()
        observation, actions = batch
        chunked_loss, _aux = model.compute_loss_with_aux(
            rng, observation, actions, train=False, gripper_dims=grip
        )
        return {"val/loss": jnp.mean(chunked_loss),
                **{f"val/{k}": v for k, v in _slot_bucket_means(chunked_loss, config.action_horizon).items()}}

    peval_step = jax.jit(
        _eval_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    eval_rng = jax.random.key(config.seed + 1)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(range(start_step, config.num_train_steps), initial=start_step,
                     total=config.num_train_steps, dynamic_ncols=True)

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)
        if val_pools and config.eval_interval > 0 and (step % config.eval_interval == 0 or step == config.num_train_steps - 1):
            val_reduced = {}
            for prefix, batches in val_pools.items():
                with sharding.set_mesh(mesh):
                    val_infos = [peval_step(jax.random.fold_in(eval_rng, i), train_state, vb)
                                 for i, vb in enumerate(batches)]
                reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(val_infos)))
                val_reduced.update({k.replace("val/", f"{prefix}/", 1): v for k, v in reduced.items()})
            # The two numbers that decide whether the history feature earned its
            # keep, logged as ratios so they are readable at a glance rather than
            # by eyeballing four curves.
            base = float(val_reduced["val/loss"])
            if base > 0:
                val_reduced["history/permuted_over_real"] = float(val_reduced["val_permuted/loss"]) / base
                val_reduced["history/random_over_real"] = float(val_reduced["val_random/loss"]) / base
                val_reduced["history/nohist_over_real"] = float(val_reduced["val_nohist/loss"]) / base
            pbar.write(f"Step {step} [val]: " + ", ".join(f"{k}={v:.4f}" for k, v in val_reduced.items()))
            wandb.log(val_reduced, step=step)
            _save_best(best_manager, train_state, data_loader, step, base)
        if val_pools and config.probe_interval > 0 and (step % config.probe_interval == 0 or step == config.num_train_steps - 1):
            wandb.log(_umi_attention_probe(config, train_state, val_pools["val"][0]), step=step)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint managers to finish")
    checkpoint_manager.wait_until_finished()
    best_manager.wait_until_finished()


def _umi_attention_probe(config: _config.UmiTrainConfig, state, val_batch) -> dict:
    """Per-prompt-segment and per-CAMERA attention allocation of the action tokens.

    The camera split is free: `diagnostics.token_groups` emits one `img/<slot>`
    group per entry in `obs.images`, so the acting wrist, the static context view
    and the masked-out idle slot are already separate groups. That answers "is
    the context camera being used at all", which for this setup is the same
    question as "is the head-camera stand-in working".

    Segments come from the tokenized prompt itself, so nothing has to be threaded
    through the Observation. Eager, on a small probe batch, using EMA params like
    eval. The probe batch comes from the `val` pool, whose history length is
    pinned to full — which is what makes the per-lag attribution well defined.
    """
    from ego2g1.train import diagnostics as _diagnostics

    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    obs, actions = jax.tree.map(lambda x: x[: config.probe_batch_size], val_batch)
    out = _diagnostics.attention_allocation(
        model, obs, actions,
        segment_masks=_diagnostics.umi_segment_masks(
            obs, sentinel_id=config.model_config().relation_sentinel_id,
            n_lags=config.n_lags, lag_ticks=config.lag_ticks,
        ),
    )
    per_layer = out["per_layer"]
    payload = {}
    for g, name in enumerate(out["group_names"]):
        key = name.replace("/", "_")
        payload[f"attn/{key}"] = float(per_layer[:, g].mean())
        payload[f"attn_first_layer/{key}"] = float(per_layer[0, g])
        payload[f"attn_last_layer/{key}"] = float(per_layer[-1, g])
    payload["attn/entropy"] = float(out["entropy_per_layer"].mean())
    return payload


if __name__ == "__main__":
    import sys

    import tyro

    if "--relation" in sys.argv:
        sys.argv.remove("--relation")
        main_relation(tyro.cli(_config.EgoRelationTrainConfig))
    elif "--umi" in sys.argv:
        sys.argv.remove("--umi")
        main_umi(tyro.cli(_config.UmiTrainConfig))
    else:
        main(tyro.cli(_config.Ego2G1TrainConfig))
