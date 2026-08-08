"""Compute both norm-stats artifacts over the TRAIN split (TRAINING_PLAN.md §3.6).

Mirrors scripts/compute_norm_stats.py semantics (repack + data transforms,
i.e. exactly what precedes Normalize in the training stack) and additionally
accumulates the E001 per-slot sigma over the (H, D_real) action grid.

Run from the openpi root:
    uv run python -m ego2g1.train.compute_norm_stats
"""

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as _normalize

from ego2g1.train import config as _config
from ego2g1.train import data_config as _data_config
from ego2g1.train import dataset as _dataset
from ego2g1.train import norm as _norm


class _PerSlotRunning:
    """Streaming per-(slot, dim) mean/variance (Chan parallel update)."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def update(self, batch: np.ndarray) -> None:  # (N, H, D)
        n = batch.shape[0]
        bmean = batch.mean(axis=0)
        bm2 = ((batch - bmean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, bmean, bm2
            return
        delta = bmean - self.mean
        tot = self.count + n
        self.mean = self.mean + delta * (n / tot)
        self.m2 = self.m2 + bm2 + delta**2 * (self.count * n / tot)
        self.count = tot

    def sigma(self) -> np.ndarray:
        if self.count < 2:
            raise ValueError("need at least 2 samples")
        return np.sqrt(np.maximum(self.m2 / self.count, 0.0))


def main(config: _config.Ego2G1TrainConfig, batch_size: int = 128):
    model_config = config.model_config()
    dataset = _dataset.create_dataset(config, model_config, split="train")
    meta = _dataset.load_extraction_meta(config.dataset_root)

    data_config = _data_config.create_data_config(
        config, model_config, norm_assets_dir=config.assets_dirs / config.repo_id, skip_norm_stats=True
    )
    transforms = [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]

    pooled = {"state": _normalize.RunningStats(), "actions": _normalize.RunningStats()}
    per_slot = _PerSlotRunning()
    raw_min = np.full(config.action_dim_actual, np.inf)
    raw_max = np.full(config.action_dim_actual, -np.inf)

    def flush(buf):
        nonlocal raw_min, raw_max
        actions = np.stack([s["actions"] for s in buf])  # (N, H, D_real)
        states = np.stack([s["state"] for s in buf])
        pooled["actions"].update(actions.reshape(-1, actions.shape[-1]))
        pooled["state"].update(states.reshape(-1, states.shape[-1]))
        per_slot.update(actions)
        flat = actions.reshape(-1, actions.shape[-1])
        raw_min = np.minimum(raw_min, flat.min(axis=0))
        raw_max = np.maximum(raw_max, flat.max(axis=0))

    buf = []
    for i in tqdm.tqdm(range(len(dataset)), desc="Computing stats (train split)"):
        sample = dataset[i]
        for t in transforms:
            sample = t(sample)
        buf.append({"actions": np.asarray(sample["actions"], dtype=np.float64),
                    "state": np.asarray(sample["state"], dtype=np.float64)})
        if len(buf) == batch_size:
            flush(buf)
            buf = []
    if buf:
        flush(buf)

    norm_stats = {k: v.get_statistics() for k, v in pooled.items()}
    per_slot_stats = _norm.PerSlotStats(
        sigma_slot=per_slot.sigma(),
        mu_slot=per_slot.mean,
        provenance={
            "extraction_config_hash": meta["config_hash"],
            "ego2g1_config_hash": config.config_hash(),
            "num_datapoints": per_slot.count,
            "val_real_episodes": list(config.val_real_episodes),
        },
    )

    problems = _norm.check_stats_sanity(
        norm_stats, per_slot_stats, config.degenerate_dim_allowlist,
        raw_min=raw_min, raw_max=raw_max,
    )
    if problems:
        raise SystemExit("Stats sanity check FAILED:\n  " + "\n  ".join(problems))

    # certify + report what the data path will do with these stats
    act = norm_stats["actions"]
    deg = _norm.degenerate_action_dims(act, config.action_dim_actual)
    span = act.q99[: config.action_dim_actual] - act.q01[: config.action_dim_actual] + 1e-6
    n_extreme = np.maximum(
        np.abs((raw_min - act.q01[: config.action_dim_actual]) / span * 2.0 - 1.0),
        np.abs((raw_max - act.q01[: config.action_dim_actual]) / span * 2.0 - 1.0),
    )
    print(f"degenerate dims (neutralized to -1 in the data path): {np.flatnonzero(deg).tolist()}")
    print(f"max |normalized| on live dims: {n_extreme[~deg].max():.1f} "
          f"(dim {int(np.flatnonzero(~deg)[n_extreme[~deg].argmax()])}); "
          f"model-space clamp: {config.model_space_clamp}")

    output_dir = config.assets_dirs / config.repo_id
    print(f"Writing pooled norm_stats.json and {_norm.PER_SLOT_FILENAME} to: {output_dir}")
    _normalize.save(output_dir, norm_stats)
    _norm.save_per_slot(output_dir, per_slot_stats)

    # eyeball report: per-slot sigma + effective E001 boost
    sigma_pooled = norm_stats["actions"].std[: config.action_dim_actual]
    gain = per_slot_stats.gain(config.per_slot_floor_c, sigma_pooled, degenerate_mask=deg)
    with np.printoptions(precision=2, suppress=True, linewidth=200):
        print(f"sigma_slot (H, {config.action_dim_actual}), slots 0/9/24/49:")
        for k in (0, 9, 24, 49):
            print(f"  slot {k:2d}: {per_slot_stats.sigma_slot[k]}")
        print(f"gain grid (c={config.per_slot_floor_c}), max per slot:")
        print(f"  {gain.max(axis=1)}")


def main_relation(config: _config.EgoRelationTrainConfig):
    """Stats for the relational config: per-(slot, dim) action quantiles + pooled
    relation z-score stats, both over the TRAIN split only.

    Reads parquet directly instead of iterating the LeRobot dataset. Neither
    statistic depends on a pixel, so going through the dataset would decode ~19k
    video frames to compute the same numbers; this way it is seconds.
    """
    from ego2g1.train import dataset as _ds

    h = config.action_horizon
    d_real = config.action_dim_actual
    grip = config.gripper_dims

    meta = _ds.assert_relation_dataset_compatible(
        config.dataset_root, config.expected_config_hash, h, config.fps,
        config.n_objects, config.hands,
    )

    print(f"Building relative action chunks from parquet (H={h}, D={d_real}) ...")
    actions = _ds.relation_raw_action_chunks(config, h, split="train")   # (N, H, D)
    relations = _ds.relation_raw_relations(config, split="train")        # (N*n_obj, rel_dim)
    print(f"  actions {actions.shape}  relations {relations.shape}")

    # Exact percentiles: (N, H, D) float64 is ~107 MB at this size, so there is
    # no reason to approximate with a streaming estimator.
    q01 = np.percentile(actions, 1, axis=0)
    q99 = np.percentile(actions, 99, axis=0)

    # What the model will actually see, so the loss weights are derived from the
    # real distribution rather than a Gaussian assumption.
    span = q99 - q01 + 1e-6
    normalized = 2.0 * (actions - q01) / span - 1.0
    if config.model_space_clamp is not None:
        normalized = np.clip(normalized, -config.model_space_clamp, config.model_space_clamp)
    normalized[..., list(grip)] = actions[..., list(grip)]   # grippers pass through at +-1
    model_space_variance = normalized.reshape(-1, d_real).var(axis=0)

    stats = _norm.RelationNormStats(
        action_q01=q01,
        action_q99=q99,
        relation_mean=relations.mean(axis=0),
        relation_std=relations.std(axis=0),
        gripper_dims=grip,
        provenance={
            "extraction_config_hash": meta["config_hash"],
            "ego2g1_config_hash": config.config_hash(),
            "num_chunks": int(actions.shape[0]),
            "num_relation_rows": int(relations.shape[0]),
            "val_source_episodes": list(config.val_source_episodes),
            "action_norm_scheme": config.action_norm_scheme,
            "model_space_variance": model_space_variance.tolist(),
            "gripper_closed_fraction": [
                float((actions[..., d] > 0).mean()) for d in grip
            ],
        },
    )

    problems = _norm.check_relation_stats_sanity(stats)
    if problems:
        raise SystemExit("Stats sanity check FAILED:\n  " + "\n  ".join(problems))

    output_dir = config.assets_dirs / config.repo_id
    print(f"Writing {_norm.RELATION_FILENAME} to: {output_dir}")
    _norm.save_relation(output_dir, stats)

    # ------------------------------------------------------------------ report
    from ego2g1.train import data_config as _dc

    w = _dc.loss_dim_weights(stats, d_real, grip, config.w_gripper)
    labels = [f"{s}_{a}" for s in ("L", "R") for a in ("dx", "dy", "dz", "rx", "ry", "rz")]
    labels += [f"{s}_grip" for s in ("L", "R")]
    with np.printoptions(precision=4, suppress=True, linewidth=200):
        print(f"\nmodel-space std per slot (target: uniform; a pooled scheme gives 17-24x spread):")
        for k in (0, 1, 9, 24, h - 1):
            print(f"  slot {k:2d}: {normalized[:, k, :].std(axis=0)}")
        print(f"\nmax |normalized| on non-gripper dims: "
              f"{np.abs(np.delete(normalized, list(grip), axis=-1)).max():.2f} "
              f"(clamp {config.model_space_clamp})")
        print(f"\nrelation z-score gains (1/std): {1.0 / np.maximum(stats.relation_std, 1e-9)}")
        print(f"relation |z|>{config.state_norm_clip} fraction: "
              f"{float((np.abs((relations - stats.relation_mean) / stats.relation_std) > config.state_norm_clip).mean()) * 100:.4f}%")
    print(f"\nloss_dim_weights (w_gripper={config.w_gripper}, mean 1):")
    for name, wi, v in zip(labels, w, model_space_variance, strict=True):
        print(f"  {name:>7s}  Var(x1)={v:7.4f}  Var(u)={1 + v:7.4f}  w={wi:7.4f}")
    tot = sum(wi * (1 + v) for wi, v in zip(w, model_space_variance, strict=True))
    gshare = sum(w[d] * (1 + model_space_variance[d]) for d in grip) / tot
    print(f"  -> gripper share of weighted MSE: {gshare * 100:.2f}% "
          f"(intent {2 * config.w_gripper / (12 + 2 * config.w_gripper) * 100:.2f}%)")


def main_umi(config: _config.UmiTrainConfig):
    """Stats for the UMI config: per-(slot, dim) action quantiles + PER-LAG
    history z-score stats, both over the TRAIN split only.

    Reads parquet directly (see dataset._umi_episode_frames): neither statistic
    depends on a pixel, so iterating the LeRobot dataset would decode ~41k video
    frames to compute the same numbers.
    """
    from ego2g1.train import dataset as _ds

    h = config.action_horizon
    d_real = config.action_dim_actual
    grip = config.gripper_dims

    meta = _ds.assert_umi_dataset_compatible(
        config.dataset_root, config.expected_config_hash, h, config.fps,
        config.hand, config.n_lags, max(config.lag_ticks),
    )

    print(f"Building relative action chunks from parquet (H={h}, D={d_real}) ...")
    actions = _ds.umi_raw_action_chunks(config, h, split="train")        # (N, H, 7)
    # full_only: lag j's stats must come from anchors where lag j is REAL, not
    # from LeRobot's clamped duplicates of the episode's first frame
    history = _ds.umi_raw_history(config, split="train", full_only=True)  # (M, n_lags, 7)
    print(f"  actions {actions.shape}  history {history.shape}")

    # Exact percentiles: float64 at this size is well under a GB, so there is no
    # reason to approximate with a streaming estimator.
    q01 = np.percentile(actions, 1, axis=0)
    q99 = np.percentile(actions, 99, axis=0)

    # What the model will actually see, so the loss weights are derived from the
    # real distribution rather than a Gaussian assumption. Every dim including
    # the gripper: it is continuous here, so nothing is exempt.
    span = q99 - q01 + 1e-6
    normalized = 2.0 * (actions - q01) / span - 1.0
    if config.model_space_clamp is not None:
        normalized = np.clip(normalized, -config.model_space_clamp, config.model_space_clamp)
    model_space_variance = normalized.reshape(-1, d_real).var(axis=0)

    # Scalar quantiles of the OBSERVED gripper, for state_mode="gripper_token".
    # Taken from lag 0 of the history (the anchor tick's gripper), which is the
    # exact value the prompt digitizes at train and serve time. Computed
    # unconditionally so one stats artifact serves BOTH modes and switching
    # state_mode needs no recompute.
    observed_grip = history[:, 0, -1]
    g01, g99 = (float(np.percentile(observed_grip, 1)),
                float(np.percentile(observed_grip, 99)))

    stats = _norm.UmiNormStats(
        action_q01=q01,
        action_q99=q99,
        history_mean=history.mean(axis=0),
        history_std=history.std(axis=0),
        gripper_dims=grip,
        gripper_q01=g01,
        gripper_q99=g99,
        provenance={
            "extraction_config_hash": meta["config_hash"],
            "ego2g1_config_hash": config.config_hash(),
            "num_chunks": int(actions.shape[0]),
            "num_history_blocks": int(history.shape[0]),
            "val_source_episodes": list(config.val_source_episodes),
            "action_norm_scheme": config.action_norm_scheme,
            "lag_ticks": list(config.lag_ticks),
            "model_space_variance": model_space_variance.tolist(),
            "gripper_raw_range": [float(actions[..., grip[0]].min()),
                                  float(actions[..., grip[0]].max())],
        },
    )

    problems = _norm.check_umi_stats_sanity(stats)
    if problems:
        raise SystemExit("Stats sanity check FAILED:\n  " + "\n  ".join(problems))

    output_dir = config.assets_dirs / config.repo_id
    print(f"Writing {_norm.UMI_FILENAME} to: {output_dir}")
    _norm.save_umi(output_dir, stats)

    # ------------------------------------------------------------------ report
    from ego2g1.train import data_config as _dc

    w = _dc.loss_dim_weights(stats, d_real, grip, config.w_gripper)
    labels = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
    with np.printoptions(precision=4, suppress=True, linewidth=200):
        print("\nmodel-space std per slot (target: uniform; a pooled scheme would leave")
        print("slot 0 at ~1/30 unit scale on this dataset):")
        for k in (0, 1, 9, 24, h - 1):
            print(f"  slot {k:2d}: {normalized[:, k, :].std(axis=0)}")
        print(f"\nmax |normalized|: {np.abs(normalized).max():.2f} "
              f"(clamp {config.model_space_clamp})")
        print(f"\nper-lag history std (lag 0's pose dims are structurally 0):")
        for j, tick in enumerate(config.lag_ticks):
            print(f"  lag {j} (t-{tick:2d}): {stats.history_std[j]}")
        z = (history - stats.history_mean) / np.maximum(stats.history_std, 1e-6)
        print(f"\nhistory |z|>{config.state_norm_clip} fraction: "
              f"{float((np.abs(z) > config.state_norm_clip).mean()) * 100:.4f}%")
    # state_mode="gripper_token": how the observed gripper lands in the bins
    from ego2g1.train import umi_transforms as _ut

    bins = config.gripper_bins
    used = np.array([_ut.digitize_gripper(v, g01, g99, bins) for v in observed_grip])
    print(f"\ngripper state quantiles: q01={g01:.4f} q99={g99:.4f} "
          f"(raw range {observed_grip.min():.4f}..{observed_grip.max():.4f})")
    print(f"  -> {bins} bins: {len(np.unique(used))} distinct occupied, "
          f"range {used.min()}..{used.max()}; "
          f"{float((used == 0).mean()) * 100:.1f}% in bin 0, "
          f"{float((used == bins - 1).mean()) * 100:.1f}% in bin {bins - 1}")
    print(f"\nloss_dim_weights (w_gripper={config.w_gripper}, mean 1):")
    for name, wi, v in zip(labels, w, model_space_variance, strict=True):
        print(f"  {name:>5s}  Var(x1)={v:7.4f}  Var(u)={1 + v:7.4f}  w={wi:7.4f}")
    tot = sum(wi * (1 + v) for wi, v in zip(w, model_space_variance, strict=True))
    gshare = sum(w[d] * (1 + model_space_variance[d]) for d in grip) / tot
    print(f"  -> gripper share of weighted MSE: {gshare * 100:.2f}% "
          f"(intent {config.w_gripper / (6 + config.w_gripper) * 100:.2f}%)")


if __name__ == "__main__":
    import sys

    if "--relation" in sys.argv:
        sys.argv.remove("--relation")
        main_relation(tyro.cli(_config.EgoRelationTrainConfig))
    elif "--umi" in sys.argv:
        sys.argv.remove("--umi")
        main_umi(tyro.cli(_config.UmiTrainConfig))
    else:
        main(tyro.cli(_config.Ego2G1TrainConfig))
