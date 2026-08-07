"""Dataset construction: LeRobot dataset + sidecar assert + boundary remap +
train/val split by REAL episode.

Replaces the reverted fork commit's data_loader.py edits: we build the
dataset ourselves and hand it to stock transform_dataset/TorchDataLoader.
"""

import dataclasses
import hashlib
import json
import pathlib

import numpy as np

from ego2g1.core import chunk_math


def load_extraction_meta(dataset_root) -> dict:
    """Read the extraction_meta.json sidecar written next to the dataset."""
    return json.loads((pathlib.Path(dataset_root) / "extraction_meta.json").read_text())


def assert_dataset_compatible(dataset_root, expected_config_hash: str | None, action_horizon: int, fps: int) -> dict:
    """Fail loud before GPU time: sidecar exists, hash matches, and the
    extraction config's horizon/fps agree with the training config."""
    meta = load_extraction_meta(dataset_root)
    if expected_config_hash is not None and meta["config_hash"] != expected_config_hash:
        raise ValueError(
            f"Dataset at {dataset_root} has extraction config_hash {meta['config_hash']}, "
            f"expected {expected_config_hash}. Regenerate the dataset or update the training config."
        )
    cfg = meta["config"]
    if int(cfg["action_horizon"]) < int(action_horizon):
        raise ValueError(f"extraction horizon {cfg['action_horizon']} < training horizon {action_horizon}")
    if float(cfg["control_hz"]) != float(fps):
        raise ValueError(f"extraction control_hz {cfg['control_hz']} != training fps {fps}")
    return meta


def _episode_lengths(dataset_root) -> list[int]:
    lengths = {}
    with (pathlib.Path(dataset_root) / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            lengths[int(rec["episode_index"])] = int(rec["length"])
    return [lengths[i] for i in range(len(lengths))]


@dataclasses.dataclass(frozen=True)
class SplitIndices:
    """Boundary-aware flat indices, split train/val by real source episode."""

    train: chunk_math.BoundaryAwareIndices
    val: chunk_math.BoundaryAwareIndices


def build_split_indices(dataset_root, action_horizon, val_real_episodes=(), allow_terminal_padding=True,
                        meta=None) -> SplitIndices:
    """Boundary-aware indices per subset. A LeRobot episode belongs to val iff
    its sidecar `source_episode` is in `val_real_episodes` — a real episode
    that was filter-split into several LeRobot episodes can never straddle
    the split.

    `meta` overrides the on-disk sidecar, which is how the relational dataset
    reuses this: its sidecar is a different schema, so `relation_extraction_meta`
    adapts it in memory rather than either rewriting the dataset or forking this
    function."""
    meta = load_extraction_meta(dataset_root) if meta is None else meta
    lengths = _episode_lengths(dataset_root)
    n = len(lengths)
    eps = [meta["episodes"][str(i)] for i in range(n)]
    known_sources = {e["source_episode"] for e in eps}
    unknown = set(val_real_episodes) - known_sources
    if unknown:
        raise ValueError(f"val_real_episodes not in dataset: {sorted(unknown)}")

    def indices_for(subset_is_val: bool) -> chunk_math.BoundaryAwareIndices:
        # Zero out non-subset episodes' valid ranges by marking every frame
        # anchor_bad; lengths/offsets stay global so flat indices are correct.
        anchor_bad = []
        for e, length in zip(eps, lengths):
            in_val = e["source_episode"] in val_real_episodes
            if in_val == subset_is_val:
                anchor_bad.append(list(e.get("anchor_bad", [])))
            else:
                anchor_bad.append(list(range(length)))
        return chunk_math.BoundaryAwareIndices(
            lengths,
            [bool(e["episode_real_end"]) for e in eps],
            action_horizon,
            allow_terminal_padding,
            anchor_bad=anchor_bad,
        )

    return SplitIndices(train=indices_for(False), val=indices_for(True))


def raw_state_pool(train_config, *, split: str = "val") -> np.ndarray:
    """Every raw 30-dim state of a split's episodes, read straight from the
    parquet — no video decode, so it costs milliseconds. Feeds the shuffled-state
    val pool (ego2g1.transforms.ShuffleState)."""
    import pandas as pd

    root = pathlib.Path(train_config.dataset_root).resolve()
    meta = load_extraction_meta(root)
    val_sources = set(train_config.val_real_episodes)
    frames = []
    for idx_str, e in meta["episodes"].items():
        i = int(idx_str)
        if (e["source_episode"] in val_sources) != (split == "val"):
            continue
        path = root / "data" / f"chunk-{i // 1000:03d}" / f"episode_{i:06d}.parquet"
        frames.append(np.stack(pd.read_parquet(path, columns=["state"])["state"].to_numpy()))
    if not frames:
        raise ValueError(f"split {split!r} has no episodes — cannot build a state pool")
    return np.concatenate(frames).astype(np.float32)


def create_dataset(train_config, model_config, *, split: str = "train"):
    """LeRobot dataset with pose/hand delta_timestamps, wrapped to expose only
    the boundary-valid datapoints of the requested split. Emits raw samples;
    all transforms (including RelativeChunkActions) are applied by stock
    transform_dataset from the DataConfig (ego2g1.data_config)."""
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    root = pathlib.Path(train_config.dataset_root).resolve()
    assert_dataset_compatible(root, train_config.expected_config_hash,
                              model_config.action_horizon, train_config.fps)

    dataset = lerobot_dataset.LeRobotDataset(
        train_config.repo_id,
        root=str(root),
        delta_timestamps=chunk_math.make_delta_timestamps(model_config.action_horizon, train_config.fps),
    )
    splits = build_split_indices(root, model_config.action_horizon, train_config.val_real_episodes)
    indices = splits.train if split == "train" else splits.val
    if len(indices) == 0:
        raise ValueError(f"split {split!r} has zero valid datapoints")

    # match stock create_torch_dataset's prompt_from_task behavior
    import openpi.transforms as _transforms
    import openpi.training.data_loader as _data_loader

    meta = lerobot_dataset.LeRobotDatasetMetadata(train_config.repo_id, root=str(root))
    wrapped = chunk_math.BoundaryAwareDataset(dataset, indices)
    return _data_loader.TransformedDataset(wrapped, [_transforms.PromptFromLeRobotTask(meta.tasks)])


# --------------------------------------------------------------------------
# UmiTrainConfig: red_block_on_yellow_block_umi and its schema
# --------------------------------------------------------------------------

# The dataset directory name, used to build `source_episode` strings — which is
# what val_source_episodes matches on. One LeRobot episode per recording here,
# so the mapping is the identity.
UMI_RAW_DIR = "red_block_on_yellow_block_umi"


def umi_extraction_meta(dataset_root) -> dict:
    """Adapt the UMI dataset to the ego2g1 sidecar schema, in memory.

    This dataset ships no `extraction_meta.json` at all, so every field below is
    synthesized — and each is a TRUE statement about it rather than a
    placeholder:

    - `config_hash`: derived by hashing the schema-defining parts of
      `info.json` (fps, robot_type, and the feature shapes). Same guarantee as a
      real one — a differently-shaped re-export yields a different digest — just
      computed on our side.
    - `source_episode`: 117 episodes, one per recording, so LeRobot episode i is
      `episode_i`. There is no source-path list to read, so the index IS the
      identity.
    - `episode_real_end = True`: correct. Nothing was filter-split; the end of
      each episode really is the end of a take, so terminal padding legitimately
      means "hold at the final pose".
    - `anchor_bad = [0]`: LOAD-BEARING, not bookkeeping. `action[t]` is the state
      at t+1 (see umi_transforms' module docstring), so the absolute pose at tick
      t is `action[t-1]` and tick 0 has no real pose at all — LeRobot would clamp
      the gather and hand back tick 1's pose silently. Excluding it costs 117 of
      41207 frames and removes an entire class of off-by-one corruption.
    """
    root = pathlib.Path(dataset_root)
    sidecar = root / "extraction_meta.json"
    if sidecar.exists():
        raw = json.loads(sidecar.read_text())
        if "episodes" in raw and "config_hash" in raw:
            return raw   # already ego2g1-schema; nothing to adapt

    info = json.loads((root / "meta" / "info.json").read_text())
    lengths = _episode_lengths(root)
    schema = {
        "fps": info["fps"],
        "robot_type": info.get("robot_type"),
        "features": {k: v.get("shape") for k, v in info["features"].items()},
    }
    config_hash = hashlib.sha256(
        json.dumps(schema, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    episodes = {
        str(i): {
            "source_file": f"{UMI_RAW_DIR}/episode_{i}",
            "source_episode": f"{UMI_RAW_DIR}/episode_{i}",
            "tick_start": 0,
            "tick_end": int(length),
            "episode_real_end": True,
            "anchor_bad": [0],
        }
        for i, length in enumerate(lengths)
    }
    return {
        "config_hash": config_hash,
        "config": schema,
        "episodes": episodes,
        "schema_version": info.get("codebase_version"),
        "variant": "umi",
        "adapted_by": "ego2g1.train.dataset.umi_extraction_meta",
    }


def _assert_umi_action_is_one_tick_ahead(dataset_root, n_episodes: int = 3) -> None:
    """Verify `action[t, 9] == observation.state[t+1, 0]` on a few episodes.

    THE assumption the whole config rests on. Everything -- the anchor, the pose
    history, the gripper history, the chunk targets -- is read out of `action` at
    offsets derived from "action[t] describes tick t+1". A re-export that changed
    that alignment would shift every input by one tick and nothing downstream
    would notice: the shapes stay right, the stats stay plausible, and the policy
    just learns a slightly wrong world.

    Measured 0.0 across all 117 episodes / 41207 frames on the dataset this was
    written against, so an exact comparison is the honest test. Checked on a few
    episodes rather than all of them because this runs before every training job.
    """
    import pandas as pd

    root = pathlib.Path(dataset_root)
    n = len(_episode_lengths(root))
    for i in range(min(n_episodes, n)):
        path = root / "data" / f"chunk-{i // 1000:03d}" / f"episode_{i:06d}.parquet"
        df = pd.read_parquet(path, columns=["action", "observation.state"])
        act = np.stack(df["action"].to_numpy()).astype(np.float64)
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float64).reshape(-1)
        err = float(np.abs(act[:-1, 9] - state[1:]).max())
        if err != 0.0:
            raise ValueError(
                f"episode {i}: action[t, 9] != observation.state[t+1, 0] (max err {err:g}). "
                "This config reads the anchor, the pose history and the gripper history "
                "out of `action` assuming action[t] describes tick t+1 "
                "(ego2g1.train.umi_transforms module docstring). That assumption does not "
                "hold for this export, so every input would be shifted by one tick."
            )


def assert_umi_dataset_compatible(dataset_root, expected_config_hash, action_horizon, fps,
                                  hand="right", n_lags=6, history_span=15) -> dict:
    """Fail loud before GPU time: schema widths, cameras, fps, horizon, lengths.

    Checks the FEATURE WIDTHS and CAMERA KEYS from info.json rather than
    trusting the config: a silently different export would otherwise surface as
    a reshape error deep inside a dataloader worker, or — worse — as a plausible
    loss on a wrong layout.
    """
    root = pathlib.Path(dataset_root)
    meta = umi_extraction_meta(root)
    if expected_config_hash is not None and meta["config_hash"] != expected_config_hash:
        raise ValueError(
            f"Dataset at {dataset_root} has (derived) schema config_hash "
            f"{meta['config_hash']}, expected {expected_config_hash}."
        )
    info = json.loads((root / "meta" / "info.json").read_text())
    if float(info["fps"]) != float(fps):
        raise ValueError(f"dataset fps {info['fps']} != training fps {fps}")

    feats = info["features"]
    got_state = int(feats["observation.state"]["shape"][0])
    got_action = int(feats["action"]["shape"][0])
    if got_state != 1:
        raise ValueError(
            f"observation.state is {got_state}-dim, expected 1 (the continuous gripper). "
            "A wider state means this export carries proprioception this config does not "
            "know how to read."
        )
    if got_action != 10:
        raise ValueError(
            f"action is {got_action}-dim, expected 10 (absolute TCP vec9 + gripper)"
        )
    _assert_umi_action_is_one_tick_ahead(root)
    other = "left" if hand == "right" else "right"
    for key in (f"observation.images.cam_{hand}_wrist", f"observation.images.cam_{other}_wrist"):
        if key not in feats:
            raise ValueError(
                f"missing camera feature {key!r}; this config needs BOTH the acting "
                f"wrist camera and the context camera (have: "
                f"{sorted(k for k in feats if k.startswith('observation.images.'))})"
            )

    # An episode must be long enough to hold the backward history AND the
    # forward chunk from at least one anchor, or it contributes nothing but
    # padded windows.
    shortest = min(_episode_lengths(root))
    if shortest <= action_horizon:
        raise ValueError(
            f"shortest episode is {shortest} frames <= action_horizon {action_horizon}: "
            "it would contribute only padded chunks"
        )
    if shortest <= history_span + 1:
        raise ValueError(
            f"shortest episode is {shortest} frames <= the history span {history_span} + 1: "
            f"no anchor in it could ever see a full {n_lags}-lag window"
        )
    return meta


def create_umi_dataset(train_config, model_config, *, split: str = "train"):
    """LeRobot dataset for the UMI schema, wrapped to the requested split.

    `action` is gathered in BOTH directions (backward for the pose history,
    forward for the chunk) because it is this dataset's only pose column; see
    umi_transforms.make_delta_timestamps.
    """
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    from ego2g1.train import umi_transforms as _ut

    root = pathlib.Path(train_config.dataset_root).resolve()
    lags = train_config.lag_ticks
    meta = assert_umi_dataset_compatible(
        root, train_config.expected_config_hash, model_config.action_horizon,
        train_config.fps, train_config.hand, train_config.n_lags, max(lags),
    )

    kwargs = {}
    if getattr(train_config, "video_backend", None):
        kwargs["video_backend"] = train_config.video_backend
    dataset = lerobot_dataset.LeRobotDataset(
        train_config.repo_id,
        root=str(root),
        delta_timestamps=_ut.make_delta_timestamps(
            model_config.action_horizon, train_config.fps, lags),
        **kwargs,
    )
    splits = build_split_indices(
        root, model_config.action_horizon, train_config.val_source_episodes, meta=meta,
    )
    indices = splits.train if split == "train" else splits.val
    if len(indices) == 0:
        raise ValueError(f"split {split!r} has zero valid datapoints")

    import openpi.training.data_loader as _data_loader
    import openpi.transforms as _transforms

    lr_meta = lerobot_dataset.LeRobotDatasetMetadata(train_config.repo_id, root=str(root))
    wrapped = chunk_math.BoundaryAwareDataset(dataset, indices)
    return _data_loader.TransformedDataset(wrapped, [_transforms.PromptFromLeRobotTask(lr_meta.tasks)])


def _umi_episode_frames(train_config, split: str):
    """Yield (episode index, `action` rows (T, 10)) per episode of a split.

    Only `action` is read, mirroring the training path exactly: it carries the
    poses (cols 0..8) and the gripper (col 9), and `action[u-1]` is the state at
    tick u. `observation.state` is the same gripper numbers one tick over and is
    never needed (umi_transforms.make_delta_timestamps).

    Straight from parquet: neither the action stats nor the history stats depend
    on a pixel, so going through the LeRobot dataset would decode ~41k video
    frames to compute the same numbers. Seconds instead of hours.
    """
    import pandas as pd

    root = pathlib.Path(train_config.dataset_root).resolve()
    meta = umi_extraction_meta(root)
    val = set(train_config.val_source_episodes)
    for idx_str, ep in meta["episodes"].items():
        i = int(idx_str)
        if (ep["source_episode"] in val) != (split == "val"):
            continue
        path = root / "data" / f"chunk-{i // 1000:03d}" / f"episode_{i:06d}.parquet"
        df = pd.read_parquet(path, columns=["action"])
        yield i, np.stack(df["action"].to_numpy()).astype(np.float64)


def umi_raw_action_chunks(train_config, action_horizon, *, split: str = "train") -> np.ndarray:
    """Every raw 7-dim relative action chunk of a split, from parquet only. (N, H, 7).

    Reproduces exactly what training will see, including the two boundary rules:
    anchors start at t = 1 (tick 0 has no real anchor — see umi_extraction_meta),
    and a window running past the end repeats the final action, which for these
    anchor-relative targets means "hold at the final pose". Same padding LeRobot
    applies, reproduced here so the stats see the training distribution.
    """
    from ego2g1.train import umi_transforms as _ut

    tf = _ut.UmiRelativeActions()
    chunks = []
    for _, act in _umi_episode_frames(train_config, split):
        length = len(act)
        pad = np.repeat(act[-1:], action_horizon, axis=0)
        padded = np.concatenate([act, pad], axis=0)
        for t in range(1, length):
            chunks.append(tf({
                # anchor = absolute pose at tick t = action[t-1][:9]
                "observation/pose_history": act[t - 1][None, :9],
                "observation/targets": padded[t: t + action_horizon],
            })["actions"])
    if not chunks:
        raise ValueError(f"split {split!r} has no episodes")
    out = np.stack(chunks).astype(np.float64)
    if out.shape[-1] != 7:
        raise ValueError(f"built {out.shape[-1]}-dim actions, expected 7")
    return out


def umi_raw_history(train_config, *, split: str = "train", full_only: bool = True) -> np.ndarray:
    """Every raw (n_lags, 7) history block of a split, from parquet only.

    (N, n_lags, 7). With `full_only` (the default) only anchors whose whole
    window exists are returned — which is what the PER-LAG stats must be
    computed over: lag j's mean/std has to come from samples where lag j is
    real, not from clamped duplicates of the episode's first frame.

    Also feeds the random-history val pool (umi_transforms.UmiStateHistory.pool),
    where full blocks are likewise the right thing to draw from.
    """
    from ego2g1.train import umi_transforms as _ut

    lags = train_config.lag_ticks
    tf = _ut.UmiStateHistory()
    span = max(lags)
    blocks = []
    for _, act in _umi_episode_frames(train_config, split):
        length = len(act)
        # anchor t needs action[t-1-lag] for every lag, i.e. t >= span + 1
        start = span + 1 if full_only else 1
        for t in range(start, length):
            # ONE gather, exactly as the training path does it: pose and gripper
            # come out of the same rows, so they cannot describe different ticks
            rows = np.stack([act[t - 1 - l] for l in lags])                  # (n_lags, 10)
            blocks.append(tf({
                "observation/pose_history": rows[:, :9],
                "observation/gripper_history": rows[:, 9:],
            })["history"])
    if not blocks:
        raise ValueError(f"split {split!r} has no episodes")
    return np.stack(blocks).astype(np.float64)


# --------------------------------------------------------------------------
# EgoRelationTrainConfig: red_block_in_pen_holder_ego and its schema
# --------------------------------------------------------------------------

# The raw recording directory the 50 source HDF5 files came from. Only used to
# build `source_episode` strings, which is what val_source_episodes matches on.
RELATION_RAW_DIR = "red_block_in_pen_holder_50"


def relation_extraction_meta(dataset_root) -> dict:
    """Adapt the relational dataset's sidecar to the ego2g1 schema, in memory.

    That dataset was written by a different pipeline, so its
    `extraction_meta.json` is `{schema_version, variant, config, sources}` — no
    `config_hash`, no per-episode block, no `episode_real_end`, no `anchor_bad`.
    Every field synthesized below is a TRUE statement about that dataset rather
    than a placeholder:

    - `config_hash`: that pipeline stamps no hash of its own, so one is derived
      here by hashing its config block. Same guarantee (a different extraction
      config yields a different digest), just computed on our side.
    - `source_episode`: the sources list is 50 paths, verified contiguous and
      ascending, so LeRobot episode i came from `episode_{i+1}.hdf5`. Read from
      the list rather than assumed, and cross-checked below.
    - `episode_real_end = True`: correct. One LeRobot episode per recording,
      whole; nothing was filter-split, so the end of each episode really is the
      end of a take and terminal padding legitimately means "hold still".
    - `anchor_bad = []`: correct. That pipeline quarantines whole episodes rather
      than flagging ticks, and exports no per-frame validity column, so no tick
      is marked bad. (Its own gates did run: measured on this dataset, zero
      frames exceed the 2.0 m/s or 720 deg/s limits it configures.)
    """
    root = pathlib.Path(dataset_root)
    raw = json.loads((root / "extraction_meta.json").read_text())
    if "episodes" in raw and "config_hash" in raw:
        return raw   # already ego2g1-schema; nothing to adapt

    lengths = _episode_lengths(root)
    sources = list(raw.get("sources", []))
    if len(sources) != len(lengths):
        raise ValueError(
            f"sidecar lists {len(sources)} sources but the dataset has {len(lengths)} episodes"
        )
    config_hash = hashlib.sha256(
        json.dumps(raw.get("config", {}), sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    episodes = {}
    for i, (src, length) in enumerate(zip(sources, lengths, strict=True)):
        stem = pathlib.Path(src).stem            # "episode_7"
        episodes[str(i)] = {
            "source_file": src,
            "source_episode": f"{RELATION_RAW_DIR}/{stem}",
            "tick_start": 0,
            "tick_end": int(length),
            "episode_real_end": True,
            "anchor_bad": [],
        }
    return {
        "config_hash": config_hash,
        "config": raw.get("config", {}),
        "episodes": episodes,
        "schema_version": raw.get("schema_version"),
        "variant": raw.get("variant"),
        "adapted_by": "ego2g1.train.dataset.relation_extraction_meta",
    }


def assert_relation_dataset_compatible(dataset_root, expected_config_hash, action_horizon, fps, n_objects,
                                       hands=("left", "right")) -> dict:
    """Fail loud before GPU time: schema widths, fps, horizon, provenance.

    Checks the FEATURE WIDTHS from info.json rather than trusting the config,
    because a silently different object count would otherwise surface as a
    reshape error deep inside a dataloader worker.
    """
    root = pathlib.Path(dataset_root)
    meta = relation_extraction_meta(root)
    if expected_config_hash is not None and meta["config_hash"] != expected_config_hash:
        raise ValueError(
            f"Dataset at {dataset_root} has (derived) extraction config_hash "
            f"{meta['config_hash']}, expected {expected_config_hash}."
        )
    info = json.loads((root / "meta" / "info.json").read_text())
    if float(info["fps"]) != float(fps):
        raise ValueError(f"dataset fps {info['fps']} != training fps {fps}")

    feats = info["features"]
    n_hands = len(hands)
    want_state = 9 * n_hands * n_objects + n_hands
    want_action = 9 * n_hands + n_hands
    got_state = int(feats["observation.state"]["shape"][0])
    got_action = int(feats["action"]["shape"][0])
    got_ref = int(feats["observation.action_reference_tcp"]["shape"][0])
    if got_state != want_state:
        raise ValueError(
            f"observation.state is {got_state}-dim, expected {want_state} "
            f"(= 9*{n_hands} hands*{n_objects} objects + {n_hands} grasp)"
        )
    if got_action != want_action:
        raise ValueError(f"action is {got_action}-dim, expected {want_action}")
    if got_ref != 9 * n_hands:
        raise ValueError(f"observation.action_reference_tcp is {got_ref}-dim, expected {9 * n_hands}")

    shortest = min(_episode_lengths(root))
    if shortest <= action_horizon:
        raise ValueError(
            f"shortest episode is {shortest} frames <= action_horizon {action_horizon}: "
            "it would contribute only padded chunks"
        )
    return meta


def create_relation_dataset(train_config, model_config, *, split: str = "train"):
    """LeRobot dataset for the relational schema, wrapped to the requested split.

    Only `action` gets delta_timestamps: `action[t]` already holds the target at
    t+1, so a window of H actions from anchor t covers targets t+1..t+H. The
    anchor's own reference pose and state are that frame's values.
    """
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    from ego2g1.train import relation_transforms as _rt

    root = pathlib.Path(train_config.dataset_root).resolve()
    meta = assert_relation_dataset_compatible(
        root, train_config.expected_config_hash, model_config.action_horizon,
        train_config.fps, train_config.n_objects, train_config.hands,
    )

    kwargs = {}
    if getattr(train_config, "video_backend", None):
        kwargs["video_backend"] = train_config.video_backend
    dataset = lerobot_dataset.LeRobotDataset(
        train_config.repo_id,
        root=str(root),
        delta_timestamps=_rt.make_delta_timestamps(model_config.action_horizon, train_config.fps),
        **kwargs,
    )
    splits = build_split_indices(
        root, model_config.action_horizon, train_config.val_source_episodes, meta=meta,
    )
    indices = splits.train if split == "train" else splits.val
    if len(indices) == 0:
        raise ValueError(f"split {split!r} has zero valid datapoints")

    import openpi.training.data_loader as _data_loader
    import openpi.transforms as _transforms

    lr_meta = lerobot_dataset.LeRobotDatasetMetadata(train_config.repo_id, root=str(root))
    wrapped = chunk_math.BoundaryAwareDataset(dataset, indices)
    return _data_loader.TransformedDataset(wrapped, [_transforms.PromptFromLeRobotTask(lr_meta.tasks)])


def relation_raw_relations(train_config, *, split: str = "train") -> np.ndarray:
    """Every raw per-object relation row of a split, straight from the parquet.

    (N * n_objects, relation_dim) — pooled across objects, which is the shape the
    z-score stats must be computed over (see norm.RelationNormStats). No video
    decode, so it costs milliseconds.
    """
    import pandas as pd

    root = pathlib.Path(train_config.dataset_root).resolve()
    meta = relation_extraction_meta(root)
    val = set(train_config.val_source_episodes)
    n_obj, n_hands = train_config.n_objects, len(train_config.hands)
    per_hand = 9 * n_obj

    rows = []
    for idx_str, ep in meta["episodes"].items():
        i = int(idx_str)
        if (ep["source_episode"] in val) != (split == "val"):
            continue
        path = root / "data" / f"chunk-{i // 1000:03d}" / f"episode_{i:06d}.parquet"
        state = np.stack(pd.read_parquet(path, columns=["observation.state"])["observation.state"].to_numpy())
        for k in range(n_obj):
            rows.append(np.concatenate(
                [state[:, h * per_hand + 9 * k: h * per_hand + 9 * k + 9] for h in range(n_hands)],
                axis=-1,
            ))
    if not rows:
        raise ValueError(f"split {split!r} has no episodes")
    return np.concatenate(rows).astype(np.float64)


def relation_raw_action_chunks(train_config, action_horizon, *, split: str = "train") -> np.ndarray:
    """Every raw 14-dim relative action chunk of a split, built from parquet only.

    (N, H, 14). Deliberately bypasses the LeRobot dataset: the action stats need
    `action` and `observation.action_reference_tcp`, both of which live in the
    parquet, so going through the dataset would decode ~19k video frames to
    compute statistics that do not depend on a single pixel. Seconds instead of
    hours, and identical numbers.

    Boundary rule matches BoundaryAwareIndices with terminal padding: an anchor t
    is valid for all t < length, and a window running past the end repeats the
    final action -- which, for these anchor-relative targets, means "hold at the
    final pose". That is the same padding LeRobot applies, reproduced here so the
    stats see exactly the distribution training will.
    """
    import pandas as pd

    from ego2g1.train import relation_transforms as _rt

    root = pathlib.Path(train_config.dataset_root).resolve()
    meta = relation_extraction_meta(root)
    val = set(train_config.val_source_episodes)
    tf = _rt.RelativeEEFRotvecActions(hands=tuple(train_config.hands))
    n_hands = len(train_config.hands)

    chunks = []
    for idx_str, ep in meta["episodes"].items():
        i = int(idx_str)
        if (ep["source_episode"] in val) != (split == "val"):
            continue
        path = root / "data" / f"chunk-{i // 1000:03d}" / f"episode_{i:06d}.parquet"
        df = pd.read_parquet(path, columns=["action", "observation.action_reference_tcp"])
        act = np.stack(df["action"].to_numpy()).astype(np.float64)
        ref = np.stack(df["observation.action_reference_tcp"].to_numpy()).astype(np.float64)
        length = len(act)
        # repeat-pad the tail so every anchor yields a full H-slot window
        pad = np.repeat(act[-1:], action_horizon, axis=0)
        act_padded = np.concatenate([act, pad], axis=0)
        for t in range(length):
            chunks.append(tf({
                "action": act_padded[t: t + action_horizon],
                "observation/action_reference_tcp": ref[t],
            })["actions"])
    if not chunks:
        raise ValueError(f"split {split!r} has no episodes")
    out = np.stack(chunks).astype(np.float64)
    if out.shape[-1] != 7 * n_hands:
        raise ValueError(f"built {out.shape[-1]}-dim actions, expected {7 * n_hands}")
    return out
