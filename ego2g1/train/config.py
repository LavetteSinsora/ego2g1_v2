"""Training configs: every knob in one frozen dataclass per experiment family.

Three families share `_CommonTrainFields` (the training-loop mechanics that have
nothing to do with the representation):

- `Ego2G1TrainConfig`     — 30-dim absolute-EEF-6d + Revo2-6-motor actions,
                            proprioception digitized into the pi05 prompt.
- `EgoRelationTrainConfig` — 14-dim relative-EEF-rotvec + binary-gripper actions,
                            object-relation vectors injected as prompt tokens
                            (HumanEgo-style; docs/datasets.md).
- `UmiTrainConfig`        — UMI: one acting arm, 7-dim relative-EEF-rotvec
                            + CONTINUOUS gripper, two wrist cameras, no scene
                            perception; recent TCP poses injected as prompt
                            tokens.

All produce stock openpi objects but are NOT registered in openpi's _CONFIGS —
ego2g1 entrypoints take a dataclass directly through tyro.

Field ORDER differs from a single flat dataclass, but `config_hash()` serializes
with sort_keys=True, so moving a field into the shared base does not change any
existing config's hash and existing checkpoints keep resolving their norm stats.
"""

import dataclasses
import hashlib
import json
import pathlib
from typing import ClassVar, Literal

import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders

from ego2g1.core import paths as _paths
from ego2g1.train import model as _model
from ego2g1.train import transforms as _transforms
from ego2g1.train import weight_loader as _ego_weight_loader


@dataclasses.dataclass(frozen=True)
class _CommonTrainFields:
    """Training-loop mechanics, shared by every experiment family.

    Every field has a default so subclasses may add their own in any order.
    """

    name: str = "ego2g1"      # name of this config
    exp_name: str = "ego2g1"  # name of specific experiment using this config

    # --- training ---
    batch_size: int = 32
    num_train_steps: int = 10000

    log_interval: int = 100 # interval of logging train loss, etc.
    save_interval: int = 1000 # interval of saving model checkpoint (for resuming training. new checkpoint saved, old deleted)
    # Interval of storing not-deleted checkpoints (for offline diagnostic).
    #
    # MUST BE A MULTIPLE OF save_interval, or most of them silently vanish.
    # orbax keeps a checkpoint permanently only if its step is divisible by
    # keep_period; everything else is pruned by max_to_keep=1 (openpi
    # checkpoints.initialize_checkpoint_dir) as soon as a newer one lands. The
    # old 2500 did NOT divide save_interval=1000, so of the ten saves in a
    # 10k-step run — 1000..9000 plus the final 9999 — only 5000 was divisible
    # by 2500 and only 5000 and 9999 survived. 2000 keeps 2000/4000/6000/8000
    # plus the final. (Step 0 is never saved: the loop guards on
    # `step > start_step`.)
    #
    # In _HASH_EXCLUDE, so changing it does not move config_hash() and existing
    # checkpoints keep resolving their norm stats.
    keep_period: int = 2000
    # eval/probe cadence: the val minimum lands around step ~1k on this dataset,
    # so 250 resolves it instead of sampling it once. The best-val checkpoint
    # (checkpoints/<name>/<exp>/best/) is re-crowned at every eval.
    eval_interval: int = 250 # interval of running eval (e.g., record loss on validation set), 0 disables
    eval_num_batches: int = 4
    probe_interval: int = 250 # interval of running attention allocation probe
    probe_batch_size: int = 2

    num_workers: int = 2
    seed: int = 42
    ema_decay: float | None = 0.99
    checkpoint_base_dir: str = "./checkpoints"
    assets_base_dir: str = "./assets"
    weight_loader_params_path: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    # --- learning rate: cosine with warmup; the decay horizon is ALWAYS
    # num_train_steps (no separate decay_steps knob — changing the run length
    # automatically rescales the schedule so LR lands on final_lr at the end)
    peak_lr: float = 2.5e-5
    warmup_steps: int = 1_000
    final_lr: float = 2.5e-6  # LR at the last step; openpi convention: peak/10
    fsdp_devices: int = 1
    wandb_enabled: bool = True
    wandb_project: str = "ego2g1"
    resume: bool = False
    overwrite: bool = False

    # Fields that do not change the produced training data or model, and so are
    # excluded from config_hash(). ClassVar => not a dataclass field.
    _HASH_EXCLUDE: ClassVar[frozenset[str]] = frozenset({
        "exp_name", "wandb_enabled", "wandb_project", "num_workers",
        "log_interval", "save_interval", "keep_period", "resume", "overwrite",
        "eval_interval", "eval_num_batches", "probe_interval", "probe_batch_size",
    })

    def __post_init__(self):
        if self.warmup_steps >= self.num_train_steps:
            raise ValueError(f"warmup_steps={self.warmup_steps} >= num_train_steps={self.num_train_steps}")

    # --- derived ---

    def lr_schedule(self) -> _optimizer.CosineDecaySchedule:
        return _optimizer.CosineDecaySchedule(
            warmup_steps=self.warmup_steps,
            peak_lr=self.peak_lr,
            decay_steps=self.num_train_steps,
            decay_lr=self.final_lr,
        )

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    def config_hash(self) -> str:
        """Hash of every field that affects the produced training data/model."""
        payload = {k: v for k, v in dataclasses.asdict(self).items() if k not in self._HASH_EXCLUDE}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class Ego2G1TrainConfig(_CommonTrainFields):
    name: str = "ego2g1_pi05" # name of this config
    exp_name: str = "ego2g1" # name of specific experiment using this config

    # --- data ---
    # default = the extraction pipeline's own output location (core.paths)
    dataset_root: str = dataclasses.field(default_factory=lambda: str(
        _paths.data_dir() / "lerobot_datasets" / "ego2g1" / "put_bottle_in_box_ego"))
    # norm stats are written to assets/<name>/<repo_id>/ — OLD checkpoints carry
    # their stats under the unsuffixed name; new-repo checkpoints start fresh on
    # the policy-compliant _ego name (docs/datasets.md)
    repo_id: str = "ego2g1/put_bottle_in_box_ego"
    # extraction config hash this training run expects (from the dataset's
    # extraction_meta.json). 7b7f8bb7… = the 2026-07-17 full re-extraction with
    # s004c_resolve smooth proprioception (102 sub-episodes, 16464 frames;
    # joint accel RMS median 29.9 -> 6.1 rad/s^2).
    expected_config_hash: str | None = "7b7f8bb70e3e7f1c"
    fps: int = 30 # data's corresponding frequency (how many actions correspond to 1 second of expected execution)
    hands: tuple[str, ...] = ("left", "right") # order of hand in action label (i.e., which hand occupies the first 15-dim of the action)
    # these must match the sidecar's source_episode strings (raw-dir name /
    # episode stem) — the re-extracted dataset records put_bottle_in_box_ego/*
    val_real_episodes: tuple[str, ...] = (
        "put_bottle_in_box_ego/episode_10", "put_bottle_in_box_ego/episode_20",
        "put_bottle_in_box_ego/episode_30", "put_bottle_in_box_ego/episode_40",
        "put_bottle_in_box_ego/episode_50", "put_bottle_in_box_ego/episode_60",
        "put_bottle_in_box_ego/episode_70", "put_bottle_in_box_ego/episode_80",
        "put_bottle_in_box_ego/episode_90",
    ) # which episodes are validation episodes and should be held-out for norm stat calculation

    # --- model ---
    action_dim: int = 32  # pi05_base padded width
    action_dim_actual: int = 30 # actual dimension of the action (loss in padded dim is masked)
    action_horizon: int = 50
    control_mode: str = _transforms.CONTROL_MODE_EEF # pi0.5 pretraining appends "<control mode> joint/end effector <control mode>" as text tokens in thhe prompt
    # pi05 convention: the (normalized) state is digitized into 256 bins and fed
    # as text tokens in the prompt ("Task: ..., State: 12 240 ...;\nAction: ").
    # WARNING: False makes the policy state-BLIND — the pi05 architecture has no
    # continuous state token (that's the pi0 path), so the prompt is the ONLY
    # way proprioception reaches the model. False also changes the prompt
    # template away from what pi05_base pretrained with — prefer state_dropout_p
    # below, which withholds the state WITHOUT touching the template.
    discrete_state_input: bool = True
    # Probability that a training sample's state digits are replaced by the word
    # "unknown" (ego2g1.transforms.STATE_SENTINEL). Reduces the policy's reliance
    # on proprioception, which real-robot evals suggest it memorizes at the
    # expense of the visual pathway. Three regimes:
    #   0.0 — baseline: real state on every sample (prompt byte-identical to stock)
    #   0.5 — dropout: coin flip per sample at TRAIN time only; val and SERVING
    #         always get the real state (the model must cope without it, but is
    #         still given it)
    #   1.0 — blind: the state is withheld at train AND val AND serving time
    # Serving mode is derived from this field alone (data_config.serve_state_mode),
    # so a checkpoint carries its own prompt semantics and cannot be mis-served.
    state_dropout_p: float = 0.0

    # --- normalization ---
    per_slot_floor_c: float = 0.1 # parameter for per dim, per time-slot normalization
    # Per-slot centering, applied ONLY to the EEF-delta dims (first 9 of each
    # hand's 15): subtract the per-slot mean before the gain, so the 1/c boost
    # lands on motion deviation rather than on constants (rot6d diagonals
    # otherwise become ~+10 targets at slot 0). Hand-command dims are absolute
    # and get gain ~1 anyway — never centered ("flat = hold still" preserved).
    per_slot_center: bool = True
    # Final |target| bound in model space (after centering + gain): caps
    # heavy-tail outliers (dim 7 reaches |normalized| ~ 25) so no single label
    # can dominate a batch. Train-side label surgery only; no serving inverse
    # exists or is needed. None disables.
    model_space_clamp: float | None = 10.0
    degenerate_dim_allowlist: tuple[int, ...] = (9, 10, 11, 12, 13, 14) # action dimensions that can have degenerate stats (e.g., all zeros when left hand never moved). Included dimensions might be filled to -1 depending on whether value fluctuation is too small.

    # --- train-time RTC ---
    rtc_training: bool = False
    rtc_d_max: int = 16  # maximum expected inference time (expressed in # of timesteps)

    def __post_init__(self):
        super().__post_init__()
        if self.action_dim_actual != 15 * len(self.hands):
            raise ValueError(
                f"action_dim_actual={self.action_dim_actual} != 15*len(hands)={15 * len(self.hands)}"
            )
        if self.model_space_clamp is not None and self.model_space_clamp <= 1.0:
            raise ValueError(f"model_space_clamp={self.model_space_clamp} must be > 1 (or None)")
        if not 0.0 <= self.state_dropout_p <= 1.0:
            raise ValueError(f"state_dropout_p={self.state_dropout_p} must be in [0, 1]")
        if self.state_dropout_p > 0.0 and not self.discrete_state_input:
            raise ValueError("state_dropout_p masks the state IN the prompt; it is meaningless "
                             "with discrete_state_input=False (which removes the prompt state entirely)")

    # --- derived ---

    def model_config(self) -> _model.Ego2G1Pi0Config:
        return _model.Ego2G1Pi0Config(
            pi05=True,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            action_dim_actual=self.action_dim_actual,
            rtc_training=self.rtc_training,
            rtc_d_max=self.rtc_d_max,
            discrete_state_input=self.discrete_state_input,
        )

    def weight_loader(self) -> _weight_loaders.WeightLoader:
        return _weight_loaders.CheckpointWeightLoader(self.weight_loader_params_path)

    def feature_flags(self) -> dict:
        """Checkpoint stamp (ego2g1.stamp): serving code must declare support
        for every flag with `required: True`."""
        return {
            "per_slot_rescale": {
                "required": self.per_slot_floor_c < 1.0,
                "floor_c": self.per_slot_floor_c,
            },
            # serving MUST add the per-slot mean back (else centered dims are biased)
            "per_slot_center": {"required": self.per_slot_center},
            # serving MUST neutralize degenerate state dims like training did
            "degenerate_neutralization": {"required": True},
            # train-side label surgery only; informational
            "model_space_clamp": {"required": False, "value": self.model_space_clamp},
            "control_mode_prompt": {"required": True, "mode": self.control_mode},
            # required ONLY for a fully blind checkpoint: serving must withhold the
            # state exactly as training did. A dropout checkpoint (0 < p < 1) is
            # served WITH the real state, which is stock behavior — informational.
            "state_masking": {"required": self.state_dropout_p >= 1.0, "p": self.state_dropout_p},
            "relative_chunk_actions": {"required": True},
            **{k: {"required": False, "value": v}
               for k, v in self.model_config().feature_flags().items()},
        }


@dataclasses.dataclass(frozen=True)
class EgoRelationTrainConfig(_CommonTrainFields):
    """HumanEgo-style relational state + 14-dim relative-EEF-rotvec actions.

    Differs from Ego2G1TrainConfig in BOTH halves of the interface, which is why
    it is a sibling rather than a mode flag:

    state   — no absolute proprioception at all. Per object, its pose in each
              TCP's frame (R^18), z-scored with stats POOLED ACROSS OBJECTS
              (required: one shared encoder + shuffled prompt order means
              per-object stats would break permutation equivariance), fed
              through a learned encoder as ONE prompt token per object; hand
              state is the words open/closed.
    action  — per hand [3 translation + 3 rotation-vector] relative to the
              anchor TCP pose, plus one binary gripper dim per hand.
    """

    name: str = "ego2g1_relation"
    exp_name: str = "relation"

    # --- data ---
    dataset_root: str = dataclasses.field(default_factory=lambda: str(
        _paths.data_dir() / "lerobot_datasets" / "ego2g1" / "red_block_in_pen_holder_ego"))
    # names the norm-stats assets dir (assets/<name>/<repo_id>/); see docs/datasets.md
    repo_id: str = "ego2g1/red_block_in_pen_holder_ego"
    # This dataset's sidecar carries no config_hash of its own (its schema is
    # {schema_version, variant, config, sources}), so the adapter in
    # ego2g1.train.dataset SYNTHESIZES one by hashing the config block. None
    # skips the assert; set it once you have run against a dataset you trust.
    expected_config_hash: str | None = None
    fps: int = 30
    hands: tuple[str, ...] = ("left", "right")
    # LeRobot video decoder. None = LeRobot's default (torchcodec). This dataset
    # is mp4v/mpeg4 rather than the old libx264, and torchcodec needs system
    # FFmpeg SHARED LIBRARIES (not just the imageio-ffmpeg binary the mac profile
    # puts on PATH) — on a box without them it fails with an opaque
    # "Could not load libtorchcodec". The file itself decodes fine under pyav, so
    # set "pyav" if you hit that.
    video_backend: str | None = None

    # Object instances, in the order the dataset's observation.state lays them
    # out (obj1, obj2, obj3 == holder, red, yellow). Must match
    # info.json's ego_relation.object_categories.
    objects: tuple[str, ...] = ("black, metal pen holder", "red cube", "yellow cube")
    # How each object is NAMED in the prompt. Shorter than the detector prompt
    # category above, and the string the attention probe groups by.
    object_prompt_names: tuple[str, ...] = ("pen holder", "red cube", "yellow cube")
    # Shuffle the order objects appear in the prompt (train only). The SAME
    # permutation is applied to the relation rows, so the name->vector pairing
    # is preserved; this is what forces order-invariance instead of letting the
    # model learn "the graspable one is always second".
    shuffle_object_order: bool = True

    # Held-out episodes, as sidecar `source_episode` strings. Every 7th of the
    # 50 recordings (14%); the adapter maps LeRobot episode i -> episode_{i+1}.
    val_source_episodes: tuple[str, ...] = (
        "red_block_in_pen_holder_50/episode_7", "red_block_in_pen_holder_50/episode_14",
        "red_block_in_pen_holder_50/episode_21", "red_block_in_pen_holder_50/episode_28",
        "red_block_in_pen_holder_50/episode_35", "red_block_in_pen_holder_50/episode_42",
        "red_block_in_pen_holder_50/episode_49",
    )

    # --- model ---
    action_dim: int = 32  # pi05_base padded width
    # 14 = 2 hands * (3 translation + 3 rotation-vector) + 2 gripper
    action_dim_actual: int = 14
    action_horizon: int = 50
    control_mode: str = _transforms.CONTROL_MODE_EEF
    relation_hidden: int = 512   # GeGLU hidden width of the relation encoder
    grasp_head: bool = True      # auxiliary per-slot grasp-probability head
    # Safeguard 2: the injected relation token's size as a FRACTION of the
    # sentinel embedding it is added to, which is read live at injection time.
    # Dimensionless, so it cannot be silently wrong the way the old absolute
    # target was (relation_v1 got 1.63 against a base of 487.76).
    #
    # 1.0 = parity: the token is rotated arctan(1) = 45 deg and the three object
    # tokens end up ~47 deg apart. Deliberately not lower: a too-large injection
    # is easy for training to shrink, a too-small one is a multiplicative
    # correction it provably cannot make (see RelationEncoder).
    relation_alpha: float = 1.0

    # --- normalization ---
    # "per_slot_quantile": per-(slot, dim) q01/q99 -> [-1, 1] directly, gripper
    #   dims exempt. Chosen for this dataset: it makes every slot unit-scale by
    #   construction (measured slot49/slot0 std ratio 1.11-1.63, max |Z| 5.30).
    # "pooled_floored_gain": the E001 scheme (pooled quantile Normalize, then
    #   gain = sigma_pooled / max(sigma_slot, c*sigma_pooled)). Retained for A/B
    #   and for reuse on future datasets.
    #   NOTE: per_slot_floor_c=0.1 is MISTUNED for this dataset. The gain
    #   saturates at 1/c = 10 but slots 0-1 need 17-24x, leaving slot 0 about
    #   1.7-2.4x under-scaled relative to slot 49. c ~= 0.03-0.04 equalizes it.
    action_norm_scheme: Literal["per_slot_quantile", "pooled_floored_gain"] = "per_slot_quantile"
    per_slot_floor_c: float = 0.1
    per_slot_center: bool = True
    model_space_clamp: float | None = 10.0
    # Relation-vector z-score clip, in sigmas. A no-op on this training data
    # (|z| > 5 occurs on 0.0000% of samples) and pure insurance against a
    # detector glitch at deployment time.
    state_norm_clip: float = 5.0

    # --- loss ---
    # Gripper weight, applied on VARIANCE-NORMALIZED dims. Decoupling scale from
    # importance is the point: with raw +-1 grippers and quantile-normalized EEF
    # dims, the gripper's Var(u)=1.83 vs the EEF's 1.10, so an unweighted MSE
    # already gives the 2 gripper dims 21.6% of the loss (not 2/14 = 14.3%).
    # After variance normalization every dim starts equal, so w_gripper=3 means
    # exactly what it reads: worth 3 EEF dims each => 2*3/(12+2*3) = 33% of the
    # loss. Without the normalization step the same 3 would mean something that
    # drifts whenever action_norm_scheme changes.
    w_gripper: float = 3.0
    # Weight of the auxiliary grasp-schedule BCE. A probe and regularizer, not a
    # competitor to the flow loss — keep it small.
    w_aux: float = 0.2

    # --- train-time RTC (unused here; kept so model_config stays uniform) ---
    rtc_training: bool = False
    rtc_d_max: int = 16

    def __post_init__(self):
        super().__post_init__()
        if self.action_dim_actual != 7 * len(self.hands):
            raise ValueError(
                f"action_dim_actual={self.action_dim_actual} != 7*len(hands)={7 * len(self.hands)} "
                "(per hand: 3 translation + 3 rotation-vector + 1 gripper)"
            )
        if len(self.objects) != len(self.object_prompt_names):
            raise ValueError(
                f"{len(self.objects)} objects vs {len(self.object_prompt_names)} prompt names"
            )
        if self.model_space_clamp is not None and self.model_space_clamp <= 1.0:
            raise ValueError(f"model_space_clamp={self.model_space_clamp} must be > 1 (or None)")
        if self.state_norm_clip <= 1.0:
            raise ValueError(f"state_norm_clip={self.state_norm_clip} must be > 1")
        if self.w_gripper <= 0.0:
            raise ValueError(f"w_gripper={self.w_gripper} must be > 0")
        if self.w_aux < 0.0:
            raise ValueError(f"w_aux={self.w_aux} must be >= 0")
        if self.w_aux > 0.0 and not self.grasp_head:
            raise ValueError("w_aux > 0 needs grasp_head=True (there is nothing to weight otherwise)")

    # --- derived ---

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    @property
    def relation_dim(self) -> int:
        """Per-object relation width: one vec9 per hand."""
        return 9 * len(self.hands)

    @property
    def state_dim(self) -> int:
        """Width of the `state` field: the flattened relation matrix.

        The 2 grasp binaries are NOT here — they reach the model as the words
        open/closed in the prompt, not as numbers.
        """
        return self.n_objects * self.relation_dim

    @property
    def gripper_dims(self) -> tuple[int, ...]:
        """Action dims carrying the binary gripper, at the TAIL of the vector
        (so the dim-group mask is a slice). Layout:
        [L_dx L_dy L_dz L_rx L_ry L_rz | R_dx ... R_rz | L_grip R_grip]."""
        n = 6 * len(self.hands)
        return tuple(range(n, n + len(self.hands)))

    def model_config(self) -> _model.Ego2G1Pi0Config:
        return _model.Ego2G1Pi0Config(
            pi05=True,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            action_dim_actual=self.action_dim_actual,
            rtc_training=self.rtc_training,
            rtc_d_max=self.rtc_d_max,
            # Only openpi's TokenizePrompt reads this; our prompt builder owns
            # the string end to end, so it never takes effect. Left at the pi05
            # default so the model config reads unsurprisingly.
            discrete_state_input=True,
            n_objects=self.n_objects,
            relation_dim=self.relation_dim,
            relation_hidden=self.relation_hidden,
            grasp_head=self.grasp_head,
            state_dim=self.state_dim,
            relation_alpha=self.relation_alpha,
        )

    def weight_loader(self) -> _weight_loaders.WeightLoader:
        # NOT the stock loader: pi05_base has no relation_encoder/grasp_head, and
        # stock _merge_params silently DROPS reference params that do not match
        # ".*lora.*". See ego2g1/train/weight_loader.py.
        return _ego_weight_loader.Ego2G1CheckpointWeightLoader(self.weight_loader_params_path)

    def feature_flags(self) -> dict:
        """Checkpoint stamp (ego2g1.stamp): serving code must declare support
        for every flag with `required: True`."""
        return {
            # serving MUST invert whichever action normalization was used
            "action_norm_scheme": {"required": True, "scheme": self.action_norm_scheme},
            # serving MUST build the relational prompt and run the encoder
            "relation_state": {
                "required": True,
                "n_objects": self.n_objects,
                "relation_dim": self.relation_dim,
                "clip": self.state_norm_clip,
            },
            # serving MUST decode rotation-VECTOR rotations, not 6d
            "relative_eef_rotvec_actions": {"required": True},
            # serving MUST expand the binary gripper to hand commands
            "binary_gripper": {"required": True, "dims": list(self.gripper_dims)},
            "control_mode_prompt": {"required": True, "mode": self.control_mode},
            # train-side only; informational
            "model_space_clamp": {"required": False, "value": self.model_space_clamp},
            "loss_gripper_weight": {"required": False, "value": self.w_gripper},
            "grasp_head": {"required": False, "value": self.grasp_head, "w_aux": self.w_aux},
            **{k: {"required": False, "value": v}
               for k, v in self.model_config().feature_flags().items()},
        }


@dataclasses.dataclass(frozen=True)
class UmiTrainConfig(_CommonTrainFields):
    """UMI single-arm wrist-camera policy (`red_block_on_yellow_block_umi`).

    Differs from both siblings in every part of the interface, which is why it
    is a third dataclass rather than a mode flag on either:

    cameras — TWO. The acting arm's wrist camera, and the static arm's camera
              standing in for the head camera this setup does not have.
    state   — no absolute proprioception and no scene perception. A window of
              recent MEASURED TCP poses, each expressed in the current pose's
              frame, plus the gripper, injected as one prompt token per lag
              (ego2g1.train.umi_transforms).
    action  — 3 translation + a rotation block relative to the anchor TCP, plus
              a CONTINUOUS gripper. One arm only; the other holds its pose and
              contributes only its camera. The rotation encoding is the
              `rotation_repr` A/B knob: 3 dims (rotvec, the default) or 6
              (rot6d), so 7 or 10 dims total.

    No grasp head. It existed to give a calibrated probability where a BINARY
    gripper's flow sample is ambiguous; with a continuous gripper there is no
    binary to be calibrated about, and defining the head's label would require
    picking a closed-threshold whose arbitrariness is exactly the problem.
    """

    name: str = "umi_wrist"
    exp_name: str = "umi"

    # --- data ---
    dataset_root: str = dataclasses.field(default_factory=lambda: str(
        _paths.data_dir() / "lerobot_datasets" / "ego2g1" / "red_block_on_yellow_block_umi"))
    # names the norm-stats assets dir (assets/<name>/<repo_id>/); see docs/datasets.md
    repo_id: str = "ego2g1/red_block_on_yellow_block_umi"
    # This dataset carries no extraction sidecar of its own, so
    # ego2g1.train.dataset.umi_extraction_meta SYNTHESIZES one by hashing
    # info.json's schema-defining fields. None skips the assert.
    expected_config_hash: str | None = None
    fps: int = 30
    # The single acting arm. Names the acting camera feature; the other arm
    # holds its pose and contributes only its camera.
    hand: str = "right"
    video_backend: str | None = None
    # Every 8th of the 117 episodes (12%), as `source_episode` strings. The
    # adapter maps LeRobot episode i -> episode_{i} (one recording each).
    val_source_episodes: tuple[str, ...] = tuple(
        f"red_block_on_yellow_block_umi/episode_{i}" for i in range(7, 117, 8)
    )

    # --- model ---
    action_dim: int = 32  # pi05_base padded width
    # 3 translation + rotation (rot_dim) + 1 continuous gripper. MUST be passed
    # together with rotation_repr: 7 for "rotvec", 10 for "rot6d". __post_init__
    # refuses a mismatch rather than deriving it, matching the sibling configs'
    # idiom (`15 * len(hands)`, `7 * len(hands)`) and keeping the value visible
    # in the checkpoint stamp.
    action_dim_actual: int = 7
    action_horizon: int = 50
    control_mode: str = _transforms.CONTROL_MODE_EEF

    # --- rotation representation (the A/B knob) ---
    # How relative rotations are encoded, in BOTH the action targets and the
    # state-history rows. Everything else about the two runs is identical --
    # dataset, normalization scheme, camera slots, loss balance (see
    # data_config.loss_dim_weights, which rescales the rotation block so its
    # share of the loss does not change with the dim count).
    #
    # "rotvec"  3 dims, the SO(3) log map. Centred on zero, minimal, unique only
    #           for |theta| < pi (measured max on this dataset at H=50: 1.385
    #           rad, so comfortable). The default and the measured baseline.
    # "rot6d"   6 dims, the first two columns of R. No wraparound and continuous
    #           in R, but redundant by 3 DOF and centred on the identity matrix.
    #           Survives the per-(slot, dim) quantile normalization intact
    #           (measured model-space std 0.35-0.47 at every slot, the same band
    #           as rotvec) -- see ego2g1.train.umi_transforms.ROTATION_REPRS.
    #
    # NOTE this changes `stats_dir`, so the two representations never share a
    # umi_stats.npz and computing one cannot clobber the other.
    rotation_repr: Literal["rotvec", "rot6d"] = "rotvec"

    # --- how proprioception reaches the model ---
    # "history"       a window of recent MEASURED TCP poses + gripper, each
    #                 encoded to one injected prompt token (the relational
    #                 config's machinery). Continuous, high precision, but it
    #                 adds a learned encoder to the param tree and gives the
    #                 policy a velocity signal it can dead-reckon from.
    # "gripper_token" NO pose history at all. The gripper alone, normalized to
    #                 [-1, 1], binned into `gripper_bins` and written into the
    #                 prompt as DIGITS — exactly how pi0.5 carries its
    #                 discretized state. The param tree stays stock (no
    #                 encoder is built), and the only thing the policy knows
    #                 about its own body is whether the gripper is open.
    #
    # Both share everything else: dataset, 7-dim action space, per-(slot, dim)
    # quantile normalization, camera slots, and the entire deploy conversion
    # path. That is why this is a field rather than a fourth config family.
    #
    # NOTE lag 0 is gathered in BOTH modes: it is the chunk's anchor, not just
    # a history token (umi_transforms.UmiRelativeActions reads poses[0]).
    state_mode: Literal["history", "gripper_token"] = "history"
    # Bins for the "gripper_token" digitization. 256 is pi0.5's own state
    # resolution, so the digits land in the same numeric range the base model
    # saw pretraining.
    gripper_bins: int = 256

    # --- state history (state_mode="history" only) ---
    # Number of PAST lags. Total injected tokens is history_lags + 1, because
    # lag 0 (the anchor's own tick) is a token too: its pose part is
    # structurally zero but it carries the CURRENT gripper, which
    # nothing else in this config would deliver.
    history_lags: int = 5
    # Ticks between lags. At fps=30, stride 3 spans lag 0 .. t-0.5 s — enough
    # of the approach to read velocity, against a 1.67 s action horizon.
    history_stride: int = 3
    # P(history length = j) for j in 0..history_lags+1 inclusive, so the tuple
    # is history_lags + 2 long. Truncation is always from the STALE end (see
    # WristStateHistory), which is what keeps "j-th token == lag j" exact and
    # is forced by relying on RoPE for lag identity.
    #
    # NOT uniform, deliberately: deployment runs at full length essentially
    # always, so uniform would spend most of training on a regime that barely
    # occurs. The mass on short lengths exists to train the episode-start
    # regime — ~2.5% of frames, and the ones where the policy decides how to
    # initiate a motion — and to double as a graded history dropout.
    #
    # This one knob subsumes both intended operating modes:
    #   MODE_A_LENGTH_PROBS  gripper only, no motion history
    #   the default          full history, with the short regime trained
    history_len_probs: tuple[float, ...] = (0.0, 0.05, 0.05, 0.05, 0.05, 0.10, 0.70)
    history_hidden: int = 512    # GeGLU hidden width of the history encoder
    # Injected token size as a FRACTION of the sentinel embedding it is added
    # to, read live at injection time. 1.0 = the token is rotated arctan(1) =
    # 45 deg. See ego2g1.train.relation.RelationEncoder for why the relative
    # form exists and what the absolute one broke.
    history_alpha: float = 1.0
    # Which camera slot the ACTING arm's wrist camera occupies. Must be a wrist
    # slot: openpi gates crop/rotate augmentation on the substring "wrist", and
    # that augmentation would put an unmodelled rotation between the acting
    # image and the anchor-relative action labels.
    acting_slot: str = "right_wrist_0_rgb"
    # The context camera goes to base_0_rgb and therefore DOES get spatial
    # augmentation — correct only while the arm carrying it holds still.
    context_is_static: bool = True

    # --- normalization ---
    action_norm_scheme: Literal["per_slot_quantile"] = "per_slot_quantile"
    model_space_clamp: float | None = 10.0
    # Per-lag z-score clip, in sigmas. Insurance against a tracking glitch at
    # deployment time.
    state_norm_clip: float = 5.0

    # --- loss ---
    # Gripper weight, applied on VARIANCE-NORMALIZED dims (data_config
    # .loss_dim_weights). Unlike the relational config, the gripper here is
    # normalized alongside every other dim, so this number means what it reads:
    # w_gripper=3 gives the gripper 3/(6+3) = 33% of the loss.
    w_gripper: float = 3.0

    # --- train-time RTC (unused here; kept so model_config stays uniform) ---
    rtc_training: bool = False
    rtc_d_max: int = 16

    # History length distribution for the no-motion-history baseline: always
    # exactly one token, the anchor's own tick, carrying the current gripper.
    MODE_A_LENGTH_PROBS: ClassVar[tuple[float, ...]] = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        super().__post_init__()
        if self.rotation_repr not in ("rotvec", "rot6d"):
            raise ValueError(f"rotation_repr={self.rotation_repr!r} must be 'rotvec' or 'rot6d'")
        expected = 3 + self.rot_dim + 1
        if self.action_dim_actual != expected:
            raise ValueError(
                f"action_dim_actual={self.action_dim_actual} != {expected} for "
                f"rotation_repr={self.rotation_repr!r} (3 translation + {self.rot_dim} "
                "rotation + 1 continuous gripper, one arm). Pass BOTH flags: "
                f"--rotation-repr {self.rotation_repr} --action-dim-actual {expected}"
            )
        if self.hand not in ("left", "right"):
            raise ValueError(f"hand={self.hand!r} must be 'left' or 'right'")
        if self.state_mode not in ("history", "gripper_token"):
            raise ValueError(f"state_mode={self.state_mode!r} must be "
                             "'history' or 'gripper_token'")
        if self.state_mode == "gripper_token":
            if self.gripper_bins < 2:
                raise ValueError(f"gripper_bins={self.gripper_bins} must be >= 2")
        else:
            # length draws are meaningless without a history to truncate
            if len(self.history_len_probs) != self.n_lags + 1:
                raise ValueError(
                    f"history_len_probs has {len(self.history_len_probs)} entries, expected "
                    f"{self.n_lags + 1} (one per achievable length 0..{self.n_lags})"
                )
            if any(p < 0 for p in self.history_len_probs):
                raise ValueError(f"history_len_probs={self.history_len_probs} has a negative entry")
            total = sum(self.history_len_probs)
            if not 0.999 <= total <= 1.001:
                raise ValueError(f"history_len_probs sums to {total}, expected 1.0")
        if self.model_space_clamp is not None and self.model_space_clamp <= 1.0:
            raise ValueError(f"model_space_clamp={self.model_space_clamp} must be > 1 (or None)")
        if self.state_norm_clip <= 1.0:
            raise ValueError(f"state_norm_clip={self.state_norm_clip} must be > 1")
        if self.w_gripper <= 0.0:
            raise ValueError(f"w_gripper={self.w_gripper} must be > 0")

    # --- derived ---

    @property
    def n_lags(self) -> int:
        """How many ticks are gathered backwards. In "history" mode these are
        the injected tokens; in "gripper_token" mode it is just the anchor."""
        return len(self.lag_ticks)

    @property
    def lag_ticks(self) -> tuple[int, ...]:
        """Tick offsets gathered backwards, most recent first.

        "gripper_token" still needs lag 0 — that is the chunk's ANCHOR, which
        every mode composes its actions onto — but nothing older.
        """
        from ego2g1.train import umi_transforms as _ut

        if self.state_mode == "gripper_token":
            return (0,)
        return _ut.lag_ticks(self.history_lags, self.history_stride)

    @property
    def injects_tokens(self) -> bool:
        """Whether a learned encoder writes into the prompt. False keeps the
        param tree bit-for-bit stock pi05."""
        return self.state_mode == "history"

    @property
    def rot_dim(self) -> int:
        """Width of the rotation block: 3 for "rotvec", 6 for "rot6d"."""
        from ego2g1.train import umi_transforms as _ut

        return _ut.rot_dim(self.rotation_repr)

    @property
    def history_dim(self) -> int:
        """Per-lag vector width: 3 translation + rot_dim rotation + 1 gripper.

        The same width as an action row, deliberately: history rows and action
        rows are the same physical quantity at different times, so a divergence
        between the two would mean the anchor composition had come apart.
        """
        from ego2g1.train import umi_transforms as _ut

        return _ut.history_dim(self.rotation_repr)

    @property
    def rot_dims(self) -> tuple[int, ...]:
        """Action dims carrying the rotation, between translation and gripper.

        Read by `data_config.loss_dim_weights` to hold the rotation BLOCK's
        share of the loss fixed as the dim count changes — without which
        rot6d would silently hand rotation 50% of the loss where rotvec gives
        it 33%, and the A/B would measure the reweighting instead of the
        representation.
        """
        return tuple(range(3, 3 + self.rot_dim))

    @property
    def gripper_dims(self) -> tuple[int, ...]:
        """The action dim carrying the continuous gripper, at the TAIL."""
        return (3 + self.rot_dim,)

    @property
    def stats_dir(self) -> pathlib.Path:
        """Where umi_stats.npz lives — assets/<name>/<repo_id>[/<rotation_repr>].

        Keyed by the rotation representation because the artifact's shape
        depends on it: the (H, D) quantile grid and the (n_lags, D) per-lag
        history stats are 7 columns wide for rotvec and 10 for rot6d. Sharing
        one path would mean `compute_norm_stats` for one representation
        overwrites the other's artifact — which for the rotvec runs already on
        disk is not a hypothetical, since resuming them reloads it.

        The DEFAULT representation deliberately keeps the unsuffixed path, so
        every checkpoint written before this field existed still resolves.
        """
        base = self.assets_dirs / self.repo_id
        return base if self.rotation_repr == "rotvec" else base / self.rotation_repr

    def model_config(self) -> _model.Ego2G1Pi0Config:
        # The injection fields are still spelled n_objects / relation_* — for
        # this config they mean "number of history tokens" and "width of one
        # history vector". Left alone deliberately: renaming them would touch
        # the relational path for no behavioural gain (ego2g1.train
        # .umi_transforms module docstring).
        return _model.Ego2G1Pi0Config(
            pi05=True,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            action_dim_actual=self.action_dim_actual,
            rtc_training=self.rtc_training,
            rtc_d_max=self.rtc_d_max,
            discrete_state_input=True,   # unused: our prompt builder owns the string
            # 0 => no encoder is built at all and the param tree is bit-for-bit
            # stock pi05 (ego2g1.train.model builds RelationEncoder only when
            # n_objects > 0, and embed_prefix short-circuits on it being None).
            n_objects=self.n_lags if self.injects_tokens else 0,
            relation_dim=self.history_dim,
            relation_hidden=self.history_hidden,
            grasp_head=False,
            state_dim=1,
            relation_alpha=self.history_alpha,
            # the injected rows are an ORDERED sequence, not an unordered set:
            # report the most-recent-vs-most-stale separation instead of the
            # object config's mean pairwise angle.
            inject_ordered=self.injects_tokens,
        )

    def weight_loader(self) -> _weight_loaders.WeightLoader:
        # NOT the stock loader: pi05_base has no relation_encoder, and stock
        # _merge_params silently DROPS reference params that do not match
        # ".*lora.*". See ego2g1/train/weight_loader.py.
        return _ego_weight_loader.Ego2G1CheckpointWeightLoader(self.weight_loader_params_path)

    def feature_flags(self) -> dict:
        """Checkpoint stamp (ego2g1.stamp): serving code must declare support
        for every flag with `required: True`."""
        return {
            # serving MUST invert the per-(slot, dim) quantile normalization
            "action_norm_scheme": {"required": True, "scheme": self.action_norm_scheme},
            # serving MUST build the history rows and run the encoder
            # Present only in "history" mode: serving must build the history
            # rows and run the encoder. A gripper_token checkpoint has no
            # encoder and must not be served as though it did.
            **({"state_history": {
                "required": True,
                "n_lags": self.n_lags,
                "history_dim": self.history_dim,
                "lag_ticks": list(self.lag_ticks),
                "clip": self.state_norm_clip,
                # the history rows use the SAME rotation encoding as the
                # actions, so serving cannot get one right and the other wrong
                "rotation_repr": self.rotation_repr,
                # measured, never commanded — see umi_transforms' docstring
                "pose_source": "measured",
            }} if self.injects_tokens else {
                # serving MUST digitize the gripper into the prompt with the
                # SAME bins and the SAME quantiles training used — a different
                # binning is a silently different state.
                "gripper_token": {
                    "required": True,
                    "bins": self.gripper_bins,
                    "pose_source": "measured",
                }}),
            # WHICH rotation decode serving must apply. The flag NAME carries
            # it, not a value inside one flag, because `stamp.check_supported`
            # gates on names: `relative_eef_rot6d_actions` is deliberately NOT
            # in `stamp.SUPPORTED_FEATURES`, so a rot6d checkpoint is REFUSED by
            # the deploy path (ego2g1/deploy/modes/umi_eef.py decodes rotvec and
            # assumes a 7-dim row) instead of being decoded with the wrong
            # geometry. Add the name there only together with the decode itself.
            **({"relative_eef_rotvec_actions": {"required": True}}
               if self.rotation_repr == "rotvec"
               else {"relative_eef_rot6d_actions": {
                   "required": True, "rot_dims": list(self.rot_dims)}}),
            # serving MUST send a continuous gripper, not a binary open/closed
            "continuous_gripper": {"required": True, "dims": list(self.gripper_dims)},
            # serving MUST place the two cameras in these slots
            "wrist_cameras": {
                "required": True,
                "acting_slot": self.acting_slot,
                "context_slot": "base_0_rgb",
                "context_is_static": self.context_is_static,
            },
            "control_mode_prompt": {"required": True, "mode": self.control_mode},
            # train-side only; informational
            "model_space_clamp": {"required": False, "value": self.model_space_clamp},
            "loss_gripper_weight": {"required": False, "value": self.w_gripper},
            "history_len_probs": {"required": False, "value": list(self.history_len_probs)},
            **{k: {"required": False, "value": v}
               for k, v in self.model_config().feature_flags().items()},
        }
