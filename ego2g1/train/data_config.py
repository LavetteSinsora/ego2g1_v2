"""Assemble the openpi DataConfig for ego2g1 (TRAINING_PLAN.md §3.5).

Stock transform_dataset fixes the order:
    repack -> data_transforms.inputs -> Normalize -> model_transforms.inputs
and inference applies outputs in reverse:
    model_transforms.outputs -> Unnormalize -> data_transforms.outputs.

Placement is normalization-critical:
- RelativeChunkActions and Ego2G1Inputs run BEFORE Normalize (and are exactly
  what compute_norm_stats sees).
- PerSlotRescale runs after Normalize (it is defined in pooled-normalized
  units) and before PadStatesAndActions (the gain grid is (H, 30)).
- AppendControlMode runs before TokenizePrompt.
- TokenizePrompt digitizes the NORMALIZED 30-dim state into the pi05 prompt.
"""

import dataclasses
import pathlib

import numpy as np

import openpi.models.tokenizer as _tokenizer
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms

from ego2g1.core import chunk_math
from ego2g1.train import norm as _norm
from ego2g1.train import transforms as _ego_transforms


def arm_dims_mask(hands: tuple[str, ...]) -> "np.ndarray":
    """The EEF/arm block (first 9) of each hand's [eef 9 | hand 6] — the dims
    the E001 per-slot treatment (both centering AND gain) applies to. Hand
    commands are absolute [0,1], anchor-independent, and carry no per-slot
    structure, so they are left to pooled quantile Normalize alone."""
    return np.tile(np.repeat([True, False], [9, 6]), len(hands))


def build_per_slot_transforms(train_config, norm_stats, per_slot):
    """The E001 transform pair from stats artifacts (norm-critical, so in one
    place used by both training and serving):
    - degeneracy mask (norm.degenerate_action_dims) drives gain exemption AND
      data-path neutralization;
    - mu_n = per-slot mean mapped into normalized units, zeroed everywhere
      centering does not apply (hand commands, degenerate dims);
    - forward clamps to train_config.model_space_clamp."""
    d_real = train_config.action_dim_actual
    act = norm_stats["actions"]
    deg = _norm.degenerate_action_dims(act, d_real)
    arm = arm_dims_mask(tuple(train_config.hands))
    # Per-slot gain is arm-only: hand dims keep gain 1 (a no-op in the rescale
    # pair), so their sole normalization is pooled quantile Normalize. Arms get
    # the full E001 floored per-slot rescale on top.
    gain = per_slot.gain(train_config.per_slot_floor_c, act.std[:d_real], degenerate_mask=deg)
    gain = np.where(arm[None, :], gain, 1.0).astype(np.float32)
    mu_n = None
    if train_config.per_slot_center:
        if per_slot.mu_slot is None:
            raise ValueError(
                "per_slot_center=True but the per-slot stats artifact has no mu_slot — "
                "re-run `python -m ego2g1.train.compute_norm_stats` with this code version"
            )
        q01, q99 = act.q01[:d_real], act.q99[:d_real]
        mu_n = (per_slot.mu_slot[:, :d_real] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        mu_n = np.where(arm & ~deg, mu_n, 0.0)
        mu_n = mu_n.astype(np.float32)
    return (
        _ego_transforms.PerSlotRescale(gain=gain, mu_n=mu_n, degenerate_mask=deg,
                                       clamp=train_config.model_space_clamp),
        _ego_transforms.PerSlotRescaleInverse(gain=gain, mu_n=mu_n),
    )


def serve_state_mode(train_config) -> str:
    """The prompt state mode a checkpoint must be SERVED with, derived from the
    one field the stamp carries. A blind checkpoint (p >= 1.0) never saw a state
    digit and must not be shown one; a dropout checkpoint (0 < p < 1) was trained
    to cope without the state but is given the real one at inference — dropout is
    a train-time regularizer, not a deployment mode."""
    return "blind" if train_config.state_dropout_p >= 1.0 else "real"


def create_data_config(
    train_config,
    model_config,
    *,
    norm_assets_dir: pathlib.Path | str,
    per_slot_dir: pathlib.Path | str | None = None,
    skip_norm_stats: bool = False,
    state_mode: str = "real",
    dropout_p: float = 0.0,
    shuffle_state_pool: np.ndarray | None = None,
) -> _config.DataConfig:
    """Build the full DataConfig. `norm_assets_dir` holds norm_stats.json (the
    config assets dir at train time; the checkpoint's assets/<asset_id>/ dir at
    serving). `per_slot_dir` holds per_slot_stats.npz — defaults to
    norm_assets_dir, but serving passes the run-level assets_ego2g1 dir, so the
    two artifacts never have to live in the same directory.

    `state_mode` / `dropout_p` select how the prompt's state segment is built
    (ego2g1.transforms.Ego2G1TokenizePrompt). Callers:
      train loop   -> "dropout", p = train_config.state_dropout_p
      val (real)   -> "real"      | val (blind)    -> "blind"
      val (shuffled) -> "real" + shuffle_state_pool
      serving      -> serve_state_mode(train_config)
    `shuffle_state_pool` is diagnostic-only and must never be set for training."""
    norm_assets_dir = pathlib.Path(norm_assets_dir)
    per_slot_dir = pathlib.Path(per_slot_dir) if per_slot_dir is not None else norm_assets_dir

    norm_stats = None
    per_slot_transforms = None
    if not skip_norm_stats:
        norm_stats = _normalize.load(norm_assets_dir)
        per_slot = _norm.load_per_slot(per_slot_dir)
        per_slot_transforms = build_per_slot_transforms(train_config, norm_stats, per_slot)

    data_inputs = [
        chunk_math.RelativeChunkActions(hands=tuple(train_config.hands)),
        _ego_transforms.Ego2G1Inputs(model_type=model_config.model_type),
    ]
    if shuffle_state_pool is not None:
        # before Normalize: the pool holds RAW states, so the rest of the stack
        # (norm, per-slot neutralization, digitization) treats them identically
        # to a genuine state.
        data_inputs.append(_ego_transforms.ShuffleState(pool=shuffle_state_pool))
    data_transforms = _transforms.Group(
        inputs=data_inputs,
        outputs=[_ego_transforms.Ego2G1Outputs(action_dim=train_config.action_dim_actual)],
    )

    model_inputs = []
    model_outputs = []
    if per_slot_transforms is not None:
        forward, inverse = per_slot_transforms
        model_inputs.append(forward)
        model_outputs.append(inverse)
    if model_config.discrete_state_input:
        # Ego2G1TokenizePrompt owns the prompt string so the state segment can be
        # withheld; with state_mode="real" it is byte-identical to stock.
        tokenize = _ego_transforms.Ego2G1TokenizePrompt(
            tokenizer=_ego_transforms.Ego2G1Tokenizer(model_config.max_token_len),
            mode=state_mode,
            dropout_p=dropout_p,
        )
    else:
        # pi0-style prompt (no state, no Task:/State:/Action: scaffold) — a
        # different prompt TEMPLATE, not just a withheld state. state_mode is
        # moot here; config.__post_init__ forbids combining the two.
        tokenize = _transforms.TokenizePrompt(
            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
            discrete_state_input=False,
        )
    model_inputs += [
        _ego_transforms.AppendControlMode(control_mode=train_config.control_mode),
        _transforms.ResizeImages(224, 224),
        tokenize,
        _transforms.PadStatesAndActions(model_config.action_dim),
    ]
    model_transforms = _transforms.Group(inputs=model_inputs, outputs=model_outputs)

    # Repack (dataset-only, never at inference): make dataset samples look
    # like what the robot client sends — image/state under observation/*,
    # pose/hand chunks kept for RelativeChunkActions.
    repack_transforms = _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "observation/image": "image",
                    "observation/state": "state",
                    **{f"pose.{h}": f"pose.{h}" for h in train_config.hands},
                    **{f"hand.{h}": f"hand.{h}" for h in train_config.hands},
                    "prompt": "prompt",
                }
            )
        ]
    )

    return _config.DataConfig(
        repo_id=train_config.repo_id,
        asset_id=train_config.repo_id,
        norm_stats=norm_stats,
        repack_transforms=repack_transforms,
        data_transforms=data_transforms,
        model_transforms=model_transforms,
        use_quantile_norm=True,  # pi05
        prompt_from_task=True,
    )


# --------------------------------------------------------------------------
# UmiTrainConfig stack
# --------------------------------------------------------------------------


def create_umi_data_config(
    train_config,
    model_config,
    *,
    stats_dir: pathlib.Path | str,
    skip_norm_stats: bool = False,
    history_length_probs: tuple[float, ...] | None = None,
    history_fixed_len: int | None = None,
    permute_history: bool = False,
    history_pool: np.ndarray | None = None,
    shuffle_gripper: bool = False,
    omit_gripper: bool = False,
) -> _config.DataConfig:
    """Assemble the UMI DataConfig (`UmiTrainConfig`).

    Placement notes, all load-bearing:

    - `norm_stats={}` makes openpi's mandatory `Normalize` step a NO-OP. It is
      inserted unconditionally by `transform_dataset` and cannot be removed
      without forking that function; an empty dict is the supported way to opt
      out, because Normalize only touches keys present in the stats. Neither of
      our normalizations (per-(slot, dim) quantile, per-lag z-score) is
      expressible as openpi's per-dim NormStats anyway.
    - Both normalizers live in `model_transforms`, so `compute_norm_stats` --
      which applies repack + data_transforms only -- observes RAW actions and
      RAW history.
    - `UmiSplitGathered` runs FIRST and is the only place that knows how
      `make_delta_timestamps` packed the backward history and the forward chunk
      into one `action` gather.
    - `UmiRelativeActions` and `UmiStateHistory` both read `pose_history[0]` as
      the anchor. One source, so the action frame and the history frame cannot
      come apart; `norm.lag_zero_pose_is_offset` checks that invariant as a
      number once the stats exist.
    - Both read `rotation_repr` from the SAME config field, so the action chunk
      and the history rows always encode rotation the same way.

    The history-length arguments select a pool:
      train         -> history_length_probs = train_config.history_len_probs
      val           -> history_fixed_len = n_lags  (full; the reference curve)
      val_permuted  -> ... + permute_history=True
      val_random    -> ... + history_pool=<other samples' rows>
      val_nohist    -> history_fixed_len = 0
    Val pools pin the length deliberately: they are asking about the history's
    CONTENT, and letting the length vary too would confound two questions.
    """
    from ego2g1.train import norm as _norm
    from ego2g1.train import umi_transforms as _ut

    stats_dir = pathlib.Path(stats_dir)
    n_lags = train_config.n_lags

    model_inputs = []
    model_outputs = []
    if not skip_norm_stats:
        stats = _norm.load_umi(stats_dir)
        if stats.rotation_repr != train_config.rotation_repr:
            raise ValueError(
                f"{stats_dir} holds {stats.rotation_repr!r} stats but the config is "
                f"{train_config.rotation_repr!r}. The action grid would be sliced to "
                "the wrong columns rather than failing on shape, so this is checked "
                "explicitly (see UmiTrainConfig.stats_dir, which keys the artifact "
                "path by representation so the two never collide in the first place)."
            )
        d_real = train_config.action_dim_actual
        model_inputs.append(_ut.NormalizeHistory(
            mean=stats.history_mean, std=stats.history_std, clip=train_config.state_norm_clip,
        ))
        if train_config.action_norm_scheme != "per_slot_quantile":
            raise NotImplementedError(
                f"action_norm_scheme={train_config.action_norm_scheme!r} is declared in the "
                "config but only 'per_slot_quantile' is wired here"
            )
        # gripper_dims=() on purpose: the gripper is CONTINUOUS here, so it has a
        # real quantile span and is normalized like every other dim. The
        # relational config exempts its gripper only because a quantile map of a
        # two-point distribution is meaningless.
        model_inputs.append(_ut.PerSlotQuantizeActions(
            q01=stats.action_q01[:, :d_real], q99=stats.action_q99[:, :d_real],
            gripper_dims=(), clamp=train_config.model_space_clamp,
        ))
        model_outputs.append(_ut.PerSlotQuantizeActionsInverse(
            q01=stats.action_q01[:, :d_real], q99=stats.action_q99[:, :d_real],
            gripper_dims=(),
        ))

    model_inputs += [
        _transforms.ResizeImages(224, 224),
        _ut.RelationTokenizePrompt(max_token_len=model_config.max_token_len),
        _transforms.PadStatesAndActions(model_config.action_dim),
    ]

    inject = train_config.injects_tokens
    if inject:
        prompt_tf = _ut.UmiPrompt(control_mode=train_config.control_mode)
    else:
        # state_mode="gripper_token": the gripper is BINNED into the prompt
        # instead of encoded into a token, so the quantiles must come from the
        # stats artifact. `skip_norm_stats` is only used by compute_norm_stats
        # itself, which never reaches the prompt builder.
        if skip_norm_stats:
            prompt_tf = _ut.UmiGripperPrompt(q01=0.0, q99=1.0,
                                             bins=train_config.gripper_bins,
                                             control_mode=train_config.control_mode)
        else:
            if stats.gripper_q01 != stats.gripper_q01:      # NaN
                raise ValueError(
                    f"{stats_dir} carries no gripper quantiles; state_mode="
                    "'gripper_token' cannot digitize without them. Re-run "
                    "`python -m ego2g1.train.compute_norm_stats --umi` with this "
                    "code version.")
            prompt_tf = _ut.UmiGripperPrompt(
                q01=stats.gripper_q01, q99=stats.gripper_q99,
                bins=train_config.gripper_bins,
                control_mode=train_config.control_mode,
                shuffle=shuffle_gripper,
                include_gripper=not omit_gripper)

    data_transforms = _transforms.Group(
        inputs=[
            _ut.UmiSplitGathered(n_lags=n_lags),
            # Both take the SAME rotation_repr from the SAME config field: the
            # action chunk and the history rows are the same physical quantity
            # at different times, so encoding them differently would be a
            # geometry mismatch the loss would happily train around.
            _ut.UmiRelativeActions(rotation_repr=train_config.rotation_repr),
            # Runs in BOTH modes: it produces `state` (the current gripper),
            # which the gripper-token prompt digitizes. In gripper_token mode
            # n_lags is 1, so the "history" it builds is just the anchor row
            # and nothing is injected from it.
            _ut.UmiStateHistory(
                rotation_repr=train_config.rotation_repr,
                length_probs=history_length_probs,
                fixed_len=history_fixed_len,
                permute=permute_history,
                pool=history_pool,
            ),
            prompt_tf,
            _ut.UmiInputs(
                model_type=model_config.model_type,
                acting_slot=train_config.acting_slot,
                context_is_static=train_config.context_is_static,
                inject=inject,
            ),
        ],
        outputs=[_ut.UmiOutputs(action_dim=train_config.action_dim_actual)],
    )

    # Dataset-only: rename the dotted LeRobot feature keys to the slash-form the
    # transforms use. Never runs at inference (the robot client sends slash keys).
    acting = f"observation.images.cam_{train_config.hand}_wrist"
    context_hand = "left" if train_config.hand == "right" else "right"
    repack_transforms = _transforms.Group(
        inputs=[
            _transforms.RepackTransform({
                "observation/image_wrist": acting,
                "observation/image_context": f"observation.images.cam_{context_hand}_wrist",
                # `action` alone carries the pose history, the gripper history
                # AND the chunk targets; UmiSplitGathered decodes the layout.
                # `observation.state` is deliberately not mapped — see
                # umi_transforms.make_delta_timestamps.
                "action": "action",
                "action_is_pad": "action_is_pad",
                "prompt": "prompt",
            })
        ]
    )

    return _config.DataConfig(
        repo_id=train_config.repo_id,
        asset_id=train_config.repo_id,
        norm_stats={},   # see docstring: makes openpi's Normalize a no-op
        repack_transforms=repack_transforms,
        data_transforms=data_transforms,
        model_transforms=_transforms.Group(inputs=model_inputs, outputs=model_outputs),
        use_quantile_norm=True,
        prompt_from_task=True,
    )


# --------------------------------------------------------------------------
# EgoRelationTrainConfig stack (loss_dim_weights is shared with the UMI stack)
# --------------------------------------------------------------------------


def loss_dim_weights(stats, action_dim_actual: int, gripper_dims, w_gripper: float,
                     rot_dims: tuple[int, ...] = (), rot_block_dims: int = 3) -> tuple[float, ...]:
    """Per-dim flow-loss weights: variance-normalize first, THEN apply w_gripper.

    THREE stages, in this order (each undone by reordering them):

      1. 1/Var(u_d)                   — put every dim on equal footing
      2. rot_block_dims / len(rot_dims) on the rotation dims — hold the rotation
         BLOCK's share of the loss fixed as its dim COUNT changes
      3. w_gripper on the gripper dims — task importance

    Stage 2 exists for the UMI config's `rotation_repr` A/B and is a no-op
    everywhere else (`rot_dims=()` by default; and for rotvec, 3/3 = 1.0, so the
    relational config and every existing rotvec run are bitwise unaffected).
    Without it, switching rotvec -> rot6d silently doubles rotation's share of
    the loss, because the weights are per-DIM and 6d spends six dims on what
    rotvec says in three. Measured on red_block_on_yellow_block_umi at
    w_gripper=3::

        translation / rotation / gripper share of the weighted MSE
        rotvec              33.3%   33.3%   33.3%
        rot6d, no stage 2   25.0%   50.0%   25.0%     <- reweighting, not repr
        rot6d, stage 2      33.3%   33.3%   33.3%

    so an A/B without it would be measuring the reweighting as much as the
    encoding. `rot_block_dims` is the reference width the block is normalized
    to (3 = "however many dims it takes, rotation is worth three EEF dims").

    Why stages 1 and 3 must be separate. The flow target is u = noise - x1, so
    Var(u_d) = 1 + Var(x1_d). With grippers left raw at +-1 and EEF dims quantile
    normalized to std ~0.32, that is 1.83 vs 1.10 -- so an UNWEIGHTED MSE already
    hands the 2 gripper dims 21.6% of the loss, not 2/14 = 14.3%. Multiplying by a
    bare 3 on top of that would land at 58%, not the intended 33%.

    Stage 1 (1/Var(u_d)) puts every dim on equal footing. Stage 3 (w_gripper on the
    gripper dims) is then a statement about task importance whose meaning does not
    move when the normalization scheme changes: w_gripper=3 gives the grippers
    2*3/(12+2*3) = 33% of the loss, and would still mean "worth 3 EEF dims each"
    under the pooled_floored_gain scheme.

    Normalized to mean 1 so the reported loss stays on the same scale as an
    unweighted run and remains comparable across weightings.
    """
    import numpy as np

    var = stats.provenance.get("model_space_variance")
    if var is None:
        raise ValueError(
            "relation_stats.npz carries no `model_space_variance` in its provenance; "
            "re-run `python -m ego2g1.train.compute_norm_stats --config relation` with "
            "this code version (the loss weights are derived from it, and guessing it "
            "would silently change what w_gripper means)"
        )
    var = np.asarray(var, dtype=np.float64)[:action_dim_actual]
    w = 1.0 / (1.0 + var)
    if rot_dims:
        w[list(rot_dims)] *= rot_block_dims / len(rot_dims)
    w[list(gripper_dims)] *= w_gripper
    w = w / w.mean()
    return tuple(float(x) for x in w)


def create_relation_data_config(
    train_config,
    model_config,
    *,
    stats_dir: pathlib.Path | str,
    skip_norm_stats: bool = False,
    shuffle_objects: bool | None = None,
    swap_relations: bool = False,
    include_objects: bool = True,
) -> _config.DataConfig:
    """Assemble the relational DataConfig.

    Placement notes, all load-bearing:

    - `norm_stats={}` makes openpi's mandatory `Normalize` step a NO-OP. It is
      inserted unconditionally by `transform_dataset` and cannot be removed
      without forking that function; an empty dict is the supported way to opt
      out, because Normalize only touches keys present in the stats. Both of our
      normalizations are per-(slot, dim) or pooled-across-objects and cannot be
      expressed in openpi's per-dim NormStats anyway.
    - Both normalizers live in `model_transforms`, so `compute_norm_stats` --
      which applies repack + data_transforms only -- observes RAW actions and RAW
      relations, which is exactly what it must measure.
    - `RelationPrompt` owns both the object order and the relation row order, so
      the name->vector pairing cannot be broken by a later edit to one of them.

    `shuffle_objects` overrides the config default: train shuffles, val does not
    (a fixed order keeps the val curve comparable across steps).
    """
    from ego2g1.train import norm as _norm
    from ego2g1.train import relation_transforms as _rt

    stats_dir = pathlib.Path(stats_dir)
    shuffle = train_config.shuffle_object_order if shuffle_objects is None else shuffle_objects

    model_inputs = []
    model_outputs = []
    if not skip_norm_stats:
        stats = _norm.load_relation(stats_dir)
        d_real = train_config.action_dim_actual
        model_inputs.append(_rt.NormalizeRelations(
            mean=stats.relation_mean, std=stats.relation_std, clip=train_config.state_norm_clip,
        ))
        if train_config.action_norm_scheme != "per_slot_quantile":
            raise NotImplementedError(
                f"action_norm_scheme={train_config.action_norm_scheme!r} is declared in the config "
                "and its stats artifact exists, but only 'per_slot_quantile' is wired here yet"
            )
        model_inputs.append(_rt.PerSlotQuantizeActions(
            q01=stats.action_q01[:, :d_real], q99=stats.action_q99[:, :d_real],
            gripper_dims=train_config.gripper_dims, clamp=train_config.model_space_clamp,
        ))
        model_outputs.append(_rt.PerSlotQuantizeActionsInverse(
            q01=stats.action_q01[:, :d_real], q99=stats.action_q99[:, :d_real],
            gripper_dims=train_config.gripper_dims,
        ))

    model_inputs += [
        _transforms.ResizeImages(224, 224),
        _rt.RelationTokenizePrompt(max_token_len=model_config.max_token_len),
        _transforms.PadStatesAndActions(model_config.action_dim),
    ]

    data_transforms = _transforms.Group(
        inputs=[
            _rt.RelativeEEFRotvecActions(hands=tuple(train_config.hands)),
            _rt.RelationPrompt(
                object_prompt_names=tuple(train_config.object_prompt_names),
                hands=tuple(train_config.hands),
                shuffle=shuffle,
                control_mode=train_config.control_mode,
                swap_relations=swap_relations,
                include_objects=include_objects,
            ),
            _rt.RelationInputs(model_type=model_config.model_type),
        ],
        outputs=[_rt.RelationOutputs(action_dim=train_config.action_dim_actual)],
    )

    # Dataset-only: rename the dotted LeRobot feature keys to the slash-form the
    # transforms use. Never runs at inference (the robot client sends slash keys).
    repack_transforms = _transforms.Group(
        inputs=[
            _transforms.RepackTransform({
                "observation/image": "observation.images.camera0",
                "observation/state": "observation.state",
                "observation/action_reference_tcp": "observation.action_reference_tcp",
                "action": "action",
                "prompt": "prompt",
            })
        ]
    )

    return _config.DataConfig(
        repo_id=train_config.repo_id,
        asset_id=train_config.repo_id,
        norm_stats={},   # see docstring: makes openpi's Normalize a no-op
        repack_transforms=repack_transforms,
        data_transforms=data_transforms,
        model_transforms=_transforms.Group(inputs=model_inputs, outputs=model_outputs),
        use_quantile_norm=True,
        prompt_from_task=True,
    )
