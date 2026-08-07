"""Data transforms for the UMI config (`UmiTrainConfig`).

One acting arm, two wrist cameras, no scene perception, no absolute
proprioception. Everything the policy knows about its own body arrives as a
short window of recent TCP poses expressed in the CURRENT pose's frame,
injected as prompt tokens the same way the relational config injects
object-relation vectors.

Dataset contract -- `red_block_on_yellow_block_umi`, read from its `info.json`
(`robot_type: Unitree_G1_Dex1_VirtualTCP_EEF_columns_right`, 117 episodes,
41207 frames, 30 fps)::

    observation.images.cam_right_wrist   acting arm's wrist camera (480x640)
    observation.images.cam_left_wrist    static arm's camera (workspace context)
    observation.state              (1,)  gripper, native units (measured 1.20 .. 5.40)
    action                        (10,)  absolute TCP pose vec9 + gripper

    vec9 = [tx, ty, tz, R[:,0], R[:,1]]   (ego2g1.core.rot6d)

THE OFF-BY-ONE THAT DEFINES EVERYTHING BELOW. This dataset stores no separate
reference-pose column: `action[t]` is the state at tick **t+1**, verified
exactly on the one column that appears in both places --
``action[t, 9] == observation.state[t+1, 0]`` holds to **0.0 across all 117
episodes / 41207 frames**. So the absolute state at tick u is ``action[u-1]``,
and the chunk anchor, the pose history AND the gripper history are all gathered
from `action` at negative offsets. `observation.state` is never read: it is the
same gripper numbers at a one-tick-different alignment, so reading it would mean
a second gather at a second offset with a second pad mask, all to obtain
bit-identical values.

Tick 0 of every episode therefore has no real anchor at all -- `action[-1]` does
not exist -- and is excluded via `anchor_bad` (ego2g1.train.dataset
.umi_extraction_meta) rather than being served a clamped one.

MEASURED, not commanded. In a UMI recording there is no commanded pose -- a
human moves the gripper and the pose is tracked -- so measured is the only
thing that exists at training time. Deployment must therefore feed measured
poses too, or the history distribution shifts by the arm's tracking error,
which is not small (it is the quantity the deploy watchdog exists to monitor).

Stack placement (mirrors relation_transforms')::

    repack -> [UmiSplitGathered, UmiRelativeActions, UmiStateHistory,
               UmiPrompt, UmiInputs]      <- SplitGathered emits both histories
           -> Normalize({})   <- deliberately a NO-OP; see data_config
           -> [NormalizeHistory, PerSlotQuantizeActions, ResizeImages,
               RelationTokenizePrompt, PadStatesAndActions]
    outputs: [PerSlotQuantizeActionsInverse] -> Unnormalize -> [UmiOutputs]

Both normalizers live in model_transforms so that `compute_norm_stats` -- which
applies repack + data_transforms only -- observes RAW actions and RAW history.

Reused wholesale from `relation_transforms`: `PerSlotQuantizeActions` and its
inverse, and `RelationTokenizePrompt` (verbatim prompt tokenization). Reused
from `relation.py`: `RelationEncoder`, which is generic -- an MLP applied
identically to every row of an (n, d) matrix, RMSNormed and scaled relative to
the embedding it is added to. The model-side fields are still spelled
`n_objects` / `relation_dim` / `relation_hidden`; for this config they mean
"number of history tokens" and "width of one history vector". They were left
alone on purpose: renaming them would touch the relational path for no
behavioural gain.
"""

import dataclasses
import random

import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model

from ego2g1.core import rot6d as _rot6d
from ego2g1.core import rotvec as _rotvec
from ego2g1.train.relation_transforms import (  # noqa: F401  (re-exported for the stack)
    RELATION_SENTINEL,
    PerSlotQuantizeActions,
    PerSlotQuantizeActionsInverse,
    RelationTokenizePrompt,
)
from ego2g1.train.transforms import CONTROL_MODE_EEF, _parse_image

# Same reserved single-token sentinel as the relational config (`<unused0>`).
# One vocabulary item serves both: nothing in the model distinguishes them, and
# the k-th occurrence always receives the k-th injected row whatever it means.
HISTORY_SENTINEL = RELATION_SENTINEL

# Per-lag vector width: 3 translation + 3 rotation-vector + 1 gripper.
HISTORY_DIM = 7

# Absolute pose width in the dataset's `action` column, before the gripper.
POSE_DIM = 9

# The prompt segment carrying the injected history tokens. Named for what it
# holds -- OBSERVED past state, not past commands. If a commanded-pose variant
# ever exists, the two must be distinguishable in the prompt of a checkpoint
# you find on disk six months from now.
HISTORY_SEGMENT = "State history:"


def lag_ticks(history_lags: int, history_stride: int) -> tuple[int, ...]:
    """Tick offsets into the past, most recent first: (0, s, 2s, ..., k*s).

    Lag 0 is included deliberately and is NOT a wasted slot. Its pose part is
    the anchor relative to itself, i.e. structurally zero -- but its gripper
    dim is the CURRENT aperture, which is the single most important
    proprioceptive number the policy has and which nothing else in this config
    would deliver (there is no gripper word in the prompt; see `UmiPrompt`).
    It also makes the two intended operating modes one knob: a history
    truncated to length 1 is exactly "current gripper state, no motion
    history".
    """
    if history_lags < 0:
        raise ValueError(f"history_lags={history_lags} must be >= 0")
    if history_stride < 1:
        raise ValueError(f"history_stride={history_stride} must be >= 1")
    return tuple(j * history_stride for j in range(history_lags + 1))


def make_delta_timestamps(action_horizon: int, fps: float, lags: tuple[int, ...]) -> dict:
    """delta_timestamps for LeRobotDataset. ONE key, both directions.

    `action` is the only column this config reads (see the module docstring's
    off-by-one):

        rows 0 .. n_lags-1   offset -(1 + lag)/fps  -> full state at t - lag
        rows n_lags ..       offset  +k/fps         -> target at t+1+k

    `observation.state` is deliberately NOT gathered even though it is the
    gripper column, because `action[u-1, 9] == observation.state[u, 0]` holds to
    0.0 across all 117 episodes / 41207 frames -- so the gripper at tick t-lag is
    already sitting in column 9 of the pose row being fetched for that same tick.
    Gathering it separately would mean a second array at a DIFFERENT offset
    (-lag/fps rather than -(1+lag)/fps, because the two columns are aligned one
    tick apart) plus a second pad mask, to obtain a bit-identical number. One
    gather has no offsets to keep in sync and no masks to disagree.

    `UmiSplitGathered` is the only place this layout is decoded.

    LeRobot clamps out-of-range offsets to the episode boundary and reports which
    entries it invented via `action_is_pad`, which `UmiStateHistory` reads.
    Ignoring it would be a silent corruption: the first k*s ticks of every episode
    would get frame 0 repeated, so the history would read "stationary" during
    exactly the approach the policy is deciding how to start.
    """
    h = int(action_horizon)
    return {"action": [-(1 + l) / fps for l in lags] + [k / fps for k in range(h)]}


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UmiSplitGathered(_transforms.DataTransformFn):
    """Split the combined `action` gather into history rows and chunk targets.

    Pure layout, isolated in one transform so that exactly one place in the
    codebase knows how `make_delta_timestamps` packed the two directions
    together. Everything downstream sees named fields.

    The gripper history comes out of column 9 of the SAME rows as the pose
    history, not from a separate `observation.state` gather -- they are the same
    numbers (see make_delta_timestamps), and taking them from one array is what
    makes it structurally impossible for the pose and the gripper to end up
    describing different ticks.
    """

    n_lags: int

    def __call__(self, data: dict) -> dict:
        if "action" not in data:
            return data
        out = dict(data)
        gathered = np.asarray(out.pop("action"), dtype=np.float64)
        if gathered.ndim != 2 or gathered.shape[-1] != POSE_DIM + 1:
            raise ValueError(
                f"action: expected (n_lags + H, {POSE_DIM + 1}), got {gathered.shape}"
            )
        if gathered.shape[0] <= self.n_lags:
            raise ValueError(
                f"action gather has {gathered.shape[0]} rows <= n_lags {self.n_lags}: "
                "make_delta_timestamps and this transform disagree about the layout"
            )
        history = gathered[: self.n_lags]
        out["observation/pose_history"] = history[:, :POSE_DIM]
        out["observation/gripper_history"] = history[:, POSE_DIM:]
        out["observation/targets"] = gathered[self.n_lags:]

        is_pad = out.pop("action_is_pad", None)
        if is_pad is not None:
            # one mask for both, because there is one gather
            out["observation/pose_history_is_pad"] = np.asarray(is_pad).reshape(-1)[: self.n_lags]
        return out


# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UmiRelativeActions(_transforms.DataTransformFn):
    """Absolute stored poses -> anchor-relative 7-dim action chunk.

    For each slot k::

        delta_k = inv(T_anchor) @ T_target_k
        out[k]  = [delta_k translation (3), log(delta_k rotation) (3), gripper]

    The anchor is row 0 of the pose history, i.e. the measured TCP at the
    observation tick. Reading it from there rather than from a field of its own
    is the point: ONE anchor, one source, so the action frame and the history
    frame cannot drift apart under a later edit.

    Rotation is a ROTATION VECTOR, not 6d. Over this dataset's 50-slot chunk the
    relative rotation is small, so a 6d encoding would sit on the identity
    matrix and the signal would live inside the encoding's own noise floor; as
    rotation vectors, measured here, slot 0 has std 0.005-0.010 rad and slot 49
    has 0.15-0.33. The log map is unique only for |theta| < pi, which bounds the
    horizon -- max observed on this dataset at H=50 is 1.21 rad, comfortable but
    re-measure before raising H. See ego2g1.core.rotvec.

    The gripper is CONTINUOUS and passes through raw, to be normalized per
    (slot, dim) alongside every other dim. Deliberately not binarized: a
    two-point target is the maximally multimodal case for flow matching, whose
    regression target near the transition is the mean of the two modes -- i.e.
    ambiguous exactly at the frames where grasp timing matters. This dataset has
    the real ramp (measured: ~14% of frames sit strictly between the open and
    closed plateaus), so there is a genuine graded signal to learn.
    """

    def __call__(self, data: dict) -> dict:
        if "observation/targets" not in data or "observation/pose_history" not in data:
            return data
        out = dict(data)
        target = np.asarray(out.pop("observation/targets"), dtype=np.float64)     # (H, 10)
        poses = np.asarray(out["observation/pose_history"], dtype=np.float64)     # (n_lags, 9)
        if poses.ndim != 2 or poses.shape[-1] != POSE_DIM:
            raise ValueError(f"pose_history: expected (n_lags, {POSE_DIM}), got {poses.shape}")

        t_anchor = _rot6d.vec9_to_se3(poses[0])                                   # (4, 4)
        t_tgt = _rot6d.vec9_to_se3(target[:, :POSE_DIM])                          # (H, 4, 4)
        delta = np.linalg.inv(t_anchor) @ t_tgt

        eef = np.empty((target.shape[0], 6), dtype=np.float64)
        eef[:, :3] = delta[:, :3, 3]
        eef[:, 3:] = _rotvec.mat_to_rotvec(delta[:, :3, :3])
        out["actions"] = np.concatenate([eef, target[:, POSE_DIM:]], axis=-1).astype(np.float32)
        return out


@dataclasses.dataclass(frozen=True)
class UmiOutputs(_transforms.DataTransformFn):
    """Trim model padding back to the 7-dim UMI action space."""

    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


# --------------------------------------------------------------------------
# state history
# --------------------------------------------------------------------------


def _sample_length(probs: tuple[float, ...]) -> int:
    """Draw a history length from `probs` (index j = P(length j)).

    stdlib `random`, NOT np.random: torch seeds Python's random per worker per
    epoch but does NOT seed numpy, so numpy would hand every dataloader worker
    the identical sequence.
    """
    return random.choices(range(len(probs)), weights=probs, k=1)[0]


@dataclasses.dataclass(frozen=True)
class UmiStateHistory(_transforms.DataTransformFn):
    """Gathered absolute poses + grippers -> (n_lags, 7) anchor-relative rows.

    Row j is the state at tick `t - lags[j]`, expressed in the anchor's frame::

        delta_j = inv(T_anchor) @ T_j
        row_j   = [delta_j translation (3), log(delta_j rotation) (3), gripper_j]

    Row 0's pose part is therefore exactly zero for every sample -- structural,
    documented, and handled by the per-lag normalizer's std floor plus an
    explicit exemption in the stats sanity check, rather than hidden. Its
    gripper dim is the current one.

    LENGTH. `length_probs` draws how many leading rows survive; rows beyond that
    are zeroed AND the prompt emits only that many sentinels, so the model
    receives fewer tokens rather than zeroed ones. Truncation is always FROM THE
    STALE END, never from the middle. That is forced by the decision to let RoPE
    carry lag identity rather than writing "t-1:"/"t-2:" labels into the prompt:
    with far-end truncation the j-th token is always tick t - lags[j], so
    position determines lag exactly; drop from the middle and the same position
    could mean two different lags, making the input ambiguous rather than merely
    sparse.

    Availability is intersected with the draw: near an episode start LeRobot
    clamps the gather and flags the invented rows, and those are dropped
    whatever the draw said. So the short-history regime is trained both by the
    episode starts that genuinely produce it and by deliberate draws -- which is
    the point, because those frames are where the policy decides how to initiate
    a motion and they are otherwise a ~2.5% tail.
    """

    length_probs: tuple[float, ...] | None = None   # train: draw; None: no truncation
    fixed_len: int | None = None                    # val: pin the length

    # --- diagnostic ablations (validation pools only; never set for training) ---
    # Permute the surviving rows. Isolates whether lag ORDER is read at all --
    # which is the experiment that VALIDATES letting RoPE carry lag identity. If
    # val loss barely moves under permutation, RoPE is not doing the job and
    # explicit per-lag text labels are required after all. This is a gate on the
    # design, not a nice-to-have.
    permute: bool = False
    # Replace the whole history with another sample's. A policy that merely USES
    # the history degrades toward the no-history loss; one that has learned to
    # DEAD-RECKON off it (extrapolate constant velocity, ignore the images) is
    # destroyed. That is the main risk of this feature and this pool detects it.
    pool: np.ndarray | None = None   # (N, n_lags, 7) raw history blocks

    def __call__(self, data: dict) -> dict:
        if "observation/pose_history" not in data:
            return data
        out = dict(data)
        poses = np.asarray(out.pop("observation/pose_history"), dtype=np.float64)
        grip = np.asarray(out.pop("observation/gripper_history"), dtype=np.float64).reshape(
            poses.shape[0], -1
        )
        if grip.shape[-1] != 1:
            raise ValueError(
                f"observation/gripper_history: expected (n_lags, 1) continuous gripper, "
                f"got {grip.shape}"
            )
        n_lags = poses.shape[0]

        is_pad = out.pop("observation/pose_history_is_pad", None)
        if is_pad is None:
            n_avail = n_lags
        else:
            is_pad = np.asarray(is_pad).reshape(-1).astype(bool)
            # padding is contiguous from the stale end (LeRobot clamps to the
            # episode start), so the first True is the count of real rows
            n_avail = int(np.argmax(is_pad)) if bool(is_pad.any()) else n_lags
        if n_avail < 1:
            raise ValueError(
                "lag 0's pose is padding, i.e. this anchor has no real pose at all. "
                "Tick 0 of every episode is like this (see the module docstring's "
                "off-by-one) and must be excluded via anchor_bad, not trained on"
            )

        t_anchor = _rot6d.vec9_to_se3(poses[0])
        t_lags = _rot6d.vec9_to_se3(poses)                                   # (n_lags, 4, 4)
        delta = np.linalg.inv(t_anchor) @ t_lags
        rows = np.empty((n_lags, HISTORY_DIM), dtype=np.float64)
        rows[:, :3] = delta[:, :3, 3]
        rows[:, 3:6] = _rotvec.mat_to_rotvec(delta[:, :3, :3])
        rows[:, 6] = grip[:, 0]

        if self.pool is not None:
            rows = np.array(self.pool[random.randrange(len(self.pool))], dtype=np.float64)
            if rows.shape != (n_lags, HISTORY_DIM):
                raise ValueError(f"history pool entry {rows.shape} vs ({n_lags}, {HISTORY_DIM})")

        if self.fixed_len is not None:
            drawn = int(self.fixed_len)
        elif self.length_probs is not None:
            drawn = _sample_length(self.length_probs)
        else:
            drawn = n_lags
        length = max(0, min(drawn, n_avail, n_lags))

        if self.permute and length > 1:
            order = list(range(length))
            while True:
                candidate = list(order)
                random.shuffle(candidate)
                if candidate != order:      # a no-op "permutation" measures nothing
                    order = candidate
                    break
            rows[:length] = rows[order]

        rows[length:] = 0.0
        out["history"] = rows.astype(np.float32)
        out["history_len"] = int(length)
        # `state` is never read by the pi05 model (there is no continuous state
        # token on that path -- see Ego2G1TrainConfig.discrete_state_input), but
        # Observation requires the field and openpi derives the batch shape from
        # it. Carry the current gripper: honest, and the right thing to find in a
        # dump when debugging.
        out["state"] = grip[:1, 0].astype(np.float32)
        return out


@dataclasses.dataclass(frozen=True)
class NormalizeHistory(_transforms.DataTransformFn):
    """z-score the history rows PER LAG, then clip.

    Per-lag, NOT pooled across lags -- and this deliberately INVERTS the rule
    the relational config follows. There, pooling across objects is a
    correctness requirement: one shared encoder plus a shuffled prompt order
    means per-object stats would make the same physical relation encode
    differently depending on which slot it landed in. Here there is no shuffling
    and lag identity is fixed by prompt position, while the lags are emphatically
    NOT exchangeable -- displacement grows with lag, so pooled stats would leave
    the near lags at a fraction of unit scale. Copying the relational rule here
    is the trap.

    Rows past `history_len` are zeros on the way in and become the constant
    `-mean/std` on the way out. They are never gathered into the prompt (only
    `history_len` sentinels are emitted), so they reach nothing; they are left
    alone rather than re-masked because a mask would have to be threaded through
    `Observation` for no observable effect.
    """

    mean: np.ndarray   # (n_lags, 7)
    std: np.ndarray    # (n_lags, 7)
    clip: float = 5.0

    def __call__(self, data: dict) -> dict:
        if "relations" not in data:
            return data
        rel = np.asarray(data["relations"], dtype=np.float32)
        if rel.shape[-2:] != self.mean.shape:
            raise ValueError(f"history {rel.shape} vs per-lag stats {self.mean.shape}")
        z = (rel - self.mean) / np.maximum(self.std, 1e-6)
        return {**data, "relations": np.clip(z, -self.clip, self.clip).astype(np.float32)}


# --------------------------------------------------------------------------
# prompt / model inputs
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UmiPrompt(_transforms.DataTransformFn):
    """Build the full prompt, with one sentinel per surviving history row::

        Task: {task} <<<control_mode>>> end effector <<<control_mode>>>
          State history: <unused0> <unused0> ... Action:

    At length 0 the whole segment is dropped, which is what makes "no
    proprioception at all" a VALUE of this config rather than a second code path.

    There is no gripper WORD. The gripper is continuous, so a word would need a
    threshold, and choosing that threshold is exactly the ambiguity that made an
    auxiliary binary grasp head not worth having. The number travels in dim 6 of
    every history token instead, at full precision.
    """

    control_mode: str = CONTROL_MODE_EEF
    # Fallback task string, used only when the sample carries no `prompt`. The
    # real task comes from the LeRobot task table via openpi's
    # PromptFromLeRobotTask ("Place the red block on top of the yellow block.").
    task: str = ""

    def __call__(self, data: dict) -> dict:
        out = dict(data)
        task = out.pop("prompt", None) or self.task
        if not isinstance(task, str):
            task = task.item()
        if not task:
            raise ValueError(
                "UmiPrompt needs a task string: either a `prompt` key "
                "(PromptFromLeRobotTask) or a non-empty `task` fallback"
            )
        out["prompt"] = self.build_prompt(task, int(out.get("history_len", 0)))
        return out

    def build_prompt(self, task: str, history_len: int) -> str:
        marker = f"<<<control_mode>>> {self.control_mode} <<<control_mode>>>"
        if history_len <= 0:
            return f"Task: {task} {marker} Action: "
        tokens = " ".join([HISTORY_SENTINEL] * history_len)
        return f"Task: {task} {marker} {HISTORY_SEGMENT} {tokens} Action: "


@dataclasses.dataclass(frozen=True)
class UmiInputs(_transforms.DataTransformFn):
    """Repack into pi0 model inputs, carrying the history through.

    Camera slots are a real decision, not bookkeeping. openpi's
    `preprocess_observation` gates spatial augmentation on the literal substring
    "wrist" in the key: keys WITHOUT it get RandomCrop(0.95) + Resize +
    Rotate(+-5 deg), keys with it get ColorJitter only.

    - The ACTING arm's wrist camera must occupy a "wrist" slot. Its geometry is
      rigidly coupled to the anchor-relative action labels, so a +-5 degree
      rotation of that image would be an unmodelled rotation between input and
      label.
    - The static arm's camera goes to `base_0_rgb` and therefore DOES get the
      spatial augmentation -- which is what you want here: it is a fixed
      workspace view (this setup's stand-in for the head camera it does not
      have) whose geometry the action labels do not depend on, so augmentation
      is pure generalization. This rests entirely on that arm holding its pose.
      `context_is_static` records the assumption in the checkpoint rather than
      leaving it implicit; if the arm ever moves, the augmentation becomes a lie.

    The remaining wrist slot is zero-filled and masked out (pi05's inputs_spec
    always declares three cameras). Masking removes its tokens from attention but
    SigLIP still encodes the zeros, so it costs about a third of the image
    compute -- worth reclaiming later by overriding `inputs_spec` and
    `preprocess_observation`'s `image_keys`, not worth deviating from
    pretraining for on a first run.
    """

    model_type: _model.ModelType
    acting_slot: str = "right_wrist_0_rgb"
    context_is_static: bool = True

    def __post_init__(self):
        if self.acting_slot not in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
            raise ValueError(
                f"acting_slot={self.acting_slot!r} must be a wrist slot: the acting "
                "camera's geometry is coupled to the action labels, and a non-wrist "
                "key silently enables crop/rotate augmentation on it"
            )
        if not self.context_is_static:
            raise NotImplementedError(
                "context_is_static=False means the context camera moves, so it must not "
                "receive base_0_rgb's spatial augmentation; that variant needs its own "
                "slot assignment and has not been designed"
            )

    def __call__(self, data: dict) -> dict:
        acting = _parse_image(data["observation/image_wrist"])
        context = _parse_image(data["observation/image_context"])
        idle_slot = (
            "left_wrist_0_rgb" if self.acting_slot == "right_wrist_0_rgb" else "right_wrist_0_rgb"
        )
        inputs = {
            "state": data["state"],
            # `relations` is the Observation field name that ego2g1.train
            # .observation_patch adds and that Ego2G1Pi0.embed_prefix reads. It
            # carries the history matrix here; the name is the relational
            # config's, kept so the injection machinery is shared verbatim.
            "relations": data["history"],
            "image": {
                "base_0_rgb": context,
                self.acting_slot: acting,
                idle_slot: np.zeros_like(acting),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                self.acting_slot: np.True_,
                idle_slot: np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs
