"""Data transforms for the relational config (EgoRelationTrainConfig).

Dataset contract (red_block_in_pen_holder_ego, `info.json` -> `ego_relation`):
  observation.state (56,)               2 hands x 3 objects x vec9 (object in
                                        that TCP's frame) + 2 grasp binaries,
                                        laid out HAND-MAJOR
  observation.action_reference_tcp (18,) absolute CURRENT TCP pose, both hands
  action (20,)                          absolute TARGET TCP pose both hands + 2
                                        grasp binaries, where action[t] is the
                                        target at min(t+1, T-1)
  vec9 = [tx, ty, tz, R[:,0], R[:,1]]   identical to ego2g1.core.rot6d, so
                                        vec9_to_se3 decodes it directly

The dataset stores actions ABSOLUTE and defers the relative transform to us
(`training_action_transform: "deferred: inv(T_current) @ T_absolute_target"`).
`action[t] == observation.action_reference_tcp[t+1]` holds byte-exactly across
all 50 episodes, so the reference field is the only pose source we need.

Stack placement (mirrors ego2g1.train.transforms' contract):
  repack -> [RelativeEEFRotvecActions, RelationPrompt, RelationInputs]
         -> Normalize({})  <- deliberately a NO-OP; see data_config
         -> [NormalizeRelations, PerSlotQuantizeActions, ResizeImages,
             RelationTokenizePrompt, PadStatesAndActions]
  outputs: [PerSlotQuantizeActionsInverse] -> Unnormalize -> [RelationOutputs]

Both normalizers live in model_transforms so that `compute_norm_stats` -- which
applies repack + data_transforms only -- sees RAW actions and RAW relations,
which is exactly what it must measure.
"""

import dataclasses
import random

import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer

from ego2g1.core import rot6d as _rot6d
from ego2g1.core import rotvec as _rotvec
from ego2g1.train.transforms import CONTROL_MODE_EEF, _parse_image

# Reserved vocabulary token marking where a relation embedding is substituted
# into the prompt. `<unused0>` is a Gemma reserved token: it encodes to exactly
# one id (7), never occurs in natural text, and already has a row in the
# pretrained embedding table -- so the prompt length and the param tree are both
# unchanged, and embed_prefix only has to overwrite the embedding at that slot.
#
# The SAME sentinel is used for every object, deliberately: a per-slot sentinel
# would leak the object's position in the prompt into its token identity and
# defeat the order shuffling. The k-th sentinel occurrence receives the k-th
# relation row, and RelationPrompt permutes names and rows together.
RELATION_SENTINEL = "<unused0>"

# Words the grasp binary becomes in the prompt. The gripper is binary in this
# dataset (the `binary` export variant dropped the continuous Revo2 motor
# commands), so a word is the honest encoding -- and it costs one token instead
# of six digits.
GRASP_WORDS = {0: "open", 1: "closed"}


def sentinel_token_id(tokenizer=None) -> int:
    """Resolve RELATION_SENTINEL to its single vocabulary id.

    Derived from the live tokenizer rather than hard-coded, and asserted to be a
    single token: a sentinel that silently split into pieces would put the
    encoder's output on only the first piece and leave the rest as garbage text.
    """
    sp = (tokenizer or _tokenizer.PaligemmaTokenizer(48))._tokenizer  # noqa: SLF001
    ids = sp.encode(RELATION_SENTINEL)
    if len(ids) != 1:
        raise ValueError(
            f"{RELATION_SENTINEL!r} tokenizes to {len(ids)} tokens ({ids}); "
            "the injection sentinel must be exactly one token"
        )
    return int(ids[0])


def make_delta_timestamps(action_horizon: int, fps: float) -> dict:
    """delta_timestamps for LeRobotDataset.

    Only `action` is gathered over the chunk: action[t] already IS the target at
    t+1, so slots 0..H-1 of the gathered window cover targets t+1..t+H. The
    anchor's own `observation.action_reference_tcp` and `observation.state` are
    that frame's values and need no delta entry.
    """
    h = int(action_horizon)
    return {"action": [k / fps for k in range(h)]}


# --------------------------------------------------------------------------
# actions
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RelativeEEFRotvecActions(_transforms.DataTransformFn):
    """Absolute stored poses -> anchor-relative 14-dim action chunk.

    Per hand, for each slot k:
        delta_k   = inv(T_current) @ T_target_k
        out[k]    = [delta_k translation (3), log(delta_k rotation) (3)]
    plus one gripper dim per hand, mapped {0 open, 1 closed} -> {-1, +1}.

    Output layout puts BOTH grippers at the tail:
        [L_dx L_dy L_dz L_rx L_ry L_rz | R_dx ... R_rz | L_grip R_grip]
    so "the gripper dims" is a slice -- the loss weighting, the normalization
    exemption and the stamp all refer to the same contiguous block.

    Rotation is a ROTATION VECTOR, not 6d. Over a 50-slot chunk the relative
    rotation is small, so a 6d encoding sits on the identity matrix: measured at
    slot 0 its diagonal entries have mean 0.99994 and std 1.4e-4, so any
    normalizer that gives them unit scale amplifies ~7000x and the signal lives
    entirely inside that noise floor. The same rotations as rotation vectors have
    std 7e-3..2e-2. The log map is unique only for |theta| < pi, which bounds the
    horizon: max observed here is 1.82 rad, comfortably inside.

    A sample without the pose keys (a state-only inference request) passes
    through untouched, matching the openpi transform-pipeline contract.
    """

    hands: tuple[str, ...] = ("left", "right")

    def __call__(self, data: dict) -> dict:
        if "action" not in data or "observation/action_reference_tcp" not in data:
            return data
        out = dict(data)
        target = np.asarray(out.pop("action"), dtype=np.float64)             # (H, 20)
        ref = np.asarray(out.pop("observation/action_reference_tcp"), dtype=np.float64)  # (18,)

        n_hands = len(self.hands)
        if ref.shape != (9 * n_hands,):
            raise ValueError(f"action_reference_tcp: expected ({9 * n_hands},), got {ref.shape}")
        if target.ndim != 2 or target.shape[-1] != 9 * n_hands + n_hands:
            raise ValueError(
                f"action: expected (H, {9 * n_hands + n_hands}), got {target.shape}"
            )
        horizon = target.shape[0]

        eef = np.empty((horizon, 6 * n_hands), dtype=np.float64)
        for h in range(n_hands):
            t_cur = _rot6d.vec9_to_se3(ref[h * 9:(h + 1) * 9])              # (4, 4)
            t_tgt = _rot6d.vec9_to_se3(target[:, h * 9:(h + 1) * 9])        # (H, 4, 4)
            delta = np.linalg.inv(t_cur) @ t_tgt                            # (H, 4, 4)
            eef[:, h * 6:h * 6 + 3] = delta[:, :3, 3]
            eef[:, h * 6 + 3:h * 6 + 6] = _rotvec.mat_to_rotvec(delta[:, :3, :3])

        grip = target[:, 9 * n_hands:] * 2.0 - 1.0                          # (H, n_hands)
        out["actions"] = np.concatenate([eef, grip], axis=-1).astype(np.float32)
        return out


@dataclasses.dataclass(frozen=True)
class RelationOutputs(_transforms.DataTransformFn):
    """Trim model padding back to the 14-dim relational action space."""

    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


# --------------------------------------------------------------------------
# state -> relations + prompt
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RelationPrompt(_transforms.DataTransformFn):
    """Split observation.state into relation rows + grasp words, and build the prompt.

    The dataset lays the 54 relation dims out HAND-MAJOR
        [left->obj0, left->obj1, left->obj2, right->obj0, right->obj1, right->obj2]
    while the encoder wants one row per OBJECT spanning both hands, so this
    interleaves:
        row_k = concat(state[9k : 9k+9], state[27+9k : 27+9k+9])

    Object order in the prompt is shuffled (train only) and the SAME permutation
    is applied to the relation rows, so the name->vector pairing is preserved.
    Owning both in one transform is deliberate: splitting them across two
    transforms would make that invariant something a future edit could break
    silently, and a mispaired object is exactly the bug that would still train to
    a plausible-looking loss.

    Emits:
      relations (n_objects, relation_dim)  RAW; NormalizeRelations z-scores it
      state     (n_hands,)                 the grasp binaries -- the only
                                           non-relational quantity there is
      prompt    str                        the full pi05-style prompt
    """

    object_prompt_names: tuple[str, ...]
    hands: tuple[str, ...] = ("left", "right")
    shuffle: bool = False
    control_mode: str = CONTROL_MODE_EEF
    # Fallback task string, used only when the sample carries no `prompt`. The
    # real task comes from the LeRobot task table via openpi's
    # PromptFromLeRobotTask, so it is not duplicated in the training config.
    task: str = ""

    # --- diagnostic ablations (validation pools only; never set for training) ---
    # Permute the relation ROWS while leaving the names in place, so "red cube"
    # is paired with another object's geometry. Isolates referential binding: if
    # the loss barely moves, the policy is solving the task from generic geometry
    # ("approach the nearest graspable thing") and ignoring which object the
    # instruction actually names -- which still scores well on a single-task
    # dataset and fails the moment the instruction matters.
    swap_relations: bool = False
    # Drop the whole `Objects:` segment. No sentinels means no injection at all,
    # so this is the relational equivalent of a state-blind pool.
    include_objects: bool = True

    def __call__(self, data: dict) -> dict:
        out = dict(data)
        task = out.pop("prompt", None) or self.task
        if not isinstance(task, str):
            task = task.item()
        if not task:
            raise ValueError(
                "RelationPrompt needs a task string: either a `prompt` key "
                "(PromptFromLeRobotTask) or a non-empty `task` fallback"
            )
        state = np.asarray(out["observation/state"], dtype=np.float64).reshape(-1)
        n_hands = len(self.hands)
        n_obj = len(self.object_prompt_names)
        expected = 9 * n_hands * n_obj + n_hands
        if state.shape[0] != expected:
            raise ValueError(
                f"observation/state: expected ({expected},) = "
                f"9*{n_hands}*{n_obj} relation + {n_hands} grasp, got {state.shape}"
            )

        per_hand = 9 * n_obj
        rows = [
            np.concatenate([state[h * per_hand + 9 * k: h * per_hand + 9 * k + 9]
                            for h in range(n_hands)])
            for k in range(n_obj)
        ]
        grasp = state[9 * n_hands * n_obj:]

        order = list(range(n_obj))
        if self.shuffle:
            # stdlib `random`, NOT np.random: torch seeds Python's random per
            # worker per epoch but does NOT seed numpy, so numpy would hand
            # every dataloader worker the identical permutation sequence.
            random.shuffle(order)

        # `order` is the name order; `row_order` is the geometry order. They are
        # the same list unless the swap ablation deliberately decouples them.
        row_order = list(order)
        if self.swap_relations and n_obj > 1:
            while True:
                candidate = list(row_order)
                random.shuffle(candidate)
                if candidate != row_order:   # a no-op "swap" would measure nothing
                    row_order = candidate
                    break

        out["relations"] = np.stack([rows[i] for i in row_order]).astype(np.float32)
        out["state"] = grasp.astype(np.float32)
        out["prompt"] = self.build_prompt(task, grasp, order)
        out.pop("observation/state", None)
        return out

    def build_prompt(self, task: str, grasp, order) -> str:
        marker = f"<<<control_mode>>> {self.control_mode} <<<control_mode>>>"
        words = []
        for name, g in zip(self.hands, grasp, strict=True):
            value = int(round(float(g)))
            if value not in GRASP_WORDS:
                raise ValueError(
                    f"{name} grasp flag is {float(g)!r}, expected the binary 0 (open) or "
                    f"1 (closed). This dataset's `binary` export variant stores exactly "
                    f"0/1; a continuous value means the state layout is wrong (most "
                    f"likely the grasp dims are not the last {len(self.hands)})."
                )
            words.append(f"{name.capitalize()} hand: {GRASP_WORDS[value]}")
        hands = " ".join(words)
        if not self.include_objects:
            return f"Task: {task} {marker} {hands} Action: "
        objects = ", ".join(f"{self.object_prompt_names[i]} {RELATION_SENTINEL}" for i in order)
        return f"Task: {task} {marker} {hands} Objects: {objects} Action: "


@dataclasses.dataclass(frozen=True)
class NormalizeRelations(_transforms.DataTransformFn):
    """z-score the relation rows with stats POOLED ACROSS OBJECTS, then clip.

    z-score rather than quantile because the measured distributions are benign:
    rotation dims have std 0.36-0.53 (so the largest z-score gain is 2.8x, versus
    7000x for the relative-action rotations) and translation dims 0.066-0.107,
    with |z| > 5 occurring on 0.0000% of samples. There is no outlier tail for
    quantile normalization to protect against, and z-score gives the encoder
    exactly unit variance.

    The clip is therefore a no-op on this training data and pure insurance
    against a detector glitch at deployment time.
    """

    mean: np.ndarray   # (relation_dim,)
    std: np.ndarray    # (relation_dim,)
    clip: float = 5.0

    def __call__(self, data: dict) -> dict:
        if "relations" not in data:
            return data
        rel = np.asarray(data["relations"], dtype=np.float32)
        z = (rel - self.mean) / np.maximum(self.std, 1e-6)
        return {**data, "relations": np.clip(z, -self.clip, self.clip).astype(np.float32)}


# --------------------------------------------------------------------------
# per-(slot, dim) action normalization
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PerSlotQuantizeActions(_transforms.DataTransformFn):
    """Per-(slot, dim) quantile normalization: q01/q99 -> [-1, 1], then clamp.

    The WHOLE action normalization -- no pooled step underneath. A pooled scheme
    normalizes by statistics dominated by late slots, leaving slot 0 at ~1/20th
    unit scale even though early slots are the only ones that ever execute;
    E001's floored per-slot gain fixes most of that but saturates at 1/c = 10
    where this dataset needs 17-24x. Per-(slot, dim) needs no floor at all and
    lands every slot at std 0.27-0.44.

    Gripper dims are exempt and pass through at +-1: they are already unit-scale,
    bounded, and a quantile map of a two-point distribution is meaningless. The
    loss weighting compensates for the resulting variance difference (see
    EgoRelationTrainConfig.w_gripper).

    The clamp is lossy and has no inverse, by design -- it bounds target
    magnitude at train time only. Measured max |normalized| here is 5.30, so the
    default 10.0 never fires on this data.
    """

    q01: np.ndarray                    # (H, D_real)
    q99: np.ndarray                    # (H, D_real)
    gripper_dims: tuple[int, ...] = ()
    clamp: float | None = 10.0

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"], dtype=np.float64)
        if actions.shape[-2:] != self.q01.shape:
            raise ValueError(f"actions {actions.shape} vs per-slot grid {self.q01.shape}")
        span = self.q99 - self.q01 + 1e-6
        out = 2.0 * (actions - self.q01) / span - 1.0
        if self.clamp is not None:
            out = np.clip(out, -self.clamp, self.clamp)
        if self.gripper_dims:
            idx = list(self.gripper_dims)
            out[..., idx] = actions[..., idx]
        return {**data, "actions": out.astype(np.float32)}


@dataclasses.dataclass(frozen=True)
class PerSlotQuantizeActionsInverse(_transforms.DataTransformFn):
    """Inverse of PerSlotQuantizeActions. MANDATORY before the model output is
    interpreted as metres/radians. Runs on the padded model output (H, D_model >=
    D_real); pad dims pass through untouched."""

    q01: np.ndarray                    # (H, D_real)
    q99: np.ndarray                    # (H, D_real)
    gripper_dims: tuple[int, ...] = ()

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        h, d_real = self.q01.shape
        if actions.shape[-2] != h or actions.shape[-1] < d_real:
            raise ValueError(f"actions {actions.shape} vs per-slot grid {self.q01.shape}")
        out = actions.astype(np.float32).copy()
        span = self.q99 - self.q01 + 1e-6
        real = (out[..., :d_real].astype(np.float64) + 1.0) / 2.0 * span + self.q01
        if self.gripper_dims:
            idx = list(self.gripper_dims)
            real[..., idx] = out[..., idx].astype(np.float64)
        out[..., :d_real] = real.astype(np.float32)
        return {**data, "actions": out}


# --------------------------------------------------------------------------
# model inputs / prompt tokenization
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RelationInputs(_transforms.DataTransformFn):
    """Repack into pi0 model inputs, carrying `relations` through.

    Single egocentric camera goes to base_0_rgb; both wrist slots are
    zero-padded and masked out (same convention as ego2g1.transforms.Ego2G1Inputs
    -- pi05's inputs_spec always declares three cameras).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        inputs = {
            "state": data["state"],
            "relations": data["relations"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RelationTokenizePrompt(_transforms.DataTransformFn):
    """Tokenize the prompt VERBATIM.

    Neither openpi's TokenizePrompt nor ego2g1's Ego2G1TokenizePrompt fits:
    both assemble a `Task: {text}, State: {digits};\\nAction: ` scaffold around
    their input, and RelationPrompt has already produced the complete string
    (including the control-mode marker mid-prompt and the object segment). So the
    only job here is encode + pad/truncate, exactly as stock does it.
    """

    max_token_len: int
    tokenizer: object | None = None

    def _sp(self):
        tok = self.tokenizer or _tokenizer.PaligemmaTokenizer(self.max_token_len)
        return tok._tokenizer  # noqa: SLF001

    def __call__(self, data: dict) -> dict:
        out = dict(data)
        prompt = out.pop("prompt", None)
        if prompt is None:
            raise ValueError("RelationTokenizePrompt requires a prompt")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        tokens = self._sp().encode(prompt, add_bos=True)
        n = len(tokens)
        if n < self.max_token_len:
            pad = [False] * (self.max_token_len - n)
            mask = [True] * n + pad
            tokens = tokens + [0] * (self.max_token_len - n)
        else:
            if n > self.max_token_len:
                raise ValueError(
                    f"prompt tokenizes to {n} tokens > max_token_len {self.max_token_len}; "
                    "truncation would silently cut the object segment (and with it the "
                    "injection sentinels), so it is refused rather than tolerated"
                )
            tokens = tokens[: self.max_token_len]
            mask = [True] * self.max_token_len
        return {
            **out,
            "tokenized_prompt": np.asarray(tokens),
            "tokenized_prompt_mask": np.asarray(mask),
        }
