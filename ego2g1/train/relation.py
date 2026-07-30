"""The two ego2g1-only modules: the object-relation encoder and the grasp head.

RelationEncoder turns each object's R^18 hand-object relation vector into ONE
prompt token embedding, which `Ego2G1Pi0.embed_prefix` substitutes into the
token stream at a reserved sentinel position.

The risk this module exists to manage: pi0.5's prefix has only ever contained
PaliGemma text-token embeddings and SigLIP image tokens. An MLP output dropped
into a token position is out of distribution for the pretrained VLM, and the two
failure modes are opposite and both quiet -- too small and attention ignores it
(an expensive no-op), too large and it destabilizes pretrained attention. Three
safeguards, all on by default:

1. GeGLU, matching Gemma's own FFN nonlinearity, so the encoder is at least
   stylistically the same kind of function as the blocks that will read it.
2. Output RMSNorm with a learned scale INITIALIZED to the mean L2 norm of the
   PaliGemma token-embedding table. The injected token therefore starts life at
   the same magnitude as a real word.
3. Final projection ZERO-initialized, so at step 0 the injected embedding is
   exactly zero and the model is bit-identical to one that ignores the object
   vectors. It has to learn to use them; nothing is injected into a pretrained
   circuit before the gradient asks for it.

`embedding_norm_ratio` is the training canary for (2) and (3): injected-token
norm over mean text-token norm. Order 1 is healthy; orders of magnitude apart
means the encoder is either being ignored or shouting.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


class GeGLU(nnx.Module):
    """Gated linear unit with a GELU gate -- Gemma's FFN nonlinearity.

    Two projections of the same input: a `gate` passed through GELU and a
    `value` left linear, multiplied elementwise. Wider than a plain MLP for the
    same output width, which is the point: the gate lets the layer suppress
    channels conditionally instead of only shifting them.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int, *, rngs: nnx.Rngs,
                 zero_init_out: bool = True):
        self.gate = nnx.Linear(in_features, hidden, rngs=rngs)
        self.value = nnx.Linear(in_features, hidden, rngs=rngs)
        # Zero-init means the whole module outputs exactly 0 at step 0 (see
        # safeguard 3). Bias is zero by nnx default, so no separate handling.
        out_kernel = nnx.initializers.zeros_init() if zero_init_out else nnx.initializers.lecun_normal()
        self.out = nnx.Linear(hidden, out_features, kernel_init=out_kernel, rngs=rngs)

    def __call__(self, x):
        return self.out(nnx.gelu(self.gate(x)) * self.value(x))


class RelationEncoder(nnx.Module):
    """(*b, n_objects, relation_dim) -> (*b, n_objects, width) token embeddings.

    Applied independently and identically to every object, which is what makes
    the prompt's object order shufflable: the encoding is a function of the
    relation vector alone, never of which slot it landed in. Object IDENTITY
    reaches the model through the adjacent text ("red cube"), not through here.
    """

    def __init__(self, relation_dim: int, hidden: int, width: int, *, rngs: nnx.Rngs,
                 target_norm: float = 1.0):
        self.mlp = GeGLU(relation_dim, hidden, width, rngs=rngs, zero_init_out=True)
        # RMSNorm with a learned per-channel scale, initialized so a unit-RMS
        # input maps to `target_norm` total L2 norm (safeguard 2). RMS -> L2 is
        # a factor of sqrt(width), hence the division.
        self.scale = nnx.Param(jnp.full((width,), target_norm / jnp.sqrt(width), dtype=jnp.float32))
        self.width = width

    def __call__(self, relations):
        x = self.mlp(relations)
        # RMSNorm (no mean subtraction, matching gemma.RMSNorm) then learned scale.
        rms = jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * self.scale.value


class GraspHead(nnx.Module):
    """Prefix-only per-slot grasp-probability head.

    Reads a pooled PREFIX hidden state -- image + prompt tokens, never the noisy
    action tokens x_t -- and predicts, for each of the H future slots and each
    hand, the logit that the gripper is closed there. A grasp SCHEDULE.

    Prefix-only is the whole design. Reading the action-expert suffix instead
    would make the head's accuracy a function of the flow timestep t (at t~1 it
    sees pure noise, at t~0 it nearly sees the answer), so the probe would mean
    something different at every query. Here it is t-independent, deterministic,
    and evaluable exactly once per inference.

    What it buys, beyond being cheap: a CALIBRATED probability where the flow
    gripper dim gives only a sample (flow matching a binary target averages the
    two modes near the decision boundary, so samples can land near 0 -- ambiguous
    exactly when it matters); a metric that is invariant to w_gripper and to the
    normalization scheme, so it can drive the weight sweep; and a task-semantic
    gradient on the prefix, which forces it to encode task PHASE.
    """

    def __init__(self, width: int, action_horizon: int, n_hands: int, *, rngs: nnx.Rngs,
                 hidden: int = 256):
        self.action_horizon = action_horizon
        self.n_hands = n_hands
        # Attention pooling over prefix tokens: a learned query scores each
        # token, so the head can look at the object/hand segments rather than
        # being dominated by the ~256 image tokens that mean-pooling would drown
        # it in.
        self.query = nnx.Linear(width, 1, rngs=rngs)
        self.mlp = GeGLU(width, hidden, action_horizon * n_hands, rngs=rngs, zero_init_out=False)

    def __call__(self, prefix_out, prefix_mask):
        """prefix_out (b, s, width), prefix_mask (b, s) -> logits (b, H, n_hands)."""
        scores = self.query(prefix_out)[..., 0]                       # (b, s)
        scores = jnp.where(prefix_mask, scores, -jnp.inf)             # padding never pools
        weights = jax.nn.softmax(scores, axis=-1)[..., None]          # (b, s, 1)
        pooled = jnp.sum(weights * prefix_out, axis=-2)               # (b, width)
        logits = self.mlp(pooled)                                     # (b, H*n_hands)
        return logits.reshape(logits.shape[0], self.action_horizon, self.n_hands)


def _find_embedding_table(tree):
    """Depth-first search for the `input_embedding` leaf.

    Searched by NAME rather than by a fixed path: the checkpoint tree and the
    live nnx tree nest it differently (`PaliGemma/llm/embedder/...` vs
    `embedder/...`), and the path is not part of any contract we control.
    """
    if isinstance(tree, dict):
        if "input_embedding" in tree and not isinstance(tree["input_embedding"], dict):
            return tree["input_embedding"]
        for value in tree.values():
            found = _find_embedding_table(value)
            if found is not None:
                return found
    return None


def paligemma_embedding_norm(params_path: str) -> float:
    """Mean L2 norm of a PRETRAINED PaliGemma token embedding AS IT ENTERS the
    residual stream — the target injected relation tokens are scaled to match
    (safeguard 2).

    Reads the checkpoint on disk. It must, for three separate reasons, all of
    which were wrong when this read the live nnx param tree from
    `Ego2G1Pi0.__init__`:

    1. It has to run OUTSIDE a trace. `__init__` executes under
       `jax.eval_shape`, where every leaf is an abstract tracer and `float()`
       raises ConcretizationTypeError.
    2. It has to read the PRETRAINED table. `__init__` runs before the weight
       loader, so the live tree there holds nnx's random init, not pi05_base —
       it would have returned a number describing noise.
    3. It has to include `gemma.Embedder.encode`'s `x *= sqrt(embed_dim)`
       (gemma.py:150). Tokens reach the stream sqrt(2048) ~= 45x larger than
       their table rows, and `embed_prefix` adds the relation delta AFTER that
       scaling. Matching the raw row norm would inject a token ~45x too small —
       exactly the "attention ignores it, expensive no-op" failure this
       safeguard exists to prevent.

    Cached next to the params, keyed by path: the value is a constant of the
    pretrained checkpoint, and restoring 12+ GB to read one array is not
    something to repeat every run.
    """
    import json
    import pathlib

    import numpy as np

    import openpi.models.model as _model_mod
    import openpi.shared.download as _download

    local = _download.maybe_download(params_path)
    cache = pathlib.Path(local).parent / "ego2g1_embedding_norm.json"
    key = str(params_path)
    if cache.exists():
        try:
            hit = json.loads(cache.read_text()).get(key)
            if hit is not None:
                return float(hit)
        except (ValueError, OSError):
            pass  # unreadable cache is not a reason to fail; just re-measure

    params = _model_mod.restore_params(local, restore_type=np.ndarray)
    table = _find_embedding_table(params)
    if table is None:
        raise ValueError(f"no `input_embedding` leaf in the params at {params_path}")
    table = np.asarray(table, dtype=np.float32)
    row_norm = float(np.mean(np.linalg.norm(table, axis=-1)))
    width = table.shape[-1]
    del params, table
    value = row_norm * float(np.sqrt(width))

    try:
        existing = json.loads(cache.read_text()) if cache.exists() else {}
        existing[key] = value
        cache.write_text(json.dumps(existing, indent=1))
    except OSError:
        pass  # a read-only asset cache just means we measure again next run
    return value


@at.typecheck
def grasp_bce_loss(
    logits: at.Float[at.Array, "b ah nh"],
    targets: at.Float[at.Array, "b ah nh"],
) -> at.Float[at.Array, ""]:
    """Mean binary cross-entropy of the grasp schedule.

    `targets` are the gripper action dims in {-1, +1} (the model's action space),
    mapped back to {0, 1} here so the head's labels and the flow head's targets
    cannot drift apart -- they are literally the same numbers.
    """
    labels = (targets + 1.0) / 2.0
    # log-sum-exp form: numerically stable for large |logits|, unlike
    # log(sigmoid(x)) written directly.
    return jnp.mean(jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def grasp_auc(logits, targets) -> float:
    """Rank-based AUC of the grasp schedule, pooled over slots/hands/batch.

    Reported instead of accuracy because the classes are imbalanced (~26% closed
    on red_block_in_pen_holder_ego), where accuracy is dominated by the majority
    class. Numpy, eager, logging only -- never inside the train step.
    """
    import numpy as np

    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = (np.asarray(targets, dtype=np.float64).reshape(-1) + 1.0) / 2.0
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")   # single-class batch: AUC is undefined, not 0.5
    # Mann-Whitney U via average ranks (ties share a rank).
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(scores)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
