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
2. Output RMSNorm, then a scale set RELATIVE to the embedding being injected
   into: `alpha * ||base||`, with ||base|| read live from the token stream at
   injection time. alpha is a dimensionless learned scalar, so "how hard am I
   perturbing this token" is a number you can read off directly.
3. Final projection ZERO-initialized, so at step 0 the injected embedding is
   exactly zero and the model is bit-identical to one that ignores the object
   vectors. It has to learn to use them; nothing is injected into a pretrained
   circuit before the gradient asks for it.

Safeguard 2 was originally an ABSOLUTE target measured offline from the
pretrained checkpoint, and that is what broke relation_v1 -- see RelationEncoder
for the post-mortem. The relative form cannot fail the same way.

`relation/rotation_deg` is the training canary. Because gemma.RMSNorm discards
per-token magnitude and keeps only DIRECTION, the injection's only possible
effect is to rotate the token it is added to; the norms themselves are a means,
not the end. arctan(alpha) is the healthy value; near 0 means the geometry never
reaches attention.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


# Bound ONCE, at import. nnx stores `kernel_init` in the graphdef's STATIC
# fields, which are compared by object identity -- and `lecun_normal()` returns a
# freshly built closure on every call. Constructing it inline would therefore make
# two otherwise identical models compare unequal, and `init_train_state` builds
# the model twice (once under jax.eval_shape, once under jit) and requires the two
# graphdefs to match. `zeros_init()` happens to return a module-level function and
# is stable either way; both are hoisted so the rule is uniform rather than
# depending on which initializer you happen to pick.
_ZEROS_INIT = nnx.initializers.zeros_init()
_LECUN_NORMAL = nnx.initializers.lecun_normal()


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
        out_kernel = _ZEROS_INIT if zero_init_out else _LECUN_NORMAL
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
                 alpha_init: float = 1.0):
        self.mlp = GeGLU(relation_dim, hidden, width, rngs=rngs, zero_init_out=True)
        # Safeguard 2, RELATIVE form. `alpha` is the injected token's size as a
        # FRACTION of the embedding it is added to, and the size of that embedding
        # is read live at injection time (see Ego2G1Pi0.embed_prefix) rather than
        # measured offline from the checkpoint.
        #
        # The offline route is what broke relation_v1: it returned 1.63 where the
        # sentinel's embedding is 487.76, so the injection rotated the token by
        # 0.2 deg and the three objects arrived at cosine 0.99999 -- identical, to
        # the transformer. Nothing could recover it, because growing a scale 284x
        # is a MULTIPLICATIVE correction driven by additive SGD steps whose
        # gradient is itself tiny precisely because the injection is tiny. The
        # trained checkpoint shows the per-channel mean moved 0.6% in 10k steps.
        #
        # Reading the base live makes that class of error impossible: the target
        # and the thing it is compared against are the same tensor. alpha is a
        # dimensionless scalar, so a wrong value is visible by inspection.
        self.alpha = nnx.Param(jnp.asarray(alpha_init, dtype=jnp.float32))
        self.width = width

    def __call__(self, relations, base_norm):
        """relations (*b, n, relation_dim), base_norm scalar -> (*b, n, width).

        `base_norm` is the L2 norm of the embedding this output will be ADDED to
        (the sentinel's, already scaled by Embedder.encode's sqrt(embed_dim)).
        """
        x = self.mlp(relations)
        # RMSNorm (no mean subtraction, matching gemma.RMSNorm). A unit-RMS vector
        # has L2 = sqrt(width), NOT 1 -- hence the division below, which is purely
        # an RMS->L2 unit conversion and is unrelated to Embedder.encode's own
        # sqrt(embed_dim) despite being the same number.
        rms = jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
        gain = self.alpha.value * base_norm / jnp.sqrt(self.width)
        # zero_init_out means x == 0 exactly at step 0, so rms -> sqrt(1e-6) and the
        # output is still exactly 0: safeguard 3 survives unchanged.
        return (x / rms) * gain


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
