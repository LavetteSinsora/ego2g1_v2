"""What do the injected sentinel embeddings actually look like?

Answers three questions about a trained relational checkpoint, without building
the 3B model -- it restores params, pulls out the relation_encoder subtree and
ONE embedding row, and runs the encoder on real relation vectors:

1. How big is the encoder's output, and how much does it vary across objects?
2. How big is the sentinel's own pretrained embedding it gets added to?
3. After the add, are the three object tokens distinguishable -- or effectively
   the same vector? Since gemma.RMSNorm discards magnitude and keeps only
   direction, the quantity that matters is the ANGLE the delta rotates the base,
   not the norm ratio itself.

Usage:
    python -m tools.inspect_relation_injection \
        --checkpoint checkpoints/ego2g1_relation/relation_v1/9999 \
        --dataset data/lerobot_datasets/ego2g1/red_block_in_pen_holder_ego
"""

import argparse
import pathlib

import numpy as np


def _cos(a, b):
    """Row-wise cosine between (N, D) arrays.

    NOTE the `axis=-1`: np.linalg.norm's second POSITIONAL arg is `ord`, not
    `axis`, so np.linalg.norm(a, -1) silently computes a matrix norm (the min
    absolute column sum) and the result is not a cosine at all. The assertion
    below is there because that mistake produces plausible-looking numbers.
    """
    cos = np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    )
    assert np.all(np.abs(cos) <= 1.0 + 1e-6), f"not a cosine: max |cos| = {np.abs(cos).max()}"
    return cos


def _arr(x):
    """Unwrap a leaf. nnx state serializes params as {'value': array} (visible in
    the sharding logs as ...['kernel'].value), while a plain released checkpoint
    stores the array directly. Accept both."""
    if isinstance(x, dict) and "value" in x:
        x = x["value"]
    return np.asarray(x, dtype=np.float64)


def _find(tree, key):
    """DFS for a named entry, returning it whatever its type.

    Deliberately NOT the `not isinstance(dict)` variant in relation.py: that one
    hunts for an array leaf, but `relation_encoder` is a SUBTREE, and a leaf may
    itself be a {'value': array} dict. Callers use _arr() to normalize.
    """
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _geglu(params, x):
    """RelationEncoder.mlp forward: out(gelu(gate(x)) * value(x))."""

    def lin(p, h):
        return h @ _arr(p["kernel"]) + _arr(p["bias"])

    def gelu(h):  # tanh approximation, matching nnx.gelu's default
        return 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h**3)))

    return lin(params["out"], gelu(lin(params["gate"], x)) * lin(params["value"], x))


def main(checkpoint: str, dataset: str, n_frames: int) -> None:
    import openpi.models.model as _model_mod

    from ego2g1.train import norm as _norm
    from ego2g1.train import relation_transforms as _rt

    ckpt = pathlib.Path(checkpoint)
    params = _model_mod.restore_params(ckpt / "params", restore_type=np.ndarray)
    # Some checkpoints wrap the tree one level ({"params": {...}}), some don't.
    while isinstance(params, dict) and set(params) == {"params"}:
        params = params["params"]
    print(f"top-level param keys: {sorted(params)}")

    enc = _find(params, "relation_encoder")
    if enc is None:
        raise SystemExit(
            "no 'relation_encoder' anywhere in the tree -- is this a relational "
            f"checkpoint? top-level keys: {sorted(params)}"
        )
    print(f"relation_encoder subtree: {sorted(enc)}   mlp: {sorted(enc['mlp'])}")
    scale = _arr(enc["scale"])
    table = _arr(_find(params, "input_embedding"))
    width = table.shape[-1]
    print(f"scale {scale.shape}, embedding table {table.shape}")
    print()

    # ---- 2. the base the delta is added to -------------------------------
    sid = _rt.sentinel_token_id()
    base = table[sid] * np.sqrt(width)          # Embedder.encode's scaling
    all_norms = np.linalg.norm(table, axis=-1) * np.sqrt(width)
    print("=" * 78)
    print("BASE (what the delta is added to)")
    print("=" * 78)
    raw = np.linalg.norm(table, axis=-1)
    print(f"  sentinel id                 : {sid}  ({_rt.RELATION_SENTINEL})")
    print(f"  embed_dim / sqrt(embed_dim) : {width} / {np.sqrt(width):.4f}")
    print()
    print("  RAW table rows (before Embedder.encode's x *= sqrt(embed_dim)):")
    print(f"    ||table[{sid}]||             : {raw[sid]:10.4f}")
    print(f"    mean over vocabulary      : {raw.mean():10.4f}")
    print()
    print("  SCALED -- this is what actually enters the prefix, and what the")
    print("  encoder's output is added to:")
    print(f"    ||embed(<unused0>)||      : {np.linalg.norm(base):10.4f}")
    print(f"    mean over vocabulary      : {all_norms.mean():10.4f}")
    print(f"    median over vocabulary    : {np.median(all_norms):10.4f}")
    print("  (relation/text_norm during training is a THIRD population: the mean over")
    print("   tokens actually in the prompt, weighted by occurrence -- so it is lower")
    print("   than the vocabulary mean, being dominated by spaces and common words.)")

    # ---- the encoder's configured magnitude ------------------------------
    print()
    print("=" * 78)
    print("ENCODER SCALE (the trained value of RelationEncoder.scale)")
    print("=" * 78)
    print(f"  ||scale||                   : {np.linalg.norm(scale):10.4f}")
    print(f"  per-channel mean / std      : {scale.mean():10.6f} / {scale.std():.6f}")
    print(f"  min / max                   : {scale.min():10.6f} / {scale.max():.6f}")
    print(f"  relative spread (std/|mean|): {scale.std() / max(abs(scale.mean()), 1e-12):10.4f}")
    print()
    print("  scale is initialized PERFECTLY UNIFORM at relation_target_norm/sqrt(width),")
    print("  so the relative spread dates the parameter: ~0 means it barely moved and")
    print("  ||scale|| still equals the value it was INITIALIZED with (i.e. the measurement")
    print("  was wrong); a large spread means training reshaped it and the init is not")
    print(f"  recoverable from here. Uniform init would give mean = ||scale||/sqrt(width) "
          f"= {np.linalg.norm(scale) / np.sqrt(len(scale)):.6f}.")
    print("  RMSNorm pins ||output|| to this, so it IS the plateau in relation/injected_norm.")
    print("  Compare against what paligemma_embedding_norm returns now: if that says ~259")
    print("  and this says ~1.7, training shrank a correct init; if both say ~1.7, the")
    print("  measurement was wrong and safeguard 2 never engaged.")

    # ---- 1. run the encoder on REAL relation vectors ---------------------
    import pandas as pd

    root = pathlib.Path(dataset)
    # train.py writes the stats to the RUN dir (and to best/), not the step dir,
    # so <run>/9999 and <run>/best/0 both resolve via the parent.
    assets = ckpt / "assets_ego2g1"
    if not assets.exists():
        assets = ckpt.parent / "assets_ego2g1"
    stats = _norm.load_relation(assets)
    frames = []
    for p in sorted((root / "data" / "chunk-000").glob("*.parquet"))[:3]:
        frames.append(np.stack(pd.read_parquet(p, columns=["observation.state"])
                               ["observation.state"].to_numpy()))
    state = np.concatenate(frames)[:n_frames].astype(np.float64)

    # hand-major (54,) -> per-object rows spanning both hands, as RelationPrompt does.
    # state is [2 hands x n_obj x 9 | 2 grasp bits], so n_obj follows from the widths.
    relation_dim = len(stats.relation_mean)          # 18 = one vec9 per hand
    n_obj = (state.shape[1] - 2) // relation_dim
    rows = np.stack([np.concatenate([state[:, 9 * k:9 * k + 9],
                                     state[:, 27 + 9 * k:27 + 9 * k + 9]], axis=1)
                     for k in range(n_obj)], axis=1)            # (N, 3, 18)
    z = (rows - stats.relation_mean) / np.maximum(stats.relation_std, 1e-6)
    z = np.clip(z, -5.0, 5.0)

    x = _geglu(enc["mlp"], z.reshape(-1, z.shape[-1]))
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    delta = ((x / rms) * scale).reshape(z.shape[0], n_obj, width)

    names = ("pen holder", "red cube", "yellow cube")
    print()
    print("=" * 78)
    print(f"ENCODER OUTPUT over {len(delta)} real frames")
    print("=" * 78)
    print(f"  {'object':14s} {'mean ||delta||':>15s} {'std':>10s}")
    for k in range(n_obj):
        d = np.linalg.norm(delta[:, k], axis=-1)
        print(f"  {names[k]:14s} {d.mean():15.4f} {d.std():10.4f}")
    print("  (RMSNorm pins the norm, so near-zero std here is expected, NOT a finding.)")

    print()
    print("  Does the encoder DISTINGUISH objects? cosine between delta directions:")
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            cos = _cos(delta[:, i], delta[:, j])
            print(f"    {names[i]:12s} vs {names[j]:12s}: cos = {cos.mean():+.6f}"
                  f"   (angle {np.degrees(np.arccos(np.clip(cos.mean(), -1, 1))):6.2f} deg)")
    print("  cos near +1 = the encoder maps every object to the same direction (collapsed).")

    # ---- 3. the actual injected tokens -----------------------------------
    injected = base[None, None, :] + delta
    print()
    print("=" * 78)
    print("INJECTED TOKENS (base + delta) -- what attention actually receives")
    print("=" * 78)
    ratio = np.linalg.norm(delta, axis=-1).mean() / np.linalg.norm(base)
    print(f"  ||delta|| / ||base||        : {ratio:10.6f}")
    print(f"  rotation of the base vector : {np.degrees(np.arctan(ratio)):10.4f} deg")
    print()
    print("  pairwise cosine between the three FINAL token embeddings:")
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            cos = _cos(injected[:, i], injected[:, j])
            print(f"    {names[i]:12s} vs {names[j]:12s}: cos = {cos.mean():.8f}"
                  f"   (angle {np.degrees(np.arccos(np.clip(cos.mean(), -1, 1))):.4f} deg)")
    print()
    print("  cos = 1.000000 to 6+ decimals means the three object tokens are the SAME")
    print("  vector as far as the transformer is concerned -- the geometry never arrives.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-frames", type=int, default=512)
    a = ap.parse_args()
    main(a.checkpoint, a.dataset, a.n_frames)
