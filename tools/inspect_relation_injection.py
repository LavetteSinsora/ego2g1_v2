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


def _find(tree, key):
    """DFS for a named leaf; same lookup relation.paligemma_embedding_norm uses."""
    if isinstance(tree, dict):
        if key in tree and not isinstance(tree[key], dict):
            return tree[key]
        for value in tree.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _geglu(params, x):
    """RelationEncoder.mlp forward: out(gelu(gate(x)) * value(x))."""

    def lin(p, h):
        return h @ np.asarray(p["kernel"], np.float64) + np.asarray(p["bias"], np.float64)

    def gelu(h):  # tanh approximation, matching nnx.gelu's default
        return 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h**3)))

    return lin(params["out"], gelu(lin(params["gate"], x)) * lin(params["value"], x))


def main(checkpoint: str, dataset: str, n_frames: int) -> None:
    import openpi.models.model as _model_mod

    from ego2g1.train import norm as _norm
    from ego2g1.train import relation_transforms as _rt

    ckpt = pathlib.Path(checkpoint)
    params = _model_mod.restore_params(ckpt / "params", restore_type=np.ndarray)
    enc = _find(params, "relation_encoder") or params["params"]["relation_encoder"]
    if "mlp" not in enc:
        enc = enc["relation_encoder"] if "relation_encoder" in enc else enc
    scale = np.asarray(enc["scale"], np.float64)
    table = np.asarray(_find(params, "input_embedding"), np.float64)
    width = table.shape[-1]

    # ---- 2. the base the delta is added to -------------------------------
    sid = _rt.sentinel_token_id()
    base = table[sid] * np.sqrt(width)          # Embedder.encode's scaling
    all_norms = np.linalg.norm(table, axis=-1) * np.sqrt(width)
    print("=" * 78)
    print("BASE (what the delta is added to)")
    print("=" * 78)
    print(f"  sentinel id                 : {sid}  ({_rt.RELATION_SENTINEL})")
    print(f"  ||embed(<unused0>)||        : {np.linalg.norm(base):10.4f}")
    print(f"  mean ||embed(token)||       : {all_norms.mean():10.4f}   <- what text_norm reports")
    print(f"  median                      : {np.median(all_norms):10.4f}")
    print("  (if the sentinel row is far from the mean, the 259 figure was the wrong anchor)")

    # ---- the encoder's configured magnitude ------------------------------
    print()
    print("=" * 78)
    print("ENCODER SCALE (the trained value of RelationEncoder.scale)")
    print("=" * 78)
    print(f"  ||scale||                   : {np.linalg.norm(scale):10.4f}")
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
            a, b = delta[:, i], delta[:, j]
            cos = np.sum(a * b, -1) / (np.linalg.norm(a, -1) * np.linalg.norm(b, -1))
            print(f"    {names[i]:12s} vs {names[j]:12s}: cos = {cos.mean():+.6f}")
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
            a, b = injected[:, i], injected[:, j]
            cos = np.sum(a * b, -1) / (np.linalg.norm(a, -1) * np.linalg.norm(b, -1))
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
