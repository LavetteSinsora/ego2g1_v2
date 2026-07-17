"""Loader equivalence test (SPEC.md verification hook).

For random valid datapoints: actions produced by the real loading path
(LeRobotDataset with delta_timestamps -> RelativeChunkActions) must equal
deltas computed directly from the work-dir stage npzs (s002_01 poses,
s002_02 hand commands, sliced by the sidecar tick ranges). Also checks that
BoundaryAwareIndices excludes exactly the tail frames of non-real-end
episodes plus the sidecar's anchor_bad (bridged) frames.

Run: uv run python -m pytest tests/test_loader_equivalence [--root ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ego2g1.core.rot6d import se3_to_vec9, vec9_to_se3
from ego2g1.data.config import load_config
from ego2g1.core.boundary import BoundaryAwareIndices, load_boundary_indices
from ego2g1.core.relative_actions import RelativeChunkActions, make_delta_timestamps

TOL = 1e-5   # f32 storage
PRE = {"left": "l", "right": "r"}


def direct_actions(cfg, work_dir, entry, t):
    """Action chunk at local frame t computed straight from stage npzs."""
    stem = Path(entry["source_file"]).stem
    pose_npz = np.load(Path(work_dir) / stem / "s002_01.npz")
    hand_npz = np.load(Path(work_dir) / stem / "s002_02.npz")
    a, b = entry["tick_start"], entry["tick_end"]
    H = cfg.action_horizon
    L = b - a
    parts = []
    for hand in cfg.hands:
        pose = pose_npz[f"pose_{PRE[hand]}"][a:b]      # (L, 9) f32
        cmds = hand_npz[f"hand_cmds_{PRE[hand]}"][a:b]  # (L, 6) f32
        # lerobot pads by repeating the episode's last frame
        idx = np.minimum(t + np.arange(H + 1), L - 1)
        T = vec9_to_se3(pose[idx].astype(np.float64))
        deltas = se3_to_vec9(np.linalg.inv(T[0]) @ T[1:])
        parts.append(np.concatenate([deltas, cmds[idx[1:]]], axis=-1))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def to_numpy(sample):
    out = {}
    for k, v in sample.items():
        out[k] = v.numpy() if hasattr(v, "numpy") else v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="dataset root (default: cfg)")
    ap.add_argument("--n", type=int, default=20, help="datapoints to check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode-video", action="store_true",
                    help="really decode video frames (needs a working backend)")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(args.root) if args.root else Path(cfg.output_root) / cfg.repo_id
    if not (root / "extraction_meta.json").exists():
        sys.exit(f"FAIL: no dataset at {root} - run s005 first")

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        sys.exit("FAIL: lerobot not installed - see s005_write_lerobot INSTALL_CMD")

    sidecar = json.loads((root / "extraction_meta.json").read_text())
    if sidecar["config_hash"] != cfg.config_hash:
        print(f"WARNING: dataset config_hash {sidecar['config_hash']} != "
              f"current cfg {cfg.config_hash}; comparing against sidecar anyway")

    entries = {int(k): v for k, v in sidecar["episodes"].items()}
    n_eps = len(entries)
    lengths = [entries[i]["tick_end"] - entries[i]["tick_start"] for i in range(n_eps)]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    real_end = [entries[i]["episode_real_end"] for i in range(n_eps)]
    H = cfg.action_horizon

    # ---- boundary indexing: excluded == tail frames of non-real-end episodes
    bidx = load_boundary_indices(root, H, cfg.allow_terminal_padding)
    assert bidx.total_frames == offsets[-1], \
        f"lerobot meta total {bidx.total_frames} != sidecar total {offsets[-1]}"
    anchor_bad = [entries[i].get("anchor_bad", []) for i in range(n_eps)]
    expected_excluded = set()
    for i in range(n_eps):
        if not (real_end[i] and cfg.allow_terminal_padding):
            expected_excluded |= set(range(offsets[i] + max(lengths[i] - H, 0),
                                           offsets[i + 1]))
        expected_excluded |= {int(offsets[i]) + int(t) for t in anchor_bad[i]}
    actual_excluded = set(range(bidx.total_frames)) - set(bidx.indices.tolist())
    assert actual_excluded == expected_excluded, (
        f"boundary mismatch: {len(actual_excluded)} excluded vs "
        f"{len(expected_excluded)} expected")
    # cross-check the standalone-constructed indices agree
    bidx2 = BoundaryAwareIndices(lengths, real_end, H, cfg.allow_terminal_padding,
                                 anchor_bad=anchor_bad)
    assert np.array_equal(bidx.indices, bidx2.indices)
    print(f"boundary indexing OK: {len(bidx)}/{bidx.total_frames} frames valid, "
          f"{len(actual_excluded)} excluded (tails + anchor_bad)")

    # ---- action equivalence on random valid datapoints
    assert float(cfg.control_hz).is_integer()
    # Video pixels are not under test (assertions cover pose/hand actions
    # only); decoding is stubbed because neither of the pinned lerobot's
    # backends works on this Mac (torchcodec needs FFmpeg dylibs, the pyav
    # path needs the removed torchvision.io.VideoReader). Pass --decode-video
    # to exercise real decoding where the environment supports it.
    if not args.decode_video:
        import torch
        import lerobot.common.datasets.lerobot_dataset as _lds
        _lds.decode_video_frames = (
            lambda path, ts, tol, backend=None: torch.zeros(len(ts), 3, 8, 8))
    dataset = LeRobotDataset(
        cfg.repo_id, root=root,
        delta_timestamps=make_delta_timestamps(H, int(cfg.control_hz)))
    transform = RelativeChunkActions(hands=cfg.hands)

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(bidx), size=min(args.n, len(bidx)), replace=False)
    worst = 0.0
    for i in picks:
        g = bidx[int(i)]
        ep = int(np.searchsorted(offsets, g, side="right") - 1)
        t = g - int(offsets[ep])
        sample = transform(to_numpy(dataset[g]))
        got = sample["actions"]
        want = direct_actions(cfg, cfg.work_dir, entries[ep], t)
        assert got.shape == (H, cfg.action_dim), got.shape
        diff = float(np.abs(got - want).max())
        worst = max(worst, diff)
        status = "ok" if diff < TOL else "MISMATCH"
        print(f"  ep {ep:3d} frame {t:3d} (global {g:6d}): max|diff| = {diff:.2e} {status}")
        if diff >= TOL:
            sys.exit(f"FAIL: action mismatch {diff:.2e} >= {TOL} at episode {ep} frame {t}")

    print(f"PASS: {len(picks)} datapoints, worst max|diff| = {worst:.2e} < {TOL}")


if __name__ == "__main__":
    main()
