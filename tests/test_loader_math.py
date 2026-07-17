"""Package-free consistency checks (numpy only; no lerobot/torch needed).

1. Loader math == deployment runtime math: RelativeChunkActions deltas,
   composed onto an anchor the way the robot client will
   (T_target = T_anchor @ delta), must reproduce the stored absolute poses
   exactly (ground-truth anchoring) - the same identity sim.chunks.selftest
   checks, but through the actual loader code path on real stage outputs.
2. vec9/6d round-trip stability on real poses.
3. BoundaryAwareIndices on the synthetic layouts.

Run: uv run python -m pytest tests/test_loader_math [episode_1]
"""

import sys
from pathlib import Path

import numpy as np

from ego2g1.data import io
from ego2g1.core.rot6d import se3_to_vec9, vec9_to_se3
from ego2g1.data.config import PipelineConfig
from ego2g1.core.boundary import BoundaryAwareIndices
from ego2g1.core.relative_actions import RelativeChunkActions, make_delta_timestamps

PRE = {"left": "l", "right": "r"}


def main():
    ep = sys.argv[1] if len(sys.argv) > 1 else "episode_1"
    cfg = PipelineConfig()
    H = cfg.action_horizon

    s0201, _ = io.load_stage(cfg, ep, "s002_01")
    s0202, _ = io.load_stage(cfg, ep, "s002_02")
    s004, _ = io.load_stage(cfg, ep, "s004")
    a, b = int(s004["subep_start"][0]), int(s004["subep_end"][0])
    assert b - a >= H + 1, "first sub-episode too short for one datapoint"

    # --- 1. loader == runtime composition
    transform = RelativeChunkActions(hands=cfg.hands)
    worst = 0.0
    for t in range(a, b - H, 7):
        sample = {}
        for hand in cfg.hands:
            sample[f"pose.{hand}"] = s0201[f"pose_{PRE[hand]}"][t:t + H + 1]
            sample[f"hand.{hand}"] = s0202[f"hand_cmds_{PRE[hand]}"][t + 1:t + H + 1]
        actions = transform(sample)["actions"]          # (H, 30)
        for i, hand in enumerate(cfg.hands):
            eef = actions[:, i * 15:i * 15 + 9]
            anchor = vec9_to_se3(s0201[f"pose_{PRE[hand]}"][t].astype(np.float64))
            for k in range(H):
                # deployment composition: T_target = T_anchor @ delta_k
                T_tgt = anchor @ vec9_to_se3(eef[k].astype(np.float64))
                stored = vec9_to_se3(
                    s0201[f"pose_{PRE[hand]}"][t + 1 + k].astype(np.float64))
                worst = max(worst, float(np.abs(T_tgt - stored).max()))
    assert worst < 1e-5, f"loader/runtime composition mismatch: {worst:.2e}"
    print(f"1. loader deltas compose back to stored poses: max err {worst:.2e} OK")

    # --- 2. vec9 round-trip on real poses
    v = s0201["pose_l"].astype(np.float64)
    rt = se3_to_vec9(vec9_to_se3(v))
    err = float(np.abs(rt - v).max())
    assert err < 1e-6, err
    print(f"2. vec9 <-> SE3 round-trip on {len(v)} real poses: max err {err:.2e} OK")

    # --- 3. boundary indexing
    idx = BoundaryAwareIndices([100, 60, 80], [False, True, False], 50, True)
    got = set(idx.indices.tolist())
    want = set(range(0, 50)) | set(range(100, 160)) | set(range(160, 190))
    assert got == want, (sorted(got - want), sorted(want - got))
    idx2 = BoundaryAwareIndices([100, 60], [True, True], 50, False)
    assert set(idx2.indices.tolist()) == set(range(0, 50)) | set(range(100, 110))
    # anchor_bad frames (bridged ticks) never anchor a datapoint, in both the
    # tail-clipped and the terminal-padding regimes
    idx3 = BoundaryAwareIndices([100, 60], [False, True], 50, True,
                                anchor_bad=[[3, 4, 5], [58]])
    want3 = (set(range(0, 50)) - {3, 4, 5}) | (set(range(100, 160)) - {158})
    assert set(idx3.indices.tolist()) == want3
    print("3. BoundaryAwareIndices synthetic layouts (+anchor_bad) OK")

    # --- 4. delta_timestamps shape
    dts = make_delta_timestamps(H, int(cfg.control_hz))
    assert len(dts["pose.left"]) == H + 1 and dts["pose.left"][0] == 0.0
    assert len(dts["hand.left"]) == H and dts["hand.left"][0] > 0.0
    print("4. delta_timestamps layout OK")
    print(f"PASS ({ep})")


if __name__ == "__main__":
    main()
