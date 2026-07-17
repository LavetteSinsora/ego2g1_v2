"""Chunk-relative action representation + anchor-mode runtime for wrist_replay.

Deployment-shaped layering:

- `ReplayChunkProvider` plays the role the policy server will play later: at
  each chunk start t0 it emits deltas Δ_k = T_h(t0)^-1 · T_h(t_k), k=1..K —
  the human wrist's relative motion in the local frame of its chunk-start
  pose. The alignment rotation B never enters here: these deltas are exactly
  what a trained policy will be asked to output.

- `ChunkRuntime` is the client-side math: given an anchor flange pose it maps
  each delta to an absolute flange target, T_target(t_k) = T_anchor · B^-1 · Δ_k · B,
  where B (per side, rotation-only) converts between the human wrist frame
  axes and the robot flange frame axes (flange target for a human pose T_h is
  G(t) = T_h(t) · B).

Anchor modes (the load-bearing distinction of this tool):

- "ground-truth": each chunk anchors on the HUMAN's recorded wrist pose at
  the boundary tick, T_anchor = T_h(t0) · B. Then T_target(t_k) == G(t_k)
  identically — IK error does NOT feed into later chunks. Use this to judge
  retargeting + IK quality in isolation.

- "measured": each chunk anchors on the ROBOT's IK-achieved flange pose at
  the boundary tick, exactly as a deployed runtime must (the real robot only
  knows its own proprioception/FK). IK tracking error feeds forward, so
  imperfect IK shows up as accumulating drift away from the human motion.
"""

import numpy as np

from . import frames

ANCHOR_MODES = ("ground-truth", "measured")


class ReplayChunkProvider:
    """Serves chunks of human-frame deltas from a precomputed wrist
    trajectory (later: swapped for a policy-server client)."""

    def __init__(self, T_h_seq, chunk_len=50):
        self.T_h = T_h_seq            # list of 4x4, one per control tick
        self.chunk_len = chunk_len

    def chunk_starts(self):
        return list(range(0, len(self.T_h) - 1, self.chunk_len))

    def get_chunk(self, start):
        """Deltas Δ_1..Δ_K for the chunk starting at tick `start`
        (K = chunk_len, or less at the episode tail)."""
        K = min(self.chunk_len, len(self.T_h) - 1 - start)
        inv0 = frames.se3_inv(self.T_h[start])
        return [inv0 @ self.T_h[start + k] for k in range(1, K + 1)]


class ChunkRuntime:
    """Maps chunk deltas to absolute flange targets under an anchor mode."""

    def __init__(self, B, mode):
        assert mode in ANCHOR_MODES, mode
        self.B = B                    # 4x4, rotation-only
        self.B_inv = frames.se3_inv(B)
        self.mode = mode

    def flange_target_for_human(self, T_h):
        """Direct retarget map G(t) = T_h(t) · B."""
        return T_h @ self.B

    def anchor_for_chunk(self, T_h_start, measured_flange):
        if self.mode == "ground-truth":
            return self.flange_target_for_human(T_h_start)
        return measured_flange

    def targets_for_chunk(self, T_anchor, deltas):
        return [T_anchor @ self.B_inv @ d @ self.B for d in deltas]


def selftest_identity(T_h_seq, B, chunk_len=50, tol=1e-9):
    """The convention check that catches every frame bug at once: in
    ground-truth mode the composed targets must reproduce G(t_k) exactly."""
    provider = ReplayChunkProvider(T_h_seq, chunk_len)
    runtime = ChunkRuntime(B, "ground-truth")
    worst = 0.0
    for start in provider.chunk_starts():
        anchor = runtime.anchor_for_chunk(T_h_seq[start], measured_flange=None)
        targets = runtime.targets_for_chunk(anchor, provider.get_chunk(start))
        for k, T_tgt in enumerate(targets, 1):
            G = runtime.flange_target_for_human(T_h_seq[start + k])
            worst = max(worst, float(np.abs(T_tgt - G).max()))
    assert worst < tol, f"anchor/delta identity violated: max abs err {worst}"
    return worst
