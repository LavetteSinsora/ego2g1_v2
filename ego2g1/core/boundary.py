"""Boundary-aware datapoint indexing (SPEC.md "Loader semantics").

A frame t of an episode of `length` frames is a VALID datapoint iff
`t + H <= length - 1` (a full chunk of H future poses exists), OR the
episode is an `episode_real_end` sub-episode and `allow_terminal_padding`
is on (LeRobot repeat-padding then means "hold pose"; everywhere else the
human kept moving and padding would be a lie) - AND, in either case, t is
not an `anchor_bad` frame (a bridged tick: its stored pose may appear
inside other datapoints' action windows, but it never anchors one). pi0
ignores `action_is_pad`, so this must be enforced by remapping indices,
not by masking. numpy only.
"""

import json
from pathlib import Path

import numpy as np


class BoundaryAwareIndices:
    """Flat-index remap over a frame-indexed dataset laid out episode by
    episode (LeRobot order): global index = episode offset + t."""

    def __init__(self, episode_lengths, real_end_flags, action_horizon,
                 allow_terminal_padding, anchor_bad=None):
        lengths = [int(x) for x in episode_lengths]
        flags = [bool(x) for x in real_end_flags]
        if len(lengths) != len(flags):
            raise ValueError(f"{len(lengths)} lengths vs {len(flags)} real_end flags")
        bad = [set(int(t) for t in b) for b in anchor_bad] if anchor_bad \
            else [set()] * len(lengths)
        if len(bad) != len(lengths):
            raise ValueError(f"{len(lengths)} lengths vs {len(bad)} anchor_bad lists")
        h = int(action_horizon)
        valid = []
        offset = 0
        for length, real_end, ep_bad in zip(lengths, flags, bad):
            if real_end and allow_terminal_padding:
                n_valid = length
            else:
                n_valid = max(length - h, 0)
            valid.extend(offset + t for t in range(n_valid) if t not in ep_bad)
            offset += length
        self.total_frames = offset
        self.indices = np.asarray(valid, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


class BoundaryAwareDataset:
    """Wrap any frame-indexable dataset so only valid datapoints are visible."""

    def __init__(self, dataset, indices: BoundaryAwareIndices):
        self._dataset = dataset
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        return self._dataset[self._indices[i]]


def load_boundary_indices(dataset_root, action_horizon, allow_terminal_padding=True):
    """Build BoundaryAwareIndices from a written dataset: episode lengths
    from lerobot meta (meta/episodes.jsonl), real_end flags + anchor_bad
    frame offsets from the extraction_meta.json sidecar."""
    root = Path(dataset_root)
    sidecar = json.loads((root / "extraction_meta.json").read_text())
    lengths = {}
    with (root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            lengths[int(rec["episode_index"])] = int(rec["length"])
    n = len(lengths)
    episode_lengths = [lengths[i] for i in range(n)]
    eps = [sidecar["episodes"][str(i)] for i in range(n)]
    real_end = [bool(e["episode_real_end"]) for e in eps]
    anchor_bad = [e.get("anchor_bad", []) for e in eps]
    return BoundaryAwareIndices(episode_lengths, real_end, action_horizon,
                                allow_terminal_padding, anchor_bad=anchor_bad)


def make_boundary_aware(dataset, dataset_root, action_horizon,
                        allow_terminal_padding=True):
    """Convenience: wrap `dataset` using meta found under `dataset_root`."""
    idx = load_boundary_indices(dataset_root, action_horizon, allow_terminal_padding)
    return BoundaryAwareDataset(dataset, idx)
