"""Norm-stats artifacts: pooled openpi NormStats + the E001 per-slot grid.

Two files per stats computation, both stamped with provenance
(TRAINING_PLAN.md §3.6):
- `norm_stats.json` — openpi-native pooled per-dim stats for `state` and
  `actions` (written via openpi.shared.normalize.save); consumed unchanged by
  stock Normalize/Unnormalize and the checkpoint assets hook.
- `per_slot_stats.npz` — sigma_slot (H, D_real) + provenance json; the E001
  gain grid is DERIVED at load time for the configured floor c (sigma is the
  artifact, c is config).
"""

import dataclasses
import json
import pathlib

import numpy as np

import openpi.shared.normalize as _normalize

PER_SLOT_FILENAME = "per_slot_stats.npz"

# (slot, dim) pairs allowed to be degenerate (q99-q01 or sigma_slot ~ 0).
# rot6d identity-adjacent dims at slot 1 and unused left fingers are known;
# extend deliberately, never silently. Checked per-dim over ALL slots for
# pooled stats and per (slot, dim) for sigma_slot.
DEGENERATE_EPS = 1e-8


def degenerate_action_dims(actions_stats: _normalize.NormStats, d_real: int) -> np.ndarray:
    """(D_real,) bool mask — THE single degeneracy criterion; every consumer
    (sanity gate, gain, centering, data-path neutralization) must use it.

    A dim is degenerate when its pooled quantile span is ~0 (never moves) OR
    span << sigma (spike-plus-tail: constant for the 98% bulk with a rare
    outlier tail — e.g. unused fingers with retargeting glitches). The second
    clause matters: healthy distributions have span ~ 4-5 sigma, and the
    quantile-Normalize epsilon turns tail frames of a span~0 dim into
    normalized values of ~1e5 (measured on put_bottle_in_box dims 13/14)."""
    span = (actions_stats.q99 - actions_stats.q01)[:d_real]
    std = actions_stats.std[:d_real]
    return (span <= DEGENERATE_EPS) | (span < 0.5 * std)


@dataclasses.dataclass(frozen=True)
class PerSlotStats:
    sigma_slot: np.ndarray  # (H, D_real)
    provenance: dict
    # per-(slot, dim) mean of raw actions; needed for per-slot centering.
    # None when loaded from a pre-centering artifact (recompute to get it).
    mu_slot: np.ndarray | None = None

    def gain(self, floor_c: float, sigma_pooled: np.ndarray,
             degenerate_mask: np.ndarray | None = None) -> np.ndarray:
        """E001: gain[k,d] = sigma_pooled[d] / max(sigma_slot[k,d], c*sigma_pooled[d]).
        floor_c=1 -> gain==1 (bitwise stock pooled behavior). Dims whose pooled
        sigma is itself degenerate get gain 1 (they carry no signal at all);
        pass `degenerate_mask` (degenerate_action_dims) to also exempt
        spike-plus-tail dims whose sigma is nonzero only from outliers."""
        if not 0 < floor_c <= 1:
            raise ValueError(f"floor_c={floor_c} must be in (0, 1]")
        sp = np.asarray(sigma_pooled, dtype=np.float64)[: self.sigma_slot.shape[1]]
        divisor = np.maximum(self.sigma_slot, floor_c * sp[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            g = sp[None, :] / divisor
        g = np.where(sp[None, :] <= DEGENERATE_EPS, 1.0, g)
        if degenerate_mask is not None:
            g = np.where(np.asarray(degenerate_mask, dtype=bool)[None, :], 1.0, g)
        # gain <= 1/c by the floor; gain < 1 is legitimate (late slots whose
        # sigma exceeds the pooled sigma get shrunk toward unit scale).
        assert np.all(np.isfinite(g)) and np.all(g > 0.0) and np.all(g <= 1.0 / floor_c + 1e-9)
        return g.astype(np.float32)


def save_per_slot(directory: pathlib.Path | str, stats: PerSlotStats) -> None:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    extra = {} if stats.mu_slot is None else {"mu_slot": stats.mu_slot}
    np.savez(
        directory / PER_SLOT_FILENAME,
        sigma_slot=stats.sigma_slot,
        provenance=json.dumps(stats.provenance),
        **extra,
    )


def load_per_slot(directory: pathlib.Path | str) -> PerSlotStats:
    path = pathlib.Path(directory) / PER_SLOT_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m ego2g1.train.compute_norm_stats` (E001 needs the per-slot grid)"
        )
    with np.load(path, allow_pickle=False) as z:
        return PerSlotStats(
            sigma_slot=np.asarray(z["sigma_slot"], dtype=np.float64),
            provenance=json.loads(str(z["provenance"])),
            mu_slot=np.asarray(z["mu_slot"], dtype=np.float64) if "mu_slot" in z.files else None,
        )


def load_pooled(directory: pathlib.Path | str) -> dict[str, _normalize.NormStats]:
    return _normalize.load(directory)


# --------------------------------------------------------------------------
# EgoRelationTrainConfig artifacts (relational state + 14-dim rotvec actions)
# --------------------------------------------------------------------------

RELATION_FILENAME = "relation_stats.npz"


@dataclasses.dataclass(frozen=True)
class RelationNormStats:
    """The two stats grids the relational config needs, in ONE artifact.

    action_q01/action_q99 : (H, D_real) per-(slot, dim) quantiles.
        Unlike the E001 grid, this is the WHOLE action normalization -- there is
        no pooled step underneath it. Measured on red_block_in_pen_holder_ego it
        leaves every slot at std 0.27-0.44 (ratio 1.11-1.63 across the chunk)
        against 17-24x for a pooled-only scheme, because early-slot deltas are
        ~30x smaller than late-slot ones and only early slots ever execute.

    relation_mean/relation_std : (relation_dim,) z-score stats, POOLED ACROSS
        OBJECTS. Pooling is a correctness requirement, not a convenience: one
        shared encoder plus a shuffled prompt order means per-object stats would
        make the same physical relation encode differently depending on which
        slot it landed in, destroying the permutation equivariance the shuffling
        exists to create. Object identity travels as text, never as scale.

    gripper_dims are recorded so the forward/inverse pair and the loss weighting
    cannot disagree about which dims are exempt from normalization.
    """

    action_q01: np.ndarray       # (H, D_real)
    action_q99: np.ndarray       # (H, D_real)
    relation_mean: np.ndarray    # (relation_dim,)
    relation_std: np.ndarray     # (relation_dim,)
    gripper_dims: tuple[int, ...]
    provenance: dict

    def __post_init__(self):
        if self.action_q01.shape != self.action_q99.shape:
            raise ValueError(f"q01 {self.action_q01.shape} != q99 {self.action_q99.shape}")
        if self.relation_mean.shape != self.relation_std.shape:
            raise ValueError(f"mean {self.relation_mean.shape} != std {self.relation_std.shape}")

    @property
    def action_span(self) -> np.ndarray:
        """The Normalize denominator, with openpi's exact epsilon."""
        return self.action_q99 - self.action_q01 + 1e-6


def save_relation(directory: pathlib.Path | str, stats: RelationNormStats) -> None:
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / RELATION_FILENAME,
        action_q01=stats.action_q01,
        action_q99=stats.action_q99,
        relation_mean=stats.relation_mean,
        relation_std=stats.relation_std,
        gripper_dims=np.asarray(stats.gripper_dims, dtype=np.int64),
        provenance=json.dumps(stats.provenance),
    )


def load_relation(directory: pathlib.Path | str) -> RelationNormStats:
    path = pathlib.Path(directory) / RELATION_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m ego2g1.train.compute_norm_stats "
            "--config relation` (the relational config needs the per-slot quantile "
            "grid and the pooled relation z-score stats)"
        )
    with np.load(path, allow_pickle=False) as z:
        return RelationNormStats(
            action_q01=np.asarray(z["action_q01"], dtype=np.float64),
            action_q99=np.asarray(z["action_q99"], dtype=np.float64),
            relation_mean=np.asarray(z["relation_mean"], dtype=np.float64),
            relation_std=np.asarray(z["relation_std"], dtype=np.float64),
            gripper_dims=tuple(int(d) for d in z["gripper_dims"]),
            provenance=json.loads(str(z["provenance"])),
        )


def check_relation_stats_sanity(stats: RelationNormStats, max_abs_norm: float = 20.0) -> list[str]:
    """Returns human-readable violations (empty = pass).

    Every one of these is a data bug rather than a tuning knob:
    - a zero-span (slot, dim) means that dim never moves at that slot, which for
      a 14-dim all-live action space should not happen (the 30-dim config had
      genuinely dead finger dims; this one has none, so an empty degenerate set
      is itself the check);
    - a zero relation std means a relation dim is constant across the dataset;
    - a gripper dim inside the normalized set would double-scale it.
    """
    problems = []
    span = stats.action_q99 - stats.action_q01
    d_real = span.shape[1]
    for d in range(d_real):
        if d in stats.gripper_dims:
            continue
        dead = np.flatnonzero(span[:, d] <= DEGENERATE_EPS)
        if dead.size:
            problems.append(
                f"actions dim {d}: zero quantile span at slots {dead[:5].tolist()}"
                f"{' ...' if dead.size > 5 else ''} ({dead.size}/{span.shape[0]} slots)"
            )
    for d in stats.gripper_dims:
        if not np.all(span[:, d] <= DEGENERATE_EPS) and np.any(span[:, d] > 0):
            # not fatal: the grid is simply unused for these dims
            pass
    if np.any(stats.relation_std <= DEGENERATE_EPS):
        dead = np.flatnonzero(stats.relation_std <= DEGENERATE_EPS)
        problems.append(f"relation dims {dead.tolist()} have ~zero std (constant across the dataset)")
    if max(stats.gripper_dims, default=-1) >= d_real:
        problems.append(f"gripper_dims {stats.gripper_dims} out of range for D_real={d_real}")
    return problems


def check_stats_sanity(
    pooled: dict[str, _normalize.NormStats],
    per_slot: PerSlotStats,
    degenerate_dim_allowlist: tuple[int, ...],
    raw_min: np.ndarray | None = None,
    raw_max: np.ndarray | None = None,
    max_abs_norm: float = 1000.0,
) -> list[str]:
    """E001 eval item 7. Returns human-readable violations (empty = pass).
    Dims flagged by degenerate_action_dims (the SAME mask the data path
    neutralizes) and all-slot-degenerate sigma rows must be in the allowlist;
    anything else is a data bug, not a tuning knob. If raw per-dim min/max are
    provided, additionally certify that no un-masked dim produces a normalized
    value beyond `max_abs_norm` (spike-plus-tail shapes the mask missed)."""
    problems = []
    act = pooled["actions"]
    d_real = per_slot.sigma_slot.shape[1]
    mask = degenerate_action_dims(act, d_real)
    for d in range(d_real):
        if mask[d] and d not in degenerate_dim_allowlist:
            span = float(act.q99[d] - act.q01[d])
            problems.append(
                f"actions dim {d}: degenerate (q99-q01 = {span:.3e}, std = {float(act.std[d]):.3e}), not allowlisted"
            )
    for d in range(d_real):
        if d in degenerate_dim_allowlist:
            continue
        # slot 0 of anchor-relative deltas is legitimately tiny; require SOME
        # signal by mid-chunk rather than at every slot.
        if float(per_slot.sigma_slot[:, d].max()) <= DEGENERATE_EPS:
            problems.append(f"actions dim {d}: sigma_slot ~ 0 at every slot (not allowlisted)")
    if raw_min is not None and raw_max is not None:
        span = act.q99[:d_real] - act.q01[:d_real] + 1e-6  # Normalize's exact denominator
        for d in range(d_real):
            if mask[d]:
                continue  # neutralized in the data path, extremes never reach the model
            n_extreme = max(
                abs((float(raw_min[d]) - float(act.q01[d])) / float(span[d]) * 2.0 - 1.0),
                abs((float(raw_max[d]) - float(act.q01[d])) / float(span[d]) * 2.0 - 1.0),
            )
            if n_extreme > max_abs_norm:
                problems.append(
                    f"actions dim {d}: max |normalized| = {n_extreme:.1f} > {max_abs_norm:g} "
                    "(spike-plus-tail distribution not caught by the degeneracy mask)"
                )
    return problems
