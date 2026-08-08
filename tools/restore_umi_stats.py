"""Restore a UMI norm-stats artifact into the assets staging dir FROM the
checkpoint that was trained with it.

WHY THIS EXISTS INSTEAD OF `compute_norm_stats --umi`.

`train.main_umi` loads `umi_stats.npz` from the assets staging dir
(`UmiTrainConfig.stats_dir`) on every start, INCLUDING a resume. If that staging
copy has been deleted -- CPFS is tight and assets are regenerable, so deleting
it is a reasonable thing to have done -- a resume cannot start.

Recomputing looks equivalent and almost certainly is: the grids are exact
percentiles over the train split, read straight from parquet, and the rotvec
code path is unchanged. "Almost certainly" is the problem. If ANYTHING that
feeds the grids has moved since the run started -- val_source_episodes,
action_horizon, the lag grid, a boundary rule in `umi_raw_action_chunks` -- the
resumed run would normalize its targets differently from its own first 9999
steps, and would train to a perfectly plausible loss while doing it. There is no
assertion anywhere that would catch that, because from step 9999's point of view
the new stats are simply the stats.

The checkpoint's own `assets_ego2g1/umi_stats.npz` is the artifact that run
actually used. Copying it back is exact by construction, and needs no argument
about what has or has not changed.

WHAT IT CHECKS BEFORE COPYING. The point is to be able to trust the result, so
it refuses rather than warns:

  - the source is the representation you asked for;
  - the source has a FULL lag grid (n_lags rows). Both state modes share one
    artifact per representation -- legitimately, since the action grid and the
    gripper quantiles do not depend on state_mode -- but only the history-mode
    file works for both, because a gripper_token run computes a 1-lag grid that
    a history run cannot use;
  - every OTHER checkpoint given on the command line agrees with the source on
    the action grid, the gripper dims and the gripper quantiles, bit for bit.
    That is what licenses restoring ONE file and resuming BOTH runs off it. A
    mismatch means the two runs were never normalizing the same way and the
    comparison between them was already invalid -- worth knowing loudly.

Usage (from the repo root, with the train profile sourced):

    python -m tools.restore_umi_stats \\
        --source   /mnt/cpfs/hxy/runs/ego2g1_v2/umi_wrist/umi \\
        --also     /mnt/cpfs/hxy/runs/ego2g1_v2/umi_wrist/umi_no_state_history \\
        --assets-base-dir /mnt/cpfs/hxy/runs/ego2g1_v2/assets

Add --force to overwrite an existing staging file (it refuses by default: an
existing file is the one training would have used, and clobbering it silently is
the failure mode this whole module exists to prevent).
"""

import argparse
import pathlib
import shutil
import sys

import numpy as np

CHECKPOINT_ASSETS = "assets_ego2g1"


def _load(run_dir: pathlib.Path):
    from ego2g1.train import norm as _norm

    d = run_dir / CHECKPOINT_ASSETS
    path = d / _norm.UMI_FILENAME
    if not path.exists():
        present = sorted(p.name for p in run_dir.glob("*")) if run_dir.exists() else []
        raise SystemExit(
            f"{path} does not exist.\n"
            f"  {run_dir} contains: {present}\n"
            "  Without the run's own copy there is nothing to restore FROM, and a "
            "recompute cannot be proven equal to what it trained on — see this "
            "module's docstring before deciding to recompute anyway."
        )
    return _norm.load_umi(d), path


def main(argv=None) -> int:
    from ego2g1.train import config as _config

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=pathlib.Path,
                   help="checkpoint dir to restore FROM; use the HISTORY-mode run, "
                        "whose full lag grid serves both state modes")
    p.add_argument("--also", action="append", default=[], type=pathlib.Path,
                   help="other checkpoint dirs that must agree with the source "
                        "(repeatable)")
    p.add_argument("--assets-base-dir", required=True, type=pathlib.Path)
    p.add_argument("--rotation-repr", default="rotvec", choices=("rotvec", "rot6d"))
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing staging file")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    cfg = _config.UmiTrainConfig(assets_base_dir=str(a.assets_base_dir),
                                 rotation_repr=a.rotation_repr,
                                 action_dim_actual=3 + (3 if a.rotation_repr == "rotvec" else 6) + 1)
    dest_dir = cfg.stats_dir
    dest = dest_dir / "umi_stats.npz"

    src, src_path = _load(a.source)
    print(f"source: {src_path}")
    print(f"  rotation_repr={src.rotation_repr}  action grid {src.action_q01.shape}  "
          f"history {src.history_mean.shape}  gripper_dims={src.gripper_dims}")
    print(f"  provenance lag_ticks={src.provenance.get('lag_ticks')}  "
          f"num_chunks={src.provenance.get('num_chunks')}")

    problems = []
    if src.rotation_repr != a.rotation_repr:
        problems.append(f"source is {src.rotation_repr!r}, not {a.rotation_repr!r}")
    if src.history_mean.shape[0] != cfg.n_lags:
        problems.append(
            f"source has a {src.history_mean.shape[0]}-lag history grid, expected "
            f"{cfg.n_lags}. It came from a gripper_token run; restore from the "
            "HISTORY-mode checkpoint instead — its grid serves both modes, this one "
            "does not")

    # The invariant that licenses one staging file for two runs.
    for other in a.also:
        o, o_path = _load(other)
        deltas = []
        if o.rotation_repr != src.rotation_repr:
            deltas.append(f"rotation_repr {o.rotation_repr!r} vs {src.rotation_repr!r}")
        for name, x, y in (("action_q01", o.action_q01, src.action_q01),
                           ("action_q99", o.action_q99, src.action_q99)):
            if x.shape != y.shape:
                deltas.append(f"{name} shape {x.shape} vs {y.shape}")
            elif not np.array_equal(x, y):
                deltas.append(f"{name} differs (max |delta| {np.abs(x - y).max():.3e})")
        if o.gripper_dims != src.gripper_dims:
            deltas.append(f"gripper_dims {o.gripper_dims} vs {src.gripper_dims}")
        for name, x, y in (("gripper_q01", o.gripper_q01, src.gripper_q01),
                           ("gripper_q99", o.gripper_q99, src.gripper_q99)):
            # NaN == NaN is False; both-NaN is agreement here (old artifacts)
            if not (x == y or (x != x and y != y)):
                deltas.append(f"{name} {x} vs {y}")
        if deltas:
            problems.append(f"{o_path} disagrees with the source: " + "; ".join(deltas))
        else:
            print(f"agrees: {o_path}  (action grid, gripper dims and quantiles "
                  f"identical; its {o.history_mean.shape[0]}-lag history grid is "
                  "unused in gripper_token mode)")

    if dest.exists() and not a.force:
        problems.append(
            f"{dest} already exists. That file is what training would load, so it is "
            "not overwritten silently. Inspect it, then pass --force if you are sure")

    if problems:
        print("\nREFUSING TO RESTORE:", file=sys.stderr)
        for x in problems:
            print(f"  - {x}", file=sys.stderr)
        return 1

    print(f"\ndestination: {dest}")
    if a.dry_run:
        print("(dry run — nothing written)")
        return 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    # Read it back through the real loader rather than trusting the copy.
    from ego2g1.train import norm as _norm

    back = _norm.load_umi(dest_dir)
    assert back.rotation_repr == src.rotation_repr
    assert np.array_equal(back.action_q01, src.action_q01)
    assert np.array_equal(back.history_mean, src.history_mean)
    print("restored and verified by re-loading through norm.load_umi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
