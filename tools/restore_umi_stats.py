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

IT PICKS THE SOURCE ITSELF, from the candidates you name. Which checkpoint's
copy is usable is not obvious and is not the operator's problem to solve:

  - the artifact must have a FULL lag grid (n_lags rows). Both state modes share
    one artifact per representation -- legitimately, since the action grid and
    the gripper quantiles do not depend on state_mode -- but only a
    history-length grid works for both, because a gripper_token COMPUTE produces
    a 1-lag grid that a history run cannot use;
  - it must carry the gripper QUANTILES. A run that predates the field
    (gripper_q01/q99 arrived with state_mode="gripper_token", in c4cecbb) has
    NaN there. That is invisible to a history run, which never reads them, and
    fatal to a gripper_token run, which refuses to digitize without them. So the
    older checkpoint of a pair can be the one that cannot serve the restore.

CONSISTENCY IS STILL ENFORCED across every candidate, because restoring ONE file
and resuming SEVERAL runs off it is only licensed if they were all normalizing
the same way:

  - action_q01/q99 and gripper_dims must be bit-identical. A difference means
    those runs never shared a normalization and comparing them was already
    invalid -- worth knowing loudly;
  - history grids of the same height must be bit-identical;
  - gripper quantiles: NaN on one side is ABSENCE, not disagreement -- an
    artifact written before the field existed is a subset of one written after,
    not a conflict with it. Two DIFFERENT non-NaN values are a real conflict.

Usage (from the repo root, with the train profile sourced):

    python -m tools.restore_umi_stats \\
        --candidate /mnt/cpfs/hxy/runs/ego2g1_v2/umi_wrist/umi \\
        --candidate /mnt/cpfs/hxy/runs/ego2g1_v2/umi_wrist/umi_no_state_history \\
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
    p.add_argument("--candidate", action="append", required=True, dest="candidates",
                   type=pathlib.Path, metavar="DIR",
                   help="checkpoint dir whose assets_ego2g1/ copy may serve as the "
                        "source (repeatable). All must agree; the first USABLE one "
                        "is restored — see this module's docstring for what usable "
                        "means, and why it is not simply the first one given")
    p.add_argument("--assets-base-dir", required=True, type=pathlib.Path)
    p.add_argument("--rotation-repr", default="rotvec", choices=("rotvec", "rot6d"))
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing staging file")
    p.add_argument("--compare-with", type=pathlib.Path, default=None,
                   help="a directory holding another umi_stats.npz (e.g. a fresh "
                        "compute_norm_stats output) to diff against the source. "
                        "REPORTS ONLY, never blocks — it answers 'would recomputing "
                        "have given the same grids?' with numbers instead of an "
                        "argument")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    cfg = _config.UmiTrainConfig(assets_base_dir=str(a.assets_base_dir),
                                 rotation_repr=a.rotation_repr,
                                 action_dim_actual=3 + (3 if a.rotation_repr == "rotvec" else 6) + 1)
    dest_dir = cfg.stats_dir
    dest = dest_dir / "umi_stats.npz"

    loaded = [(*_load(d), d) for d in a.candidates]
    problems = []

    print("candidates:")
    for st, path, _ in loaded:
        q = ("absent (predates the field)" if st.gripper_q01 != st.gripper_q01
             else f"{st.gripper_q01:.4f}..{st.gripper_q99:.4f}")
        print(f"  {path}\n"
              f"    rotation_repr={st.rotation_repr}  action {st.action_q01.shape}  "
              f"history {st.history_mean.shape}  gripper_dims={st.gripper_dims}\n"
              f"    gripper quantiles {q}  "
              f"lag_ticks={st.provenance.get('lag_ticks')}  "
              f"num_chunks={st.provenance.get('num_chunks')}")

    # --- every candidate must describe the SAME normalization -----------------
    ref, ref_path, _ = loaded[0]
    for o, o_path, _ in loaded[1:]:
        deltas = []
        if o.rotation_repr != ref.rotation_repr:
            deltas.append(f"rotation_repr {o.rotation_repr!r} vs {ref.rotation_repr!r}")
        pairs = [("action_q01", o.action_q01, ref.action_q01),
                 ("action_q99", o.action_q99, ref.action_q99)]
        # History grids are only comparable at equal height: a gripper_token
        # COMPUTE legitimately produces a shorter one.
        if o.history_mean.shape == ref.history_mean.shape:
            pairs += [("history_mean", o.history_mean, ref.history_mean),
                      ("history_std", o.history_std, ref.history_std)]
        for name, x, y in pairs:
            if x.shape != y.shape:
                deltas.append(f"{name} shape {x.shape} vs {y.shape}")
            elif not np.array_equal(x, y):
                deltas.append(f"{name} differs (max |delta| {np.abs(x - y).max():.3e})")
        if o.gripper_dims != ref.gripper_dims:
            deltas.append(f"gripper_dims {o.gripper_dims} vs {ref.gripper_dims}")
        for name, x, y in (("gripper_q01", o.gripper_q01, ref.gripper_q01),
                           ("gripper_q99", o.gripper_q99, ref.gripper_q99)):
            # NaN is ABSENCE, not disagreement: an artifact written before the
            # field existed is a SUBSET of one written after, not a conflict with
            # it. `umi` predates gripper_q01/q99 (they arrived with
            # state_mode="gripper_token", in c4cecbb) and treating its NaN as a
            # mismatch blocked a restore that was entirely correct. Two DIFFERENT
            # non-NaN values remain a real conflict.
            if x != x or y != y:
                continue
            if x != y:
                deltas.append(f"{name} {x} vs {y}")
        if deltas:
            problems.append(f"{o_path} disagrees with {ref_path}: " + "; ".join(deltas))
    if not problems and len(loaded) > 1:
        print(f"\nall {len(loaded)} candidates agree on the action grid and gripper "
              "dims — one staging file legitimately serves every run")

    # --- pick the first USABLE candidate --------------------------------------
    # Usable is not the same as "first given": the older checkpoint of a pair can
    # be the one that cannot serve, because it predates the gripper quantiles.
    src = src_path = None
    skipped = []
    for st, path, _ in loaded:
        why = []
        if st.rotation_repr != a.rotation_repr:
            why.append(f"is {st.rotation_repr!r}, not {a.rotation_repr!r}")
        if st.history_mean.shape[0] != cfg.n_lags:
            why.append(f"has a {st.history_mean.shape[0]}-lag history grid, not "
                       f"{cfg.n_lags} — a history run would raise in NormalizeHistory")
        if st.gripper_q01 != st.gripper_q01:
            why.append("carries no gripper quantiles (predates the field) — a "
                       "gripper_token run refuses to digitize without them")
        if why:
            skipped.append(f"{path}: " + "; ".join(why))
        else:
            src, src_path = st, path
            break
    if src is None:
        problems.append("no candidate can serve as the source:\n      "
                        + "\n      ".join(skipped))
    else:
        for s in skipped:
            print(f"\nskipped as source: {s}")
        print(f"\nchosen source: {src_path}")

    # Reported, never enforced: a difference here does not make the restore
    # wrong (the source is what the run trained on, by definition), it tells you
    # whether recomputing WOULD have been safe. Equal grids mean the pipeline is
    # reproducible and the caution cost nothing; unequal grids mean recomputing
    # would have silently changed the target distribution mid-run.
    if a.compare_with is not None and src is not None:
        from ego2g1.train import norm as _norm

        other = _norm.load_umi(a.compare_with)
        print(f"\ncompare-with: {a.compare_with}/{_norm.UMI_FILENAME}")
        same = True
        for name, x, y in (("action_q01", other.action_q01, src.action_q01),
                           ("action_q99", other.action_q99, src.action_q99),
                           ("history_mean", other.history_mean, src.history_mean),
                           ("history_std", other.history_std, src.history_std)):
            if x.shape != y.shape:
                print(f"  {name:13s} SHAPE {x.shape} vs {y.shape}")
                same = False
            elif np.array_equal(x, y):
                print(f"  {name:13s} identical")
            else:
                d = np.abs(x - y)
                print(f"  {name:13s} DIFFERS  max |delta| {d.max():.3e}  "
                      f"max rel {np.abs(d / np.maximum(np.abs(y), 1e-12)).max():.3e}")
                same = False
        print("  -> a recompute would have been bit-identical" if same else
              "  -> a recompute would have CHANGED the normalization: restoring was "
              "necessary, not just cautious")

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
