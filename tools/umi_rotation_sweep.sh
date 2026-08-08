#!/usr/bin/env bash
# THE rotation-representation A/B: one command, start to finish.
#
#     source /mnt/cpfs/hxy/ego2g1_v2/envs/ppu-train.sh
#     tmux new -s sweep
#     bash /mnt/cpfs/hxy/ego2g1_v2/tools/umi_rotation_sweep.sh
#     # detach C-b d, reattach `tmux attach -t sweep`
#
# There is nothing to run before or after this, and nothing to run by hand if it
# stops: it is SAFE TO RERUN at any point. Every training run passes --resume,
# which openpi's initialize_checkpoint_dir handles in all three cases (no
# directory yet -> fresh; directory but no checkpoints -> fresh; checkpoints ->
# continue), so a crashed or killed run picks up from its last save instead of
# refusing to start or silently restarting from zero.
#
# WHAT IT DOES
#
#   preflight   environment, dataset, pi05 params, tests, resume targets
#   restore     the ROTVEC staging stats, from the checkpoints that trained on
#               them, if the staging copy is missing (automatic; see below)
#   stats       the ROT6D staging stats (new, so computed fresh)
#   run 1       rot6d + state history     10k steps
#   run 2       rot6d + gripper token     10k steps
#   run 3       rotvec + state history    resume 9999 -> 20k
#   run 4       rotvec + gripper token    resume 9999 -> 20k
#
# WHY THE TWO STATS PATHS DIFFER. The rot6d grids do not exist yet, so computing
# them is the only option and is correct. The rotvec grids are what two 9999-step
# runs already trained against, and `train.main_umi` reloads them from the
# STAGING dir on resume -- so if that copy is gone, recomputing is the tempting
# move and the wrong one. A recompute is almost certainly identical (exact
# percentiles, straight from parquet, unchanged code path) but cannot be SHOWN
# identical, and if anything feeding the grids has moved since those runs started
# the resume would normalize differently from its own first 9999 steps and train
# to a perfectly plausible loss doing it. Each run saved the artifact it used
# into its own checkpoint, so this script copies that back instead -- exact by
# construction, no argument required. See tools/restore_umi_stats.py.
#
# FAILURE POLICY. Preflight, restore and stats are FATAL: every run depends on
# them, so continuing past a failure only burns GPU-hours producing garbage.
# Training runs are INDEPENDENT -- one OOM must not cost the other three -- so
# failures are recorded, the sweep continues, and the exit code is nonzero with a
# summary at the end.

set -u -o pipefail

# ---------------------------------------------------------------- configuration
# Defaults match the workspace layout in docs/ppu_venv_setup.md. Every one is
# overridable from the environment; nothing below this block needs editing.

WS="${WS:-/mnt/cpfs/hxy}"
REPO="${REPO:-$WS/ego2g1_v2}"
RUNS="${RUNS:-$WS/runs/ego2g1_v2}"                  # checkpoints + logs
ASSETS="${ASSETS:-$RUNS/assets}"                    # norm-stats staging
PI05_PARAMS="${PI05_PARAMS:-$WS/cache/openpi/openpi-assets/checkpoints/pi05_base/params}"
LOGDIR="${LOGDIR:-$RUNS/logs/rotation_sweep_$(date +%Y%m%d_%H%M%S)}"

# The two ROTVEC runs already on disk, resumed to 20k as the control arm.
EXP_RV_HIST="${EXP_RV_HIST:-umi}"
EXP_RV_TOK="${EXP_RV_TOK:-umi_no_state_history}"
# The two ROT6D runs, trained from scratch.
EXP_R6_HIST="${EXP_R6_HIST:-umi_6DRot_with_history}"
EXP_R6_TOK="${EXP_R6_TOK:-umi_6DRot_no_history}"

FROM_SCRATCH_STEPS="${FROM_SCRATCH_STEPS:-10000}"   # matches the rotvec baselines
RESUME_TO_STEPS="${RESUME_TO_STEPS:-20000}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-9999}"

# A stray norm-stats file lands here if `compute_norm_stats --umi` is ever run
# WITHOUT --assets-base-dir: the default is <repo>/assets, inside the pull-only
# checkout. Not used for anything -- but if it exists it is a free answer to
# "would recomputing have matched?", so the restore step diffs against it.
STRAY_ASSETS="${STRAY_ASSETS:-$REPO/assets}"

# --video-backend pyav is mandatory here: torchcodec cannot load its native
# library on PPU. --assets-base-dir and --checkpoint-base-dir keep tens of GB out
# of the pull-only checkout, and compute_norm_stats and train MUST agree on the
# assets value or training cannot find the stats it just wrote.
COMMON=(--video-backend pyav
        --weight-loader-params-path "$PI05_PARAMS"
        --assets-base-dir "$ASSETS"
        --checkpoint-base-dir "$RUNS"
        --resume)

# Two flags, always together: action_dim_actual is VALIDATED against
# rotation_repr rather than derived from it, so forgetting one fails at config
# construction instead of building a mismatched normalization grid.
ROT6D=(--rotation-repr rot6d --action-dim-actual 10)
ROTVEC=(--rotation-repr rotvec --action-dim-actual 7)

# "no state history" is state_mode=gripper_token: the gripper alone, binned into
# the prompt as pi05-style digits, and no learned encoder in the param tree.
HIST=(--state-mode history)
TOK=(--state-mode gripper_token)

mkdir -p "$LOGDIR"
FAILED=()
SUMMARY=()

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------- preflight: environment

say "preflight: environment"
cd "$REPO" || die "no checkout at $REPO (override with REPO=...)"

[ -n "${VIRTUAL_ENV:-}" ] || die "no venv active. Run first:
    source $REPO/envs/ppu-train.sh
NEVER 'uv run' or 'uv sync' on this box — both target ./.venv and resolve NVIDIA
wheels from uv.lock, which is a silent CPU fallback."

python -c "import jax,sys; n=jax.device_count(); print(f'  jax backend={jax.default_backend()} devices={n}');
sys.exit(0 if jax.default_backend()=='gpu' and n>0 else 1)" \
    || die "jax is not on the PPU backend — re-source envs/ppu-train.sh and read its verification block"

# The from-scratch runs load pi05_base. Checked HERE rather than discovered 40
# minutes in, after preflight has already passed.
[ -d "$PI05_PARAMS" ] || die "pi05_base params not found at
    $PI05_PARAMS
The two from-scratch runs cannot start without them. Set PI05_PARAMS=... (the
config default is a gs:// URL, which would need network access from this pod)."
echo "  pi05_base params: $PI05_PARAMS"

python - "$ASSETS" "$RUNS" <<'PY' || die "dataset not readable — check EGO2G1_DATA"
import pathlib, sys
from ego2g1.train import config as _config
c = _config.UmiTrainConfig(assets_base_dir=sys.argv[1], checkpoint_base_dir=sys.argv[2])
root = pathlib.Path(c.dataset_root)
print(f"  dataset: {root}")
if not (root / "meta" / "info.json").exists():
    print(f"  MISSING: {root}/meta/info.json", file=sys.stderr)
    sys.exit(1)
print(f"  assets staging: {c.stats_dir}  (rot6d -> {c.stats_dir}/rot6d)")
PY

# ------------------------------------------------------------ preflight: tests

say "preflight: tests"
# test_legacy_config_hash_is_pinned CANNOT pass on this box and says nothing
# about this change: config_hash() covers `dataset_root`, whose default derives
# from EGO2G1_DATA (or <repo>/data), so the hash is a function of the machine's
# data path AND the checkout location. The constant was pinned on a Mac checkout.
# Deselected BY NAME rather than by dropping the test files — the rot6d tests are
# new, and silencing the suite to get past one environment-dependent assert is
# how a real bug hides.
python -m pytest tests/train/test_umi.py tests/train/test_relation.py -q \
    --deselect tests/train/test_relation.py::test_legacy_config_hash_is_pinned \
    2>&1 | tee "$LOGDIR/00_tests.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "tests failed — see $LOGDIR/00_tests.log"

# --------------------------------------------- restore the rotvec staging stats

RV_STATS="$ASSETS/umi_wrist/ego2g1/red_block_on_yellow_block_umi/umi_stats.npz"
if [ -f "$RV_STATS" ]; then
    say "rotvec stats: already staged"
    echo "  $RV_STATS"
else
    say "rotvec stats: restoring from the checkpoints that trained on them"
    RESTORE=(--source "$RUNS/umi_wrist/$EXP_RV_HIST"
             --also   "$RUNS/umi_wrist/$EXP_RV_TOK"
             --assets-base-dir "$ASSETS")
    # Free evidence if a stray recompute is lying around: reports whether it
    # would have matched. Never blocks either way.
    if [ -f "$STRAY_ASSETS/umi_wrist/ego2g1/red_block_on_yellow_block_umi/umi_stats.npz" ]; then
        RESTORE+=(--compare-with "$STRAY_ASSETS/umi_wrist/ego2g1/red_block_on_yellow_block_umi")
    fi
    python -m tools.restore_umi_stats "${RESTORE[@]}" 2>&1 | tee "$LOGDIR/01_restore.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "could not restore the rotvec stats — see $LOGDIR/01_restore.log"
fi

# --------------------------------------------- preflight: the resume targets

say "preflight: resume targets"
python - "$RUNS" "$ASSETS" "$EXP_RV_HIST" "$EXP_RV_TOK" "$RESUME_FROM_STEP" <<'PY' \
    2>&1 | tee "$LOGDIR/02_preflight.log"
import json, pathlib, sys
from ego2g1.train import config as _config, norm as _norm

runs, assets, exp_hist, exp_tok, step = sys.argv[1:6]
step = int(step)
bad = []
rv = _config.UmiTrainConfig(assets_base_dir=assets, checkpoint_base_dir=runs)

# The staged artifact the two resumes will load.
s = _norm.load_umi(rv.stats_dir)
if s.rotation_repr != "rotvec":
    bad.append(f"{rv.stats_dir} holds {s.rotation_repr!r} stats, not 'rotvec'")
elif s.history_mean.shape[0] != rv.n_lags:
    # Both state modes share ONE artifact per representation, legitimately: the
    # action grid and the gripper quantiles do not depend on state_mode. Only the
    # per-lag history grid does, and only the HISTORY run reads it, so the full
    # grid is the one that serves both. A 1-lag grid (from a gripper_token
    # compute) would crash the history run on its first batch.
    bad.append(f"{rv.stats_dir} has a {s.history_mean.shape[0]}-lag history grid; the "
               f"history run injects {rv.n_lags} tokens and would raise in "
               "NormalizeHistory on the first batch")
else:
    print(f"  OK  staged rotvec stats: action {s.action_q01.shape} history "
          f"{s.history_mean.shape} gripper q01={s.gripper_q01:.4f} q99={s.gripper_q99:.4f}")

for exp, mode in ((exp_hist, "history"), (exp_tok, "gripper_token")):
    d = pathlib.Path(runs) / rv.name / exp
    if not d.exists():
        parent = pathlib.Path(runs) / rv.name
        siblings = sorted(p.name for p in parent.glob("*")) if parent.exists() else []
        bad.append(f"{d} does not exist (set "
                   f"EXP_RV_{'HIST' if mode == 'history' else 'TOK'}=...); "
                   f"what IS there: {siblings}")
        continue
    steps = sorted(int(p.name) for p in d.glob("[0-9]*") if p.name.isdigit())
    if step not in steps:
        bad.append(f"{d} has no step {step} to resume from (found {steps[-5:]})")
    else:
        print(f"  OK  {exp}: step {step} present (all: {steps})")

    stamp = d / "ego2g1_stamp.json"
    if stamp.exists():
        cfg = json.loads(stamp.read_text()).get("ego2g1_config", {})
        # Both keys are ABSENT on stamps written before their knob existed, and
        # the absence PINS the value: there was only the rotvec path before
        # rotation_repr, and only the history path before state_mode (which
        # arrived with gripper_token, in c4cecbb). `umi` is such a checkpoint.
        got_repr = cfg.get("rotation_repr", "rotvec")
        got_mode = cfg.get("state_mode", "history")
        if got_repr != "rotvec":
            bad.append(f"{stamp}: rotation_repr={got_repr!r}, expected 'rotvec'")
        if got_mode != mode:
            bad.append(f"{stamp}: state_mode={got_mode!r}, expected {mode!r}")

    # The run's own copy of the artifact it trained against. It records the
    # STAGING file, not the run's config -- train.main_umi copies, it does not
    # recompute -- so a gripper_token run legitimately carries the 6-lag grid.
    # Only the history run constrains it.
    own = d / "assets_ego2g1"
    if (own / _norm.UMI_FILENAME).exists():
        o = _norm.load_umi(own)
        n = o.history_mean.shape[0]
        if o.rotation_repr != "rotvec":
            bad.append(f"{own}: {o.rotation_repr!r} stats, expected 'rotvec'")
        elif mode == "history" and n != rv.n_lags:
            bad.append(f"{own}: {n}-lag grid but {exp} injects {rv.n_lags} tokens")
        else:
            note = "" if mode == "history" else " (unused in this mode)"
            print(f"  OK  {exp}: trained against a {n}-lag {o.rotation_repr} grid{note}")

if bad:
    print("\nPREFLIGHT FAILED:")
    for b in bad:
        print(f"  - {b}")
    sys.exit(1)
print("\npreflight OK")
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || die "preflight failed — see $LOGDIR/02_preflight.log"

# ------------------------------------------------------------ rot6d norm stats

# Deliberately NO --state-mode: the action grid and the gripper quantiles are
# identical in both modes, and only the history-mode (full lag) grid serves both.
# Skipped if already present, so a rerun does not rewrite an artifact a partially
# trained run is already resuming against.
R6_STATS="$ASSETS/umi_wrist/ego2g1/red_block_on_yellow_block_umi/rot6d/umi_stats.npz"
if [ -f "$R6_STATS" ]; then
    say "rot6d stats: already computed"
    echo "  $R6_STATS"
else
    say "rot6d stats: computing"
    # Read the printed block: model-space std should be ~0.35-0.47 at EVERY slot
    # and the three block shares should be 33/33/33, matching rotvec. If they are
    # not, the comparison is confounded before a single step is taken.
    python -m ego2g1.train.compute_norm_stats --umi "${ROT6D[@]}" \
        --video-backend pyav --assets-base-dir "$ASSETS" \
        2>&1 | tee "$LOGDIR/03_stats_rot6d.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "rot6d norm stats failed — see $LOGDIR/03_stats_rot6d.log"
fi

# ------------------------------------------------------------------- the runs

run() {
    local tag="$1"; shift
    local log="$LOGDIR/${tag}.log"
    say "$tag  ->  $log"
    local t0=$SECONDS
    python -m ego2g1.train.train --umi "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    local mins=$(( (SECONDS - t0) / 60 ))
    if [ "$rc" -eq 0 ]; then
        SUMMARY+=("  OK    $tag  (${mins}m)  $log")
    else
        SUMMARY+=("  FAIL  $tag  (${mins}m, rc=$rc)  $log")
        FAILED+=("$tag")
        printf '\033[1;31m%s FAILED (rc=%s) — continuing with the next run\033[0m\n' "$tag" "$rc"
    fi
}

# --- the experiment: rot6d, from scratch, both proprioception modes ----------
run "04_rot6d_history" \
    "${COMMON[@]}" "${ROT6D[@]}" "${HIST[@]}" \
    --exp-name "$EXP_R6_HIST" --num-train-steps "$FROM_SCRATCH_STEPS"

run "05_rot6d_gripper_token" \
    "${COMMON[@]}" "${ROT6D[@]}" "${TOK[@]}" \
    --exp-name "$EXP_R6_TOK" --num-train-steps "$FROM_SCRATCH_STEPS"

# --- the control: more compute on the rotvec runs already trained -----------
#
# WARM RESTART, not a continuation. The LR schedule's decay horizon is always
# num_train_steps, so asking for 20k re-derives the cosine as if the run had
# always been 20k long: LR climbs from the ~2.5e-6 these checkpoints ended at
# back to ~1.4e-5 at step 10k, then decays to 2.5e-6 at 20k. That is the intended
# second phase of learning, but it is NOT "the same run, longer" -- read the loss
# curve accordingly.
run "06_rotvec_history_resume20k" \
    "${COMMON[@]}" "${ROTVEC[@]}" "${HIST[@]}" \
    --exp-name "$EXP_RV_HIST" --num-train-steps "$RESUME_TO_STEPS"

run "07_rotvec_gripper_token_resume20k" \
    "${COMMON[@]}" "${ROTVEC[@]}" "${TOK[@]}" \
    --exp-name "$EXP_RV_TOK" --num-train-steps "$RESUME_TO_STEPS"

# -------------------------------------------------------------------- summary

say "summary"
printf '%s\n' "${SUMMARY[@]:-}"
echo
echo "logs: $LOGDIR"
if [ "${#FAILED[@]}" -gt 0 ]; then
    printf '\033[1;31m%d run(s) failed: %s\033[0m\n' "${#FAILED[@]}" "${FAILED[*]}"
    echo "Rerunning this script is safe: every run passes --resume, so the finished"
    echo "ones cost one step each and the failed ones continue from their last save."
    exit 1
fi
echo "all four runs finished"
