#!/usr/bin/env bash
# One-shot sequential sweep for the rotation-representation A/B, meant to be
# launched inside ONE tmux window on the PPU box and left alone for a day.
#
#     tmux new -s sweep
#     bash /mnt/cpfs/hxy/ego2g1_v2/tools/umi_rotation_sweep.sh
#     # detach with C-b d; reattach with `tmux attach -t sweep`
#
# WHAT IT RUNS, in order:
#
#   preflight   tests + the four things that would waste a day if wrong
#   stats       norm stats for rot6d (rotvec's are already on disk, untouched)
#   run 1       rot6d + state history          from scratch, 10k steps
#   run 2       rot6d + gripper token          from scratch, 10k steps
#   run 3       rotvec + state history         RESUME 9999 -> 20k
#   run 4       rotvec + gripper token         RESUME 9999 -> 20k
#
# WHY THIS ORDER. The two new runs go first because they are the experiment;
# the resumes are the "does more compute help" control and are the ones you can
# most afford to lose to a crash. It also means an early abort leaves the
# already-trained rotvec checkpoints untouched at step 9999.
#
# FAILURE POLICY, deliberately split:
#   preflight and stats are FATAL — every run depends on them, so continuing
#   past a failure just burns GPU-hours producing garbage.
#   training runs are INDEPENDENT — one OOM must not cost you the other three.
#   Failures are recorded and reported in the summary, and the exit code is
#   nonzero if any run failed.
#
# The rotvec runs are RESUMED, so they reload assets this script must not have
# touched. That is why rot6d stats go to their own directory
# (UmiTrainConfig.stats_dir) and why preflight asserts the rotvec artifact is
# still rotvec before anything is written.

set -u -o pipefail

# ---------------------------------------------------------------- configuration
# Everything machine-specific lives here. Nothing below this block should need
# editing between runs.

WS="${WS:-/mnt/cpfs/hxy}"
REPO="${REPO:-$WS/ego2g1_v2}"
RUNS="${RUNS:-$WS/runs/ego2g1_v2}"
ASSETS="${ASSETS:-$RUNS/assets}"
PI05_PARAMS="${PI05_PARAMS:-$WS/cache/openpi/openpi-assets/checkpoints/pi05_base/params}"
LOGDIR="${LOGDIR:-$RUNS/logs/rotation_sweep_$(date +%Y%m%d_%H%M%S)}"

# Experiment names. The two ROTVEC ones must match the runs already on disk —
# preflight fails loudly (and lists what IS there) rather than starting a fresh
# run under a name that does not exist, which is what `--resume` would otherwise
# quietly do.
EXP_RV_HIST="${EXP_RV_HIST:-umi_hist}"
EXP_RV_TOK="${EXP_RV_TOK:-umi_nohist}"
EXP_R6_HIST="${EXP_R6_HIST:-umi_6DRot_with_history}"
EXP_R6_TOK="${EXP_R6_TOK:-umi_6DRot_no_history}"

FROM_SCRATCH_STEPS="${FROM_SCRATCH_STEPS:-10000}"   # matches the rotvec baselines
RESUME_TO_STEPS="${RESUME_TO_STEPS:-20000}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-9999}"        # what preflight expects to find

# Mandatory on this box: torchcodec cannot load its native library on PPU.
COMMON=(--video-backend pyav
        --weight-loader-params-path "$PI05_PARAMS"
        --assets-base-dir "$ASSETS"
        --checkpoint-base-dir "$RUNS")

# The rot6d config is TWO flags, always together: action_dim_actual is validated
# against rotation_repr rather than derived from it, so a config that changes one
# and forgets the other fails at construction instead of building a mismatched
# normalization grid.
ROT6D=(--rotation-repr rot6d --action-dim-actual 10)
ROTVEC=(--rotation-repr rotvec --action-dim-actual 7)

# "no state history" is state_mode=gripper_token: the gripper alone, binned into
# the prompt as pi05-style digits, and NO learned encoder in the param tree.
HIST=(--state-mode history)
TOK=(--state-mode gripper_token)

mkdir -p "$LOGDIR"
FAILED=()
declare -a SUMMARY

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------- preflight

say "preflight"
cd "$REPO" || die "no checkout at $REPO"

[ -n "${VIRTUAL_ENV:-}" ] || die "no venv active — run: source $REPO/envs/ppu-train.sh
(NEVER 'uv run'/'uv sync' on this box: they target ./.venv and resolve NVIDIA
wheels from uv.lock, which is a silent CPU fallback)"

python -c "import jax,sys; n=jax.device_count(); print(f'jax backend={jax.default_backend()} devices={n}');
sys.exit(0 if jax.default_backend()=='gpu' and n>0 else 1)" \
    || die "jax is not on the PPU backend — re-source envs/ppu-train.sh and read its verification block"

say "preflight: tests"
python -m pytest tests/train/test_umi.py tests/train/test_relation.py -q \
    2>&1 | tee "$LOGDIR/00_tests.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "tests failed — see $LOGDIR/00_tests.log"

# The four things that would waste a day if wrong, each checked as a fact rather
# than assumed: the rotvec stats exist and are still ROTVEC (this script writes
# rot6d stats and must not have clobbered them), and both resume targets exist
# at the step we expect to continue from.
say "preflight: resume targets and existing artifacts"
python - "$RUNS" "$ASSETS" "$EXP_RV_HIST" "$EXP_RV_TOK" "$RESUME_FROM_STEP" <<'PY' \
    2>&1 | tee "$LOGDIR/01_preflight.log"
import pathlib, sys
from ego2g1.train import config as _config, norm as _norm

runs, assets, exp_hist, exp_tok, step = sys.argv[1:6]
step = int(step)
bad = []

rv = _config.UmiTrainConfig(assets_base_dir=assets, checkpoint_base_dir=runs)
print(f"rotvec stats dir: {rv.stats_dir}")
try:
    s = _norm.load_umi(rv.stats_dir)
    if s.rotation_repr != "rotvec":
        bad.append(f"{rv.stats_dir} holds {s.rotation_repr!r} stats, not 'rotvec' — "
                   "the resumed runs would normalize against the wrong grid")
    else:
        print(f"  OK  rotation_repr={s.rotation_repr} action grid {s.action_q01.shape} "
              f"history {s.history_mean.shape}")
    # BOTH state modes share one stats file per representation — legitimately,
    # because the action grid and the gripper quantiles do not depend on
    # state_mode. Only the PER-LAG history grid does: it is (n_lags, D), and
    # n_lags is 6 in history mode but 1 in gripper_token mode. So an artifact
    # computed in gripper_token mode is (1, D) and serves gripper_token only —
    # a history run loading it dies in NormalizeHistory on the first batch,
    # twenty minutes into a job. The reverse is fine (NormalizeHistory is a
    # no-op when nothing is injected), which is why this script computes stats
    # in the DEFAULT history mode and lets both runs share them.
    if s.history_mean.shape[0] != rv.n_lags:
        bad.append(
            f"{rv.stats_dir} has a {s.history_mean.shape[0]}-lag history grid but the "
            f"history run needs {rv.n_lags}. It was computed in gripper_token mode; "
            "recompute it WITHOUT --state-mode (history is the default) — the "
            "resulting artifact serves both modes")
except FileNotFoundError as e:
    bad.append(f"rotvec stats missing: {e}")

for exp, mode in ((exp_hist, "history"), (exp_tok, "gripper_token")):
    d = pathlib.Path(runs) / rv.name / exp
    if not d.exists():
        siblings = sorted(p.name for p in (pathlib.Path(runs) / rv.name).glob("*")) \
            if (pathlib.Path(runs) / rv.name).exists() else []
        bad.append(f"{d} does not exist (set EXP_RV_{'HIST' if mode=='history' else 'TOK'}=...); "
                   f"what IS there: {siblings}")
        continue
    steps = sorted(int(p.name) for p in d.glob("[0-9]*") if p.name.isdigit())
    if step not in steps:
        bad.append(f"{d} has no step {step} to resume from (found {steps[-5:]})")
    else:
        print(f"  OK  {d} has step {step} (all: {steps})")
    stamp = d / "ego2g1_stamp.json"
    if stamp.exists():
        import json
        cfg = json.loads(stamp.read_text()).get("ego2g1_config", {})
        got = cfg.get("rotation_repr", "rotvec")   # absent on pre-knob stamps
        if got != "rotvec":
            bad.append(f"{stamp} says rotation_repr={got!r}, expected 'rotvec'")
        if cfg.get("state_mode") != mode:
            bad.append(f"{stamp} says state_mode={cfg.get('state_mode')!r}, expected {mode!r}")

if bad:
    print("\nPREFLIGHT FAILED:")
    for b in bad:
        print(f"  - {b}")
    sys.exit(1)
print("\npreflight OK")
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || die "preflight failed — see $LOGDIR/01_preflight.log"

# ----------------------------------------------------------------- norm stats

# rot6d stats land in <assets>/umi_wrist/<repo_id>/rot6d/ (UmiTrainConfig
# .stats_dir keys the path by representation), so this cannot touch the rotvec
# artifact the two resumes reload. Read the printed block: the per-slot
# model-space std should be ~0.35-0.47 at EVERY slot and the three block shares
# of the weighted MSE should be 33/33/33, matching rotvec.
#
# ONE file for BOTH rot6d runs, and deliberately NO --state-mode here. The
# action quantile grid and the gripper quantiles are identical in the two modes;
# only the per-lag history grid differs, and it must be the 6-lag (history-mode)
# version. That one serves gripper_token too, because with nothing injected
# `NormalizeHistory` never fires. Computing it in gripper_token mode would give
# a 1-lag grid that the history run cannot use.
say "norm stats: rot6d"
python -m ego2g1.train.compute_norm_stats --umi "${ROT6D[@]}" \
    --video-backend pyav --assets-base-dir "$ASSETS" \
    2>&1 | tee "$LOGDIR/02_stats_rot6d.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "rot6d norm stats failed — see $LOGDIR/02_stats_rot6d.log"

# ------------------------------------------------------------------ the runs

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
run "03_rot6d_history" \
    "${COMMON[@]}" "${ROT6D[@]}" "${HIST[@]}" \
    --exp-name "$EXP_R6_HIST" --num-train-steps "$FROM_SCRATCH_STEPS"

run "04_rot6d_gripper_token" \
    "${COMMON[@]}" "${ROT6D[@]}" "${TOK[@]}" \
    --exp-name "$EXP_R6_TOK" --num-train-steps "$FROM_SCRATCH_STEPS"

# --- the control: more compute on the rotvec runs already trained -----------
#
# WARM RESTART, not a continuation. The LR schedule's decay horizon is always
# num_train_steps, so asking for 20k re-derives the cosine as if the run had
# always been 20k long: LR climbs from the ~2.5e-6 these checkpoints ended at
# back to ~1.4e-5 at step 10k, then decays to 2.5e-6 at 20k. That is the
# intended "second phase of learning", but it is NOT "the same run, longer" —
# read the loss curve accordingly.
run "05_rotvec_history_resume20k" \
    "${COMMON[@]}" "${ROTVEC[@]}" "${HIST[@]}" \
    --exp-name "$EXP_RV_HIST" --num-train-steps "$RESUME_TO_STEPS" --resume

run "06_rotvec_gripper_token_resume20k" \
    "${COMMON[@]}" "${ROTVEC[@]}" "${TOK[@]}" \
    --exp-name "$EXP_RV_TOK" --num-train-steps "$RESUME_TO_STEPS" --resume

# -------------------------------------------------------------------- summary

say "summary"
# ${...[@]:-} rather than ${...[@]}: an empty array under `set -u` is an error
# on older bash, and this block must never be the thing that hides a result.
printf '%s\n' "${SUMMARY[@]:-}"
echo
echo "logs: $LOGDIR"
if [ "${#FAILED[@]}" -gt 0 ]; then
    printf '\033[1;31m%d run(s) failed: %s\033[0m\n' "${#FAILED[@]}" "${FAILED[*]}"
    exit 1
fi
echo "all four runs finished"
