# ego2g1.train + ego2g1.serve — π₀.₅ fine-tune and policy serving

`src/openpi` stays bit-stock — that is the load-bearing design decision,
inherited from the fork and kept here. Every deviation lives in this repo:
the Pi0 subclass (`train/model.py`), per-slot normalization (`train/norm.py`),
the per-token adaRMS change as a runtime symbol rebind with a source
fingerprint guard (`train/gemma_patch.py` — an unpatched load fails loud via
`train/stamp.py` feature flags, never silently). openpi itself is the pinned
submodule at `third_party/openpi`; its `src/openpi` there is identical to
upstream at the fork point, so the pin is interchangeable with an upstream
commit.

## Train

```bash
# PPU box: borrowed venv + device pinning — NOT uv (docs/environments.md)
source envs/ppu-train.sh
python -m ego2g1.train.train --exp-name my_run

# norm stats first if the dataset changed (writes assets/<name>/<repo_id>/)
python -m ego2g1.train.compute_norm_stats
```

Config = one frozen dataclass (`train/config.py`, via tyro). Things that bite:

- **`expected_config_hash`**: copy it from the dataset's
  `extraction_meta.json`. Training asserts it — a dataset regenerated under
  different extraction knobs refuses to train silently-wrong.
- **norm-stats identity** = `assets/<config.name>/<config.repo_id>/`. New-repo
  runs use `ego2g1/put_bottle_in_box_ego`; OLD checkpoints carry stats under
  the unsuffixed name inside their own `assets/` and keep working. Changing
  `repo_id` orphans nothing that ships inside a checkpoint, but re-computes
  stats for new runs.
- Two stats artifacts ship per checkpoint: pooled-quantile `norm_stats.json`
  (openpi-native) + `per_slot_stats.npz` (the E001 per-slot rescale grid,
  floor c=0.1). Deployment MUST apply PerSlotRescaleInverse before pooled
  Unnormalize — guarded by the stamp flags.
- The control-mode marker (`<<<control_mode>>> end effector <<<control_mode>>>`)
  is appended by a model transform, never baked into the dataset.

E-series deviations (full history in the old repo's OPENPI_EDITS.md): E001
per-slot floored normalization — implemented; E002 per-token adaRMS — the
gemma_patch, bitwise-stock when unused (test-pinned); E003 multi-timestep
flow — designed, gated on profiling.

## Serve

```bash
source envs/ppu-serve.sh          # pins ONE PPU; allocator flags are booby-trapped
python -m ego2g1.serve --checkpoint checkpoints/ego2g1_pi05/<exp>/best/<step>
```

The serve stack rebuilds the exact training transform stack from the
checkpoint stamp and prefers checkpoint-carried norm stats. RTC sampling is
chosen by the checkpoint's `rtc_training` flag, not a CLI flag.

**Latency reality check** (docs/jitter_root_cause.md): the 2026-07-16 rollout
measured 1.3–5.1 s per chunk against a 0.4 s budget — the robot froze 8.7 s
mid-motion and lurched. Before trusting any timing strategy, measure: warm the
server, run a few inferences, and compare against `d`·dt. Deploy-side refuses
budgets the measured latency can't honor; serve-side profiling of the PPU
path (image decode / transforms / forward) is the open item.

## Tests

```bash
uv run python -m pytest tests/train -q     # 139 pass; includes the bitwise-stock guard
```
