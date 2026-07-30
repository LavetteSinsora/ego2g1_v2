# ego2g1.data — Pico recordings → LeRobot dataset

The staged pipeline from the original `data_extraction/`, ported intact (same
content-addressed cache: every stage's config fields hash through its
dependency closure, so changing a knob re-runs exactly the stages it can
affect). One deliberate semantic change: **proprioception is now solved from
the smoothed targets** (s004c below) — the fix for the measured fact that the
old dataset's `arm_qpos` carried ~26 rad/s² of hand-tracking noise straight
onto the robot (docs/jitter_root_cause.md).

## Stages

| stage | scope | does |
|---|---|---|
| s001 | per-ep | resample Pico streams onto the uniform 30 Hz grid; tracker-spike filter |
| s003_placement | per-ep | rigid Pico→robot placement `S` (reach + ready-pose + torso clearance) |
| b_calib | global | fixed wrist→flange rotation `B` (dataset-mean chordal) |
| s003_state | per-ep | IK on RAW targets → the s004 filter signals (ik err, self-clearance). Its arm_qpos is diagnostic-only now |
| s002_01 / s002_02 | per-ep | EEF pose labels `G(t)=pelvis⁻¹·S·T_wrist·B` / Revo2 hand commands |
| s004 | per-ep | per-tick quality filters → strict split into sub-episodes |
| s004b_smooth | per-ep | SavGol on action labels within each good run |
| **s004c_resolve** | per-ep | **re-solve IK on the smoothed targets** → final `state_eef_*`/`arm_qpos`. Seeded per span from s003's proven branch; posture task tracks the previous solution (`resolve_smooth_cost=0.05`, xr_teleoperate's `‖q−q_last‖²` transplanted to mink) |
| s005 | global | write the LeRobot dataset + `extraction_meta.json` sidecar |

Measured effect of s004c (worst-joint accel RMS, rad/s²):

| episode | raw s003 | resolved |
|---|---|---|
| episode_1 | 26.5 | 3.8 |
| episode_2 (elbow at its limit) | 24.6 | 7.4 |
| episode_3 | 25.6 | 7.3 |

EEF cost ≤ 0.21 cm mean vs the smoothed targets; per-tick `resolve_pos_cm_*`
is stored, so a filter can gate residual workspace-edge events later. Set
`resolve_proprio=false` to reproduce the old (jittery) behavior for hardware
A/B replay.

## Where the raw episodes live

Raw Pico recordings live OUTSIDE this repo (they are multi-GB and per-machine).
The pipeline reaches them through `PipelineConfig.episodes_dir`
(`ego2g1/data/config.py`; default `data/<name>/`): either copy/move the episode
hdf5 dirs under `data/` (git-ignored) or point `episodes_dir` at their real
location. **Never symlink them in** — this repo must clone standalone, and
`tests/test_no_symlinks.py` enforces that.

## Run

```bash
# everything, all episodes (raw hdf5 in data/put_bottle_in_box_ego/)
uv run python -m ego2g1.data.run_pipeline --jobs 4

# knobs / partial runs
uv run python -m ego2g1.data.run_pipeline --through s004c_resolve --limit 3
uv run python -m ego2g1.data.run_pipeline --stages s004c_resolve --force
uv run python -m ego2g1.data.dashboard episode_1 -o report.html

# a Unitree teleop recording -> same dataset schema (no human retarget stages)
uv run python -m ego2g1.data.teleop_import --source data/put_bottle_in_box_teleop \
    --repo-id ego2g1/put_bottle_in_box_teleop
```

Interfaces and array schemas: `ego2g1/data/SPEC.md` (ported; module paths in
it still reference the old package name in places — the arrays are the
contract, and the loader equivalence tests pin them).

## Sharp edges

- `datasets` MUST stay 3.6.0 and lerobot at the openpi pin (both in
  pyproject) — see docs/environments.md for the double failure mode.
- The config hash changed vs the old repo (new stage + path fields):
  old-checkpoint norm stats do not match datasets extracted here; retrain.
- The cache lives in `ego2g1/data/work/` (git-ignored). `--force` re-runs a
  stage in place; deleting an episode's dir re-runs everything for it.
