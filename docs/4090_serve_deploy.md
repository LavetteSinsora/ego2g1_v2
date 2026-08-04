# The 4090 box: serve + deploy, one machine, `relation_eef`

One RTX 4090 running BOTH `ego2g1.serve` (policy) and `ego2g1.deploy`
(executor + live perception) — a role none of the other three profiles
(`docs/environments.md`) cover: unlike the PPU box, this is a normal CUDA
target `uv sync` handles directly; unlike the robot PC, it isn't a
donor-venv/stacked install. `envs/4090.sh` is the profile for it.

```bash
source envs/4090.sh
```

sources the normal uv-managed `.venv`, sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`
(serve's JAX and the perception detector's torch share this one card — see the
profile's header for why that's the opposite of `envs/ppu-serve.sh`'s rule),
auto-detects the DDS NIC + robot/camera host, and — if the files already exist
— exports `EGO2G1_TASK_CONFIG` / `EGO2G1_STEREO_CALIB` / `EGO2G1_CAMERA_EXTRINSIC`
so `runner.py` doesn't need `--task-config`/`--stereo-calib`/`--camera-extrinsic`
repeated on every invocation (an explicit flag still overrides).

Prerequisites before any of this runs — see `docs/deploy.md` for the general
deploy story and the prior conversation's `relation_eef` checklist for the
full detail:

- checkpoint rsync'd from the PPU under `checkpoints/ego2g1_pi05/<exp>/best/`
  (trained with `EgoRelationTrainConfig` — check `ego2g1_stamp.json`'s
  `config_class`);
- `uv sync --group train --group perception` (perception includes `deploy`);
- `ego2g1/deploy/perception/task_config.yaml` authored from
  `task_config.example.yaml`, objects matching the checkpoint positionally;
- `stereo_calib.npz` / `camera_calib.npz` at the repo root current for
  today's camera mount;
- `BRAINCO_CLOSED_POSE` in `ego2g1/deploy/gripper_calib.py` measured (still
  `np.ones(6)` placeholder otherwise).

## Serve

One terminal/tmux pane:

```bash
source envs/4090.sh
python -m ego2g1.serve --checkpoint checkpoints/ego2g1_pi05/<exp>/best/<step>
```

## Deploy — pre-flight (run once per session; skip 2–4 once done and unchanged)

```bash
source envs/4090.sh   # same shell or a second pane — picks up EGO2G1_IFACE etc.

# 1. DDS sanity
python -m ego2g1.deploy.check listen --iface "$EGO2G1_IFACE"

# 2. only if the camera mount moved since last calibration
python -m ego2g1.deploy.check stereo-capture
python -m ego2g1.deploy.check handeye-capture
python -m ego2g1.deploy.perception.handeye_calib <samples.npz>   # _cli_solve -> camera_calib.npz

# 3. only if BRAINCO_CLOSED_POSE is still the np.ones(6) placeholder
python -m ego2g1.deploy.check hand-jog --hand left
python -m ego2g1.deploy.check hand-jog --hand right
# -> hand-edit ego2g1/deploy/gripper_calib.py with the printed vectors

# 4. detection sanity, NO actuation at all — eyeball the box/mask overlay
python -m ego2g1.deploy.perception_preview --camera-host "$EGO2G1_CAMERA_HOST" \
    --eye left --prompts "obj0:...,obj1:...,obj2:..."   # match your task-config prompts

# 5. round-trip latency vs. the mode's budget
python -m ego2g1.deploy.check latency --host 127.0.0.1

# 6. full relation_eef code path, mock executor, no robot
python -m ego2g1.deploy.runner --host 127.0.0.1 --action-mode relation_eef \
    --prompt "..." --dry-run
```

## Deploy — the live run

```bash
python -m ego2g1.deploy.runner --host 127.0.0.1 --port 8000 \
    --action-mode relation_eef --mode sync --dashboard \
    --network-interface "$EGO2G1_IFACE" --camera-host "$EGO2G1_CAMERA_HOST" \
    --prompt "..."
```

`--task-config`/`--stereo-calib`/`--camera-extrinsic` come from `envs/4090.sh`'s
exports (see above) as long as `ego2g1/deploy/perception/task_config.yaml`
exists — pass them explicitly to override. `--mode rtc` raises
`NotImplementedError` for `relation_eef` — stick to `sync` (or another
non-RTC strategy). Dashboard's reset-to-episode is also not implemented for
this mode.

## Testing scripts (offline, no hardware, no serve process)

```bash
uv run python -m pytest tests/ -q                     # whole suite, 203+ pass on a working setup
uv run python -m pytest tests/deploy tests/deploy/perception tests/core/test_relation_layout.py -q
```

The second line is the `relation_eef`-relevant subset specifically:

| file | what it pins |
|---|---|
| `tests/core/test_relation_layout.py` | 56-dim state layout / grip-dim slicing matches config |
| `tests/deploy/test_relation_conversion.py` | the real-episode EEF→joint conversion tracking |
| `tests/deploy/test_runner_relation_eef.py` | the runner's `_observe_relation` / `_build_relation_adapter` wiring |
| `tests/deploy/test_task_config.py` | `load_task_config` / `validate_against_server_metadata` |
| `tests/deploy/perception/test_{detector,depth,tracker,orientation,latch,relation_perception}.py` | each perception-cascade stage in isolation, all with fakes/no GPU |
| `tests/deploy/perception/test_{stereo_calib,handeye_calib,touch_calib}.py` | the calibration math itself |

On a machine where `openpi` isn't importable (`train` group not synced),
tests importing `ego2g1.train.config` fail/error for that reason alone —
unrelated to anything `relation_eef`-specific. `uv sync --group train
--group deploy --group perception` on the 4090 resolves it.
