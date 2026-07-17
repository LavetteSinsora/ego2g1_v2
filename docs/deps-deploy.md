# deploy dependencies

What `uv sync --group deploy` should end up meaning. Recorded here (not merged
into pyproject.toml by the deploy port — integration owns that file).

## Path dep (vendored)

```
unitree_deploy @ third_party/unitree_deploy        # editable path dep
```

Install today, from the repo root:

```bash
VIRTUAL_ENV=$PWD/../.venv uv pip install -e third_party/unitree_deploy
```

Vendored from the old repo's `unitree-deploy-main/` (Unitree's deploy stack,
BSD-3-Clause, v0.0.5) minus `unitree_deploy.egg-info/` and `__pycache__/`. The
copy already carries the Python-3.12 fix (mutable `np.ndarray`/`list`
dataclass defaults → `field(default_factory=...)` in
`unitree_deploy/robot_devices/arm/configs.py`);
`tests/test_deploy_vendored.py` pins it. The 52 MB of
`robot_devices/assets/g1/` (URDF + meshes) is load-bearing: gravity
compensation (`g1_arm_ik.solve_tau` = `pin.rnea`) reads
`g1_body29_hand14.urdf`.

Its pyproject declares `lerobot==0.4.1`, `pyrealsense2`, `meshcat`,
`matplotlib`, `logging_mp`, `opencv-python`, `tyro`, `draccus`, `mujoco` —
heavier than what the deploy layer actually imports (arm + brainco + image
client + env need: numpy, scipy, torch, draccus, opencv, logging_mp, tyro).
Installing with `--no-deps` and adding the short list below is the leaner
path if `lerobot==0.4.1` fights the repo's own lerobot pin — it did not in
the current `.venv`, which installs it editable with deps unresolved.

## Not on PyPI

```
unitree_sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python
```

DDS transport (`ChannelFactoryInitialize`, LowCmd_/LowState_ IDL, CRC). Pulls
in `cyclonedds` — already in the repo `.venv`. Note the repo memory: this
box pushes/pulls GitHub via https only.

## Binary/conda-grade wheels

```
pin        # pinocchio — gravity-comp rnea in unitree_deploy/robot_devices/arm/g1_arm_ik.py
casadi     # the vendored IK variant imports it at module level
```

Both must go into the uv venv itself (`VIRTUAL_ENV=$PWD/.venv uv pip install
pin casadi`), NOT a conda pip — see memory `unitree-deploy-env-setup.md` for
the failure mode.

## Rest of the deploy group

```
torch              # unitree_deploy's send_action takes tensors (already in .venv)
opencv-python      # recorder mp4 sink, check camera rung
pandas + pyarrow   # replay_dataset reads LeRobot parquet directly
tyro               # every CLI entry point
openpi-client      # websocket PolicyClient (from the openpi fork's packages/openpi-client)
draccus, logging_mp  # unitree_deploy internals
```

`ego2g1.deploy` itself needs mujoco/mink only in `relative_eef` mode
(kinematics.py imports them lazily); a joint-mode robot PC can skip them.

## Machine parameters (env/CLI, documented defaults)

| what | default | where |
|---|---|---|
| DDS network interface | none (join default domain) | `--network-interface` (runner, replay_*, check) |
| robot subnet | 192.168.123.x, deploy box on-subnet | docs/deploy.md |
| head camera host | 192.168.123.164 | `--camera-host` (runner, check camera) |
| policy server | 127.0.0.1:8000 | `--host/--port` |
