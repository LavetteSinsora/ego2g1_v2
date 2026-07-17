# ego2g1 v2

Egocentric human recordings (Pico headset) → π₀.₅ fine-tune → Unitree G1-D +
BrainCo Revo2, as **one self-contained repo**: clone it on a fresh machine, `uv
sync`, and every role works. Successor to the original ego2g1 tree, which grew
five loose copies of openpi and zero dependency manifests.

The refactor also bakes in the deploy jitter root cause
(docs/jitter_root_cause.md): the judder was never the servo or the executor — it
was hand-tracking noise converted faithfully by IK into joint zig-zag, plus
serve latencies 10× the timing budget. Smoothing is therefore an explicit,
measured stage of target generation here, not an afterthought.

## Layout

```
ego2g1/            the one importable package; layers only import downward
  core/            math + physical constants (joint order, frames, chunk math); numpy-only
  kin/             G1+Revo2 model: FK, IK, self-collision (mujoco + mink)
  data/            Pico recordings -> LeRobot dataset pipeline + dashboard
  train/           pi0.5 fine-tune; imports openpi as a library, src/openpi stays stock
  serve/           websocket policy server (+ latency self-check)
  deploy/          joint-chunk execution on the proven unitree_deploy interpolator;
                   action modes: relative-EEF (IK here) and joint-space (no IK)
assets/            G1 + Revo2 MJCF/URDF — the single copy
data/              raw recordings + generated datasets (git-ignored; docs/data.md)
tools/             teleop (WebXR bare-hand), lift column, diagnostics
third_party/       openpi (submodule), unitree_deploy (vendored)
envs/              per-machine profiles: mac dev, PPU train/serve, robot PC
docs/              runbooks + the know-how that used to live in heads
```

The policy side ends at **timestamped joint chunks**; the execution side starts
there. Everything above that boundary runs and tests offline.

## Bootstrap

```bash
git clone --recursive <url> && cd ego2g1_v2
uv sync                    # core + kin + data
uv sync --group train      # + openpi (submodule path dep; PPU box uses envs/ instead)
uv sync --group deploy     # + unitree_deploy/DDS (robot PC)
uv run python -m pytest tests/ -q         # 203 pass on a working setup

# per-machine env facts (proxy, PPU allocator traps, robot subnet):
source envs/<machine>.sh   # mac-dev | ppu-train | ppu-serve | robot
```

## The whole lifecycle, one screen

Raw recordings → dataset → checkpoint → serving → robot. Full detail per
area in docs/; machine profiles in envs/ (see docs/environments.md).

```bash
# 1. DATA (mac-dev) — Pico hdf5 in data/put_bottle_in_box_ego/ -> LeRobot dataset
uv run python -m ego2g1.data.run_pipeline --jobs 4          # docs/data.md
uv run python -m ego2g1.data.dashboard episode_1 -o report.html   # verify by eye
uv run python -m ego2g1.data.teleop_import --source data/<teleop-rec> --repo-id ego2g1/<name>_teleop

# 2. TRAIN (PPU box — borrowed venv, NOT uv)                # docs/training.md
source envs/ppu-train.sh
python -m ego2g1.train.compute_norm_stats                   # after any dataset change
python -m ego2g1.train.train --exp-name my_run

# 3. SERVE (PPU box, one pinned NPU)
source envs/ppu-serve.sh
python -m ego2g1.serve --checkpoint checkpoints/ego2g1_pi05/<exp>/best/<step>

# 4. DEPLOY (robot PC / this mac on the 192.168.123.x subnet)   # docs/deploy.md
uv sync --group deploy                                      # once; needs CYCLONEDDS_HOME
uv run python -m ego2g1.deploy.check listen                 # then walk the rung ladder
uv run python -m ego2g1.deploy.replay_dataset --dataset data/lerobot_datasets/ego2g1/put_bottle_in_box_ego
uv run python -m ego2g1.deploy.check latency --host <serve-box>
uv run python -m ego2g1.deploy.runner --host <serve-box> --port 8000 --mode sync

# 5. TELEOP (optional; bare-hand WebXR)                     # tools/teleop/README.md
uv sync --group teleop
uv run python -m tools.teleop --sim                         # mjpython on macOS
```

## Where things stand

- Extraction, training, serving, deploy, teleop all ported; 203 tests pass.
- The re-extracted dataset (`data/lerobot_datasets/ego2g1/put_bottle_in_box_ego`,
  extraction hash `7b7f8bb7…`) carries smooth proprioception: joint accel RMS
  median 29.9 → 6.1 rad/s² vs the old pipeline. Training expects this hash.
- Deploy is built on unitree_deploy's proven 500 Hz interpolating executor with
  two first-class action modes (joint / relative-EEF); its EEF→joint conversion
  measured 42.5× smoother than the old path on synthetic Pico-grade noise.
- Hardware-unverified: damp() e-stop internals, vendor connect() init ramp,
  Brainco motor order, and the serve-latency fix (docs/deploy.md "open risks").
  The dataset-replay A/B (old jittery dataset vs this one) is the first thing
  to run on the robot.
