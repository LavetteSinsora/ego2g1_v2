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
uv sync --group train      # + jax/openpi   (see envs/ for the PPU box)
uv sync --group deploy     # + DDS/unitree_deploy (robot PC)
```

Status: under construction — porting from the original tree. Per-area docs land
with each port.
