# deploy

Run an ego2g1 checkpoint — or a recording — on the real G1-D, smoothly, on
purpose. Everything here is downstream of one measurement
(`docs/jitter_root_cause.md`): **the executor was never the problem.** On a
constant command the arm holds to <1 mrad; the judder came from (1) noisy EEF
targets that a fully-converged IK faithfully turned into 26+ rad/s² joint
zig-zag, and (2) serve latencies of 1.3–5.1 s against a 0.4 s timing budget,
which turned "async replanning" into freeze-then-lurch. So the deploy layer
does three things and refuses to do more:

1. **Vendors the proven executor.** `third_party/unitree_deploy` is Unitree's
   own stack — a 500 Hz interpolating arm thread with a `max_pos_speed` slew
   cap, soft `drive_to_waypoint` first ramps, and gravity-comp tau feedforward
   (`pin.rnea` over the vendored URDF). ZH drives this arm smoothly through
   exactly this code. We wrap it (`ego2g1/deploy/executor.py`); we do not
   rewrite it.
2. **Fixes the target path, where the jitter is actually made.** EEF chunks
   go through OneEuroSE3 → mink IK with the posture task retargeted to the
   previous solution each tick at cost 0.05 → a 4-tap joint filter. That
   posture trick is xr_teleoperate's `0.1·‖q − q_last‖²` smoothness cost
   transplanted; measured on the extraction re-solve it cut worst-joint accel
   RMS from ~25 to 4–7 rad/s² at ≤1.7 cm max EEF cost. The synthetic deploy
   test (`tests/test_deploy_conversion.py`) reproduces it: raw converged IK
   83 rad/s², the deploy pipeline 2.0 — 42× — with 12 mm mean / 22 mm max
   position error against the noise-free targets.
3. **Measures latency before the robot moves, and refuses.** The runner times
   real warmup inferences (real frame size — the wire cost is part of the
   number) and will not start a timing strategy whose budget the measured p95
   cannot honor. A refusal at the terminal beats a lurch on the arm.

## The action-mode boundary

The policy⇄execution contract is **timestamped joint chunks**. A chunk enters
in one of two modes and everything downstream (strategies, clamp, executor)
is mode-blind:

| mode | chunk | what happens |
|---|---|---|
| `joint` | (H, 14) or (H, 26) absolute joints | validated, hand-padded, **no IK, no mujoco import** — ZH-style, and the slot a future joint-space ego2g1 policy drops into |
| `relative_eef` | (H, 30) anchor-relative EEF deltas + hand cmds | anchor = FK(measured q at the obs tick); compose → OneEuroSE3 → DualArmIK (posture-tracks-last @ 0.05, seed carried row to row) → JointFilter |

The mode is read from the checkpoint's own `control_mode` in the serve
handshake (`--action-mode auto`); override only to test a mismatch on purpose.
Conversion happens **whole-chunk at inference time** (ZH's adapter pattern),
because the async strategies blend old/new chunks at the seam and that blend
is only well-defined in joint space — averaging two vec9 rotations is not a
rotation.

Executor row layout (verified against unitree_deploy's `g1_motors` +
`brainco_motors`, pinned by `tests/test_deploy_vendored.py`): `[0:14]` arm
L7+R7 in DualArmIK order, `[14:20]` left Brainco
[thumb, thumbAux, index, middle, ring, pinky] in [0,1], `[20:26]` right.

## Install

```bash
# from ego2g1_v2/ — the repo .venv already carries mujoco/mink/torch/etc.
VIRTUAL_ENV=$PWD/../.venv uv pip install -e third_party/unitree_deploy
# plus, once per machine (see docs/deps-deploy.md for the full story):
#   unitree_sdk2py (git install), pin, casadi, openpi-client
```

Machine specifics are CLI/env parameters with the current lab values as
defaults: the deploy box sits on the robot subnet (192.168.123.x — it IS the
robot PC; no bridge), head camera `--camera-host 192.168.123.164`, DDS
`--network-interface` unset (join the running domain rather than fight for
it), policy server `--host/--port` wherever `python -m ego2g1.serve` runs.

## The rung ladder

Walk it in order; each rung gates the next. 2, 3 and 8 need no robot.

```bash
python -m ego2g1.deploy.check listen        # 1. DDS subscribe only. Proves the domain,
                                            #    topics, and the Brainco bridge.  [robot]
python -m ego2g1.deploy.check fk  --dataset data/lerobot_datasets/ego2g1/put_bottle_in_box
                                            # 2. FK(stored joints) == stored state.
                                            #    Joint order, waist==0, flange, pelvis,
                                            #    vec9 — in one shot.           [offline]
python -m ego2g1.deploy.check ik  --dataset ...
                                            # 3. deploy IK tracks the stored poses;
                                            #    also times one solve vs the 33 ms
                                            #    budget.                       [offline]
python -m ego2g1.deploy.check camera        # 4. one frame to disk. LOOK AT IT next to
                                            #    a training frame — a wrong viewpoint
                                            #    fails silently.                 [robot]
python -m ego2g1.deploy.check hand-sweep --hand right --motor 2
                                            # 5. one Brainco motor at a time. WATCH
                                            #    which finger moves — the order is
                                            #    plausible but hardware-unverified. [robot]
python -m ego2g1.deploy.replay_dataset --dataset ...
                                            # 6. stored JOINTS through the executor.
                                            #    No IK. Proves the plumbing.     [robot]
python -m ego2g1.deploy.replay_dataset --dataset ... --from-eef
                                            # 6b. re-solve IK offline from the stored
                                            #    poses through the deploy solver — the
                                            #    eef->joint path in isolation.   [robot]
python -m ego2g1.deploy.check replay-actions --dataset ...
                                            # 7. ACTION-shaped chunks through the REAL
                                            #    conversion (measured anchor, OneEuro,
                                            #    IK, filter, clamp). Proves the
                                            #    transforms. Run 6 first so this is
                                            #    interpretable.                  [robot]
python -m ego2g1.deploy.check latency --host <serve-box>
                                            # 8. round trip + per-mode budget verdicts.
                                            #    Run on the server box AND here; the
                                            #    difference is the network.  [no robot]
```

Rung 6 is also **the hardware A/B** for the extraction fix: the old dataset's
stored joints carry ~26 rad/s² (IK'd from unsmoothed Pico targets) and judder
through ANY executor including this one; the re-extracted dataset (IK
re-solved after label smoothing) should replay quiet. Same robot, same
executor — the only variable is the data. `replay_dataset` prints the source
accel RMS before it moves anything so you know what to expect.

## Running a policy

```bash
python -m ego2g1.deploy.runner --host <serve-box> --port 8000 \
    --prompt "put the bottle in the box" --mode sync
```

`--mode` is one of five consumers (ported from zh_deploy_inference's
strategies, all kept):

| mode | consumption | hard latency budget (refused if p95 exceeds) |
|---|---|---|
| `sync` | one chunk, re-infer when drained; the arm HOLDS during inference | none — slow is a pause, never a splice |
| `async` | newest chunk wins, skipping the rows that elapsed | chunk duration H/fps |
| `temporal_ensembling` | every live chunk votes, exp-weighted | H/fps |
| `temporal_smoothing` | blend the unexecuted overlap linearly | max_latency_steps/fps (default 8/30 = 267 ms) |
| `rtc` | temporal_smoothing + RTC prefix (prev chunk re-anchored via FK, `d`, `n_prefix`) to the server | max_latency_steps/fps |

**Get smoothness in `sync` first.** Async/RTC buy reactivity only once the
serve latency actually fits; that is what rung 8 and the startup self-check
are for. The measured PPU-box latencies (1.3–5.1 s) fit NO timing mode — the
runner will print the refusal and exit, which is the correct behavior until
serving is profiled and fixed.

Loop semantics worth knowing (all ported, all deliberate):

- ticks are paced with `precise_wait` (sleep, then spin the last ~1 ms);
- every waypoint is stamped **one control period past the end of the current
  cycle** (`t_cycle_end + dt`, exactly unitree_deploy's `UnitreeEnv.step`),
  so the 500 Hz interpolator never extrapolates;
- the first commanded motion is the vendor's own `drive_to_waypoint` soft
  ramp — expect the arm to move to its init pose at `connect()`;
- the observation's hand block is the **last command**, never encoders
  (training never saw an encoder; encoders stall against a grasped object).

## Safety

Three independent gates, all tripping to `damp()` — never to "stop sending"
(the firmware executes the last message forever; silence is a HOLD, not a
stop). G1-D reality: no lower body, fixed/suspended base, direct `rt/lowcmd`
(never `rt/arm_sdk`), legs/waist held at measured by the vendored controller.

1. **Clamp** between strategy and executor: max 0.15 rad per 30 Hz tick.
   Turns a bad splice or a mis-normalized chunk into lag instead of a lurch.
2. **Watchdog**: state staleness (>0.2 s), camera staleness (>0.5 s), plan
   starvation (>2 s, duration-based so startup doesn't false-trip), IK
   tracking error (>10 cm — "the frames are wrong").
3. **damp() e-stop** (`executor.damp()`): stops the vendor's publish thread,
   then publishes kp=0/kd=2 on all 35 motors, five times. Latched — send()
   is dead afterwards. ctrl-C in `replay_*` damps; the vendor's normal
   `close()` (drive to init pose) is only used on clean exits.

## Recording

Every runner session writes `recordings/<task>_<stamp>/events.jsonl` +
`head.mp4` + `meta.json` (see `deploy/recorder.py` for the event kinds). This
is not optional: the freeze-lurch was diagnosed from exactly this stream in
the old repo. `--no-record` exists for dry runs only.

## When it still judders

```bash
python -m ego2g1.deploy.sniff_lowcmd --seconds 5     # our stack IDLE: any traffic
                                                     # = a foreign publisher owns the
                                                     # wire. Fix THAT first.
python -m ego2g1.deploy.measure_rate --seconds 10    # true rt/lowstate rate + stalls;
                                                     # a starved link starves lowcmd too
python -m ego2g1.deploy.replay_diag --dataset ...    # instrumented rung 6: loop-timing
                                                     # vs servo-tracking verdict + npz
```

`replay_diag` prints the source data's accel RMS up front — if that is ~26
rad/s², the judder is in the data and no deploy knob will remove it.

## Observability

The jitter was only diagnosable because the old loop could be watched live and
replayed offline (`docs/jitter_root_cause.md`); both tools are ported. Both
are pure add-ons: the loop's threads gain no code, telemetry is PULLED —
`telemetry()` reads existing state under existing locks, nothing pushes.

**Live dashboard** — `--dashboard` on the runner serves a one-page monitor
(default `:8080`, own daemon thread, ~10 Hz HTTP pulls): active chunk +
pointer, the commanded 26-dim row, inference light, DelayBudget stats,
clamp/watchdog counters, camera frame. GET is pure telemetry; the only live
control is the E-STOP button (POST → `watchdog.trip` → `damp()`). Off by
default.

```bash
python -m ego2g1.deploy.runner --host <serve-box> --prompt "..." --dashboard
python -m ego2g1.deploy.dashboard --demo     # page dev: synthetic data, no hardware
```

**Session reconstruction** — any recording (`recordings/<task>_<stamp>/`,
written by every non-`--no-record` run) rebuilds at any monotonic time t,
offline, no robot/JAX/mujoco:

```bash
python -m ego2g1.deploy.replay_record recordings/<session>           # timeline
python -m ego2g1.deploy.replay_record recordings/<session> --at 12.3 # state at t
```

```python
from ego2g1.deploy.replay_record import Session
s = Session("recordings/...")
s.at(t)         # chunk state + pointer, commanded row, phase, frame ids
s.chunk_at(t)   # the (H, 26) chunk itself + index
```

Async modes are reconstructed by re-feeding the REAL buffer classes the
recorded install/pop sequence, so the splice math cannot drift from live. To
make that exact, `infer_result` events carry the converted joint chunk
(`actions`) and `meta.json` the strategy params — the only schema additions.
