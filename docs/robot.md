# The robot: G1-D specifics, network, bring-up

What is specific to *our* robot and site — the facts that are true whether you are
deploying a policy, teleoperating, or just moving the lift column. The deploy loop's
own runbook (dashboard, recording, replay) is [deploy.md](deploy.md).

> **SAFETY** — the G1-D lowcmd path has **no balance controller**: every joint, legs
> included, is held by our position PD. Robot on a stand or suspended, remote in
> hand. Ctrl-C damps; the remote is the thing that always works.

## No lower body, no `arm_sdk`

The robot is a **G1-D**: fixed/suspended base, the lower body is irrelevant. Do not
design around balance, `ai_sport`, or the `rt/arm_sdk` blend-weight path (the
architecture in Unitree's `G1_DEPLOYMENT.md` §1 is stale for this robot) — we command
**`rt/lowcmd` directly**. The pipeline's kinematics assume exactly this: the MJCF is a
fixed base, and pelvis→flange depends only on the 14 arm joints — no IMU, no world
frame, no legs.

The firmware runs `tau = kp·(q*−q) + kd·(dq*−dq) + tau_ff` on the **last** command it
received. Two consequences:

- **"stop publishing" is NOT a stop** — the firmware holds the last setpoint forever.
  The only real stop is a damping command: `damp()` sets kp=0, kd=2 on every joint and
  **latches** (after it, `send_arm`/`send_hands` are no-ops). That is the e-stop.
- every joint needs an explicit policy. Per `ego2g1.deploy.dds`:

| joints (G1_29 indices) | command | gains (kp/kd) |
|---|---|---|
| arms 15–28 (2×7) | the control loop's targets | shoulder/elbow 80/3, wrist 40/1.5 |
| waist 12–14 | **pinned at 0 rad** — training froze the waist there, and waist=0 is what makes pelvis→flange arm-only | 300/3 |
| legs 0–11 | **held at whatever they measured at connect time** — harmless on a fixed base | 300/3 |
| 29–34 | unused | — |

## Lift column (升降)

Driven over DDS, not serial/GPIO; the lift firmware is on the control board, so it
responds whenever the robot is powered — teleop need not be running.

| topic | type | meaning |
|---|---|---|
| `rt/cmd_hispeed` | `geometry_msgs/Point32` | `.z` = vertical **velocity** in [−1,1] (+up, −down, 0 hold), publish continuously at ~30 Hz |
| `rt/hispeed_state` | `geometry_msgs/Point32` | `.y` = height (m) |

Operator tool: `./tools/lift/move_g1d.sh` (arrow keys, auto-stop on release, hard
±0.30 velocity clamp, z=0 guaranteed on exit).

## Network map

The deploy machine sits **on the robot's subnet** — it *is* the "robot PC" in the rung
ladder; only policy serving lives elsewhere (the PPU box). `source envs/robot.sh` sets
all of this up (iface auto-detect included).

| what | where |
|---|---|
| robot subnet | `192.168.123.x` |
| robot board | `192.168.123.164` — ssh `unitree@192.168.123.164` (password `123`) |
| head camera | `image_server` on the robot board (ZMQ), host `192.168.123.164` |
| robot-side python with `unitree_sdk2py` | `/home/unitree/miniconda3/envs/tv/bin/python` |
| DDS | direct subscribe/publish: `--iface` = the NIC holding your 192.168.123.x address, `--domain 0` |
| policy server | NOT here — PPU box via ssh tunnel (below) |

`check listen` should see `rt/lowstate` immediately; if not, the iface is wrong.

## Bring-up

Walk `python -m ego2g1.deploy.check --help` before every session — the rungs catch
most deployment bugs with the model out of the loop.

```bash
source envs/robot.sh                                   # venv + iface/domain/camera

# rung — DDS alive? subscribe-only, nothing commanded
python -m ego2g1.deploy.check listen --iface $EGO2G1_IFACE

# rung — camera alive? (must show the RAW head frame, not the wire-resized one)
python -m ego2g1.deploy.check camera --host $EGO2G1_CAMERA_HOST

# rung — Brainco hand motor order (flagged UNVERIFIED in dds.py; a permuted order
# means the thumb closes when you curl your ring finger)
python -m ego2g1.deploy.check hand-sweep --iface $EGO2G1_IFACE
```

### Serve on the PPU box, deploy here

Nothing assumes co-location: `ego2g1.serve` binds `0.0.0.0:8000`; deploy takes
`--host/--port` for the policy server, separate from the DDS/camera args. Tunnel
rather than exposing the port — the deploy entrypoint never plumbs an api_key
through, so **the tunnel is the auth**:

```bash
# on the PPU box
EGO2G1_PPU=15 source envs/ppu-serve.sh                 # pin ONE card (docs/environments.md)
python -m ego2g1.serve --checkpoint checkpoints/<exp>/<step>

# here
ssh -N -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -L 8000:localhost:8000 user@ppu
python -m ego2g1.deploy --host 127.0.0.1 --port 8000 --task "..." --blocking
```

**Warm the server before connecting the robot**: the first infer triggers an XLA
compile that takes minutes; the loop's starvation watchdog arms only after the first
chunk lands, precisely so that compile can't damp a robot that hasn't moved. If the
tunnel dies mid-episode, the websocket call just blocks, the trajectory drains, and
the watchdog damps at 1.0 s — correct, but with a WAN in the loop it's a live risk.

### The wire budget is the delay budget

`DelayBudget.max_d` = 20 ticks — **~667 ms at 30 Hz, a hard ceiling** on round trip
(upload + inference + download) at p95. Past it, chunks splice without RTC's
continuity guarantee and the seams rest on the joint clamp alone. The loop logs
`budget={'d':…, 'p95_ms':…, 'saturated':…}` every 2 s — that is the go/no-go readout;
run `--blocking` first and watch it before enabling async/RTC. (Measured on the old
stack: PPU serve latencies of 1.3–5.1 s against a 0.4 s budget produced an 8.7 s
freeze then a lurch — see [jitter_root_cause.md](jitter_root_cause.md).)

Client-side `--image-resize 224 224` (default: on) is what makes a remote server
viable: it puts ~150 KB on the wire instead of the ~0.9–2.7 MB raw frame, and latency
is `d`. **224×224 is the only safe value** — the server still runs its own
aspect-preserving `ResizeImages(224,224)`, which is a no-op at that size; any other
value letterboxes the letterbox, a quiet train/serve viewpoint mismatch that reads as
a bad policy. `--image-resize None` hands the resize back to the server — LAN only.

### Start where training started (`--start-from-episode`)

The loop seeds from the measured joints and composes deltas onto them, so it is
well-defined from any posture — which is exactly the trap: every training episode
starts somewhere particular, and starting a rollout elsewhere makes the first
observation out-of-distribution. A bad start pose reads as a bad checkpoint.

```bash
python -m ego2g1.deploy --start-from-episode 0 --dataset data/lerobot_datasets/<repo_id> ...
```

ramps (rate-limited, 0.5 rad/s) to that episode's first posture before the loop
starts. The same fact drives teleop: engaging from arms-hanging gave **189 mm** IK
tracking error (watchdog trip); engaging from a training start pose gave **0.2 mm**
(`tools/teleop/README.md`).
