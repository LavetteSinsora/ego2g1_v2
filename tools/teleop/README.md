# tools/teleop (was `human_hand_teleoperate`)

Teleoperate G1-D with your **bare hands**. No controllers, and nothing installed on the
PICO — its browser opens a `vuer` page, enters WebXR, and streams hand tracking back.

The point is not convenience. This runs the human's hands through **the same retargeting
code that produced the training labels** and into **the same control path the policy
uses**, so it is the only end-to-end test of the retarget we have. If you can pick up the
bottle by moving your own hand, then `S`, `B`, the flange convention, the Revo2 mount
rotation and the Brainco motor order are all correct. Today a policy failure and a
retargeting bug look identical.

## How a hand pose becomes a flange target

The training label factors as `G(t) = pelvis⁻¹ · S · T_wrist(t) · B = C · R_w · B_R`
(orientation) and `C · p_w + const` (position), with `C = pelvis⁻¹·S` a heading yaw.

**Default mode — absolute orientation, position relative to engage:**

```
R_target(t) = C · R_wrist(t) · B_R                         # ABSOLUTE, fixed correspondence
p_target(t) = p_engage + C · ( p_wrist(t) − p_wrist_engage ) # RELATIVE to the clutch anchor
```

- **Orientation is absolute** because a rotation has no origin, no scale, no workspace
  limit — so it maps through one fixed transform and your hand pose ↔ flange pose is a
  fixed correspondence. Rotate your hand 90°, the flange rotates 90° the same way, every
  time, regardless of when/where you engaged. That's what makes it feel natural.
- **Position is relative** because it has all three problems (Pico's origin is set at
  headset boot; your reach exceeds the G1's; targets must be reachable). The engage
  anchor is the live stand-in for the placement fit's translation; re-engaging (the
  clutch) re-picks it when you run out of workspace. Displacement is rotated by the same
  fixed `C`, so "world right → base right" no matter how your hand is turned.

`C` is a single session-fixed **heading yaw** — the only unknown absolute orientation
adds is how your Pico "forward" relates to the robot's "forward." It's estimated once at
the first engage from a matched pose (`set_heading`), and it has no workspace cost.

Verified: with `C` set to the exact `pelvis⁻¹·S`, absolute mode reproduces the pipeline's
own labels `G(t)` to 2.2e-16 (`check replay`, `tests/test_cancellation.py`).

**Relative mode** (`orientation="relative"`, kept for A/B) differences both orientation
and position against engage: `G_target = G_engage · B⁻¹ · (T_w_engage⁻¹ T_w) · B`, in
which `S` and the world frame cancel without being known. Both modes record the identical
body-frame action downstream, so the choice is pure operator feel.

Consequence either way: the per-episode placement fit — the only genuinely non-causal
stage in the pipeline — **is not needed at teleop time**, and the Pico world frame's
*orientation* is handled by `C` (relative mode) or `C` cancels (relative-position), so
**it does not matter where the headset sits** (neck, chest, tripod), only that its SLAM
is stable and your hands are in its FOV.

## Anchor where you intend to work

The thing that replaces `S` is **the anchor**. Offline, `refine_placement` fits `S` so the
human's hands land inside the robot's reachable set. Teleop has no `S`, so `G_engage`
decides where your hands map to — and it has to be a pose your motion is reachable *from*.

Measured, on `episode_1`:

| anchor | IK tracking error |
|---|---|
| arms hanging (`NOMINAL_ARM_QPOS`) | **189 mm** — outside the workspace, trips the watchdog |
| the training start pose (`--start-from-episode`) | **0.2 mm** |

So ramp to a real episode's start pose, and engage with your hands in a matching posture.
This is what `--start-from-episode` is for, and it is close to mandatory.

## Run it

```bash
# rung 1 — settle the Brainco motor order FIRST. It is flagged UNVERIFIED in deploy/dds.py,
# and a permuted order means the thumb closes when you curl your ring finger.
.venv/bin/python -m ego2g1.deploy.check hand-sweep

# rung 2 — tracker only, no robot. Also answers the mounting question (see below).
.venv/bin/python -m tools.teleop.check stream --B ego2g1/data/work/_global/b_calib.npz

# rung 2b — does the live wrist frame agree with the one B was fitted in?
.venv/bin/python -m tools.teleop.check measure-c

# rung 3 — the live code path over offline ground truth. No headset, no robot.
.venv/bin/python -m tools.teleop.check replay --B ego2g1/data/work/_global/b_calib.npz

# rung 4 — teleoperate a DYNAMIC MuJoCo G1 (table + bottle). No robot.
# macOS MUST use mjpython (the viewer owns the main-thread GUI loop); Linux: python.
.venv/bin/mjpython -m tools.teleop --sim \
    --B ego2g1/data/work/_global/b_calib.npz \
    --dataset data/lerobot_datasets/ego2g1/put_bottle_in_box --start-from-episode 0

# the real thing
.venv/bin/python -m tools.teleop \
    --B ego2g1/data/work/_global/b_calib.npz \
    --dataset data/lerobot_datasets/ego2g1/put_bottle_in_box --start-from-episode 0
```

`--sim` swaps **only** `G1DDS` for `SimDDS` — same source, retargeter, IK, trajectory,
clamp and watchdog — so what you feel there is what the robot does. It is *dynamic*, not
kinematic: gravity, contacts and the model's own position servos, so the arm genuinely
lags its command and you can actually knock the bottle over. `--sim-realtime 0.5` runs at
half speed (easier to grasp). Keys are the same, but press them **with the viewer window
focused**. This is the one part of the package that imports the **old repo's**
`data_extraction` (for the composite model; future home `ego2g1.kin.g1_hands`, see
`INTEGRATION.md`), so `--sim` only runs where that is importable — the robot PC
doesn't need it.

`b_calib.npz` comes from the pipeline — it must be the **same `B`** the labels used:

```bash
.venv/bin/python -m ego2g1.data.run_pipeline --through b_calib \
    --set episodes_dir=$PWD/data/put_bottle_in_box_ego
```

Keys: `e` engage / re-anchor · `space` disengage · `q` damp and quit. Your reach is bigger
than the G1's, so you *will* run out of workspace — disengage, bring your hands back,
re-engage. Re-engaging re-anchors **position** (so nothing jumps) but keeps the **heading
and orientation correspondence** fixed — that's the point of absolute mode: you can clutch
to reposition without your orientation ever re-zeroing.

The **first** engage also fixes the heading `C`, so make it count: ramp the robot to a
sensible start pose (`--start-from-episode`) and engage with your hands matching it. At
that first engage the orientation ramps from the robot's current pose to your absolute
pose over ~0.5 s, so even a small mismatch won't snap.

## Mounting

All three work geometrically (the world frame cancels). The open question is only whether
the WebXR **immersive session survives** off-head — the proximity sensor may suspend it.
`check stream` answers this in fifteen minutes; it prints the dropout %.

1. **Neck / chest** (`--display-mode pass-through`, the default) — you watch the real robot
   with your own eyes and the PICO is a pure sensor. Best acuity. **Try this first.**
2. **Head-worn** (`--display-mode ego`) — robot's view inset in the real world. Guaranteed
   to keep the session alive. Note the robot's camera frame is small and the PICO panel is
   low-acuity, so the display plane wants scaling up; the operator's view and the
   *policy's* 224×224 observation are two different images from the same camera.
3. Custom OpenXR/TCP app — not bound to a browser session. Only if 1 fails and 2 is
   intolerable. Not built.

## Out-of-view robustness

Hands leave the neck mount's FOV constantly, so dropouts are the normal case, not an
error. Two layers handle them:

- **Freeze detection** (`source.py`): when a hand leaves view, televuer HOLDS its last
  pose with `active` still true — a frozen hand looks perfectly valid. We treat
  byte-identical consecutive frames as stale (live tracking never repeats to the bit), so
  a frozen hand goes inactive and `age()` stops advancing.
- **Graceful re-acquisition** (`retarget.py`): a dropped hand holds its arm pose and grip
  (never extrapolates, never opens a grasp). When it returns after a gap, it is
  **re-anchored** exactly like a per-hand engage — position picks up from where the arm
  was holding (no teleport) and orientation ramps in over ~0.3 s (no snap), regardless of
  how far the operator moved while out of view. Engage and re-acquire are the same
  operation (`_anchor_hand`).

Total loss (both hands) **holds** and resumes seamlessly; only a prolonged loss (default
5 s, "operator gone") drops to IDLE requiring a deliberate re-engage. Damp is reserved for
genuine faults (stale robot state, IK divergence, thrown thread) — never for a glance away.

## Rates

The `30 Hz` in `deploy/` was never the robot's command rate — the arm has always emitted at
**500 Hz** and the hands at **200 Hz**, interpolating between knots. 30 Hz was the rate the
*policy* produced knots at, and the whole chunk/RTC/async-inference apparatus exists to
hide ~400 ms of policy latency. A human hand has none of that, so none of it is here: the
control thread runs at the tracker's rate (~60 Hz) and pushes one knot per sample.

Measured budget per tick: hand retarget **0.45 ms** (both hands) + mink IK **~0.95 ms** ≈
1.4 ms of a 16.7 ms tick.

Finger commands pass through a **One-Euro smoother** (on by default; adaptive — smooths hard
when the hand is still, near-zero lag on a fast close) before the velocity clamp; disable it
with `--no-finger-smooth` (or tune `--smooth-min-cutoff` / `--smooth-beta`).

## Layout

| file | what |
|---|---|
| `source.py` | `VuerSource` (live WebXR) and `Hdf5Source` (replay). Both emit the **same `(26,7)` array `load_episode` produces**, so live and offline share one code path. |
| `retarget.py` | the two arm-target modes (absolute default / relative), the heading `C`, the anchor/clutch. Fingers go through `HandRetargeter.step`, the offline solve (proven bit-identical). |
| `loop.py` | `TeleopLoop` — 500 Hz arm / 200 Hz hand emitters + control at the tracker's rate. Reuses deploy's `Kinematics`, `TrajectoryBuffer`, `Clamp`, `Watchdog`, `G1DDS`. |
| `calib.py` | open-hand calibration, and the `C` (wrist-convention) measurement. |
| `check.py` | the bring-up ladder. |
| `sim.py` | `--sim`: `SimDDS` (a physics-stepped stand-in for the six-method DDS seam) + the table/bottle scene + the passive viewer. Dynamic, so grasping is real. |
| `_vendor/` | self-contained copies of everything above the package needs (see below). |

WebXR gives 25 joints, OpenXR 26; the difference is OpenXR's `palm` at index 0, which no
line of the retargeter reads. So `openxr_idx = webxr_idx + 1` and row 0 stays zero.

## Self-contained (`_vendor/`)

This package imports **nothing outside its own directory** — it runs on the robot PC
without the rest of the repo present. Everything it needs is vendored into `_vendor/`:

- `_vendor/de/` — the retarget from `data_extraction` (relative imports, copied byte-identical).
- `_vendor/eg/` — the deploy control path from `ego2g1` (DDS, mink IK, safety, trajectory,
  ramp; absolute `ego2g1.*` imports rewritten to relative), plus `eg/deploy/_g1_sim/` (the
  G1 sim + assets, built fresh from `data_extraction`, as `deploy/_g1_sim` does).

The copies are NOT a fork: `_vendor/MANIFEST.json` records each source's hash, and
`tests/test_vendor_drift.py` fails if the source drifted — so the retarget here cannot
silently diverge from the code that made the training labels. External pip deps
(`numpy`, `mujoco`, `mink`, `h5py`, `unitree_sdk2py`, `televuer`/`vuer`, `pandas`) are the
only things imported from outside.

Rebuild the copies after changing `data_extraction`/`ego2g1` (**old repo only** — the
builder copies from the old tree; in v2 the plan is the reverse, de-vendoring into
`ego2g1.*` per `INTEGRATION.md`):

```bash
.venv/bin/python -m tools.teleop._vendor._build
```

## What is not done

- **`record.py`** — writing teleop episodes in the s005 LeRobot schema. The observation
  image must come from the **robot's head camera** (`deploy/camera.py`), not the PICO's:
  the policy's observation at deploy is the robot's view.
  `ego2g1/data/teleop_import.py` is the template.
- Rungs 4–6 (sim viewer, robot arms-only, robot + hands) are procedure, not code.
