# Perception v2 — SAM 3 pipeline, stage I/O, and deployment timeline

Supersedes the detector/tracker/orientation cascade of `relation_deploy_plan.md`
§5.3. Same contract at the boundaries: `RelationPerception.observe(...)` still
returns a 56-dim hand-major relation vector, `GraspLatch` still owns the
grasp state machine, `orientation.py` still owns rotation stabilization.
What changes is everything between the camera and those boundaries.

**All latency numbers in this document are unmeasured estimates** used to
show the *shape* of the schedule. Replace them with real numbers from
`ego2g1/deploy/perception_latency.py` on the actual 4090 before treating any
of the timing conclusions as load-bearing. The stage decomposition and the
parallelism structure do not depend on the specific values; the choice of
tracker rate and reseed placement does.

---

## 1. What this replaces, and what survives

| Current | v2 |
|---|---|
| GroundingDINO-tiny → box → SAM2 → mask, every detector tick | SAM 3 image detector, ~1 Hz, **reseed only** |
| Nothing between detector ticks | SAM 3 video tracker, 10 Hz, real masks |
| `ObjectTracker` constant-velocity prediction | **deleted** |
| `ObjectTracker` MAD outlier rejection + OneEuroSE3 | **kept**, now fed at 10 Hz |
| Orientation held at `nominal_rotations` forever | Orient Anything V2 on masked crops, ~1 Hz |
| FK on freshest joint state | FK on the joint state **latched at camera timestamp** |
| `GraspLatch` | kept, consistency check now visibility-gated |

Net deletion is small. This is a re-timing and a model swap, not a rewrite of
the state assembly.

---

## 2. Design invariants

These are the rules the rest of the document assumes. Each exists because the
obvious alternative is wrong.

**I1 — SAM 3, not SAM 3.1.** SAM 3.1 checkpoints have no `transformers`
integration, and Meta's own repo video API is offline-only (`init_state`
loads the whole video; there is no `add_frame`). Streaming exists only for
`facebook/sam3` via `transformers`. 3.1's Object Multiplex is a *tracker-only*
optimization worth roughly 15% at our 3-object count, because both share one
vision backbone per frame and that fixed cost dominates at low N. Revisit 3.1
only if tracking *accuracy* becomes the limiter.

**I2 — Push frames, never masks.** The tracker holds an internal memory bank
of past frames and masks plus the original conditioning frame. Feeding our own
previous mask back in as the sole conditioning signal is self-conditioning with
no corrective anchor, and drifts. `model(inference_session=..., frame=...)`,
nothing else.

**I3 — No native PCS fusion.** `Sam3VideoModel`'s built-in detector/tracker
reconciliation runs `run_detection` unconditionally every frame (no flag skips
it), and in streaming mode its hotstart pruning is disabled. It solves
open-set unknown-cardinality detection. Our roster is 3 fixed slots validated
against the checkpoint at connect time. Reconciliation is per-slot IoU; slots
are never born and never killed.

**I4 — One timestamp per state vector.** The relation vector is
`inv(flange) @ object`. Object pose is ~60–100 ms old by the time it is a 3D
point; FK is instantaneous. Differencing them injects `arm_velocity × latency`
directly into the policy's input — 6 cm at 0.3 m/s and 200 ms, an order of
magnitude above mask-centroid noise. Every frame carries `t_capture`, and FK
is evaluated at `q(t_capture)`.

**I5 — Reseed and orientation live off the critical path.** Neither needs to
be co-timed with policy inference. Both run mid-window, both consume the same
fresh detector masks.

**I6 — Occlusion gates everything that depends on the object being visible.**
Centroid, median depth, orientation, and the latch consistency check all
degrade to garbage — not to noise — when the gripper occludes the object. Each
carries an explicit visibility gate and a hold path.

---

## 3. Stages

Rates are nominal. `⟂` marks stages that run concurrently with the stage above.

### Continuous

**S0 · Capture** — 30 Hz
Camera driver → `(rgb_left, rgb_right, t_capture)`. Latches `q(t_capture)`,
the 14-dim arm joint state, into the same record (I4). Only every 3rd frame is
forwarded to S1/S2; the rest are dropped.

**S8 · Flange FK** — 30 Hz, ~0.1 ms
In: `q(t)` · Out: `T_camera_flange[hand]` = `inv(T_pelvis_camera) @ Kinematics.flange_poses(q)[hand]`.
`T_pelvis_camera` is the static AX=XB hand-eye solve (valid because waist is
pinned at 0 and `head_link` is rigid to pelvis). Free; run it wherever needed.
The instance that reaches the state vector must use `q(t_capture)`, not `q(now)`.

### The 10 Hz measurement loop

**S1 · Track** — 10 Hz, GPU, ~50 ms *(est)*
In: `rgb_left` → processor → `pixel_values`; live `Sam3VideoInferenceSession`.
Out: per slot — low-res mask, tracker score, occlusion/visibility score.
`Sam3TrackerVideoModel.forward(inference_session=session, frame=pixel_values[0])`.
Session is long-lived: prompt text embeddings are cached in it, so recreating
it per tick would re-pay the text encoder.

**S2 · Depth** ⟂ — 10 Hz, CPU, ~25 ms *(est)*
In: `(rgb_left, rgb_right)` + `stereo_calib` · Out: depth map + validity mask.
StereoSGBM. **Fully independent of S1** until the join in S3 — start both the
instant the stereo pair lands. This is the pipeline's only true parallelism and
the reason the 10 Hz loop costs ~50 ms rather than ~75 ms.

**S3 · Join → 3D** — 10 Hz, CPU, <1 ms
In: masks (S1) + depth (S2) + `K_left`
Out: per slot — `(X, Y, Z)` in camera frame + `valid: bool`.
Centroid = mask mean (unchanged from `Detection.centroid_uv`). `Z` = median
depth over `mask ∩ depth_valid`. Back-projection `X = (u-cx)·Z/fx`,
`Y = (v-cy)·Z/fy`.
**Guards:** invalid if mask area below threshold, if visibility score below
threshold, or if too few valid depth pixels survive the intersection. Invalid
emits no measurement rather than a bad one — textureless objects give SGBM
holes, and a median over four surviving pixels is not a depth.

**S4 · Filter** — 10 Hz, CPU, <1 ms
In: `(X,Y,Z)` + validity · Out: filtered position per slot.
The surviving half of `tracker.py`: causal median + k·MAD outlier rejection
against past accepted residuals, then OneEuroSE3. Constant-velocity prediction
is gone — with a measurement every 100 ms there is nothing to extrapolate
across. The filter is what stops one bad mask/depth sample reaching the policy,
and it now has 10 samples per policy tick to work with instead of 1.

### Mid-window, ~1 Hz, off the critical path

**S5 · Reseed** — 1 Hz, GPU, ~100 ms *(est)*
In: `rgb_left`, 3 `detector_prompt` strings, current tracked masks
Out: per slot, one of — *agree* (no action) · *re-anchor* (inject detection
mask as a new conditioning frame for that `obj_id`) · *no confident detection*
(no action, coast).
Per-slot rule: take the best-scoring detection for that slot's prompt, compute
IoU against the tracked mask. High IoU → agree. Confident detection but low IoU
→ the tracker drifted or jumped objects → re-anchor. Nothing confident → coast
and let S3's guards and the latch handle it. **Never spawns, never deletes.**

**S6 · Orientation** — 1 Hz, GPU, ~30 ms *(est)*
In: `rgb_left` + the fresh S5 masks → 3 crops · Out: 3 rotations, camera frame.
Orient Anything V2, **all three crops in one batched forward pass** — three
sequential calls pay the launch/weight overhead three times for the same work.
Feeds `orientation.py`'s stabilizer + symmetry snap; raw per-tick estimates are
jumpy and flip on symmetric objects, so nothing raw reaches the state vector.
V2 is chosen over V1 specifically because it models 0..N valid front faces
rather than assuming a unique one — most manipulation objects are symmetric and
V1 would be confidently wrong. Runs on S5's masks because those are the
highest-quality masks in the window.
**Gated:** skipped (rotation held) when visibility is low or the object is
latched — during a grasp, rotation comes from the rigid transform, never from
an occluded crop.

### Per policy tick

**S7 · Latch** — 30 Hz, CPU, <1 ms
In: `hand_cmds`, `T_camera_flange`, filtered object position, visibility score
Out: resolved pose per object — live-tracked or latch-propagated.
`GraspLatch` unchanged except the consistency check, which now: (a) suspends
entirely below a visibility threshold, (b) requires divergence sustained over
several ticks rather than a single sample, (c) uses a generous threshold. All
three exist because the gripper occludes the object exactly when the latch is
carrying, the visible sliver's centroid is not the object's centre, and a
spuriously-dropped latch is worse than a slightly-stale one.

**S9 · Assemble** — 1 Hz, CPU, <1 ms
In: resolved object poses + `T_camera_flange[hand]` **at the same `t_capture`**
Out: 56-dim hand-major vector + 2 grasp binaries.
`relation[obj, hand] = inv(T_camera_flange[hand]) @ T_camera_object[obj]`.
Note this is algebraically identical to the current pelvis-frame route — the
hand-eye extrinsic appears exactly once either way, so working in camera frame
buys no calibration accuracy. It must match training's
`T_left_tcp_object = compose(invert(left[frame]), object_pose)` exactly.

**S10 · Policy** — 1 Hz, remote, ~80 ms *(est, incl. network)*
In: 56-dim state · Out: 14-dim action chunk × horizon. Runs on the remote
server, so it contends with nothing locally — the 10 Hz loop keeps running
through it.

---

## 4. Resource model

Three resources, and only one real overlap.

| | Stages | Notes |
|---|---|---|
| **GPU** | S1 (10 Hz), S5 (1 Hz), S6 (1 Hz) | **Serial.** Cannot overlap each other. |
| **CPU** | S2 (10 Hz), S3, S4, S7, S8, S9 | S2 dominates; the rest are microseconds. |
| **Remote** | S10 | Free concurrency with everything local. |

**The only true parallelism is S1 (GPU) ‖ S2 (CPU).** Everything else labelled
"concurrent" is just cheap work hiding in the noise.

The GPU being serial is the constraint that shapes the schedule: at 10 Hz each
slot is ~50 ms of tracker plus ~50 ms idle, and a ~100 ms reseed does not fit
in that idle gap. It must displace a tracker step.

**That displacement is free**, because the reseed *produces masks of its own*.
The slot where S5 runs still yields a 2D measurement — it comes from the
detector instead of the tracker. No measurement is lost; only the timing
jitters. This is why S5 sits mid-window and why S6 immediately follows it.

---

## 5. Timeline — one 1 s policy window

Policy period 1000 ms (30 control ticks at 30 Hz). Capture grid offset so that
one measurement completes *just before* the policy boundary rather than just
after.

```
 t (ms)   0    100   200   300   400   500   600   700   800   900  1000
          |     |     |     |     |     |     |     |     |     |     |
 capture  C₀    C₁    C₂    C₃    C₄    ·     C₅    C₆    C₇    C₈    C₉
          ↑                                                           ↑
       policy                                                      policy
        tick                                                        tick
                                                                    (n+1)

 GPU     [ Tk ][ Tk ][ Tk ][ Tk ][──── Det ────][Ori][ Tk ][ Tk ][ Tk ][ Tk ]
 CPU     [Sgbm][Sgbm][Sgbm][Sgbm][Sgbm]         ·   [Sgbm][Sgbm][Sgbm][Sgbm]
         └join─┘                                     (S3→S4 after each)

 legend   Tk = S1 tracker    Det = S5 reseed    Ori = S6 orientation
          Sgbm = S2 depth    · = slot displaced by reseed
```

Captures land at t = −60, 40, 140, … 940 ms; each completes ~60 ms later, so a
fresh measurement is ready *at* each 100 ms mark, including the policy
boundary. Nine tracker steps + one detector step = ten measurements per window.

### Critical path at the policy boundary

This is the only chain the policy tick actually waits on.

| | Stage | Δ | cumulative |
|---|---|---|---|
| t=940 | S0 capture, latch `q(940)` | — | 0 ms |
| | S1 track ‖ S2 depth | 50 ‖ 25 | 50 ms |
| | S3 join → 3D | <1 | 51 ms |
| | S4 filter | <1 | 51 ms |
| | S8 FK at `q(940)` | <1 | 51 ms |
| | S7 latch resolve | <1 | 52 ms |
| | S9 assemble 56-dim | <1 | 52 ms |
| t≈992 | S10 send → policy | 80 | 132 ms |

**~52 ms** of local perception on the critical path, and the state describes
t=940 ms — a 60 ms lag, all of it accounted for and none of it skewed between
the object and flange sides (I4).

What is deliberately *not* on this path: the ~100 ms reseed and the ~30 ms
orientation pass, both completed at t≈560. The rotation reaching the policy at
t=1000 is ~440 ms old. That is an accepted trade — orientation is slow-moving,
symmetry-snapped and stabilized, and held outright during latch. Position, the
fast-changing quantity, is 60 ms old.

Running the reseed at the boundary instead would have put ~130 ms of
detector+orientation work directly in front of the policy tick, roughly
tripling perception latency for no accuracy gain.

---

## 6. Cold start

Before the first policy inference, once:

1. **S0** capture one stereo pair, latch `q(t₀)`.
2. **S5-init** SAM 3 image detector over all 3 prompts → 3 masks.
   Fail loudly if any roster slot has no confident detection — starting with a
   missing object is not a recoverable state, unlike losing one mid-episode.
3. Open the `Sam3VideoInferenceSession`, bind each mask to its roster slot's
   `obj_id`. Slot↔`obj_id` binding is fixed for the episode.
4. **S6** orientation on all 3 crops, batched → seed `orientation.py`.
5. **S2/S3** depth + back-projection → seed `S4`'s filter history.
6. Warm up: run one throwaway S1 step so the first real tick does not pay
   CUDA graph / autocast compilation.

Cost is roughly 100 (det) + 30 (ori) + 25 (depth) + warmup ≈ **200 ms**, paid
once, before the episode clock starts.

Episode reset must call `session.reset_inference_session()` — the SAM 3
session is stateful, which is the one wart in hiding it behind the otherwise
stateless `ObjectDetector` interface.

---

## 7. Failure modes and their guards

| Failure | Where it bites | Guard |
|---|---|---|
| Object occluded by gripper | S3 centroid + median depth both sample the gripper | Visibility gate in S3; latch takes over (S7) |
| Latch check fires during occlusion | S7 drops the latch exactly when needed | Visibility-gated, sustained-divergence, generous threshold (I6) |
| Textureless object | SGBM holes → median over ~nothing | Valid-pixel-count floor in S3 → no measurement |
| Tracker drifts / jumps object | Silent wrong object for a full second | S5 re-anchor at 1 Hz |
| Tracker loses object entirely | Empty mask | S3 invalid → S4 holds → S5 re-acquires |
| Orientation flip on symmetric object | Rotation jumps 180° in state | `orientation.py` symmetry snap; V2's 0..N front faces |
| Arm moving fast | 6 cm relation error | I4 timestamp-matched FK |
| One bad depth sample | Garbage 3D point → policy | S4 MAD rejection |

The one failure with no guard: **all three objects lost simultaneously** for
longer than the latch horizon. That is an abort condition, not something to
paper over.

---

## 8. Numbers to measure before building

The schedule above is only valid if the estimates hold. In rough order of how
much they'd change the design:

1. **S1 tracker step**, 3 objects @ 1008 px, bf16, on the 4090. If this is
   >100 ms, 10 Hz tracking is not viable and the whole cadence shifts.
2. **S5 detector pass**, 3 prompts. Sets how much of the window the reseed
   displaces.
3. **S2 SGBM** at the real resolution. If it exceeds S1, the CPU becomes the
   bottleneck and the S1‖S2 overlap stops helping.
4. **S6 Orient Anything V2**, 3 crops batched vs sequential — confirm the
   batching win is real before depending on it.

Secondary, but they gate the design rather than the schedule:

- **Is depth wired at all?** `relation_deploy_plan.md` §5.2 notes `HeadCamera`
  reads RGB only, no depth channel today. Every 3D position here depends on
  stereo. This is a prerequisite, not a detail.
- **Is 1 Hz policy inference real or a placeholder?** A 1 s action chunk
  amplifies every staleness figure above. If it is actually 2–5 Hz, the reseed
  no longer fits mid-window and S5/S6 need their own slower cadence.
- **Orient Anything V2 on our actual crops.** A 2026 research model on
  cluttered tabletop crops with a robot arm in frame is the least-proven
  component here. Keep the "hold `nominal_rotations`" fallback wired and
  switchable, and budget for it not working.
