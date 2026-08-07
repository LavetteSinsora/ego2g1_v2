# Perception v2 — implementation plan

Replaces the detector/tracker/orientation cascade of `relation_deploy_plan.md`
§5.3.

- **§2 is measured** on the deploy 4090. Not estimated.
- **§3 is decisions**, each with the evidence that forces it. Overturn one by
  overturning its evidence, not by preference.
- **§4–§7 are the build.** §8 is the order. §9 is what is still unknown.

Decision IDs are grouped by concern: **M**odel, **T**iming, **R**esources,
**S**ignals.

---

## 1. Deliverable

At every policy inference, one **56-dim hand-major relation vector** — each
object's pose in each hand's flange frame — plus two grasp binaries. Everything
below exists to produce it.

Two poses are needed in a common frame: **object pose** (position + rotation)
and **flange pose**. The flange is free — FK on measured joints. The object is
the entire problem.

The roster is fixed: `task_config.objects`, validated against the checkpoint at
connect time. Three known instances, one per text prompt, order load-bearing
for the packing.

---

## 2. Measured baseline

RTX 4090 (23.5 GB), 640×480 stereo, 3 prompts, `transformers` 5.14.1,
torch 2.7.1+cu126. p95 unless noted.

### 2.1 Latency

| stage | p95 | note |
|---|---|---|
| SAM 3 — 1 session, 3 prompts, detect+track | **130 ms** | plateaus by decile 3; p50 129, max 132 |
| Orient Anything V2 — 1 crop @ 518 px | **54 ms** | |
| Orient Anything V2 — 2 crops @ 518 px | **83 ms** | ⇒ ~24 ms fixed + ~30 ms/crop; 3 crops ≈ 110 ms |
| StereoSGBM | **16 ms** | CPU; hides inside the SAM 3 stage |
| join — centroid + median depth + backproject | **1.2 ms** | free |
| **perception round** — sam3‖sgbm → join → orient | **221 ms** | → 4.5 Hz free-running |
| perception round, serial | 233 ms | ⇒ GPU‖CPU overlap is worth 12 ms |
| policy round trip (loopback) | **124 ms** | server 57 ms, wire+encode **66 ms (54%)** |
| policy first call | 11.5 s | XLA compile — never with the robot live |

Policy server: `horizon 50, dim 14, fps 30, control_mode relation_eef`.
Budgets from `core/latency.py`: async **500 ms**, temporal_smoothing **267 ms**,
sync unbounded.

### 2.2 Memory

| | |
|---|---|
| SAM 3 session, initial | 2.1 GB |
| SAM 3 session growth | **+11.5 MB/frame, linear, no plateau** |
| Orient Anything V2, resident / peak | 10.2 GB / 12.6 GB |
| perception combined peak | **12.8 GB** |
| jax at `mem_fraction=0.5` | 11.75 GB |
| **total demand** | **24.5 GB on a 23.5 GB card** |

Over budget before the leak is counted. **This, not latency, is the binding
constraint.**

### 2.3 Established constraints

- **One RTX 4090 for everything.** The PPU is unavailable for deployment.
- **Objects can move while ungrasped.** No frame bookkeeping corrects this; only
  a shorter round does. Drives T2, T3 and §8's ordering.
- **`relation_eef`, no RTC training.** Serve with `--no-rtc`; the plain sampler
  is used when no `prev_chunk` arrives anyway.
- **Rough orientation is sufficient.** Licenses R2's resolution cut.

### 2.4 Known defect

Only **2 of 3** prompts detect — `black pen holder` never appears, run after
run. Not latency: a roster slot empty for the whole episode. Blocks everything
(§9 Q1).

---

## 3. Decisions

### 3.1 Model and session — M

#### M1 · One SAM 3 session holding all prompts. Never one session per object.

`Sam3VideoModel.forward` computes the vision backbone **once per frame** and
shares it:

```python
vision_embeds  = self.detector_model.get_vision_features(pixel_values)  # once
all_detections = self.run_detection(..., vision_embeds=vision_embeds)   # loops prompts
vision_feats   = self.get_vision_features_for_tracker(vision_embeds)    # reuses
```

`run_detection` loops prompts, but the loop body is only the small
text-conditioned head. **N prompts cost one backbone + N cheap heads, not N
forward passes.**

Separate sessions would cost N backbones and buy nothing: one session already
gives every object its own memory bank, and cross-prompt association is blocked
outright — `_associate_det_trk` zeroes IoU between different prompt ids, so a
"cup" detection can never bind to a "plate" track.

#### M2 · SAM 3, not SAM 3.1.

3.1 has no `transformers` integration, and Meta's repo video API is offline-only
(`init_state` loads a whole video; there is no `add_frame`). Streaming exists
only for `facebook/sam3`. 3.1's Object Multiplex is a tracker-only optimisation
batching ≥16 objects per pass — worth ~15% at N=3, since the shared backbone
dominates at low object counts. Revisit only if tracking *accuracy* limits us.

#### M3 · Detection and tracking on every frame. No separate reseed tier.

Both share the backbone (M1), so the detector costs little beyond the tracker.
In exchange we get native association, reconditioning and keep-alive for free,
and no custom per-slot reconciliation.

Carried caveat: in streaming mode the **hotstart heuristics are disabled** (they
need future frames), so duplicates and unmatched tracklets are pruned less
aggressively than offline. With a fixed 3-slot roster this matters little —
birth/death is what hotstart governs.

#### M4 · `kernels` stays uninstalled.

transformers imports it unconditionally (`modeling_utils.py`:
`from kernels import get_kernel`). Current `kernels` needs
`huggingface-hub >= 1.x`; `train` resolves 0.36.2 via openpi's pin, whose
validator rejects `str | None`. The venv is shared across groups, so a `kernels`
installed for `perception-v2` **breaks `ego2g1.serve` at import**.

Cost: SAM 3 skips NMS, hole filling and sprinkle removal — masks keep holes,
duplicate detections survive, and the 130 ms figure is a slight under-estimate.
Accepted against a serve path that will not start. Revisit only if openpi's pin
moves.

### 3.2 Timing and consistency — T

#### T1 · Free-running async perception. No fixed cadence.

Rounds run back to back; one finishes, the next starts. There is no reseed to
schedule and no tracker Hz to tune. Whatever rate results **is** the rate, and
it is reported rather than assumed.

#### T2 · One instant. The whole observation is `t_capture`.

```
relation = inv( T_pelvis_flange(FK at t_capture) ) @ T_pelvis_object(t_capture)
```

Image, object poses, flange FK and grasp binaries all come from the same frame.
The joint state is latched **at capture**, and nothing is advanced to send time.

An earlier revision composed the stale object pose with **fresh** FK at send
time — free, exact, continuously available; the GPS/IMU pattern. Two things
kill it:

1. **The policy also receives the image.** `observation/image` is the frame from
   `t_capture`. Advancing FK but not the image desynchronises two modalities
   that were synchronised in every training sample: the image shows the arm
   where it was, the state vector claims where it is now. Nothing downstream can
   reconcile that.
2. **Objects move (§2.3).** Fresh-FK composition is only *correct* under object
   staticity. That assumption is false here.

What we give up: the arm's motion between capture and execution is corrected by
*assuming the commanded chunk was followed* rather than measured. Acceptable —
the arm is executing our own chunk, and it is exactly what RTC's `d` assumes.

Compensation moves to the action side (T3).

#### T3 · A replan waits for the in-flight round. `d` is therefore constant.

Do **not** grab the newest completed snapshot and send immediately. Let the
running round finish and send *its* snapshot.

| | `t_send − t_capture` | `d` |
|---|---|---|
| newest completed, send now | **P … 2P** | 10–17 ticks, varies per call |
| **wait for the in-flight round** | **exactly P** | ~10 ticks, **constant** |

`d` is sent *with* the request and the RTC guidance mask depends on it, so
determinism beats marginal freshness. `DelayBudget`'s own docstring agrees:
*"Deliberately a FIXED budget rather than a per-call prediction… Deterministic
beats adaptive."*

`d` does not depend on when the request arrived — the observation is always the
start of the round we waited for, and we always send at its end, so
`d = P + L`.

It also decides whether the chunk arithmetic closes. Horizon 50 at 30 fps is
1.67 s:

- constant `d` = 10.4 ticks → **39.6 usable slots = 1.32 s** > 1 s replan ✓
- varying `d` up to 17 ticks → 33 slots = 1.10 s, at the edge once slip is added

**The value:**

```
P  perception round   221 ms = 6.6 ticks
L  policy             124 ms = 3.7 ticks
   d = P + L          345 ms = 10.4 ticks
   × 1.15 headroom    397 ms = 11.9   ->  d = 12
```

12 ticks (400 ms) fits the 500 ms async budget with ~100 ms spare. **16 ticks
(533 ms) would exceed it** and `startup_self_check` would refuse the mode.

Asymmetry that makes erring high correct: `d` too small means executing past
the frozen prefix — a lurch at the seam; `d` too large only over-constrains and
loses a little reactivity.

**`DelayBudget.observe()` is currently fed policy latency only.** It would
report `d ≈ 4` and silently under-commit. Feeding it `P + L` is §8 step 9.

#### T4 · Every snapshot binds to a 30 Hz control tick.

The camera is asynchronous to the control loop, but action slots are integer
ticks. On arrival a frame binds to the **nearest control tick** `n_capture`, and
the snapshot's flange FK is that tick's FK — already computed by the control
loop.

- `d = (n_send − n_capture) + ceil(L × 30)`, whole ticks
- action slot *k* means control tick `n_capture + k`
- handover to the new chunk at slot `d`, at tick `n_capture + d`

Quantisation error ≤ half a tick (16.7 ms) ≈ 5 mm at arm speed — well under
every other term. It also removes any need to interpolate the chunk: slots land
on real control ticks by construction.

During the wait the robot is not idle; it is finishing the previous chunk, and
RTC's guidance mask makes the handover continuous.

### 3.3 Resources — R

#### R1 · Prune the memory bank. Never reset it.

Memory attention reads exactly two things:

```python
conditioning_outputs, unselected = self._select_closest_cond_frames(...)  # anchors
for relative_temporal_offset in range(self.num_maskmem - 1, 0, -1):       # recent window
    previous_frame_idx = frame_idx - relative_temporal_offset
    output_data = ...["non_cond_frame_outputs"].get(previous_frame_idx, ...)
```

`non_cond_frame_outputs` is only ever read for `t-1 … t-(num_maskmem-1)`. Older
entries are written once and **never read again** — that dead storage is the
entire 11.5 MB/frame growth.

So: delete non-conditioning entries older than `num_maskmem`; keep all
conditioning frames. **Provably zero quality impact** — the deleted entries are
unreachable. Conditioning frames are the long-lived anchors and accumulate 16×
slower (`recondition_every_nth_frame=16`).

Strictly better than any reset, partial or total: no identity loss, no
re-acquisition gap, no discontinuity in the state vector.

**Implemented** in `perception_v2_latency.py` (`Sam3.prune()`, `--prune`). The
trace reports `memory_allocated` (current) — **not** `max_memory_allocated`,
which only ever rises and can never show a prune working — plus per-object
stored-entry counts. Expect non-cond entries to plateau at
~`num_maskmem × num_objects`.

#### R2 · Orientation stays in the loop, at reduced resolution.

Required online (objects move) and necessarily *after* SAM 3, since it consumes
the mask crop. On one GPU that makes SAM 3 → orient serial; pipelining buys
nothing because both saturate the same card.

The lever is resolution. `preprocess_images` hardcodes 518 px and applies **no
mean/std normalisation** (ToTensor 0..1, bicubic resize on the longest side to a
multiple of 14, pad to square with white). Cost scales with `(size/14)²`:

| size | tokens | relative | 3 crops |
|---|---|---|---|
| 518 | 1369 | 1.00× | ~110 ms |
| 336 | 576 | 0.42× | ~46 ms |
| 252 | 324 | 0.24× | ~26 ms |

Second lever, VRAM: upstream keeps **fp32 parameters** and relies on internal
autocast, which is why a 5 GB checkpoint occupies 10.2 GB. Casting the weights
should roughly halve it (`--orient-cast-weights`).

Third: a **latched object needs no orientation inference at all** (§6), so
during manipulation the stage drops a crop.

#### R3 · Policy and perception co-resident, arbitrated by the replan rule.

Without MPS, kernels from two processes are **time-sliced, not co-scheduled** —
GPU work is *additive*. You cannot overlap perception's 213 ms with the policy's
57 ms; you only choose the interleaving. Real concurrency exists only between
GPU and CPU work (the 12 ms SGBM overlap).

Three risks:

1. **VRAM collision.** jax preallocates a fixed fraction; torch grows and
   fragments. An OOM lands mid-rollout with the arm moving. Cap both
   (`mem_fraction`, `torch.cuda.set_per_process_memory_fraction`) so it refuses
   at startup, and rely on R1 to make torch's footprint bounded.
2. **Latency tail.** A request landing mid-round queues behind in-flight
   kernels; `DelayBudget` takes a quantile, so the tail inflates `d`.
3. **Two CUDA contexts**, 300–500 MB each.

**Arbitration needs no separate mechanism.** Because a replan already waits for
a round boundary (T3), the policy naturally runs in the gap between rounds — no
lock, no mid-iteration yield, no torn snapshot. The timing rule *is* the
arbitration. Cost is ~6% of perception duty, which is additive work paid
regardless.

### 3.4 Signals and degradation — S

#### S1 · One visibility signal, three consumers. Never orient on an unusable crop.

`Sam3VideoSegmentationOutput` carries two per-object scores, and their
difference is the signal:

```python
obj_id_to_score          # detection: did the DETECTOR re-find it this frame?
obj_id_to_tracker_score  # tracker:   memory-propagation confidence
```

Detection present ⇒ independently re-detected ⇒ genuinely visible, crop is real.
Detection absent but tracker score high ⇒ **the mask is memory propagation** ⇒
occluded, crop is a guess. The session also records `obj_id_to_last_occluded`.

⚠ `postprocess_outputs` returns only
`object_ids / scores / boxes / masks / prompt_to_obj_ids` and **drops the
tracker score** — capture it from the raw output first, or all three gates below
silently disable.

`crop_usable` = detection matched this frame **and** mask area above a fraction
of that object's running maximum **and** tracker score above threshold.

| consumer | when unusable |
|---|---|
| **orientation** | do not call Orient Anything — **hold the last usable rotation**. A sliver crop returns nonsense that poisons `orientation.py`'s reference and the symmetry snap. |
| **latch divergence** | suspend the check. No evidence beats bad evidence. |
| **depth sampling** | reject the median — it may be reading the gripper. |

**The governing asymmetry: position survives occlusion, orientation does not.**
A visible-sliver centroid is biased but bounded by the object's extent; an
orientation estimate from the same sliver can be off by 180°. So **position
keeps updating from perception while orientation holds.**

#### S2 · Keep the robust filter. Delete only the extrapolation. Re-express every window.

`tracker.py` holds two separable things: constant-velocity prediction, and
causal median+k·MAD outlier rejection plus OneEuroSE3 smoothing. With a fresh
measurement every round the prediction is dead weight. The filter is not — it is
the only thing between one bad mask/depth sample and a garbage state vector.

But **every tick-based constant was tuned for `observe()` at 30 Hz** and is
wrong at 2–4.5 Hz:

| parameter | was | at 4.5 Hz | action |
|---|---|---|---|
| `confirm_window_ticks: 12` | 0.4 s | 2.7 s | replace with displacement gate (§6) |
| `max_track_loss_ticks: 3` | 0.1 s | 0.7 s | re-express in seconds |
| MAD history window | 30 Hz samples | 4.5 Hz samples | re-tune |
| OneEuro cutoffs | 30 Hz | 4.5 Hz | re-tune — cutoffs are rate-relative |

Express every window in **seconds or metres** and convert using the measured
round rate at runtime. The loop is free-running, so any constant in samples is
wrong the moment the rate moves.

---

## 4. Architecture

Two processes, one box, one GPU.

```
┌─ serve process ─────────────────┐   ┌─ deploy process ─────────────────────┐
│ ego2g1.serve (jax, --no-rtc)    │   │ perception thread — free-running     │
│ mem_fraction capped             │◄──┤   SAM 3 ‖ SGBM → join → orient       │
│ 57 ms inference                 │ws │   prune, publish snapshot            │
│                                 │──►│ control thread — 30 Hz               │
└─────────────────────────────────┘   │   FK, latch, compose, execute        │
                                      └──────────────────────────────────────┘
```

### 4.1 Perception round

1. `read_stereo()` → `(left, right, t_capture)`; bind to nearest control tick
   `n_capture` and take that tick's FK (T4).
2. **SAM 3** (GPU) ‖ **SGBM** (CPU) — *one* camera read handed to both; reading
   separately serialises them behind `HeadCamera`'s lock.
3. **Prune** (R1).
4. **join** → per-object 3D in camera frame, with guards.
5. **Orient** on crops where `crop_usable` (S1); else hold.
6. To pelvis frame: `T_pelvis_object = T_pelvis_camera @ T_camera_object`.
7. **Publish** one immutable snapshot by atomic reference swap.

### 4.2 Snapshot

```python
@dataclasses.dataclass(frozen=True)
class PerceptionSnapshot:
    t_capture: float
    n_capture: int                                  # control-tick index (T4)
    rgb_left: np.ndarray                            # the frame the policy is sent
    flange_pelvis: dict[str, np.ndarray]            # FK at t_capture, per hand
    object_pose_pelvis: dict[str, np.ndarray | None]
    det_score: dict[str, float | None]              # S1
    tracker_score: dict[str, float]                 # S1
    mask_area_px: dict[str, int]
    crop_usable: dict[str, bool]                    # S1
    round_s: float                                  # for staleness accounting
```

Immutable, published by rebinding one attribute. A consumer that can observe a
half-updated state gets torn reads that look exactly like perception noise. The
snapshot **owns the image** — that pairing is T2.

### 4.3 Replan sequence

```
replan due  →  wait for the in-flight round to finish        (≤ P, timeout 1.5P)
            →  it publishes its snapshot
            →  run the policy in the gap before the next round
            →  next round starts
```

On timeout, fall back to the last completed snapshot and accept a larger `d` for
that call rather than starving the control loop.

---

## 5. Modules

### 5.1 `perception/sam3_source.py` — new

```python
class Sam3Source:
    def step(self, rgb) -> Sam3Result     # masks, boxes, det+tracker scores
    def prune(self) -> int                # R1
    def slot_masks(self, res) -> dict[str, np.ndarray | None]
```

- One session, `add_text_prompt(session, all_prompts)` (M1).
- `init_video_session(..., dtype=dtype)` **and** cast `pixel_values` to the
  weight dtype. The processor always emits float32; the mismatch surfaces as
  *"input and bias type should be the same"* on the first conv.
- Capture `obj_id_to_tracker_score` from the **raw** output before
  `postprocess_outputs` drops it (S1).
- `prune()` drops `non_cond_frame_outputs` below `frame_idx − num_maskmem`, per
  object. Read `num_maskmem` / `max_cond_frame_num` off the config at runtime.
- `slot_masks` maps `prompt_to_obj_ids` back to roster slots. Assumes **one
  instance per prompt**; on a collision take the highest score and log it
  (§9 Q5).

### 5.2 `perception/orientation_v2.py` — new

Every item is a deliberate deviation from upstream's demo path:

- Import **by path** — it is a git repo, not a package, and there is no
  `inference.py`; entry points live in `utils/app_utils.py`.
- **Stub `rembg`** if absent: `app_utils` imports it at module scope, but we crop
  from the SAM 3 mask, which beats its matting guess. Never call it.
- **`tgt=None`** (S=1). Passing the same crop twice builds an S=2 sequence and
  doubles the work for one answer.
- **Batch all crops** in one forward. `inf_single_batch` takes `(B,S,C,H,W)` and
  its forward handles `B>1`; only the output unpacking hardcodes `[0]`.
- **Handle both output ranks.** S=1 returns `(B, D)`, S>1 returns `(B, S, D)`.
  Assuming rank-3 produces
  `argmax(): Expected reduction dim 1 to have non-zero size`.
- **Skip `val_fit_alpha`** in the loop — it runs a scipy fit every call. The
  symmetry parameter is needed **once at seed**, to fix the episode's symmetry
  group; then freeze it. A group that changes per frame destroys the reference
  tracking the snap depends on.
- **Parameterised input size** (R2), replicating upstream preprocessing exactly.
- Optional bf16 weight cast (R2).
- **Never called when `crop_usable` is false** (S1).

Raw output never reaches the state vector — it goes through `orientation.py`'s
stabiliser and symmetry snap. That snap is a **gauge choice, not a
quantisation**: `measured @ S` for `S` in the symmetry group describes the
identical physical pose, and picking the representative nearest the previous one
is what stops a stationary object's rotation jumping between equivalent
matrices. With `identity_symmetry_group()` it is an exact pass-through.

### 5.3 `perception/async_perception.py` — new

```python
class AsyncPerception:
    def start(self) / stop(self)
    def latest(self) -> PerceptionSnapshot | None      # atomic read
    def wait_for_current_round(self, timeout_s) -> PerceptionSnapshot   # T3
```

- One thread; publishes by rebinding a single attribute.
- `wait_for_current_round` is the replan primitive (§4.3) — it is also the GPU
  arbitration (R3).
- Records `round_s` so staleness is observable, not inferred.
- Logs loudly if two consecutive replans consume the same snapshot: perception
  is slower than the policy period and the design assumption has inverted.

### 5.4 `perception/relation_perception.py` — modify

- `observe()` consumes a snapshot instead of driving detector/depth.
- Composes from that snapshot's own `flange_pelvis` and `object_pose_pelvis`
  (T2). **No fresh FK.** The image sent must be that snapshot's `rgb_left`.
- Reports `n_send − n_capture` so the caller can set `d`.
- Latch resolution and 56-dim packing unchanged in shape.

### 5.5 `perception/tracker.py` — modify

Delete constant-velocity prediction; keep MAD rejection + OneEuroSE3; re-express
windows in seconds (S2).

### 5.6 `deploy/perception_v2_latency.py` — extend

- **`--with-policy`**: perception free-running while the policy is hit at the
  chunk rate. Reports round time *under contention*, policy p50/p95 *under
  contention*, combined VRAM peak, resulting `d`. Today's 221 ms was measured
  with the policy idle and the 124 ms with perception idle — **neither
  describes the running system.**

---

## 6. Kinematic latching

Per hand. Only `graspable: true` objects are eligible.

### 6.1 Why

When the gripper closes, perception degrades and prediction becomes exact at the
same moment:

- the visible mask's centroid is pulled toward whatever is uncovered, and that
  bias *moves as the hand rotates* — a signal that looks like object motion and
  is not
- median depth may sample the gripper
- **orientation has no partial-credit answer** from an occluded crop — the
  strongest single reason to latch
- meanwhile the object now moves rigidly with the hand, so FK gives its pose
  exactly and for free

Latch the **full pose**, position and rotation together — mixing sources would
make them inconsistent.

### 6.2 Two clocks

| | rate | provides |
|---|---|---|
| control loop | 30 Hz | grasp command, FK, hand displacement |
| perception | 2–4.5 Hz | object poses, visibility |

Anything about the *hand* is instant and exact. Anything about the *object*
arrives a few times per second and may be degraded.

### 6.3 States

```
UNLATCHED ──grasp closes near eligible object──> CANDIDATE
    ^                                                │
    │                                     hand travels 3–5 cm
    │                                     and object followed
    │                                                v
    └──── gripper opens, or sustained ──────────  LATCHED
          divergence while crop_usable
```

### 6.4 UNLATCHED → CANDIDATE: freezing the transform

Trigger at control rate on the grasp command; gate on an eligible object within
`latch_distance_m` (~5 cm) from the most recent snapshot.

**Take each term from where it is valid:**

```
T_flange_object = inv( T_flange(t_grasp) ) @ T_object(t_stale)
                        ↑ exact FK at closure   ↑ last crop_usable snapshot
```

Between the last clean look (`t_stale`) and closure (`t_grasp`), **the hand
moves a great deal** — it is reaching and closing — while **the object does not
move at all**, since nobody has touched it. So the hand term must be fresh and
the object term may be stale, position *and* rotation.

Two failure modes this avoids:

- both from `t_stale` → encodes *"the object sits 5 cm from my palm"* and
  predicts that forever
- both at `t_grasp` → derives the object pose from an already-occluded mask

**This is not a contradiction of T2.** There is no image to stay consistent
with — this is an internal transform at a physical event — and the object is
specifically at rest, so the staticity assumption that fails globally holds
locally.

Keep a ring buffer of ~3 snapshots to find `t_stale`. Log how stale it was: if
latches start failing, that distinguishes a bad transform from a bad grasp.

Record the flange position at freeze — the reference for confirmation.

### 6.5 CANDIDATE → LATCHED: confirm on displacement

Wait until the flange has translated ≥ `confirm_displacement_m` (3–5 cm) from
the freeze point, comparing on every snapshot in between.

Displacement, not a time window, because the test is *"did the object move with
the hand?"* — and if the hand has barely moved, a failed grasp is
indistinguishable from a good one no matter how many samples you take. At 2 Hz
this may be one observation, but **one observation after 5 cm of travel is
conclusive where three after 5 mm are not.**

A wall-clock timeout backstops a stationary hand.

### 6.6 The divergence test — position only, on relative motion

**Position only.** Orientation under occlusion can be wrong by 180°; including
it fires the check exactly when the latch is carrying (S1).

**Relative, not absolute.** Absolute comparison eats the occlusion bias at full
magnitude — a sliver centroid sits 2–3 cm off while `position_tol_m` is 2 cm, so
a *successful* grasp can read as divergence. Compare displacements:

```
succeeded:  |Δobject − Δflange| ≈ 0
failed:     |Δobject − Δflange| ≈ |Δflange| = 3–5 cm
```

The bias is roughly constant over a short window, so it is common-mode and
cancels; only bias *drift* from hand rotation leaks in. This turns a 2-vs-3 cm
judgement into 0-vs-5 cm.

Three guards: **suspend** when `crop_usable` is false; require divergence
**sustained** across several snapshots; use a **generous** threshold — a
spuriously-dropped latch is worse than a slightly stale one.

### 6.7 While LATCHED, and exit

Object pose = `T_flange(t_capture) @ T_flange_object`, evaluated at the
snapshot's own `t_capture` so T2 holds. **No orientation inference for that
object** — one crop fewer, ~30 ms.

Exit on gripper open (instant, control rate) or sustained divergence while
`crop_usable`.

---

## 7. Configuration

```
jax    XLA_PYTHON_CLIENT_MEM_FRACTION   ~0.55   (envs/4090.sh; currently 0.5,
                                                 sized before SAM 3 existed)
torch  set_per_process_memory_fraction   below the measured combined peak
serve  --no-rtc
```

Target budget, contingent on R1 and the bf16 cast:

```
Orient (bf16 weights)        ~5.1 GB
SAM 3 + pruned session       ~3.5 GB
                             ────────
perception                   ~8.6 GB
jax @ 0.55                  ~12.9 GB
CUDA contexts ×2             ~0.8 GB
                             ────────
                            ~22.3 GB   fits 23.5 GB
```

Both perception figures need confirming (§8 steps 2–3) before this is more than
arithmetic.

`config.py`'s `detector_period_ticks` and `orientation_period_ticks` are
meaningless under T1 — delete or repurpose.

---

## 8. Order of work

Each step gated on the previous, because each can invalidate what follows.

| # | work | gate |
|---|---|---|
| 1 | Fix the missing third detection (§9 Q1) | 3/3 slots detect reliably |
| 2 | `--prune` benchmark run, 3000 frames (R1) | VRAM flat at the tail |
| 3 | Orient bf16 + `--orient-size` sweep (R2) | resident ≤6 GB; quality acceptable |
| 4 | `--with-policy` combined benchmark (§5.6) | real contention numbers; combined peak fits |
| 5 | Diagnose the 66 ms wire cost | probe says fixed (→ TCP_NODELAY) or payload-bound |
| 6 | `Sam3Source`, `OrientationV2`, `AsyncPerception` | benchmark numbers reproduce in the real classes |
| 7 | Wire `relation_perception`, single-instant compose (T2) | open-loop eval against recorded data |
| 8 | `tracker.py` / `latch.py` (§6, S2) | latch survives a full grasp; no tick-based constant remains |
| 9 | Replan-waits-for-round + tick binding (T3, T4); `d = P + L` into `DelayBudget`; `startup_self_check` | `d` constant across calls; over-budget modes refused before the robot moves |

Steps 1–3 are cheap and can invalidate the memory plan. Do not skip to 6.

---

## 9. Open questions

**Blocking**

- **Q1. Why does `black pen holder` never detect?** Try prompt variants, check
  `score_threshold_detection` (0.5) and `new_det_thresh` (0.7). Also:
  `task_config.example.yaml` documents `detector_prompt` ending in `" ."` —
  that is **GroundingDINO's** phrase separator, wrong for SAM 3. Fix the schema
  doc or it will be copied into the real config.
- **Q2. How fast do objects move when ungrasped?** Hand-over speed (~0.5 m/s)
  over 300 ms is 15 cm and would dominate everything, forcing object-motion
  extrapolation into the design. Slow sliding needs nothing.
- **Q3. Which control mode?** `d` = 12 ticks against a 15-tick async budget is
  workable but not roomy. `sync` has no budget but pauses visibly per chunk.
- **Q4. What fills a slot with no detection?** Hold last / zeros / nominal /
  abort. Q1 makes this path live immediately.
- **Q5. Can a prompt ever yield two instances?** One-instance-per-prompt
  underpins slot mapping (§5.1).
- **Q6. Does the training encoding match?** Must equal
  `T_left_tcp_object = compose(invert(left[frame]), object_pose)` exactly.
  Camera- and pelvis-frame routes are algebraically equivalent, but "equivalent"
  has to be checked.

**Spec, fill in before §8 step 6**

- Q7. Orientation on all objects, or only graspable / only the target?
  (54 ms vs ~110 ms)
- Q8. Symmetry group per object — cube, identity, or from Orient V2's
  `ref_alpha_pred` at seed then frozen?
- Q9. `nominal_rotations` per object, for the hold fallback.
- Q10. Thresholds: `crop_usable` (det score, area fraction, tracker score),
  minimum mask area, minimum valid-depth pixels.
- Q11. Filter re-tune for 2–4.5 Hz: MAD `k` and history, OneEuro cutoffs.
- Q12. Latch: `confirm_displacement_m`, sustained-divergence window,
  `position_tol_m`.
- Q13. What marks `t_grasp` — gripper feedback or command + measured delay? Same
  signal as the grasp binary's source.
- Q14. Can both hands contend for one object?
- Q15. Episode reset: trigger, and what resets (session, filter, latch,
  orientation reference).
- Q16. Abort policy when all objects are lost beyond the latch horizon.

**Prerequisites**

- Q17. `task_config.yaml` does not exist — `envs/4090.sh` reports
  `task-config=<none>`.
- Q18. Confirm the checkpoint's `train_config.objects` matches it verbatim and
  in order.
- Q19. `camera_calib.npz` (`T_pelvis_camera`) current?
- Q20. `BRAINCO_CLOSED_POSE` measured?
