# Perception v2 — implementation notes

Companion to `perception_v2_pipeline.md`. That document is the plan; this one
records what building it actually turned up.

Three kinds of entry, kept separate on purpose:

- **§1 Deviations** — where the code does something the plan does not say.
  Each has a reason and a cost. These are decisions, and they can be overturned.
- **§2 Optimizations available** — real headroom found while building, not yet
  taken. Each says what it buys and what it risks.
- **§3 Tests that must run on hardware** — what the 110 CPU tests cannot cover,
  in the order it should be run.

Status: **§8 steps 6–8 are written and unit-tested; nothing is wired into the
runner.** Steps 1–5 all need the 4090 and have not been run.

---

## 0. What exists

New package `ego2g1/deploy/perception/v2/`, alongside the v1 cascade rather
than replacing it. v1 is what `modes/relation_eef.py` builds today and it
runs; deleting a working path before its replacement has been validated on
hardware is how you end up with neither. At §8 step 7 (open-loop eval passes)
the cutover is: delete the v1 modules, move these up one level, drop the `v2`
package. Nothing imports both.

| module | plan | lines | GPU? |
|---|---|---|---|
| `snapshot.py` | §4.2, T2, T4 | 220 | no |
| `sam3_source.py` | §5.1, M1/M3/R1/S1 | 560 | model only |
| `orientation_v2.py` | §5.2, R2/S1 | 350 | model only |
| `object_tracker.py` | §5.5, S2 | 250 | no |
| `latch.py` | §6 | 460 | no |
| `async_perception.py` | §5.3, T1/T3/R3 | 330 | model only |
| `relation.py` | §5.4, T2 | 250 | no |
| `config.py` | §7 | 190 | no |
| `timing.py` | T3/T4 | 80 | no |

110 tests in `tests/deploy/perception/v2/`, all passing, none needing a GPU,
weights, or a camera. Full suite: 429 passed (up from 319); the 4 failures and
5 errors are pre-existing and are openpi not being installed on macOS.

The pure/torch split is deliberate and load-bearing: everything that can
silently corrupt a state vector — slot mapping, the visibility gates, the
filter, the latch, the packing — is numpy-only and tested. What is untested is
the model call itself.

---

## 1. Deviations from the plan

### 1.1 One visibility gate became two — `mask_usable` and `crop_usable`

**The plan contradicts itself here.** S1 lists three consumers of one signal
(orientation, latch divergence, depth sampling) and then states the governing
asymmetry:

> position survives occlusion, orientation does not … So **position keeps
> updating from perception while orientation holds.**

If one gate drives all three, then rejecting the depth sample also stops
position updating — exactly what the asymmetry says must not happen. During
every reach the object is partly occluded by the approaching hand, so a single
strict gate would freeze the object's position for the whole approach, which
is when the policy most needs it.

Resolved by splitting along the line the asymmetry itself draws:

| gate | admits | consumers | why |
|---|---|---|---|
| `mask_usable` | enough pixels, not flagged occluded | position, depth | a biased centroid is still **bounded** by the object's extent |
| `crop_usable` | additionally re-detected, substantially complete, confident tracker | orientation, latch divergence | both failure modes are **unbounded** — 180° errors, spuriously dropped latches |

`crop_usable` implies `mask_usable`; `PerceptionSnapshot` enforces it.

**Residual risk:** a partly-occluded mask can still give a median depth that
reads the gripper. That is handed to the tracker's causal MAD gate, which is
the layer designed for one bad sample among good ones. If bring-up shows depth
errors surviving that, tighten `min_area_fraction` before re-merging the gates.

### 1.2 The outlier gate works on speed (m/s), not displacement (m)

The plan's S2 argues at length that any window in *ticks* is wrong at a
varying rate. The identical argument applies to a threshold in *metres*:
`min_residual_m: 0.01` means 0.3 m/s at 30 Hz and 0.045 m/s at 4.5 Hz. At the
free-running rate it would reject an object sliding at walking pace.

So the statistics are gathered on speed and compared in metres at the current
`dt`:

```
threshold_m = max(min_residual_m, robust_speed_threshold(history) * dt)
```

`min_residual_m` survives as an absolute floor (depth quantisation is worth
several millimetres regardless of rate). New knob `max_speed_m_s = 1.5` — the
speed floor, stated as a physical claim about the task rather than a tuning
number. Tested at 30 Hz and 4.5 Hz: same accept/reject decisions on the same
physical trajectory.

The residual history is also trimmed **by age**, not by count, for the same
reason — a fixed-length deque spans a different amount of real time at every
rate, so the threshold would silently retune itself whenever the loop sped up.

### 1.3 Smoothing is position-only — `OneEuro`, not `OneEuroSE3`

Found by a failing test. `OneEuroSE3` slerps rotation toward each new sample,
so a fresh orientation took ~3 rounds (0.7 s) to be reflected. Three reasons
that is wrong here, only the first of which is about latency:

1. Rotation updates are already sparse and already gated (only on a usable
   crop). Each one is the freshest trustworthy information available;
   smoothing adds lag to the quantity with the least to spare.
2. `OrientationRefiner` picks each rotation's symmetry branch relative to
   **its own** last output. Reporting a *smoothed* rotation downstream means
   the snap's reference and the reported value are different quantities — two
   clocks on one variable.
3. The MAD gate is position-only. Smoothing a quantity with no outlier gate is
   the wrong pairing: a bad rotation still arrives, just more slowly.

Position is smoothed, rotation passes through. This also deleted the
quaternion machinery from the module.

**Open:** OneEuro's own cutoffs are still at the 30 Hz defaults. At 4.5 Hz
`min_cutoff=1.0` gives α ≈ 0.58 versus 0.17 at 30 Hz — far less smoothing.
That may be correct (Nyquist) or may want retuning; it is plan Q11 and needs
recorded data. Untuned, the filter is close to a pass-through, which is the
safe direction.

### 1.4 The divergence baseline is the first post-closure snapshot, not the freeze

§6.4 says "record the flange position at freeze — the reference for
confirmation." Used as the divergence origin, that breaks the test.

**Two different things get frozen at closure, and the plan only names one.**

*The prediction transform* answers "where is the object now?" once it is
hidden. It pairs the flange at closure (`t_grasp`, exact FK) with the object as
last cleanly seen (`t_stale`), and §6.4's asymmetry for it is correct.

*The divergence baseline* answers a different question — "did the object come
with me?" — and it is a **change** measurement, so what it needs from its two
endpoints is not accuracy but **comparability**.

That is the whole basis of §6.6's relative test: the occlusion bias `b` (a
partly-covered mask pulls the centroid toward whatever is still visible) is
large — 2–3 cm against a 2–3 cm tolerance — so the test only works if `b`
**cancels**, which requires both endpoints to carry the *same* bias.

`t_stale` is by construction the last **usable** look, i.e. an unoccluded one,
`b ≈ 0`. Every post-closure observation is occluded, `b ≈ 2 cm`. Differencing
across that boundary:

```
Δobject = (true_now + b) − (true_stale + 0) = Δtrue + b
Δflange =  true_now      −  true_stale
divergence = |b| ≈ 2 cm    ← on a PERFECTLY SUCCESSFUL grasp
```

The bias does not cancel; it appears at full magnitude. That is exactly the
absolute-comparison failure §6.6 exists to avoid, reintroduced through the
choice of origin. Take both endpoints from post-closure snapshots and `b`
subtracts out, leaving `Δtrue`.

A second, independent reason: a closing gripper frequently **nudges** the
object a centimetre or two. So `object(t_stale) ≠ object(t_grasp)`, and a
Δobject measured from `t_stale` carries that nudge while Δflange (measured
from `t_grasp`) does not.

So the transform keeps §6.4's asymmetry, and the divergence test takes its
baseline from the first admitted observation after closure. The confirmation
travel gate uses the same baseline — it exists to guarantee the divergence
test has signal, so it must measure the same interval, or it can pass while
the divergence interval is still near zero.

Also: on CANDIDATE → LATCHED the confirming observation becomes the first
LATCHED baseline rather than clearing it. Clearing costs a full round before
drop detection can start, for nothing.

### 1.5 Confirmation is asymmetric — one agreement confirms, N disagreements reject

§6.5 and §6.6 read as if the sustained-divergence requirement applies
throughout. Implemented asymmetrically instead:

- after `confirm_displacement_m` of travel, **one** agreement latches —
  a stationary object does not follow a hand 4 cm, so this is physically
  conclusive (§6.5 says exactly this: "one observation after 5 cm of travel is
  conclusive where three after 5 mm are not");
- **`divergence_sustain` (2)** disagreements are required to reject, because a
  lone disagreement is precisely what one bad depth sample looks like, and a
  spuriously-dropped latch is worse than a slightly stale one.

While LATCHED the divergence test differences **consecutive** observations
rather than working from a fixed origin. §6.6's cancellation argument requires
the occlusion bias to be common-mode across the window being differenced; over
a long carry it drifts as the hand rotates, so a fixed origin would accumulate
that drift and eventually cross the threshold on a perfectly good latch.
Tested with a 60 cm carry under a constant 2 cm bias.

### 1.6 `divergence_gate` is configurable, defaulting to the plan's rule

**The knob most likely to need flipping during bring-up.** Two failure modes
that point in opposite directions:

- `"crop"` (plan's rule, default): if the hand occludes the object badly
  enough that it is never re-detected during a grasp, **no candidate ever
  confirms and the latch is dead weight.** Diagnostic:
  `reason="candidate_timeout"` with `usable_observations == 0`.
- `"mask"`: a memory-propagated mask follows the tracker's belief, which under
  a grasp follows the hand — so it can spuriously **agree** and confirm a
  failed grasp.

Only hardware says which one actually happens. Both settings are tested.

### 1.7 The perception round owns the trackers; `relation.py` is thin

The plan's §5.4 has `relation_perception.observe()` consuming a snapshot. It
does not say where the filtering happens. Put in the round, because the latch's
divergence test reads `snapshot.object_pose_pelvis` and must read *filtered*
poses — raw depth noise would trip it — and because the snapshot should be a
self-contained statement of what perception believes.

Consequence worth stating: **the snapshot carries perception's poses, not
latch-resolved ones.** A snapshot that already contained the latch's own
prediction would make the divergence test compare a value with itself and
always agree. Resolution happens in `relation.py` at packing time.

Threading follows: the control thread drives `on_snapshot` (detecting new
rounds by `seq`) and `on_control_tick`, so the latch state machines are
single-threaded despite spanning two rates.

### 1.8 CANDIDATE reports the tracked pose, not its own prediction

Not specified in §6. A candidate is an unconfirmed hypothesis; reporting its
rigid prediction would feed the policy a hallucinated "the object is in my
hand" relation during exactly the window where that is most likely false —
which is the failure v1's latch was built to prevent.

### 1.9 Prompt normalisation strips GroundingDINO's `" ."`

`task_config.py`'s documented YAML has `detector_prompt: "a red cube ."`. That
trailing `" ."` is GroundingDINO's phrase separator and is noise to SAM 3's
text encoder. **This is a live suspect for §2.4's defect** (one of three
prompts never detects). Stripped with a loud warning rather than rejected,
because the failure it causes is silent — a roster slot simply empty for the
whole episode — and a config copied from the v1 example is exactly how it gets
in. Duplicate prompts across slots are refused outright.

### 1.10 Retired config keys fail loudly

`detector_period_ticks` and `orientation_period_ticks` are not merely
defaulted away — loading a config that sets either raises. An operator who
sets one is expressing an intent the design no longer has a way to obey, and
silently ignoring it means they believe they tuned something they did not.

---

## 2. Optimizations available, not taken

Ordered by value. None is on the critical path; all are recorded so the
measurement that justifies them can be planned.

### 2.1 The 66 ms wire cost is over half the policy round trip

Measured: 124 ms total, 57 ms server-side, **66 ms wire+encode on loopback**
for a ~150 KB payload. That is anomalous by two orders of magnitude and it
feeds straight into `d`. If the existing payload probe (`check latency`) says
*fixed*, it is Nagle/delayed-ACK — one `TCP_NODELAY` on the websocket socket,
and `d` drops from 12 ticks to ~10. If it says *payload-bound*, shrink the
image or drop a copy in `_prepare_image`.

**Best latency-per-effort in the whole system.** §8 step 5.

### 2.2 Orientation resolution — the largest remaining compute lever

`(size/14)²` scaling, and §2.3 establishes rough orientation is sufficient:

| size | 3 crops | saving |
|---|---|---|
| 518 | ~110 ms | — |
| 336 | ~46 ms | 64 ms |
| 252 | ~26 ms | 84 ms |

64 ms off a 221 ms round is ~30%, taking the loop from 4.5 Hz to ~6.4 Hz and
`d` from 12 to 10. `orient.size` is already a config knob and preprocessing is
parameterised faithfully. Needs an accuracy check, not an implementation.

### 2.3 Latched objects skip orientation — already implemented, unmeasured

`OrientAnythingV2.estimate(skip=...)` drops crops for rigidly-held objects.
During single-object manipulation that is ~30 ms per round for free. Wired
through `PerceptionRound(latched_objects=...)`; the saving only appears once
the runner is connected.

### 2.4 bf16 weight cast — the VRAM lever, and the budget does not close without it

Upstream keeps fp32 parameters and relies on internal autocast, which is why a
5 GB checkpoint occupies 10.2 GB resident. §7's target budget (22.3 GB of
23.5 GB) **assumes this cast works.** Implemented behind `orient.cast_weights`,
off by default because the accuracy cost is unmeasured. §8 step 3.

### 2.5 Persistent SGBM worker

`PerceptionRound` holds one `ThreadPoolExecutor` for the life of the rollout
instead of creating one per round. Sub-millisecond, but it was pure churn on
the hot path and it showed in round jitter, which is what `d` is sized from.
**Already taken.**

### 2.6 The running area maximum decays

Not in the plan. Without decay the per-object maximum is a permanent
high-water mark, so an object once close to the camera and since placed
further away is marked unusable **for the rest of the episode**. Decays at
0.995/round — half-life ~30 s at 4.5 Hz. Also: only a re-detected mask may
raise the maximum, since a memory-propagated mask can drift larger while the
object is hidden and would make every later real detection look like a
collapse.

### 2.7 Not worth doing yet

- **SAM 3.1 Object Multiplex** — ~15% at N=3, no `transformers` integration,
  no streaming API. M2 already rules it out; revisit only if tracking
  *accuracy* limits us.
- **Reduced SGBM resolution** — 16 ms on the CPU, fully hidden inside the
  130 ms GPU stage. Free until SAM 3 gets faster than it.
- **Pipelining SAM 3 and orientation** — both saturate the same card, and
  without MPS kernels from two processes are time-sliced, not co-scheduled.
  GPU work is additive; there is nothing to overlap.

---

## 3. Tests that must run on hardware

The 110 CPU tests cover every decision that can silently corrupt a state
vector. They cannot cover the model calls, the memory behaviour, or any
absolute frame convention. In dependency order.

### H1 — the missing third detection (BLOCKING, §8 step 1)

`black pen holder` never detects, run after run. A roster slot empty for the
whole episode blocks everything downstream.

1. Check the task config for a trailing `" ."`. `normalize_prompt` now strips
   it and warns — if the warning fires, that was probably it.
2. Try prompt variants (`"pen holder"`, `"black cylindrical container"`).
3. Check `score_threshold_detection` (0.5) and `new_det_thresh` (0.7).
4. Fix `task_config.example.yaml`, or the `" ."` gets copied into the real
   config again.

**Gate:** 3/3 slots detect reliably across a full episode.

### H2 — prune at 3000 frames (BLOCKING, §8 step 2)

```
uv run --group perception-v2 python -m ego2g1.deploy.perception_v2_latency \
    --prompts "..." --frames 3000 --prune
```

`Sam3Source.prune()` is the same logic the benchmark validated, but the
argument for it being lossless is structural, not measured: memory attention
reads `non_cond_frame_outputs` only for `t-1 … t-(num_maskmem-1)`.

**Gate:** `memory_allocated` (not `max_memory_allocated`) flat at the tail;
non-cond stored entries plateau near `num_maskmem × n_objects`. If it still
climbs, something else in the session retains, and R1 does not close.

Also assert `Sam3Source.num_maskmem` matches the config at runtime — prune
correctness depends on it, and a wrong value deletes entries the model reads.

### H3 — orientation VRAM and quality (§8 step 3)

Sweep `--orient-cast-weights` × `--orient-size {518, 336, 252}`.

**The bf16 cast is assumed to work** (operator decision), so §7's budget
closes and this is a quality spot-check rather than a gate.

**Gate:** orientation on a static object stable to within a few degrees across
sizes. Confirm resident VRAM lands near the predicted ~5.1 GB while you are
there — the assumption is cheap to verify in the same run.

### H4 — the orientation convention: RESOLVED by reading the extraction pipeline

Training uses the **same model** —
`data_extraction_zh/third_party/humanego_runtime/preprocess/OrientAnything.py`,
`pose_method: vlm` (`configs/default.yaml:128`). So the absolute canonical
frame is irrelevant; what matters is reproducing *that file's* decode. That
turns an open question into a diff — and the diff found **three** mismatches,
not one:

| | training | first draft here | now |
|---|---|---|---|
| angle -> matrix | `Rz(ro) @ Rx(el) @ Ry(**+**az)` | `Ry(**-**az)` | fixed, pinned by test |
| background | rembg ON (`do_rm_bkg=True`) | raw crop | SAM 3 mask -> white |
| non-anchor rotation | **constructed**: x toward anchor, y from model | raw model output | `compose_relational_rotation` |

The azimuth sign was a mirrored rotation on every object — a fixed remapping
of what training saw, still a valid rotation, undetectable downstream. Exactly
the failure class this section existed to worry about, found by reading rather
than measuring.

The anchor rule is `anchor_key = obj_keys[0]` (`CamTriangulator.py:197`): the
**first roster entry** keeps the model's rotation, every other slot's is
constructed relative to it. Getting that wrong restructures 2 of 3 slots.
`PerceptionRound` defaults `anchor_id` to `objects[0].instance_id`, so deploy
and extraction agree by construction.

`TRAINING_ANGLES_TO_MATRIX` is a verbatim port kept solely as the reference
the parameterised version is pinned against — 500 random angle triples, exact
agreement. If either drifts, the test fails.

**Still open, both minor and both cheaper to fix by re-extracting than by
reverse-engineering:**

- **Fill colour.** `background_preprocess` composites onto some specific
  background; white is assumed here because upstream `preprocess_images` pads
  to square with white (1.0), so fill and pad agree. Three lines against the
  installed `orient_anything` package settles it.
- **Crop framing.** Training crops a CoTracker-keypoint bbox with a 40 px pad;
  deploy crops the SAM 3 mask bbox with a 15% fractional pad, squared. Neither
  shears — upstream resizes the longest side and pads — but they frame the
  object differently. If it shows as an accuracy gap, re-extract with the
  deploy crop rather than reproduce a keypoint bbox that has no deploy
  analogue.
- **Which object is `obj_keys[0]`?** Confirm it is the roster order the
  checkpoint advertises, not the `instance_counts` dict order.

### H5 — SAM 3 output field names

`Sam3Source._raw_map` looks for `obj_id_to_score`, `obj_id_to_tracker_score`
and `obj_id_to_last_occluded` on both the raw output and the session, because
transformers has moved these across versions. If none is found it logs an
error once and the gate degrades to detection score and mask area.

**Gate:** the error does not fire. Assert `tracker_score` varies across frames
— a constant 0.0 means it silently disabled, and all three S1 gates lose an
input without failing.

### H6 — `--with-policy` contention (§8 step 4)

Today's 221 ms was measured with the policy idle and the 124 ms with
perception idle. **Neither describes the running system.** Without MPS, GPU
work from two processes is additive.

**Gate:** round time and policy p95 under contention; combined VRAM peak under
the cap; resulting `d` still ≤ 12 ticks. Feed the measured `P` and `L` into
`timing.delay_ticks` and check it against `core/latency.budget_for("async")`.

### H7 — end-to-end open-loop (§8 step 7)

Replay recorded stereo through `PerceptionRound.step()` (it takes a clock and
a `read_stereo` callable precisely so this works offline) and compare the
56-dim vector against the training encoding:

```
T_left_tcp_object == compose(invert(left[frame]), object_pose)
```

**Gate:** camera- and pelvis-frame routes agree to numerical precision — they
are algebraically equivalent, but "equivalent" has to be checked (plan Q6).

### H8 — a full grasp through the latch (§8 step 8)

The state machine is thoroughly unit-tested against synthetic trajectories;
what those cannot produce is the real occlusion pattern of a closing BrainCo
hand.

**Gate:** latch engages on a successful grasp and survives the carry; releases
cleanly; does **not** engage on a deliberately failed grasp. Log
`usable_observations` and `stale_s` on every attempt — if candidates time out
with `usable_observations == 0`, flip `divergence_gate` to `"mask"` (§1.6).

### H9 — the parameters nobody has measured

Plan Q10–Q12, all currently bring-up defaults:

- `VisibilityConfig`: `min_det_score`, `min_tracker_score`,
  `min_area_fraction`, `min_area_px`
- `ObjectTracker`: `max_speed_m_s`, `mad_scale`, `history_s`, OneEuro cutoffs
- `LatchConfig`: `confirm_displacement_m`, `position_tol_m`,
  `divergence_sustain`, `max_stale_s`

Set these from a recorded episode, not from a live rollout. Every one is
already a config knob and recorded in `meta.json`.

### H10 — Q2, which could invalidate T2

**How fast do objects move while ungrasped?** Hand-over speed (~0.5 m/s) over
one 300 ms round is 15 cm — that would dominate every other error term and
force object-motion extrapolation back into the design, after S2 just took it
out. Slow sliding needs nothing. Measure before trusting the single-instant
composition on a task that involves handing anything over.

---

## 4. Still open before the runner can be wired

Plan §8 step 9, plus prerequisites. None of it is code that exists yet.

- `DelayBudget.observe()` is still fed **policy latency alone** — it would
  report `d ≈ 4` and silently under-commit by nearly a full perception round.
  `timing.delay_ticks(P, L)` computes the right number and is tested against
  the plan's worked example; nothing calls it.
- The replan path (`wait_for_current_round` → policy → tick-bound chunk) is
  not in `runner.py`. `AsyncPerception` provides the primitive; the loop does
  not use it.
- `startup_self_check` must refuse over-budget modes using `P + L`, not `L`.
- **Q3: which control mode?** `d = 12` against a 15-tick async budget is
  workable but not roomy.
- **Q4: what fills a slot with no detection?** `state_for` currently raises,
  which is right for warm-up and wrong for mid-rollout. H1 makes this live.
- Prerequisites Q17–Q20: `task_config.yaml` does not exist,
  `train_config.objects` unverified, `camera_calib.npz` currency unknown,
  `BRAINCO_CLOSED_POSE` unmeasured.
