# data_extraction — offline SAM 3 + Orient Anything V2 over recorded episodes

Raw material for the capability experiment: for **every frame** of an
egocentric episode, for **every object** in the roster, a mask, a bounding box
and an orientation. The dashboard reads what this writes.

This is not the deploy pipeline. It runs the same weights and reuses the same
slot mapping, gates and angle decode from
[ego2g1/deploy/perception/v2/](../ego2g1/deploy/perception/v2/), but it spends
the offline budget on things the robot cannot have.

```
uv run --group perception-v2 python -m data_extraction.extract \
    --episodes data/raw_hdf5/ego2g1/red_block_in_pen_holder_ego/episode_2.hdf5 \
    --prompts "red block,yellow block,black pen holder" \
    --out-dir data_extraction/out
```

Point `--episodes` at the directory to do all 50; the models load once. **Run
it on the PPU box** — two full SAM 3 passes plus ~1800 orientation crops per
610-frame episode.

---

## What offline buys, and where it lives in the source

Each of these is a real capability difference, verified in
`transformers/models/sam3_video/modeling_sam3_video.py`, not a guess.

| | streaming (deploy) | offline (here) |
|---|---|---|
| **hotstart tracklet removal** | disabled | **on** |
| **hotstart retraction scope** | n/a | **whole video** |
| **direction** | forward only | **forward + reverse** |
| **orientation gate** | `crop_usable` only | **every mask**, gate recorded |
| **orientation batch** | 3 crops (one frame) | **24 crops across frames** |
| **input size** | 336–518 (latency) | 518 |

### 1. Hotstart tracklet removal

`_process_hotstart` guards both of its removal rules with `if not streaming:`.
Streaming keeps every tracklet it ever births; offline deletes tracklets the
detector failed to match for 8 frames, and duplicates that overlap an earlier
tracklet for 8 frames.

`streaming` is derived from `frame is not None` inside `forward` — the
**argument shape selects the behaviour**. Pass `frame=` and you get the
streaming path; pass `frame_idx=` against a session that already holds the
video and you get the offline one. The plan's M3 names this as a cost of
streaming ("duplicates and unmatched tracklets are pruned less aggressively
than offline"); here we just don't pay it.

### 2. Global hotstart retraction — better than upstream's own iterator

`propagate_in_video_iterator` buffers only `hotstart_delay` (15) frames before
yielding, and `postprocess_outputs` filters against `hotstart_removed_obj_ids`
**as it stands when it is called**. So a tracklet retracted at frame 400 still
appears in frames 0–384 of the iterator's output.

Holding the whole pass, we keep each frame's chosen tracklet id and apply the
**final** removed set to every frame afterwards. This is strictly more
filtering than the upstream iterator performs and it is only possible offline.

### 3. Reverse propagation — the reason coverage reaches every frame

`forward(..., reverse=True)` runs memory attention from the end of the video.
This fixes the structural weakness of a causal tracker: an object does not
exist until the detector first fires on it, so every earlier frame is empty —
and in an egocentric recording the interesting object is routinely occluded,
out of frame or motion-blurred at frame 0.

The report prints how many `(frame, slot)` masks the reverse pass contributed
that forward alone did not have. That number **is** the answer to "was the
second pass worth it".

### 4. Orientation on everything, with the gate recorded beside it

Deploy runs Orient Anything only where `crop_usable` (S1), because online
there is no way to find out afterwards whether a sliver crop's confident
answer was wrong. Here we run it on **every mask** and record `crop_usable`,
`mask_usable` and the first failing gate test next to the result.

That makes two questions measurable which are currently unanswerable:

- did the gate throw away a **good** orientation?
- did an orientation it **admitted** turn out to be garbage?

Answering them is how `VisibilityConfig`'s numbers stop being bring-up
defaults (plan Q10).

The only thing still skipped is a mask too small to cut a crop from
(`crop_from_mask` returns `None` below `min_side_px`) — there is no image to
run. That is the stated boundary: **no mask, no orientation**; occlusion alone
never skips a frame.

---

## Three more signals

### Stereo depth — from the episode's own calibration

The recording already carries per-eye intrinsics **and** per-eye SDK
extrinsics, so nothing external needs measuring. [stereo.py](stereo.py) derives
the left→right rigid transform OpenCV wants and then **checks it against
physics** rather than trusting it: a baseline outside 20–200 mm, or eyes more
than 5° apart, refuses to produce depth instead of producing confident nonsense.
On this dataset it derives 64.2 mm and 0.000° — which is itself the proof that
the scalar-last quaternion reading is right.

Each object gets a **median depth over its own mask** (same robust statistic
`join_to_camera` uses — the centroid supplies the direction, the mask supplies
the range) plus a 3D point in the camera frame. `--save-depth-map` also stores
the full per-frame map as uint16 millimetres; it's off by default because it
adds 50–150 MB and the per-object depths are what the depth is *for*.

One trap handled: `StereoSGBMDepthSource.estimate` returns depth on the
**rectified** grid while SAM 3 masks are cut on the **raw** image. Sampling one
with the other is a silent, plausible-looking error. The mask is pushed through
the same maps SGBM used, and the back-projected point is rotated by `R1.T` back
into the raw left frame, so everything in the file shares one frame. (Deploy's
`PerceptionRound` does *not* do this — it has only ever run against the e2e
bench's identity placeholder calibration, where rectification is a no-op. Here
it's measured: `rectify_shift_px` is in the metadata.)

### Presence — "is this concept in the frame at all?"

SAM 3's detector has a dedicated presence token, and `Sam3DetectorOutput`
documents the relationship: `final_scores = pred_logits.sigmoid() *
presence_logits.sigmoid()`. `run_detection` computes exactly that and then
**throws the presence factor away** — only the product survives into
`obj_id_to_score`. So a low score cannot distinguish *"the concept is absent"*
from *"it's here but nothing localised it"*, and those have opposite fixes.

Captured with a forward hook on the detector (no upstream code reimplemented).
It is **per prompt, not per instance**, so it exists even on frames where
nothing was tracked — which is the case it's for. When a roster slot never
fills (the plan's §2.4 defect), the report now says which of the two problems
it is:

```
[FAIL] slot(s) ['obj2'] never produced a mask in EITHER direction.
  obj2: max presence 0.031 — SAM 3 never sees the concept at all;
        the PROMPT is wrong for this scene, not the thresholds.
```

### Symmetry order α — how many ways the object looks the same

Not a network head: upstream's `val_fit_alpha` fits von Mises densities to the
**azimuth distribution** (the 360-bin sigmoid, not its argmax) for
α ∈ {1, 2, 4}, then applies a confidence floor. So it needs the distribution,
which is why `angles_for_crops` can now return it — no second forward.

```
1  one unambiguous front (a mug with a handle)
2  two-fold — 180° apart is the same (a book, a box)
4  four-fold — 90° apart is the same (a cube, a plain holder)
0  NO CONFIDENT CALL — a real answer, not a missing one
-1 not measured (the fit was unavailable or disabled)
```

**0 and -1 are different and must stay different.** This is the measurement
plan Q8 asks for, and §5.2 defers online because the fit is a scipy `curve_fit`
per crop and the group has to be frozen at seed anyway. Offline there's no
budget and no reason to freeze: running it every frame turns the question into
data. A stable non-zero α is a trustworthy symmetry group; an α that flickers
between 1 and 2 means the model won't commit, and every rotation on those
frames is worth less than its confidence suggests. `--no-fit-symmetry` skips it.

---

## Output

One `.h5` per episode plus a `.meta.json` sidecar. Full layout is documented at
the top of [store.py](store.py). Per roster slot, all length-F:

```
mask (F,H,W) uint8 gzip   box_xyxy   det_score   presence   tracker_score
mask_area_px   occluded   source   mask_usable   crop_usable   gate_reason
azimuth_deg   elevation_deg   roll_deg   R_cam (F,3,3)   orient_skip   alpha
depth_m   point_cam (F,3)   depth_px
```

Four conventions every consumer must respect:

- **`det_score` NaN means the detector did not re-find the object** — the mask
  is memory propagation and the crop is a guess. It is not a missing
  measurement, it is the S1 signal. Never read it as zero.
- **`presence` NaN** means only that the probe couldn't read it. Presence is
  per-prompt and legitimately present on frames with no mask.
- **`alpha` 0 vs −1** — 0 is "the fit declined", −1 is "the fit never ran".
- **`source`** says which pass won the frame (`0` none, `1` forward, `2`
  reverse, `3`/`4` both). `2` marks frames deploy provably cannot have.

Masks gzip to roughly 1–2 MB per episode (from 560 MB raw) and are chunked one
frame deep, so `mask[i]` is a single chunk read.

---

## The dashboard

```
uv run python -m data_extraction.dashboard \
    --extraction data_extraction/out/episode_2.h5
```

Writes `episode_2_dashboard.html` beside it — one self-contained file, no
server, no network. Needs only numpy/h5py/Pillow, **not** the `perception-v2`
group, so it runs on a laptop against files the PPU box produced. ~14 MB for a
610-frame episode at the default `--scale 0.5`.

A frame player with the overlays drawn on canvas, plus per-frame state strips.

### The centre pixel is a plain unweighted mean

`SlotObservation.centroid_uv` is `xs.mean(), ys.mean()` over the mask's
non-zero pixels — not the bbox centre, not a median, not a distance transform.
The dashboard builds a real `SlotObservation` and calls that method, so the dot
on screen is the pixel the deploy loop back-projects. It is computed at full
resolution and scaled afterwards, so `--scale` never moves it.

The **bbox centre** is drawn too, as a hollow dashed marker. The gap between
the two is the occlusion bias `latch.py` warns about: under partial occlusion
the mean is pulled toward whatever is still uncovered, and that pull moves as
the hand rotates. Seeing them separate is how you tell an occluded frame from
a moving object.

### The rotation: columns of R, drawn from the centroid

`R = Rz(roll) @ Rx(elevation) @ Ry(azimuth)` about camera axes (OpenCV: X
right, Y down, Z forward), pinned by test to the training decode.

**R is `R_camera←object`** — its *columns* are the object's canonical basis
vectors in camera coordinates, so drawing column *i* from the centroid draws
axis *i* of the object's own frame. Two independent confirmations in the deploy
code: `compose_relational_rotation` reads `R_model[:, 1]` as a camera-frame
direction and returns `stack([x, y, z], axis=1)`, and the snapshot uses R as
the rotation block of `T_camera_object`.

**…and the convention is unvalidated.** `angles_to_matrix`'s own docstring
flags the axis assignment and all three signs as a reasoned default that has
never been measured, and plan Q6 asks whether it matches the frame the training
labels used at all. So transpose and per-angle sign flips are **live controls**
in the page: flip until the triad stays glued to the object as it turns, and
you have measured it. The page prints the resulting `convention:` block for
pasting into the perception config, and self-checks its own JS port of the
decode against the matrices stored in the file.

Two display facts to keep in mind:

- Axes are **orthographic** — endpoint = centroid + L·(dₓ, d_y). This extraction
  is monocular, so there is no depth and no perspective. Direction and
  foreshortening are right; apparent length is not metric.
- An axis pointing along the view ray has no in-plane direction, which is the
  *common* case, not an edge case. Those render as a surveyor's glyph — ⊙
  toward the camera, ⊗ away — instead of vanishing under the centroid dot.
- X/Y/Z keep the red/green/blue convention, but red↔green is the worst pair for
  deuteranopia, so **every axis is also labelled with its letter**.

### Controls

Play/scrub/step (`space`, `←`/`→`, `shift` for ×10, `home`/`end`), speed, loop.
Toggles for mask outline, mask fill, bbox, mean centroid, bbox centre, axes,
labels, axis length, and "only `crop_usable`" — that last one shows exactly
what the deploy loop would have drawn.

The panel also reads out per-frame **presence**, **median depth + 3D point**,
and **symmetry α** with its meaning spelled out.

The strips below are one row per object, switchable between **gate quality**
(no mask / memory propagation / detected-but-crop-rejected / `crop_usable`),
**which SAM 3 pass** (none / forward / reverse / both), and **presence**. Click
or drag to seek. The reverse-only frames are the ones the online loop provably
cannot have; a bright presence row above an empty quality row is an object that
is visibly there and was never localised.

---

## Notes and limits

- **`compose_relational_rotation` is NOT applied.** Training gives only the
  anchor object the model's raw rotation and constructs every other object's
  from the direction to the anchor — which needs camera-frame 3D translations,
  which need depth this monocular pass does not produce. What is written is
  the **raw** model rotation per slot. The relational form is a pure function
  of (raw rotation, own position, anchor position) and can be applied later
  without re-running anything; the anchor id is in the metadata.
- **Roster order is load-bearing.** `--prompts` order defines the slots, and
  `obj0` is the anchor. Episode attrs (`object_prompts_json`) are a fallback
  only, and are usually coarser than the task — these episodes list two
  objects there and three in the instruction.
- **Prompts from the recorder end in `" ."`**, GroundingDINO's phrase
  separator, wrong for SAM 3. `normalize_prompt` strips it and warns.
- **`kernels` is still absent** (deploy's M4), so SAM 3 skips NMS, hole filling
  and sprinkle removal here too — masks keep holes. Unlike deploy, there is no
  reason it must stay absent: nothing in this directory imports `ego2g1.serve`.
  Installing it in a PPU-only venv is the cheapest available mask-quality win
  and would make this extraction strictly better than the deploy path.
- **Memory bank pruning** uses the corrected horizon
  `max(num_maskmem, max_object_pointers_in_encoder)` = 16, not `num_maskmem` =
  7 — see `Sam3Source.prune`. Mirrored for the reverse pass, where the live
  window is *above* the cursor.
- Host RAM holds the preprocessed video at ~6 MB/frame (`--max-frames` to cap).

Tests: `tests/data_extraction/`, CPU-only, no weights.
