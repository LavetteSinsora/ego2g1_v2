# Relational-policy serve/deploy: implementation plan

Status: **approved decisions locked below; ready for staged implementation.**
Companion docs: [deploy.md](deploy.md) (existing joint/relative_eef deploy loop —
unchanged by this plan), [datasets.md](datasets.md), [robot.md](robot.md),
`../TRAINING_PLAN.md`-style prior art at the training side
(`ego2g1/train/config.py`'s `EgoRelationTrainConfig`).

## 0. What changed and why this doc exists

A checkpoint trained with `EgoRelationTrainConfig`
(`ego2g1/train/config.py:247`) speaks a completely different interface than
every existing deploy path:

| | old (`Ego2G1TrainConfig`, `relative_eef`) | new (`EgoRelationTrainConfig`) |
|---|---|---|
| state in | 30-dim absolute FK proprioception, digitized into the prompt | **no proprioception at all** — 56-dim object-in-hand relation vectors + 2 grasp binaries, injected as learned prompt tokens |
| action out | (H,30): per hand vec9 (6D rotation) + 6-dim continuous Revo2 command | (H,14): per hand [3 translation + 3 **rotation-vector**] + **1 binary** gripper dim |
| hand | Revo2, 6 continuous motors/hand | **BrainCo**, binary open/close only |
| perception | none (proprioceptive only) | **live object detection + depth + hand-relative geometry**, every tick |

`ego2g1/serve` and `ego2g1/deploy` were not touched by the training-side
work (verified: `git log --stat` on the two relational-training commits
touches only `ego2g1/{core,train}/*`). This is 100% new work. Everything
below is scoped to make it real, in a way that composes with the existing
mode-blind deploy architecture rather than forking it.

**Guiding fact that simplifies everything downstream**: both the model's
input (object pose in hand-TCP frame) and output (pose delta from the
current TCP anchor) are *relative* quantities — `inv(T_a) @ T_b` — which
are invariant to whatever absolute parent frame `T_a`/`T_b` were expressed
in, as long as both sides of one relation share the same parent frame at
that instant. This is confirmed by the dataset's own action contract,
which describes execution as `T_g1[t+1] = T_g1[t] @ delta_T[t]` — compose
onto the robot's *own* measured anchor, never trust an absolute training
frame. Deploy therefore never needs to reproduce the training pipeline's
Pico/camera0 geometry; it only needs **internal self-consistency at each
tick** (object pose and hand pose in one common, robot-fixed frame) plus
**one thing that does NOT cancel out: rotational-convention agreement**
between training's synthetic "TCP" orientation and the robot's actual
flange orientation. That single open risk is called out explicitly in
§4 and gated by a bring-up rung, not assumed silently.

## 1. Decisions locked (from discussion)

1. **TCP/flange rotation correspondence**: assume `TCP_TO_INWARD_PALM`'s
   convention already matches the BrainCo/flange mount; validate with a new
   bring-up rung before trusting it on hardware (§4.4). Do not build a
   from-scratch measured calibration up front.
2. **Perception architecture**: tiered cascade — detector ~2 Hz, orientation
   refresh ~0.2 Hz, fast tracker ~20-30 Hz, **latch only after a
   motion-consistency confirmation window**, not on the grasp bit alone
   (§5.4 — this is the refinement you asked for).
3. **Camera extrinsics**: build a lightweight **touch calibration** tool
   that reuses the perception pipeline's own detector against FK ground
   truth (§6), not a formal AprilTag hand-eye rig. CAD/datasheet numbers are
   a bootstrap only, never trusted as final (§6.1 explains why).
4. **Staging**: Phase 1 (serve fix + offline replay validation, no camera/
   DINO needed) → Phase 2 (live perception + calibration tooling) → Phase 3
   (hardware bring-up rungs), each gated by tests before the next starts.
   Implementation to be parallelized across subagents once you confirm this
   doc; §9 is the literal task list.

## 2. Current-state map (context subagents need, so they don't re-derive it)

- `ego2g1/serve/policy.py`: `config_from_stamp` hardcodes
  `_config.Ego2G1TrainConfig(**cfg_dict)` (line ~68); `create_policy`
  always calls `_data_config.create_data_config` (the old 30-dim stack).
  Neither knows `EgoRelationTrainConfig` exists.
- `ego2g1/train/stamp.py`'s `SUPPORTED_FEATURES` **already** lists every
  relational feature flag (`action_norm_scheme`, `relation_state`,
  `relative_eef_rotvec_actions`, `binary_gripper`, ...) — the stamp guard
  is ready; only the config *reconstruction* is not.
- `ego2g1/deploy/actions.py` + `policy_adapter.py`: a clean, already-proven
  boundary. Whatever mode is added, `executor.py`/`strategies.py`/
  `runner.py`'s core loop stay mode-blind — they only ever see `(H,26)`
  absolute joint rows out of `actions.py`'s converters. This is the
  extension point; do not touch the executor.
- `ego2g1/deploy/kinematics.py`'s `Kinematics.flange_poses(arm_q14)` already
  gives FK flange pose **in the pelvis frame** from measured joints — this
  is the "robot's own anchor" the new adapter composes deltas onto, exactly
  like `RelativeEEFPolicyAdapter` does today for the old 30-dim mode.
- Training's `ego2g1/train/relation_transforms.py` is the single source of
  truth for exact math (`RelativeEEFRotvecActions`, `RelationPrompt`,
  `RelationInputs`) — every deploy-side transform must be the *inverse* or
  *analogue* of something already in that file. Re-read it before
  implementing §3/§4; do not re-derive the math from scratch.
- `ego2g1/core/rotvec.py` (Rodrigues log/exp, pure numpy) already exists
  from the training work — reuse it verbatim, do not reimplement.
- `ego2g1/kin/g1.py`'s MJCF (`assets/unitree_g1/scene_fixed_base.xml`) has
  a `head_link` body (`pos="0.0039635 0 -0.044"` relative to its parent —
  this is the exact offset the training session's plan quoted) but **no
  camera site**. Waist is pinned at 0 in deployment, so `head_link`'s pose
  relative to pelvis is constant and FK-computable the same way
  `flange_pose`/`base_pose` already work — a `head_pose()` method is a
  small, low-risk addition to `G1Backend`.
- `ego2g1/deploy/camera.py`'s `HeadCamera` reads only RGB
  (`cam_{eye}_high`, split from one wide stereo frame) via
  `unitree_deploy`'s `ImageClientCamera` — **no depth channel is wired
  today**. See §5.2's open question — this needs one fact from you.

## 3. New action mode: `relation_eef` (deploy-side)

Mirrors the existing `relative_eef` mode's shape (§ old `RelativeEEFChunks`/
`RelativeEEFPolicyAdapter` in `actions.py`/`policy_adapter.py`), swapping
the pose encoding and the gripper handling.

### 3.1 `ego2g1/core/relation_layout.py` (new, tiny)

Analogue of `core/layout.py` for the 14-dim action / 56-dim relation state.
Pure constants + slice helpers, no framework imports (matches `layout.py`'s
own numpy-only style):

```python
HANDS = ("left", "right")
EEF6_DIM = 6          # [dx,dy,dz,rx,ry,rz]
ACTION_DIM = 14        # 2 * (6 + 1), grippers at the TAIL
EEF6 = {h: slice(i*6, i*6+6) for i, h in enumerate(HANDS)}
GRIP = {h: slice(12+i, 12+i+1) for i, h in enumerate(HANDS)}  # matches
    # EgoRelationTrainConfig.gripper_dims: tuple(range(6*len(hands), 6*len(hands)+len(hands)))
RELATION_DIM_PER_OBJECT = 18   # 9 (left-TCP-frame) + 9 (right-TCP-frame)
```

Note the action layout is **not** the same as `core/layout.py`'s 30-dim one
— do not reuse `layout.EEF`/`layout.HAND` for this mode.

### 3.2 `ego2g1/deploy/actions.py`: `RelativeEEFRotvecChunks`

New class, same shape of contract as `RelativeEEFChunks`:

```python
class RelativeEEFRotvecChunks:
    mode = "relation_eef"

    def __init__(self, kin=None, *, fps=30, ik_iters=25, posture_cost=0.05,
                 collision_min_dist=0.005, one_euro_kwargs=None,
                 closed_pose: dict[str, np.ndarray] | None = None):
        # same Kinematics wiring as RelativeEEFChunks (reuse, don't fork)
        # closed_pose: {"left": (6,), "right": (6,)} — see §7

    def convert(self, actions, arm_q14, hand_cmds: dict) -> np.ndarray:
        # actions: (H, 14) from the server, gripper dims RAW model-space
        # (server does NOT unnormalize them — PerSlotQuantizeActionsInverse
        # exempts gripper_dims by construction; verify this against
        # ego2g1/train/relation_transforms.py before assuming it, don't
        # trust this comment blindly)
        #
        # anchor = self.kin.flange_poses(arm_q14)   # pelvis frame, per hand
        # self.kin.ground(arm_q14)
        # for each row k, each hand h:
        #     delta_T = se3.compose(se3.se3_from_rot(rotvec.rotvec_to_mat(row[rx:rz])),
        #                            translation=row[dx:dz])   # local delta
        #     target = anchor[h] @ delta_T
        #     target = OneEuroSE3.filter(target, dt)           # unchanged pattern
        # out[:, ARM] = self.kin.solve(targets)                 # unchanged
        # frac = np.clip((row[grip] + 1.0) / 2.0, 0.0, 1.0)     # -1/+1 -> [0,1]
        # out[:, HAND[h]] = frac * self.closed_pose[h]          # see §7
```

Reuse `Kinematics`, `OneEuroSE3`, `JointFilter`, `DualArmIK` completely
unchanged — only the *target pose construction* differs (rotvec decode
instead of vec9/6D decode) and the *gripper expansion* (binary→6, not a
passthrough clip). Do not copy `RelativeEEFChunks`'s body; import and
reuse `core.rotvec.rotvec_to_mat`.

### 3.3 `ego2g1/deploy/policy_adapter.py`: `RelationPolicyAdapter`

Analogue of `RelativeEEFPolicyAdapter`, but the observation it builds is
**not** proprioceptive FK state — it is the live relation state from a new
perception module (§5):

```python
class RelationPolicyAdapter:
    mode = "relation_eef"

    def __init__(self, client, prompt="", *, perception, converter=None, kin=None, ...):
        self._perception = perception   # §5's RelationPerception instance

    def infer(self, request: dict) -> dict:
        arm_q = request["arm_q"]
        hand_cmds = request["hand_cmds"]          # last commanded, scalar per hand now
        flange = self._kin.flange_poses(arm_q)     # pelvis frame anchor, same as today
        percept = self._perception.observe(request["image"], flange, hand_cmds)
        # percept: {"state": (56,) float32 hand-major relation vector,
        #           "objects_visible": {...}, "latch": {...}}  (§5 return contract)
        out = self._client.infer(request["image"], percept["state"],
                                  request.get("prompt", self.prompt))
        out["actions"] = self._converter.convert(out["actions"], arm_q, hand_cmds)
        out["percept"] = percept   # surfaced for the recorder — do not drop silently,
                                    # a bad relation state is exactly as undiagnosable
                                    # after the fact as a bad served chunk was pre-safety-layer
        return out
```

**Important, precise wire-format fact** (re-verify against
`relation_transforms.py` before coding, this is the load-bearing detail):
`RelationPrompt` expects `observation/state` as the **raw 56-dim hand-major
concatenation** — `[left→obj0(9), left→obj1(9), left→obj2(9),
right→obj0(9), right→obj1(9), right→obj2(9), grasp_left, grasp_right]`,
in the **training config's fixed object order** (`objects`/
`object_prompt_names`, e.g. `("pen holder","red cube","yellow cube")` for
this checkpoint) — **not shuffled** at serve time (serve must build
`create_relation_data_config(..., shuffle_objects=False)`, see §4.2). All
z-scoring, per-object regrouping, and prompt-text construction (including
the open/closed words and the `<unused0>` object sentinels) happen
**server-side**. The deploy client sends raw geometry + a bare task string
only — it does not need to replicate any of the encoder-side transforms.
This mirrors exactly how the existing 30-dim client already works (send
raw state, let the server's transform stack do everything), so no new
wire-protocol concept is needed, only a new payload shape.

`observation/action_reference_tcp` (18-dim) is **not required for plain
inference** — it only matters for RTC prefix construction, which this plan
defers (§8, out of scope for v1; `EgoRelationTrainConfig.rtc_training =
False` anyway).

## 4. Serve-side fix (`ego2g1/serve/`)

### 4.1 `ego2g1/train/stamp.py`

Add a `config_class` field written by `write_stamp`:

```python
stamp = {
    ...,
    "config_class": type(train_config).__name__,  # "Ego2G1TrainConfig" | "EgoRelationTrainConfig"
}
```

Backward compatible: old checkpoints have no such key; `config_from_stamp`
must default missing key to `"Ego2G1TrainConfig"` (never guess toward the
newer class). This generalizes cleanly to any future third config family —
do this properly once rather than special-casing two classes forever.

### 4.2 `ego2g1/serve/policy.py`

- `config_from_stamp` dispatches on `stamp.get("config_class",
  "Ego2G1TrainConfig")` to build either `_config.Ego2G1TrainConfig(**cfg)`
  or `_config.EgoRelationTrainConfig(**cfg)`.
- `create_policy` branches on `isinstance(train_config,
  _config.EgoRelationTrainConfig)`:
  - build the model via `train_config.model_config()` (already relation-
    aware — `n_objects`/`relation_dim`/`grasp_head` all come from the
    config, no changes needed in `ego2g1/train/model.py`);
  - resolve relation norm stats (`relation_stats.npz`, `ego2g1/train/
    norm.py`'s `RELATION_FILENAME`/`load_relation`) via a new
    `resolve_relation_norm_assets(...)`, parallel to the existing
    `resolve_norm_assets` (same checkpoint→run→training-assets search
    order, same "falls back to training assets, WARN" behavior — copy the
    *pattern*, not the pooled/per_slot-specific code);
  - call `_data_config.create_relation_data_config(train_config,
    model_config, stats_dir=..., shuffle_objects=False)` (note the
    explicit `shuffle_objects=False` — serving must be deterministic; only
    training shuffles) instead of `create_data_config`;
  - build `transforms=[InjectDefaultPrompt(...), *data_cfg.data_transforms
    .inputs, *data_cfg.model_transforms.inputs]` — **no `Normalize` step**
    for this branch (`create_relation_data_config` already returns
    `norm_stats={}`, and the real normalization is baked into
    `model_transforms.inputs` as `NormalizeRelations`+
    `PerSlotQuantizeActions`) — do not insert `transforms.Normalize`
    here, it would be a silent no-op that looks intentional but papers
    over a missing step if the data_config ever changes.
  - `Ego2G1Policy` itself (the RTC wrapper) can likely be reused as-is
    for the plain (non-RTC) path — verify `infer()`'s plain branch
    (`sampler is _rtc.Sampler.PLAIN`) truly doesn't touch anything 30-dim-
    specific (it doesn't, from inspection — the RTC-prefix branch is the
    only dimension-coupled code, and it's simply unreachable while no
    `prev_chunk` is ever sent).
  - extend the `metadata["ego2g1"]` block with everything the deploy
    client needs to auto-configure instead of hand-typing object order:
    `"objects": list(train_config.objects)`,
    `"object_prompt_names": list(train_config.object_prompt_names)`,
    `"n_objects": train_config.n_objects`, and
    `"control_mode": "relation_eef"` (a new value; the existing client
    reads `control_mode` off the handshake already — extend, don't
    special-case).

### 4.3 New CLI ergonomics

No new entrypoint needed — `python -m ego2g1.serve --checkpoint ...`
should just work once `create_policy` dispatches correctly, for *either*
checkpoint family, from either the PPU box or a plain NVIDIA machine.

### 4.4 New bring-up rung: TCP-orientation sanity check

Add `python -m ego2g1.deploy.check tcp-orientation` (new subcommand in
`ego2g1/deploy/check.py`, alongside the existing `fk`/`ik`/`camera`/
`hand-sweep` rungs): drive each arm through a handful of known joint
configurations, at each one print the FK flange orientation *as the
`TCP_TO_INWARD_PALM`-style convention would encode it* next to a plain-
English description ("+X should point out of the palm, +Z should point
toward the fingertips" or whatever the convention actually says once
re-read from `data_extraction_zh/src/ego_relation/s1_pico_mode2/tcp.py`) —
a human eyeballs it once, on video or in the MuJoCo replay viewer, before
any policy chunk reaches the real arm. This is the validation half of
decision #1 in §1: cheap, human-in-the-loop, catches a 90°-family
rotation error immediately (which is the failure mode a fixed-axis-
convention bug produces) without needing a measured calibration rig.

## 5. Live perception pipeline (new: `ego2g1/deploy/perception/`)

New subpackage, imported lazily (mirrors `kinematics.py`'s "mujoco/mink
imported only in `__init__`" discipline — a `joint`-mode or
`relative_eef`-mode deploy must never pay for DINO/torch-vision imports).

### 5.1 Object prompt configuration

New `ego2g1/deploy/perception/task_config.py`:

```python
@dataclasses.dataclass(frozen=True)
class ObjectSpec:
    instance_id: str
    category: str            # matches train_config.objects[i], POSITIONAL
    detector_prompt: str      # what we feed to the live detector, e.g. "a red cube ."
    graspable: bool = True

@dataclasses.dataclass(frozen=True)
class DeployTaskConfig:
    objects: tuple[ObjectSpec, ...]   # order MUST match the checkpoint's train_config.objects
    hands: tuple[str, ...] = ("left", "right")
```

Loaded from a small YAML (mirrors `data_extraction_zh/configs/default.yaml`'s
`task.objects` block — same shape, deliberately, so a task config can be
ported by copy-paste rather than reinvented) OR, preferably, **cross-
checked against the server's own `object_prompt_names`/`objects` metadata
at connect time** (§4.2) — refuse to start if the operator's local task
config's object order/count doesn't match the checkpoint's, the same
"fail loud before it can mis-serve" philosophy as `stamp.check_supported`.

### 5.2 Depth source — **resolved: StereoSGBM on the existing RGB pair, no native depth available**

Researched directly: `third_party/xr_teleoperate` (the sibling `image_server`
source) is Unitree's own official repo. Its `cam_config_server.yaml` and
Unitree's own published G1-D spec sheet both confirm the head module is a
**passive stereo-RGB "HD Binocular Camera"** (generic module, 125° FOV,
60 mm baseline per datasheet) — **not** a RealSense or any active-depth
sensor; that's an optional *wrist* accessory in Unitree's own docs, not
what's configured here (this project's wrist cameras are also `type: uvc`).
`image_server.py` does have a `RealSenseCamera` class with a
`get_depth_frame()` method, but it's dead code end-to-end even if the
hardware were swapped: `ImageServer` never passes `enable_depth=True` to
it, no config file exposes that knob, and even when populated the ZMQ/
WebRTC publish calls only ever push color bytes — depth is never put on
the wire to any client, regardless of hardware. Getting native depth would
require hardware replacement plus non-trivial patches to Unitree's own
server, not a config flip.

**Decision: build `DepthSource` around OpenCV `StereoSGBM` on the existing
`cam_left_high`/`cam_right_high` rectified pair** — the same algorithm
`data_extraction_zh/src/ego_relation/s1_pico_mode2/stereo_depth.py`
already uses on the Pico's own stereo pair — as the *only* implementation,
not a placeholder-pending-something-better. Keep it behind a `DepthSource`
interface (`estimate(rgb_left, rgb_right) -> depth_map`) so a future
hardware change is still a swap, not a rewrite, but do not build a second
implementation speculatively.

**New prerequisite this surfaces**: there is no stereo calibration data
(K matrix, baseline, rectification maps) anywhere in either repo for this
camera — the "60 mm baseline" is a datasheet nominal, not a per-unit
calibration. `StereoSGBM` needs real calibration to produce metric depth.
Add a standard checkerboard/ChArUco stereo calibration pass as a
prerequisite step in Phase 2, before the touch calibration in §6 (touch
calibration solves camera→pelvis *extrinsics* given already-metric object
positions; it cannot fix a wrong stereo *intrinsic* calibration
upstream of that). This is ordinary, well-understood tooling
(`cv2.stereoCalibrate`) — not a new research problem, just a step that was
missing from the original task list and is added here as task 6b (§9).

### 5.3 Detector cascade

`ego2g1/deploy/perception/detector.py` / `tracker.py` / `orientation.py`,
implementing the tiered design already sketched in the prior training
session (never built, now the target):

- **Detector, ~2 Hz**: GroundingDINO + SAM2 over `DeployTaskConfig`'s
  `detector_prompt`s (same models `data_extraction_zh` uses — reuse the
  weights/wrapper from `data_extraction_zh/third_party/` rather than
  re-vendoring; check licensing/packaging before assuming a straight
  import works across repos, `data_extraction_zh` is its own uv project).
  Produces per-object 2D mask + centroid.
- **Depth lift**: mask centroid (or masked-region median, more robust to
  mask-boundary noise) → 3D point in camera-optical frame via depth +
  intrinsics.
- **Orientation, ~0.2 Hz**: reuse `data_extraction_zh`'s cube-symmetry
  snapping (`s2_object_relations/stereo_fusion.py`'s
  `_nearest_symmetric_rotation` — cheap, no VLM call, appropriate for this
  task's known cube/holder geometry) rather than the training pipeline's
  VLM-based `pose_method: vlm`, which is explicitly flagged there as not
  real-time. If a future task needs free-form orientation, that's the
  fallback to reach for, not the default.
- **Fast tracker, ~20-30 Hz**: between detector refreshes, track each
  object's *position* by re-projecting last-known 3D point + local optical
  flow or a simple 3D Kalman constant-velocity filter — do not run the
  detector every tick, that's the whole point of the cascade. Smooth the
  resulting per-object SE(3) with `ego2g1.kin.filters.OneEuroSE3` (already
  in this repo, already used for target smoothing in `actions.py`) rather
  than inventing a new filter.
- All stages reference `data_extraction_zh/src/ego_relation/s1_pico_mode2/
  smoothing.py`'s outlier-repair *technique* (residual-from-midpoint
  detection, robust MAD threshold) as prior art for jump rejection — but
  that code is offline/acausal (uses future frames); the live tracker
  needs a **causal** analogue (e.g. reject-and-hold-last-good-estimate
  when a new detection's residual from the OneEuro-filtered trajectory
  exceeds a robust threshold, instead of Savitzky-Golay repair).

### 5.4 Grasp confirmation / kinematic latching — the refinement you flagged

Training's `latch_object_poses`
(`data_extraction_zh/src/ego_relation/s2_object_relations/encoding.py:167`)
latches on the grasp bit alone (distance-gated), which is fine offline
because the human demonstrator's grasp signal is fairly reliable and
episodes are curated. **At deployment this is exactly backwards**: a
policy-commanded "closed" hand frequently does *not* mean the object was
actually picked up (missed grasp, object slipped, wrong approach), and
feeding the model a hallucinated "object rigidly follows the hand" relation
when the object is actually still sitting on the table is a direct route
to erratic, confidently-wrong behavior — worse than not latching at all.

Proposed state machine, per hand, in `ego2g1/deploy/perception/latch.py`:

```
UNLATCHED  -- hand closes AND nearest graspable object within
              latch_distance_m (same threshold class as training) -->
CANDIDATE  -- for a short confirmation window (e.g. 0.3-0.5 s / ~10-15
              ticks at 30 Hz), keep running the detector/tracker on the
              candidate object as if unlatched, AND compute what its pose
              WOULD be if rigidly attached to the hand (T_tcp_object frozen
              at candidate-entry) -- compare the two:
                agree  (tracked and rigid-predicted pose stay within a
                        tight position/rotation tolerance, i.e. the
                        object's motion CONVERGES with the hand's) -->
                    LATCHED: trust the rigid prediction from here,
                    stop depending on (noisy, possibly occluded-by-the-
                    hand-itself) live tracking for this object
                diverge (tracked pose keeps moving independently of the
                        hand, or tracking is lost/occluded without ever
                        agreeing) -->
                    back to UNLATCHED: treat the grasp as a MISS -- keep
                    using live-tracked (or last-known, held) pose, and
                    critically, do NOT silently keep reporting an object
                    pose that pretends success
LATCHED    -- hand opens --> UNLATCHED (release), object pose resumes
              from wherever the rigid prediction left it (re-acquire via
              detector on the next ~2 Hz cycle rather than assuming it
              fell exactly in place)
```

This directly implements the "continuously detect both object and hand
poses and check if their movement converges or diverges" idea — it is a
confirmation gate on top of the existing distance+grasp-bit heuristic, not
a replacement for it (the distance/grasp-bit test still decides *which*
object is a latch candidate; the convergence test decides whether the
latch is *real*). Surface the latch state in `percept["latch"]` (per §3.3)
so the recorder captures it — an unconfirmed/missed grasp is exactly the
kind of event that needs to be visible in post-hoc session review, same
principle as the existing tracking-error/clamp telemetry.

### 5.5 `RelationPerception.observe(...)` — the module's public contract

```python
class RelationPerception:
    def __init__(self, task_config: DeployTaskConfig, camera_intrinsics,
                 T_pelvis_camera, *, depth_source, detector, tracker, latch):
        ...

    def observe(self, image, flange_poses: dict, hand_cmds_last: dict) -> dict:
        """flange_poses: {hand: (4,4)} PELVIS frame, from Kinematics.flange_poses.
        Returns {"state": (56,) float32 hand-major relation vector matching
                 RelationPrompt's expected layout (§3.3), "objects": {...
                 per-object debug pose/visibility...}, "latch": {hand: state}}.
        Internally: lift each tracked object's pose into PELVIS frame via
        T_pelvis_camera (the ONE new calibration, §6), then relation[obj] =
        compose(invert(flange_poses[h]), object_pose_pelvis) per hand --
        this is the exact inverse of encoding.py's
        `T_left_tcp_object = compose(invert(left[frame]), object_pose)`."""
```

Grasp binaries in the returned state are simply `hand_cmds_last[h]`
thresholded/rounded — no new logic, reuse what `runner.py` already tracks
as "last commanded" per hand (generalized from a (6,) vector to a scalar
for this mode).

## 6. Camera-to-robot calibration ("touch calibration")

### 6.1 Answering directly: what does this calibration solve, and why not just use a spec sheet?

The thing we need is `T_pelvis_camera`: the camera's position **and**
orientation relative to the pelvis/FK chain, as one rigid 4×4 transform.
This is what lets us take "object detected at pixel (u,v), depth d" and
turn it into "object is at this position in the same frame my FK flange
pose is in" — without it, the relation vectors fed to the policy are
built from two unrelated coordinate systems and are simply wrong.

A camera's product datasheet gives you **intrinsics** (focal length,
principal point, distortion) — those are a property of the lens/sensor
alone and are genuinely fine to take from a spec sheet or the vendor's own
calibration file. **Extrinsics** (where the camera sits relative to
*this specific robot's* kinematic chain) are not a lens property — they
depend on how the physical housing was mounted onto *this* head shell,
which is not something a component datasheet can tell you, for three
concrete reasons that matter here specifically:

1. Assembly tolerance: even a "designed to spec" CAD mounting point is
   typically only accurate to a few mm / ~1° in practice — that sounds
   small, but at ~0.3–0.6 m arm reach, a 1° orientation error alone
   already produces ~5–10 mm of object-position error, and the errors
   compound (translation error + rotation error × reach).
2. This is not hypothetical for this project: `data_extraction_zh`'s own
   measured accuracy floor, from real captured data, is an 11.5 mm
   *median* (up to 101 mm) left/right object-pose disagreement, on a
   system whose own config explicitly flags `calibration_verified: false`.
   A deployed camera mount with no calibration at all is likely to be
   worse than that, not better.
3. `TCP_TO_INWARD_PALM`-class fixed rotations already established in this
   codebase (and `b_calib.npz`'s measured `B` for the sibling pipeline)
   exist precisely because "what the CAD says the relationship should be"
   and "what it measurably is" turned out to differ enough to matter —
   that is the team's own prior experience on the *exact* class of problem.

So: **yes, use whatever CAD/datasheet number exists as a bootstrap/sanity
bound** (it tells you the right ballpark and lets you catch a gross sign/
axis error immediately), but do not ship it as the final number.

### 6.2 The proposed lightweight procedure

Rather than a formal AprilTag/checkerboard hand-eye rig (more accurate in
theory, but a new piece of tooling, a printed target, and a separate
solver to build and validate), reuse infrastructure this plan is building
anyway: the perception pipeline's own detector + the arm's own FK.

1. Place one of the task's known objects (e.g. the red cube — already a
   DINO-detectable, already-modeled-geometry target) somewhere in the
   workspace.
2. Command the arm (open-loop, via the existing `runner.reset_to_episode`-
   style ramped motion, or a small new calibration script) to touch the
   object with a known point on the gripper, at **several different arm
   configurations** (different shoulder/elbow angles reaching the *same*
   physical point, plus repeats with the object moved) — each touch gives
   one (FK flange position, "true" object position) correspondence pair.
3. At each touch, also run the live detector+depth pipeline on the object
   *before* the hand occludes it, giving one (camera-frame position,
   FK-frame position) pair.
4. Solve the rigid transform between the two point sets with a **Kabsch
   fit** — code that already exists twice in this codebase in the exact
   needed shape (`ego2g1/core/hand/retarget.py`'s `_kabsch`, and
   `data_extraction_zh`'s `robust_rigid_fit`) — reuse one of them rather
   than writing a third.
5. Store `T_pelvis_camera` as a calibration asset next to `b_calib.npz`
   (same directory convention, `ego2g1/data/work/_global/` or a new
   `camera_calib.npz`), with a manifest recording how many touch points,
   residual error, and date — mirroring `b_calib.manifest.json`'s pattern.

This gets you a *measured* calibration using only things this plan is
building anyway (detector, depth, FK), no new hardware target, and
produces a residual-error number you can use as a go/no-go gate (if a
handful of touches don't agree to within, say, 1-2 cm, something more
fundamental is wrong and needs to be found before trusting the policy at
all). A formal hand-eye rig remains an option if this residual turns out
too large to be usable — but it's the fallback, not the default.

## 7. Binary gripper → BrainCo motor mapping

New constants module, `ego2g1/deploy/perception/gripper_calib.py` (or
alongside `core/hand/constants.py` — same file family):

```python
# Manually measured per hand: the Revo2 motor command vector, in
# core.hand.constants.MOTOR_ORDER, that represents "the BrainCo hand fully
# closed around this task's objects." Open is always all-zero (0=open is
# already the hand's own convention, core/hand/constants.py). YOU set these
# by hand per the original request -- there is no principled way to derive
# them from the binary training signal, which never specified individual
# finger commands.
BRAINCO_CLOSED_POSE: dict[str, np.ndarray] = {
    "left":  np.array([..., ..., ..., ..., ..., ...], dtype=np.float32),  # TODO: measure
    "right": np.array([..., ..., ..., ..., ..., ...], dtype=np.float32),  # TODO: measure
}
```

Deploy-side interpolation (in `RelativeEEFRotvecChunks.convert`, §3.2):
`frac = clip((raw_grip_dim + 1) / 2, 0, 1)`; `cmd = frac * closed_pose`.
Linear interpolation rather than a hard threshold, because the model's
raw flow-sampled gripper value is a continuous sample near a bimodal
target, not a clean binary — a hard threshold would throw away exactly
the "the model is unsure" signal that a calibrated hysteresis (à la
training's own `_grasp_hysteresis` in `encoding.py`) is designed to use.
If open-loop interpolation chatters in practice, add the same
close/1.35×-open hysteresis band training already validated, rather than
inventing a new scheme.

## 8. Explicitly out of scope for this pass (do not build yet)

- **Inference-time RTC for `relation_eef` mode.** `EgoRelationTrainConfig`
  wasn't trained with `rtc_training`, and the reanchor-prefix math would
  need its own design (rotvec deltas instead of vec9, gripper dims
  passed through the per-slot quantizer rather than Normalize). Note for
  later: the "feed prev_chunk through the same input-transform chain"
  trick `serve/policy.py`'s `Ego2G1Policy.infer` already uses for the old
  mode appears directly portable (`RelativeEEFRotvecActions` early-returns
  on a bare `"actions"` key exactly like `RelativeChunkActions` does), but
  this needs its own verification pass, not a silent assumption.
  Track as follow-up, not blocking v1.
- **Native depth sensor integration** — resolved as not applicable (§5.2):
  this G1-D's head camera is a passive stereo-RGB module with no depth
  sensor and no wired depth path in Unitree's own server software. Do not
  revisit this unless the physical hardware changes.
- **A formal AprilTag hand-eye calibration rig** — only if touch
  calibration's residual proves unusable (§6.2).
- **Multi-task / arbitrary object-count generalization beyond what
  `DeployTaskConfig` already allows** — the checkpoint itself is trained
  for exactly 3 fixed objects in a fixed order; supporting a
  variable-object-count checkpoint is a training-side change first.

## 9. Task breakdown (for parallel subagent execution)

Ordered by phase; within a phase, tasks with no listed dependency on each
other can run in parallel. Every task ends in a concrete, runnable test —
subagents should not report a task done without it passing.

**Phase 1 — serve fix + offline validation (no camera/robot needed)**

1. `stamp.py` `config_class` field + `config_from_stamp` dispatch (§4.1,
   §4.2). Test: round-trip a synthetic `EgoRelationTrainConfig`'s stamp
   through `write_stamp`→`read_stamp`→`config_from_stamp` and assert the
   reconstructed config equals the original (`dataclasses.asdict` diff).
2. `resolve_relation_norm_assets` + `create_policy`'s relation branch
   (§4.2). Test: build a real (or fixture) relational checkpoint dir and
   assert `create_policy` returns a working `Ego2G1Policy` whose
   `.infer()` on a hand-built 56-dim state + dummy image + prompt produces
   a `(H,14)` action array with finite values and correct
   `metadata["ego2g1"]["objects"]`/`control_mode`.
3. `core/relation_layout.py` (§3.1). Test: round-trip slice consistency
   against `EgoRelationTrainConfig.gripper_dims`.
4. `RelativeEEFRotvecChunks` + `RelationPolicyAdapter` (§3.2, §3.3), fed by
   a **mocked** `RelationPerception` that replays a real episode's stored
   `entities/poses.npz` relation vectors (from `data_extraction_zh`'s own
   output, or the exported LeRobot dataset's `observation.state` column)
   instead of a live camera. Test: replay a full training episode's
   recorded state/action sequence through
   server→adapter→converter→`(H,26)` joints and sanity-check the resulting
   joint trajectory against the dataset's own stored joints for smoothness
   (reuse the existing `tests/test_deploy_conversion.py`-style accel-RMS
   check already used for the old mode). This is the single most
   important test in this plan — it validates the entire action-decode
   path with zero hardware and zero perception risk.
5. `ego2g1/deploy/check.py`'s `tcp-orientation` rung (§4.4). Test: runs
   headless against `MockExecutor`/a fixture arm config and prints without
   crashing; the "human eyeballs it" part is manual, but the rung itself
   must be automatable in CI up to that point.

**Phase 2 — live perception + calibration (needs Phase 1 done + your
answer on §5.2's depth source)**

6. `task_config.py` (§5.1) + server-metadata cross-check.
7. `DepthSource` interface + StereoSGBM fallback implementation (§5.2).
8. Detector/tracker/orientation cascade (§5.3) — largest task, likely
   needs splitting further once §5.2 is answered and the exact
   `data_extraction_zh` model-loading path is confirmed reusable across
   repos (check its packaging/license before vendoring).
9. Latch state machine (§5.4) — unit-testable in complete isolation with
   synthetic converging/diverging trajectories, no detector or robot
   needed. Write this test first.
10. `RelationPerception.observe` (§5.5), wiring 6-9 together.
6b. **Stereo intrinsic/extrinsic calibration** for the head camera pair
    itself (checkerboard/ChArUco, `cv2.stereoCalibrate`) — a prerequisite
    for task 7's `StereoSGBM` `DepthSource` to produce metric depth at
    all; do this before or alongside task 7, not after.
11. Touch-calibration script (§6.2), reusing an existing Kabsch
    implementation. This solves camera→pelvis *extrinsics* only and
    assumes task 6b's stereo calibration is already correct — order
    matters, do not run this before 6b.
12. `gripper_calib.py` scaffold (§7) — the constants themselves are yours
    to measure by hand; the interpolation code around them is testable
    with placeholder values.

**Phase 3 — hardware bring-up** (manual, rung-ladder style, matches
`docs/deploy.md`'s existing philosophy — not subagent-parallelizable,
this is you + the real robot): run the new `tcp-orientation` rung, run
touch calibration, run `replay-actions`-equivalent for `relation_eef` mode,
then a first gated (`--dashboard`) rollout with tight safety limits before
anything unattended.
