# data_extraction — internal interface spec (contract for all modules)

Read this before writing or reviewing any module. `config.py` is the single
source of truth for parameters; this file is the single source of truth for
interfaces. If an implementation needs to deviate, update this file in the
same change.

## Ground rules

- **Self-contained**: no imports from `wrist_replay/` or `pico2usable/`.
  Their code is MIGRATED (copied + adapted) into this package. Allowed deps:
  numpy, scipy, h5py, mujoco, mink, PIL, imageio; `lerobot` only inside
  `s005_write_lerobot/` and `loader/` behind guarded imports.
- All intra-package imports are relative (`from ..common import frames`).
- Run everything from the repo root with the repo venv:
  `.venv/bin/python -m data_extraction.run_pipeline ...`
- Quaternions are **wxyz everywhere after ingest**; the only xyzw→wxyz
  reorder happens in `common/episode.py` (and raw hand arrays, which stay in
  the raw Pico convention on purpose — see s001 docstring).
- 6D rotation = first two **columns** of R concatenated; `vec9(T) = [t(3), 6d(R)(6)]`
  (`common/rot6d.py`). Loader, dashboard, and stages must all use these helpers.

## Module map (migration sources)

| dest | migrated from | adjustments |
|---|---|---|
| `common/frames.py` | `wrist_replay/frames.py` | verbatim |
| `common/episode.py` | `wrist_replay/episode_data.py` | verbatim (imports → relative) |
| `sim/g1.py` | `wrist_replay/sim.py` | asset path → `data_extraction/assets/unitree_g1/scene_fixed_base.xml`; add `base_pose()` (pelvis body 4×4 in world) and `world_to_base(T)` |
| `sim/placement.py` | `compute_placement`, `refine_placement`, `calibrate_alignment`, `first_valid_tick` from `wrist_replay/replay.py` | verbatim logic; weights/margin come from cfg args |
| `sim/chunks.py` | `wrist_replay/retarget.py` | verbatim (ReplayChunkProvider, ChunkRuntime, selftest_identity) |
| `hand/constants.py` `hand/fk_tables.py` `hand/retarget.py` `hand/screen.py` | `pico2usable/hand2brainco/{constants,fk_tables,retarget,screen_replay}.py` | bare imports → package-relative; asset paths → `data_extraction/assets/revo2/`; `screen.py` keeps only `HandSim` + `blocked_mask` + thresholds (no CLI) |
| `assets/unitree_g1/` | `wrist_replay/assets/unitree_g1/` | copied |
| `assets/revo2/` | `pico2usable/hand2brainco/assets/` | copied (xml_left, xml_right, fk_tables_*.npz) |
| `dashboard/template.html` | adapted from `wrist_replay/template.html` | same `/*__DATA_JSON__*/` injection token |

## Stage outputs (npz schemas)

Stored via `common/io.py: save_stage(cfg, ep_name, stage, arrays, meta, source_sig)`.
`ep_name` = episode stem (`episode_1`); global stages use `ep_name=None`.
T = number of control ticks. Sides: `l`/`r` suffixes.

### s001 (per episode)
- `ticks_ns (T,) i64`, `cam_match (T,) i32`, `cam_gap_ms (T,) f64`
- `{l,r}_pos (T,3)`, `{l,r}_quat (T,4) wxyz`, `{l,r}_valid (T,) bool` — wrist, MuJoCo world
- `{l,r}_hand_pos (T,26,3) f32` raw Pico frame, `{l,r}_hand_wrist_quat (T,4) f32 xyzw` raw

### s003_placement (per episode)
- `S (4,4)` — rigid placement, Pico-MJ-world → robot world
- meta: `k0`, refine shift/yaw, per-side reach check (`n_out`, worst overshoot)

### b_calib (global)
- `B_left (4,4)`, `B_right (4,4)` — wrist→flange alignment (rotation-only)
- meta: `mode`, per-episode spread (geodesic deg to the global B), mount rpy

### s003_state (per episode)
- `state_eef_{l,r} (T,9)` vec9 — flange pose in **pelvis frame** (FK-achieved if
  `proprio_source=ik_fk`, else the target pose)
- `arm_qpos (T,14) f32` — [left 7, right 7] in `sim.g1.ARM_JOINTS` order
- `ik_pos_cm_{l,r} (T,)`, `ik_ori_deg_{l,r} (T,)` — per-tick IK tracking error
  vs the target flange pose (zeros when `proprio_source=direct`)

### s002_01 (per episode)
- `pose_{l,r} (T,9)` vec9 — canonical **target** flange pose in pelvis frame:
  `G(t) = pelvis⁻¹ · S · T_wrist(t) · B` (this is what actions difference)
- meta: `selftest_max_err`, `max_tick_rot_deg` per side (gates: hard-fail on violation)

### s002_02 (per episode)
- `hand_cmds_{l,r} (T,6) f32` in [0,1] MOTOR_ORDER, `hand_cmds_raw_{l,r}`
- `hand_residual_{l,r} (T,5) f32` m, `hand_snap_{l,r} (T,2) bool`, `hand_valid_{l,r} (T,) bool`
- meta: calib frame/scales per side, summary stats

### s004 (per episode)
- per-filter masks, all `(T,) bool`, True = BAD: `bad_gap_{l,r}`, `bad_cam`,
  `bad_vel_{l,r}`, `bad_ik_{l,r}`, `bad_hand_blocked_{l,r}`,
  `bad_hand_contact_{l,r}`, `bad_hand_residual_{l,r}`, and combined `bad_any (T,)`
  (respecting `filter_hands_independently` × `cfg.hands`)
- `anchor_bad (T,) bool` — bridged ticks: interior bad runs
  ≤ `cfg.bridge_max_ticks` do NOT split the episode; their ticks stay inside
  the sub-episode (poses may appear inside action windows) but must never
  anchor a datapoint (enforced by the loader)
- `subep_start (K,) i32`, `subep_end (K,) i32` (exclusive), `subep_real_end (K,) bool`
  (computed on `bad_any` after bridging)
- meta: per-filter bad-tick counts, `n_subepisodes`, `ticks_kept`,
  `ticks_bridged`, `ticks_total`

## LeRobot dataset (s005 output)

Written with the openpi-pinned lerobot rev to `cfg.output_root/cfg.repo_id`
(`root=` arg; never the default HF cache). `fps = cfg.control_hz`. One
LeRobot episode per sub-episode. Features per frame:

| key | shape | dtype | contents |
|---|---|---|---|
| `image` | (H,W,3) | video | nearest camera frame for the tick (from source HDF5 jpegs) |
| `state` | (30,) | f32 | per hand in cfg.hands order: eef vec9 (pelvis frame) + 6 hand motors — from s003_state + s002_02 |
| `pose.left` / `pose.right` | (9,) | f32 | s002_01 canonical target flange pose (vec9) |
| `hand.left` / `hand.right` | (6,) | f32 | s002_02 commands |
| `arm_qpos` | (14,) | f32 | s003_state |

Task (language prompt) = `cfg.task_prompt`.

Sidecar `extraction_meta.json` at the dataset root:
```json
{"config_hash": "...", "config": {...cfg.to_dict()...},
 "episodes": {"<episode_index>": {
    "source_file": "...", "source_episode": "put_bottle_in_box/episode_1",
    "tick_start": 0, "tick_end": 143, "episode_real_end": true,
    "anchor_bad": [37, 38, 39],
    "S": [[...]], "B_left": [[...]], "B_right": [[...]],
    "filter_stats": {"bad_gap": 3, ...}}}}
```

## Loader semantics (openpi side, `loader/`)

- Datapoint at frame t: gather `pose.*` at offsets `[0..H]/fps` (H+1 rows;
  row 0 = anchor), `hand.*` at `[1..H]/fps`.
- `RelativeChunkActions` transform: `Δ_k = vec9_to_se3(pose_0)⁻¹ @ vec9_to_se3(pose_k)`
  → `se3_to_vec9`, concat per hand [eef 9 | hand 6] in cfg.hands order →
  `actions (H, 30)`. Also emits `state` unchanged.
- Boundary rule: frames with `t + H > subep_len - 1` are INVALID datapoints
  unless the sub-episode has `episode_real_end` and `allow_terminal_padding`
  (LeRobot repeat-padding then means "hold pose"). Frames listed in the
  sidecar's `anchor_bad` (bridged ticks) are ALSO invalid as anchors, in
  either regime. Enforced by an index-remapping wrapper dataset, because
  pi0 ignores `action_is_pad`.
- Training config must assert the dataset's `config_hash`.

## Verification hooks every module must keep

- `common/episode.py: verify_up_axis` runs on every load (hard-fail).
- `sim/chunks.py: selftest_identity` runs in s002_01 per episode (hard-fail).
- s002_01 continuity gate: max single-tick rotation ≤ `cfg.max_tick_rotation_deg`.
- Loader equivalence test (`tests/test_loader_equivalence.py`): decoded batch
  actions == deltas computed directly from s002_01 poses, to 1e-6.
