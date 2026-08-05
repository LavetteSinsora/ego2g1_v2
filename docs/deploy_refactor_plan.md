# Deploy refactor: plan

Status: **implemented (2026-08-05), with two consciously deferred items.**
Everything below landed as planned except: (a) §7's internal split of
check.py into seven rung modules — check.py moved to `tools/check.py`
wholesale and keeps its `python -m ego2g1.deploy.check <rung>` dispatcher;
the per-rung file split + the §6.3 CLI mixins remain follow-up work; (b) the
converter/adapter classes stayed in `actions.py`/`policy_adapter.py` (widely
referenced by docs/tests) rather than physically moving into `modes/*.py` —
the mode objects in `modes/` wire them, which is what the extensibility
property needed. The fifth-mode smoke test
(tests/test_deploy_modes.py) is the executable proof of §10's first bullet.
Companion docs: [deploy.md](deploy.md) (the current architecture and its
rationale — still accurate, and everything it calls load-bearing stays
load-bearing), [relation_deploy_plan.md](relation_deploy_plan.md) (how the
relation mode was added — the growth that motivated this plan),
[4090_serve_deploy.md](4090_serve_deploy.md), [jitter_root_cause.md](jitter_root_cause.md).

## 0. Why, and what is immovable

`ego2g1/deploy/` is ~11.7k lines across ~30 flat modules. Semantically it is
six clean planes — the mode-blind execution spine, the policy-mode layer, the
perception cascade, the bring-up trust ladder, the recording/replay
observability layer, and the machine-environment layer — but the file layout
doesn't reflect them, and four contracts that everything depends on are held
by **convention** (grep + docstrings) rather than by **code**:

1. **The mode contract.** "Modes differ only in `actions.py`/`policy_adapter.py`"
   was true for two modes; adding `relation_eef` leaked branches into ~8
   files (`runner.py:102/140/249/268/366/494`, a parallel
   `_observe_relation`, `dashboard.py`, `safety.py`, `replay_mujoco.py`,
   `recorder.py` docs). A fourth mode today means editing all of them.
2. **The row-streaming contract.** The observe→clamp→future-stamp→send→pace→
   damp-on-Ctrl-C loop is written four times (`replay_dataset.py:184`,
   `replay_diag.py:63`, `check.py:1264`, `replay_relation_openloop.py:187`)
   — two of the four have **no clamp and no watchdog at all**.
3. **The recording schema.** `recorder.py:15-32`'s authoritative event list
   is missing four kinds the code now emits (`percept`, `latch`,
   `hand_state`, `latency_check_refused`); `meta.json` is assembled ad hoc
   at two call sites with different key sets, and
   `replay_record.Session._make_buffer` silently defaults missing strategy
   params.
4. **The telemetry shape.** The dashboard's dict is hand-copied three times
   (`runner.telemetry()`, `perception_preview.PreviewLoop.telemetry()`,
   `replay_dashboard.ReplayLoop.telemetry()`) plus a fourth partial copy in
   `_build_replay_relation_telemetry` — structural typing that rots whenever
   the page gains a field.

The refactor turns those four conventions into code, and moves files so the
directory tree coincides with the planes. **Immovables — not up for
change**, because they are the measured, hard-won parts
(docs/jitter_root_cause.md):

- The policy⇄execution contract stays *timestamped (H, 26) absolute joint
  chunks*, converted whole-chunk at inference time. Everything below the
  adapter stays mode-blind.
- The vendored executor is wrapped, never rewritten. `damp()` semantics,
  future-stamping (`t_cycle_end + dt`), `precise_wait` pacing: byte-for-byte.
- Fail-loud philosophy: startup latency refusal, fail-loud config
  validation, "a refusal at the terminal beats a lurch on the arm".
- The rung-ladder entrypoints (`python -m ego2g1.deploy.check <rung>`,
  `...deploy.runner`, all `replay_*`) keep working — muscle memory and every
  doc reference are assets. Moves happen behind re-export shims.
- The docstrings' institutional memory (why each threshold, which incident
  motivated which guard) is **moved with the code, never summarized away**.
- No behavior change on the hot path except where a change *adds* missing
  safety (see §3).

## 1. Target package layout

```
ego2g1/deploy/
  _util.py            precise_wait, dds_init(iface) — shared leaf helpers
  core/
    runner.py         DeployRunner + CLI main (thinner: no mode branches)
    strategies.py     unchanged
    safety.py         Clamp/Watchdog/limits + per-mode sanity via modes (§2)
    latency.py        unchanged
    executor.py       UnitreeExecutor/MockExecutor (+ §8 hand-clip fix)
    session.py        NEW — ExecutorSession (§3)
    kinematics.py     unchanged
    client.py         unchanged (+ stale-docstring fix)
    fast_crc.py       unchanged (+ new bit-identity test)
  modes/
    base.py           NEW — DeployMode protocol + registry + EEFModeBase
    joint.py          JointChunks + JointPolicyAdapter
    relative_eef.py   vec9 decode + passthrough hands, on EEFModeBase
    relation_eef.py   rotvec decode + BRAINCO expansion + perception wiring
  perception/         internals unchanged; + config.py (§6)
  record/
    schema.py         NEW — typed event kinds + build_meta + SCHEMA_VERSION (§4)
    recorder.py       Recorder/RecorderSwitch/NullRecorder (moved)
    session_reader.py replay_record.Session (moved, + schema validation)
  ui/
    telemetry.py      NEW — TelemetrySnapshot + build_telemetry (§5)
    dashboard.py      page + HTTP server + overlay drawing (moved)
    replay_dashboard.py  ReplayLoop, much smaller after §5
    perception_preview.py
  tools/
    check/            one module per rung family + thin tyro dispatcher (§7)
    replay_dataset.py, replay_diag.py, replay_mujoco.py,
    replay_relation_openloop.py, measure_rate.py, sniff_lowcmd.py
    cli.py            NEW — shared Args dataclass mixins (§6.3)
  # top-level shims so every documented invocation keeps working:
  runner.py           `from .core.runner import *` (+ __main__ passthrough)
  check.py, replay_*.py, dashboard.py, ...   same pattern
```

The shims are one-line re-exports; `tests/test_no_symlinks.py`-style, a new
test asserts every historical `python -m ego2g1.deploy.<name>` entrypoint
still resolves.

## 2. `modes/`: the mode registry (kills convention #1)

### 2.1 The protocol

One object per policy family owns *everything* that differs between
families. `DeployRunner`, `dashboard`, `recorder`, and the replay tools stop
asking "which mode am I in" and start asking the mode object:

```python
# modes/base.py
class DeployMode(Protocol):
    name: str                        # "joint" | "relative_eef" | "relation_eef"
    supports_rtc: bool
    supports_reset_to_episode: bool

    def build_adapter(self, client, args) -> Adapter:
        """Adapter + converter + (relation) perception wiring, including the
        fail-loud required-config checks currently in
        runner._build_relation_adapter (runner.py:648-708)."""

    def build_observation(self, executor, camera, last_hands, prompt) -> dict:
        """Replaces runner._observe / _observe_relation (runner.py:136-173)
        AND _build_probe (runner.py:711-731) — one definition, used by both
        the loop and the startup latency probe."""

    def initial_hand_state(self) -> dict:
        """(6,)-vectors for joint/relative_eef, scalar fracs for relation
        (runner.py:114-115)."""

    def hand_state_from_row(self, row, adapter) -> dict:
        """Replaces the runner.py:249-253 branch (incl.
        _hand_frac_from_command for relation)."""

    def sanity_check_chunk(self, raw_chunk) -> bool:
        """Per-mode model-space sanity. Absorbs safety.py:161's dead 30-dim
        check and ADDS the missing (H, 14) relation check (§8)."""

    def telemetry_extras(self, adapter) -> dict | None:
        """The dashboard's per-mode panel — relation's build_relation_telemetry
        (runner.py:509-556) moves here; None for other modes."""

    def drain_recorder_events(self, adapter, since_t) -> list[Event]:
        """runner.run()'s step 4b (runner.py:264-290) — percept/latch/
        hand_state events for relation, empty elsewhere."""

MODES: dict[str, DeployMode]         # registry, keyed by server control_mode
def resolve(action_mode: str, control_mode: str) -> DeployMode
    # absorbs runner._resolve_action_mode (runner.py:630-645)
```

### 2.2 `EEFModeBase`: deduplicating the two EEF stacks

`actions.RelativeEEFChunks` (actions.py:106) and `RelativeEEFRotvecChunks`
(:185) are ~95% identical; likewise the two adapters' `infer` tails
(policy_adapter.py:122-133 vs :296-305). The honest difference is two
methods:

```python
class EEFChunksBase:                 # anchor→compose→OneEuroSE3→IK→JointFilter, once
    def decode_pose(self, row6_or_9) -> SE3: ...        # overridden
    def expand_hand(self, raw, hand) -> np.ndarray: ... # overridden: clip-passthrough
                                                        # vs frac × BRAINCO_CLOSED_POSE
```

`relative_eef.py` and `relation_eef.py` each become the base + ~20 lines.
The per-slot residual profile, `last_targets`, `reset()` — written once.
`hand_cmds` stops being a required-but-unread argument of the rotvec
converter (actions.py:247, which forces runner.py:168's fabrication).

### 2.3 What `DeployRunner` becomes

`relation_mode`, `_observe_relation`, `_hand_frac_from_command`,
`_relation_telemetry`, `_build_relation_adapter`, `_build_probe`, and
`_resolve_action_mode` all leave `runner.py`. The loop body keeps its exact
step structure (0 gate → 1 observe → 2 wait → 3 pop/clamp/send → 4 tracking
→ 4b drain → 5 pace) but steps 1, 3's hand bookkeeping, 4b, and the
telemetry call go through `self.mode`. Diff target: `runner.py` shrinks
~250 lines and contains zero occurrences of the string `relation`.

A fourth mode (joint-space ego2g1 policy, relation-v2 with live
orientation) = one file in `modes/` + one registry entry. That is the
extensibility criterion this whole section is judged by.

## 3. `core/session.py`: ExecutorSession (kills convention #2)

The one way rows reach hardware:

```python
class ExecutorSession:
    """Owns the invariants every hardware-touching tool must share:
    sanity_check_joint_row → Clamp → future-stamp (t_cycle_end + dt) →
    executor.send → precise_wait, plus KeyboardInterrupt → damp() and
    optional Watchdog integration. Recording is a constructor arg so no
    caller can forget it."""

    def __init__(self, executor, *, fps, limits: SafetyLimits,
                 recorder=NullRecorder(), watchdog=None, clock=..., wait=...)
    def send_row(self, row26) -> None          # the runner's per-tick path
    def stream(self, rows: Iterable[np.ndarray],
               confirm: str | None = None) -> None   # replay tools' loop,
               # incl. the first-send soft-ramp sleep and the y/N prompt
    def ramp_to(self, q14, hands, *, ramp_s, max_speed, settle_s) -> dict
               # reset_to_episode's body (runner.py:378-418)
```

Call sites converted, in order of what they gain:

| caller | today | after |
|---|---|---|
| `replay_dataset.py:184-225` | **no clamp, no watchdog** | both, by construction |
| `replay_diag.py:63-106` | **no clamp, no watchdog** | both (instrumentation hooks stay) |
| `check.py replay-actions :1264-1327` | own Clamp, arg named `max_step` | shared, arg named `max_joint_step` |
| `replay_relation_openloop.py:187-265` | own Clamp, own loop | shared |
| `runner.run()` step 3 + `reset_to_episode` | canonical copy | calls `send_row`/`ramp_to` |

`precise_wait` moves from `runner.py:62` to `_util.py`; the four
`from .runner import precise_wait` imports (replay_dataset.py:44,
replay_diag.py:35, check.py:1294, replay_relation_openloop.py:63) stop
dragging the whole runner (tyro Args, perception config) into every
diagnostic. The 7×-duplicated DDS bootstrap (check.py:79/:808/:1084/:1172,
measure_rate.py:39, sniff_lowcmd.py:41, executor.py:107) becomes
`_util.dds_init(iface)` — executor.py keeps its singleton-race comment on
the helper.

## 4. `record/schema.py`: the recording contract (kills convention #3)

```python
SCHEMA_VERSION = 2          # v1 = everything recorded before this refactor

EVENT_KINDS = {             # one entry per kind; the docstring table is
  "obs":            ObsEvent,            # GENERATED from this dict
  "action":         ActionEvent,
  "infer_result":   InferResultEvent,
  "clamp":          ClampEvent,
  "tracking":       TrackingEvent,
  "latency_check":  LatencyCheckEvent,
  "latency_check_refused": ...,
  "worker_error":   ..., "estop": ..., "rearm": ..., "reset": ...,
  "percept":        PerceptEvent,        # the four kinds recorder.py:15-32
  "latch":          LatchEvent,          # currently forgets
  "hand_state":     HandStateEvent,
}

def build_meta(*, mode, action_mode, fps, horizon, strategy_params,
               source: str, **extra) -> dict
    # ONE constructor for meta.json, used by runner.main (runner.py:770-780)
    # AND replay_relation_openloop (:211-216) — the openloop tool stops
    # omitting the strategy params Session._make_buffer needs, and starts
    # using record.new_session() naming instead of its hand-rolled stamp.
```

Enforcement is deliberately lightweight — this is a lab, not a wire
protocol:

- `Recorder.log(kind, ...)` asserts `kind in EVENT_KINDS` (a typo'd kind
  today just vanishes into the JSONL).
- `Session.__init__` warns on unknown kinds and on `meta.json` missing
  declared strategy params (today: silent default at
  replay_record.py:89-96), and reads `schema_version` (absent ⇒ v1, and
  the existing old-recording fallbacks in `replay_mujoco.py` become the
  documented v1 path rather than incidental `get(...)` guards).
- A test walks every `rec.log("...` call site in the package (grep-based)
  and asserts the kind is declared — the docstring can never silently
  drift again.

Old recordings stay readable forever: v1 handling is additive, nothing is
rewritten.

## 5. `ui/telemetry.py`: one snapshot builder (kills convention #4)

Invert the duck-typing. Instead of three loops each hand-assembling the
page's dict, one dataclass and one builder:

```python
@dataclasses.dataclass
class TelemetrySnapshot:            # every key dashboard.js reads, typed
    ...
    def to_json(self) -> dict

def build_telemetry(*, strategy_view, executor_view, safety_view,
                    mode_view=None, replay_view=None) -> TelemetrySnapshot
```

- `DeployRunner.telemetry()` (runner.py:429-495), `PreviewLoop.telemetry()`
  (perception_preview.py:153-174) and `ReplayLoop.telemetry()`
  (replay_dashboard.py:348-397) all become thin providers feeding
  `build_telemetry`. Missing planes render "n/a" — the page logic is
  unchanged.
- **The recorded form becomes the canonical form.** The dashboard's
  perception overlay stops reaching through `loop.adapter.perception`'s
  eight live attributes (dashboard.py:219-270) and instead draws from the
  `percept`-event shape (`RelationPerception.debug_snapshot()` — which the
  recorder already logs verbatim). Live rendering is then literally "replay
  of the current instant": `replay_dashboard`'s `_ReplayPerception` stand-in
  (~130 lines of `_Replay*` dataclasses) collapses, and live/replay overlay
  can never diverge again, because they are the same code path fed the same
  shape. `_build_replay_relation_telemetry` (replay_dashboard.py:400-441)
  is deleted outright.
- The pull-only guarantee (nothing pushes from the hot loop; existing locks
  only) is preserved — providers read the same state the current
  `telemetry()` methods read.

## 6. Configuration with one owner (§ perception/config.py, tools/cli.py)

### 6.1 `RelationDeployConfig`

Everything currently constructor-only on `RelationPerception`
(relation_perception.py:177-183: `detector_period_ticks`,
`orientation_period_ticks`, `nominal_rotations`, `symmetry_groups`,
`latch_config`, `tracker_kwargs`) plus the SGBM parameters (depth.py) gets a
frozen dataclass, YAML-loadable, env-overridable, with the current values as
defaults. `runner.Args` gains a single `--perception-config` path (the
three artifact paths stay as-is, they are a different kind of thing:
per-rig facts, not tuning). Retuning `latch_distance_m` on hardware stops
requiring a source edit.

The loaded config is written verbatim into `meta.json` (via §4's
`build_meta(extra=...)`), so a replayed session knows the thresholds that
produced it.

### 6.2 Calibration provenance

`touch_calib._cli_solve` and `handeye_calib._cli_solve` both write
`T_pelvis_camera` with different companion keys and different quality
thresholds. Unify: both write `{T_pelvis_camera, method, solved_iso,
residual_summary}` (keeping their method-specific extras), and
`runner._build_relation_adapter`'s loader logs the method+residual it
loaded. Nothing else changes — two solvers, one artifact contract.

### 6.3 CLI mixins + lab defaults

`tools/cli.py`: dataclass mixins (`RobotArgs`: `network_interface`,
`max_pos_speed`; `RunArgs`: `dry_run`, `yes`, `fps`; `IKArgs`: `ik_iters`,
`posture_cost`, `collision_min_dist`; `RecordArgs`) composed into each
tool's `Args`. Ends the drift: `ik_iters` is 40 in replay_dataset.py:153
and 25 everywhere else; `iface` vs `network_interface` across check rungs.
Defaults live once, in the mixin, sourced from `kinematics.py`'s own
signature where applicable.

Lab constants collapse into one module the env profiles document:
`192.168.123.164` (today in camera.py:34, runner.py:607, check.py ×3,
perception_preview.py), and the SSH credentials + robot-side paths
currently *in source* at remote_image_server.py:30-42 move to env vars with
the current values as documented defaults in `envs/robot.sh`/`envs/4090.sh`.

## 7. Splitting `check.py` (1393 lines, 11 jobs)

One module per rung family under `tools/check/`, dispatcher preserved:

| new module | rungs (current check.py line) |
|---|---|
| `dds.py` | `listen` (:71) |
| `kin.py` | `fk` (:134), `ik` (:174), `tcp-orientation` (:335) |
| `camera.py` | `camera` (:410), `stereo-capture` (:455) |
| `hands.py` | `hand-sweep` (:1070), `hand-jog` (:1134) — shares the termios jog loop |
| `calib_capture.py` | `handeye-capture` (:611) — imports hands.py's jog |
| `replay_actions.py` | `replay-actions` (:1244) — on ExecutorSession (§3) |
| `latency.py` | `latency` (:1338) — reuses `LatencyReport.summary()` instead of its second formatter |
| `__main__.py` | the same `tyro.extras.subcommand_cli_from_dict` dispatch, same rung names |

`python -m ego2g1.deploy.check <rung>` behavior is pinned by a test that
invokes each rung's `--help` through the shim. Rung functions adopt the
§6.3 mixins so flags stop varying by rung.

## 8. Correctness fixes folded in (each small, each opportunistic)

| fix | where | why |
|---|---|---|
| Clip hands in `MockExecutor.send` | executor.py:284 vs :164 | dry-run currently accepts what the real robot would alter |
| Loud warning when `BRAINCO_CLOSED_POSE` is the placeholder | gripper_calib.py:21, checked in relation mode's `build_adapter` | live deploy on `np.ones(6)` today warns nowhere |
| Add (H, 14) relation chunk sanity; retire dead 30-dim `sanity_check_model_action` | safety.py:161 → `modes/*.sanity_check_chunk` | the only mode with live perception feeding it has no model-space check |
| Cross-check `len(objects)`/`hands` vs `relation_layout` at task-config load | task_config.py + relation_layout.py:46-49 | a 2-object YAML passes load + server check, then shape-errors deep in `infer()` |
| Unify `task_config.hands` vs `relation_layout.HANDS` usage | runner.py:114/161/720 vs relation_perception.py | silent desync for nonstandard hands |
| Fix `orientation_period_ticks=6 # ~0.2 Hz` comment (6 ticks @30fps = 5 Hz) | relation_perception.py:178 | inert today (estimator is None) but wrong by 25× |
| Reconcile "mask-median" wording vs mean-centroid implementation | relation_perception.py:29 vs detector.py:87 | doc/impl mismatch on the seam that sets 3D position |
| One `rotation_angle_deg` in `core/se3.py` | latch.py:118, orientation.py:55, handeye_calib.py:86 (two formulas) | three implementations, one of them different |
| `touch_calib` stops importing `core.hand.retarget._kabsch` (private) | touch_calib.py:32 | promote `_kabsch` → `core` public helper |
| Move stereo_calib's module-scope `__import__("cv2")` behind the lazy guard | stereo_calib.py:294-296 | the one hole in the package-wide lazy-import rule |
| Test pinning `fast_crc` bit-identity + `damp()`'s vendor-internal attrs | fast_crc.py, executor.py:234 | the e-stop's assumptions are pinned by no test today |
| Update `PolicyClient` docstring for 56-dim state | client.py:11 | stale since relation mode |
| Route `replay_relation_openloop` through `record.new_session` + `build_meta` | replay_relation_openloop.py:210-216 | naming convention + missing strategy params (§4) |

Deliberately **not** fixed here (bigger than a fold-in, tracked separately):
the inert orientation-estimator pipeline (C6 — needs a real estimator
decision first), `ObjectQuery` vs `ObjectSpec` duplication (dies naturally
if/when detector grows a second consumer), and any SGBM parameter
re-derivation (depth.py:118-128's own flagged TODO — a hardware task).

## 9. Phased task breakdown

Ordered by risk; each phase leaves the suite green and the robot runnable.
Every task ends in a concrete test, per this repo's standing rule.

**Phase 0 — mechanical prep (zero behavior change)**

1. `_util.py`: move `precise_wait`, add `dds_init`. Rewrite the 4+7 import
   sites. Test: `import ego2g1.deploy.replay_dataset` no longer imports
   `tyro` (assert via `sys.modules`).
2. §8's test-only and doc-only rows (fast_crc pin, damp() pin, docstring
   fixes). Test: the new pins themselves.

**Phase 1 — recording schema**

3. `record/schema.py` + `build_meta` + `SCHEMA_VERSION`; recorder asserts
   kinds; Session warns + version-reads; openloop tool converted. Test:
   grep-walk test over `rec.log(` call sites; round-trip a v1 (fixture)
   and v2 session through `Session`.

**Phase 2 — ExecutorSession**

4. `core/session.py`; convert runner step 3 + `reset_to_episode`. Test:
   existing `test_deploy_runner.py` (future-stamp, clamp, trip seams)
   passes unmodified — the seams are the spec.
5. Convert the four replay/check tools. Test: `replay_dataset` against a
   fixture episode now logs `clamp` events when fed a step > limit
   (previously impossible); `check replay-actions --help` unchanged.

**Phase 3 — the mode registry (the big one)**

6. `modes/base.py` + registry + `resolve()`; port `joint`. Test:
   `_resolve_action_mode`'s existing unit test re-targeted at `resolve()`.
7. `EEFChunksBase`; re-express `relative_eef` and `relation_eef` on it.
   Test: `test_deploy_conversion.py` and `test_relation_conversion.py`
   pass unmodified (accel-RMS, identity-chunk, gripper decode — the
   conversion behavior is pinned, only its packaging moves).
8. Move observation/probe/hand-state/drain/telemetry-extras into the mode
   objects; delete runner's `relation_mode` branches. Test:
   `test_runner_relation_eef.py` end-to-end passes; new assert that
   `core/runner.py` contains no string `"relation"`.
9. New-mode smoke test: register a trivial fifth mode in-test and run a
   MockExecutor rollout through it — the extensibility claim, executable.

**Phase 4 — telemetry unification**

10. `ui/telemetry.py`; convert the three loops. Test: snapshot-compare
    `to_json()` keys against a checked-in fixture of the page's reads
    (extracted once from dashboard.js); `test_deploy_dashboard.py` passes.
11. Overlay draws from the `percept` shape; delete `_ReplayPerception`/
    `_build_replay_relation_telemetry`. Test: encode_perception_frame
    output identical (pixel-compare on a fixture) fed live-shaped vs
    recorded-shaped input.

**Phase 5 — layout move + check split**

12. Create `core/ modes/ record/ ui/ tools/`, move files, add top-level
    shims. Test: every documented `python -m ego2g1.deploy.<x>` entrypoint
    `--help`s successfully via the shim.
13. Split `check.py` per §7 + CLI mixins (§6.3). Test: per-rung `--help`
    golden test; `ik_iters` default asserted equal across all tools.

**Phase 6 — config unification**

14. `RelationDeployConfig` + `--perception-config` + meta.json embedding.
    Test: YAML round-trip; a recorded session's meta contains the latch
    thresholds; overriding `latch_distance_m` via YAML reaches the
    `GraspLatch`.
15. Calibration provenance keys + lab-defaults extraction to env profiles.
    Test: both calib CLIs' outputs load through one loader that reports
    `method`; grep test that `192.168.123.` and the SSH password appear
    only in `envs/` and defaults modules.

## 10. What success looks like

- Adding a policy mode touches `modes/<new>.py` + the registry. Nothing
  else. (Phase 3 task 9 is the executable proof.)
- No path to hardware exists without clamp + sanity + damp-on-interrupt.
- A recorded session is self-describing: schema version, strategy params,
  perception thresholds, calibration provenance — and every replay tool
  reads the same typed schema the recorder wrote.
- The live dashboard and the replay dashboard are one rendering fed two
  providers, incapable of divergence.
- `docs/deploy.md`'s architecture section stays true; only its file paths
  update.
