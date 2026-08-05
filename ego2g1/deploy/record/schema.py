"""The recording contract: every event kind a session may contain, and the one
constructor for `meta.json` (docs/deploy_refactor_plan.md §4).

Before this module, the authoritative event list was a docstring in
recorder.py that had silently drifted (it was missing `percept`, `latch`,
`hand_state`, and `latency_check_refused`), and `meta.json` was assembled ad
hoc at two call sites with different key sets — `replay_record.Session` then
silently defaulted the strategy params a replay of an openloop recording
needed. Now:

  * `Recorder.log()` refuses an undeclared kind (a typo'd kind used to just
    vanish into the JSONL, unreadable by every replay tool);
  * `build_meta()` is the only way a meta dict is assembled, and it requires
    the strategy params explicitly;
  * `SCHEMA_VERSION` is written into meta.json. Version 1 is everything
    recorded before this module existed (no `schema_version` key); readers
    treat a missing key as v1 and keep the old fallbacks. Bump the version
    when an event's MEANING changes, not when a purely additive kind/field
    appears.

Enforcement is deliberately lightweight — declared kinds and required meta
keys, not per-field validation of every payload. This is a lab recording
format, not a wire protocol; the goal is that the schema can never silently
drift from the code again, which `tests/test_deploy_record_schema.py`'s
grep-walk pins.
"""

from __future__ import annotations

import dataclasses

SCHEMA_VERSION = 2

# The four constructor params replay_record.Session._make_buffer needs to
# rebuild the PRODUCTION strategy buffer with the recorded values. Required
# in every meta.json — even for sync-only recordings, where they are inert
# but honest (a reader must never have to guess).
STRATEGY_PARAM_KEYS = ("inference_hz", "exp_weight_m",
                       "max_latency_steps", "min_smooth_steps")


@dataclasses.dataclass(frozen=True)
class EventSpec:
    """One declared event kind. `required` are the fields every emitter must
    include (`t` and `kind` are stamped by `Recorder.log` itself and not
    listed). Fields beyond `required` are allowed — several kinds carry
    optional diagnostics (e.g. infer_result's relative_eef-only
    `flange_targets`)."""

    doc: str
    required: tuple[str, ...] = ()


EVENT_KINDS: dict[str, EventSpec] = {
    "latency_check": EventSpec(
        "the startup self-check's LatencyReport, as a dict",
        ("mode", "verdict")),
    "latency_check_refused": EventSpec(
        "the runner refused to start: measured latency cannot honor the mode"),
    "obs": EventSpec(
        "per tick: state age + measured arm_q (replay_mujoco renders the body "
        "from it)",
        ("step", "state_age", "arm_q")),
    "infer_result": EventSpec(
        "per inference: latency, splice info, and `actions` — the converted "
        "(H, 26) joint chunk, so replay_record can rebuild the buffers "
        "exactly; plus per-mode diagnostics (slot_errors_m, raw_chunk, "
        "request_state, flange_targets)",
        ("latency", "horizon")),
    "action": EventSpec(
        "per tick: the popped joint row exactly as sent (post-clamp)",
        ("step", "row")),
    "clamp": EventSpec(
        "the per-tick clamp actually limited a step",
        ("step", "max_step")),
    "tracking": EventSpec(
        "per chunk (EEF modes): worst IK tracking error, metres",
        ("worst_m",)),
    "worker_error": EventSpec(
        "the async inference worker died",
        ("error",)),
    "estop": EventSpec(
        "damp() was called, with the watchdog's reason",
        ("reason",)),
    "rearm": EventSpec(
        "stale plans dropped + filters/clamp re-grounded (gated start, "
        "pause-resume, after a reset ramp)",
        ("why",)),
    "reset": EventSpec(
        "a dashboard reset-to-episode, with the landing residual",
        ("episode", "residual")),
    # --- relation_eef only -------------------------------------------------
    "percept": EventSpec(
        "per tick: RelationPerception.debug_snapshot() — detections, tracked "
        "poses, latch geometry, masks (on detector-cadence ticks)",
        ("step",)),
    "latch": EventSpec(
        "a GraspLatch state transition (perception's own event, re-stamped: "
        "original tick time kept as event_t)",
        ("event_t",)),
    "hand_state": EventSpec(
        "a commanded hand open/closed transition (perception's own event, "
        "re-stamped like `latch`)",
        ("event_t",)),
}


def build_meta(*, mode: str, action_mode: str, fps: int, horizon: int,
               strategy_params: dict, source: str, **extra) -> dict:
    """The ONE constructor for a session's meta.json base dict.

    `mode`: the strategy (sync/async/...). `action_mode`: joint/relative_eef/
    relation_eef. `strategy_params`: must contain every STRATEGY_PARAM_KEYS
    entry — pass the real values even when the mode doesn't use them.
    `source`: which tool recorded this ("runner", "replay_relation_openloop",
    ...) — a session directory should say what produced it. `extra`: anything
    else worth keeping (host/port, prompt, dataset, per-run config); keys must
    not collide with the base keys.

    The recorder's `start()` later merges its own clock epochs
    (t0_monotonic/t0_wall/started_iso/cameras) — those are capture-time facts,
    not construction-time ones, and stay out of here.
    """
    missing = [k for k in STRATEGY_PARAM_KEYS if k not in strategy_params]
    if missing:
        raise ValueError(
            f"build_meta: strategy_params missing {missing} — replay_record."
            "Session rebuilds the production buffer from these; a recording "
            "without them can only be replayed by guessing. Pass the real "
            "values (they are inert for sync but must still be recorded).")
    base = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "mode": mode,
        "action_mode": action_mode,
        "fps": int(fps),
        "horizon": int(horizon),
        **{k: strategy_params[k] for k in STRATEGY_PARAM_KEYS},
    }
    collisions = set(base) & set(extra)
    if collisions:
        raise ValueError(f"build_meta: extra keys collide with base keys: "
                         f"{sorted(collisions)}")
    return {**base, **extra}
