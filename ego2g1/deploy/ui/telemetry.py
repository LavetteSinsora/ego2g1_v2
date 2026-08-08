"""TelemetrySnapshot: the ONE declared shape of the dashboard's /state JSON
(docs/deploy_refactor_plan.md §5).

Before this module the page's dict was hand-assembled in four places
(DeployRunner.telemetry, perception_preview.PreviewLoop, replay_dashboard
.ReplayLoop, dashboard._DemoLoop) — pure structural typing that silently
rotted whenever the page gained a field. Now every producer constructs this
dataclass; a typo'd or missing key is a TypeError at the producer, and the
page's key set is enumerable for tests.

`relation_panel()` is the same idea for the relation_eef overlay panel: it
consumes the RECORDED `percept` event shape (`RelationPerception
.debug_snapshot()`'s own JSON) + the event history, so the live dashboard
and the replay dashboard build the panel from the same code path fed the
same shape — live rendering is "replay of the current instant".
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .. import actions as _actions
from ...core import layout


def executor_row_groups() -> list[dict]:
    """The page's row-group legend for the 26-dim executor row — was
    duplicated in three telemetry builders."""
    groups = [{"label": "L-arm", "start": 0, "stop": 7},
              {"label": "R-arm", "start": 7, "stop": _actions.ARM_DOF}]
    for h in layout.HANDS:
        groups.append({"label": f"{h[0].upper()}-hand",
                       "start": _actions.HAND[h].start,
                       "stop": _actions.HAND[h].stop})
    return groups


@dataclasses.dataclass
class TelemetrySnapshot:
    """Every key dashboard.js reads. Defaults are the honest "n/a" for a
    producer that doesn't have that plane (a preview tool has no strategy;
    a replay has no live budget)."""

    now: float
    mode: str = "?"
    server_rtc: bool = False
    active: bool = False
    recording: bool = False
    has_dataset: bool = False
    task: str = ""
    horizon: int = 0
    fps: int = 30
    dim: int = _actions.ROBOT_DIM
    ready: bool = False
    index: int = 0
    wall_slot: int | None = None
    trigger: int | None = None
    d: int | None = None
    action_row: list | None = None
    row_slot: int | None = None
    groups: list = dataclasses.field(default_factory=executor_row_groups)
    inferring: bool = False
    pending: bool = False
    worker_dead: bool = False
    last_splice: dict = dataclasses.field(default_factory=dict)
    stats: dict = dataclasses.field(
        default_factory=lambda: {"ticks": None, "chunks": None, "votes": None})
    budget: dict | None = None
    runway_s: float | None = None
    camera_age: float | None = None
    clamped_ticks: int = 0
    watchdog: dict = dataclasses.field(
        default_factory=lambda: {"tripped": False, "reason": None})
    arm_q: list | None = None
    state_age: float | None = None
    estopped: bool = False
    # The per-mode panel (DeployMode.telemetry_extras) or None. Carries a
    # "kind" discriminator so the page can render the right card; panels from
    # recordings written before that field existed are all relation_eef.
    relation: dict | None = None
    replay: dict | None = None      # replay_dashboard's scrub state or None (live)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def relation_panel(snapshot: dict | None, events: list | None) -> dict | None:
    """The dashboard's relation panel (objects/hands/events lists), built
    from the RECORDED `percept` shape — `RelationPerception.debug_snapshot()`
    live, or a `percept` event from a recorded session. One builder for
    both; the replay side previously re-implemented this against a
    reconstructed object graph and drifted.

    Old recordings' hand entries may predate the `hand_closed` field — the
    fallback ("any non-unlatched latch implies closed") reproduces what the
    replay builder always did for them.
    """
    if snapshot is None:
        return None
    objects = []
    for oid, o in (snapshot.get("objects") or {}).items():
        pose = o.get("tracked_pose")
        objects.append({
            "instance_id": oid,
            "detected_this_tick": bool(o.get("detected_this_tick", False)),
            "tracked": bool(o.get("tracked", False)),
            "depth_m": o.get("depth_m"),
            "confidence": (float(o["confidence"])
                          if o.get("confidence") is not None else None),
            "box_xyxy": ([float(x) for x in o["box_xyxy"]]
                        if o.get("box_xyxy") is not None else None),
            "position_pelvis": ([float(v) for v in np.asarray(pose)[:3, 3]]
                               if pose is not None else None),
        })

    hand_states = []
    for hand, h in (snapshot.get("hands") or {}).items():
        state = h.get("state", "unlatched")
        closed = h.get("hand_closed")
        if closed is None:                      # pre-field recording fallback
            closed = state != "unlatched"
        hand_states.append({
            "hand": hand,
            "hand_closed": bool(closed),
            "state": state,
            "candidate_object": h.get("candidate_object"),
            "latched_object": h.get("latched_object"),
            "ticks_in_candidate": int(h.get("ticks_in_candidate", 0)),
            "reason": h.get("reason"),
        })

    return {"kind": "relation", "objects": objects, "hands": hand_states,
            "events": list(events or [])}
