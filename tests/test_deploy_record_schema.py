"""Phase-1 refactor pins: the recording schema is code, not convention
(docs/deploy_refactor_plan.md §4, §9 task 3).

  * every string-literal `*.log("<kind>", ...)` call site in the deploy
    package uses a kind declared in record/schema.py (the grep-walk that
    makes docstring drift impossible);
  * `Recorder.log` refuses an undeclared kind;
  * `build_meta` requires the strategy params and rejects key collisions;
  * a freshly recorded (v2) session round-trips through `Session` with its
    schema_version intact; a v1 session (meta without the key) still loads.
"""

import json
import pathlib
import re

import numpy as np
import pytest

from ego2g1.deploy import recorder as _recorder
from ego2g1.deploy.record import schema as _schema
from ego2g1.deploy.replay_record import Session

DEPLOY = pathlib.Path(__file__).resolve().parent.parent / "ego2g1" / "deploy"

# `<recorder-ish>.log("kind", ...)`: match any attribute call .log("...") but
# exclude logging (logger.log / logging.*) — the deploy code only ever calls
# .log with a string literal on recorder-shaped objects.
_LOG_CALL = re.compile(r"(?<!logger)\.log\(\s*\"([a-z_]+)\"")


def test_every_literal_log_kind_is_declared():
    undeclared = {}
    for p in DEPLOY.rglob("*.py"):
        for kind in _LOG_CALL.findall(p.read_text()):
            if kind not in _schema.EVENT_KINDS:
                undeclared.setdefault(p.name, []).append(kind)
    assert not undeclared, (
        f"undeclared recorder event kinds {undeclared} — declare them in "
        "ego2g1/deploy/record/schema.py")


def test_dynamic_drain_kinds_are_declared():
    # runner's step-4b drain dispatches variable kinds; pin the two it maps to
    assert "latch" in _schema.EVENT_KINDS
    assert "hand_state" in _schema.EVENT_KINDS


def test_recorder_refuses_undeclared_kind(tmp_path):
    rec = _recorder.Recorder(tmp_path / "s", meta=_meta())
    rec.start()
    try:
        with pytest.raises(ValueError, match="undeclared recorder event kind"):
            rec.log("obz", step=0)   # the typo class the assert exists for
    finally:
        rec.stop()


def _meta(**extra):
    return _schema.build_meta(
        mode="sync", action_mode="joint", fps=30, horizon=8,
        source="test",
        strategy_params={"inference_hz": 4.0, "exp_weight_m": 0.01,
                         "max_latency_steps": 8, "min_smooth_steps": 10},
        **extra)


def test_build_meta_requires_strategy_params():
    with pytest.raises(ValueError, match="strategy_params missing"):
        _schema.build_meta(mode="sync", action_mode="joint", fps=30, horizon=8,
                           source="test", strategy_params={"inference_hz": 4.0})


def test_build_meta_rejects_key_collisions():
    with pytest.raises(ValueError, match="collide"):
        # inference_hz reaches **extra (not a named param) and collides with
        # the base key the strategy_params dict already produced
        _meta(inference_hz=9.0)


def test_v2_round_trip_and_v1_fallback(tmp_path):
    # v2: recorded through build_meta
    rec = _recorder.Recorder(tmp_path / "v2", meta=_meta(prompt="x"))
    rec.start()
    chunk = np.zeros((4, 26))
    rec.log("infer_result", latency=0.1, horizon=4, mode="sync",
            start_timestep=0, actions=chunk)
    rec.log("action", step=0, row=chunk[0])
    rec.stop()
    s = Session(tmp_path / "v2")
    assert s.schema_version == _schema.SCHEMA_VERSION
    assert s.meta["source"] == "test"
    got, idx = s.chunk_at(s.span()[1])
    assert got.shape == (4, 26) and idx == 1

    # v1: a pre-schema session (hand-written meta, no schema_version)
    v1 = tmp_path / "v1"
    v1.mkdir()
    (v1 / "meta.json").write_text(json.dumps({"mode": "sync", "fps": 30}))
    (v1 / "events.jsonl").write_text(
        json.dumps({"t": 1.0, "kind": "action", "step": 0,
                    "row": [0.0] * 26}) + "\n")
    s1 = Session(v1)
    assert s1.schema_version == 1
    assert s1.at(2.0)["action_row"] is not None
