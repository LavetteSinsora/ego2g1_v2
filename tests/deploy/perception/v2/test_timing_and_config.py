"""The `d` arithmetic (T3/T4) and the config's fail-loud contract (§7)."""

from __future__ import annotations

import pytest

from ego2g1.deploy.perception.v2.config import PerceptionV2Config
from ego2g1.deploy.perception.v2.timing import (
    chunk_arithmetic_closes, delay_ticks, usable_slots,
)

# The plan's §2.1 measurements on the 4090.
P, L = 0.221, 0.124


# --- d ----------------------------------------------------------------------

def test_d_matches_the_plans_worked_example():
    """P + L = 345 ms = 10.4 ticks; x1.15 headroom = 397 ms = 11.9 -> 12."""
    assert delay_ticks(P, L) == 12


def test_feeding_policy_latency_alone_under_commits_by_a_whole_round():
    """`DelayBudget.observe()` is currently fed policy latency only, which
    would report d ~= 4 and silently execute past the frozen prefix — a lurch
    at the seam. The observation becomes available when the ROUND started,
    not when the request was sent."""
    assert delay_ticks(0.0, L) == 5
    assert delay_ticks(P, L) == 12


def test_d_fits_the_async_budget_but_a_slower_round_would_not():
    """12 ticks = 400 ms fits the 500 ms async budget. 16 ticks = 533 ms would
    exceed it and `startup_self_check` would refuse the mode."""
    assert delay_ticks(P, L) / 30.0 < 0.5
    assert delay_ticks(0.4, 0.15) / 30.0 > 0.5


def test_d_saturates_rather_than_growing_without_bound():
    """Unbounded, a slow round yields d > H: the chunk installs with zero
    usable slots and the replan trigger goes negative — the loop silently
    stops planning."""
    assert delay_ticks(5.0, 5.0, max_d=20) == 20
    assert delay_ticks(0.0, 0.0) == 1


def test_negative_latency_is_an_error():
    with pytest.raises(ValueError):
        delay_ticks(-1.0, 0.1)


def test_the_chunk_arithmetic_closes_at_the_constant_d():
    """Constant d = 12 leaves 38 slots = 1.27 s against a 1 s replan. A
    varying d of 17 leaves 33 = 1.10 s, which is at the edge once slip is
    added — the reason T3 waits for the in-flight round."""
    closes, seconds = chunk_arithmetic_closes(50, 12, 1.0)
    assert closes and seconds == pytest.approx(38 / 30)
    _, tight = chunk_arithmetic_closes(50, 17, 1.0)
    assert tight < seconds


def test_usable_slots_never_goes_negative():
    assert usable_slots(50, 80) == 0


# --- config -----------------------------------------------------------------

def _write(tmp_path, text):
    path = tmp_path / "perception.yaml"
    path.write_text(text)
    return path


def test_defaults_load_without_a_file():
    cfg = PerceptionV2Config.load(None)
    assert cfg.sam3.prune is True
    assert cfg.orient.size == 518
    assert cfg.latch.divergence_gate == "crop"


def test_retired_cadence_keys_fail_loudly(tmp_path):
    """An operator who sets one is expressing an intent the design no longer
    has a way to obey — perception is free-running, there is no cadence."""
    path = _write(tmp_path, "detector_period_ticks: 15\n")
    with pytest.raises(ValueError, match="free-running"):
        PerceptionV2Config.load(path)


def test_unknown_top_level_key_fails(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        PerceptionV2Config.load(_write(tmp_path, "wobble: 3\n"))


def test_typoed_nested_knob_fails_rather_than_keeping_the_default(tmp_path):
    """The operator believes they retuned something they did not — worse than
    a crash."""
    path = _write(tmp_path, "latch:\n  confirm_displacment_m: 0.04\n")
    with pytest.raises(ValueError, match="unknown 'latch' key"):
        PerceptionV2Config.load(path)


def test_nested_values_override(tmp_path):
    path = _write(tmp_path, """
sam3:
  prune: false
orient:
  size: 336
  cast_weights: true
visibility:
  min_det_score: 0.7
latch:
  divergence_gate: mask
tracker:
  max_speed_m_s: 2.0
""")
    cfg = PerceptionV2Config.load(path)
    assert cfg.sam3.prune is False
    assert cfg.orient.size == 336 and cfg.orient.cast_weights is True
    assert cfg.visibility.min_det_score == 0.7
    assert cfg.latch.divergence_gate == "mask"
    assert cfg.tracker == {"max_speed_m_s": 2.0}


def test_an_invalid_latch_gate_is_caught_at_load(tmp_path):
    path = _write(tmp_path, "latch:\n  divergence_gate: sometimes\n")
    with pytest.raises(ValueError, match="divergence_gate"):
        PerceptionV2Config.load(path)


def test_the_orientation_convention_is_recorded(tmp_path):
    """The most important field in meta.json: a rotation in the wrong
    canonical frame is still a valid rotation, so nothing can detect it after
    the fact. Knowing which convention produced a recording is the only way to
    reinterpret it."""
    import json

    cfg = PerceptionV2Config.load(
        _write(tmp_path, "convention:\n  azimuth_sign: 1.0\n"))
    blob = json.loads(json.dumps(cfg.as_dict()))
    assert blob["convention"]["azimuth_sign"] == 1.0
