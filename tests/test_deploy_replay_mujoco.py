"""The MuJoCo session replayer's reconstruction, headless (no viewer, no
display): measured pose lookup, commanded-FK markers, per-slot flange-target
markers, worst-error location, and the old-recording fallbacks.
"""

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from ego2g1.core import layout
from ego2g1.deploy import actions as _actions
from ego2g1.deploy.replay_mujoco import _Replayer, check

H = 5
T0 = 1000.0
DT = 0.1


def make_session(tmp_path, *, with_measured=True, with_targets=True):
    """A tiny hand-written sync session: one chunk, H action ticks."""
    d = tmp_path / "sess"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(
        {"mode": "sync", "fps": 30, "t0_monotonic": T0, "t0_wall": 0.0}))

    rng = np.random.default_rng(0)
    actions = np.zeros((H, _actions.ROBOT_DIM))
    actions[:, 0] = np.linspace(0.01, 0.05, H)          # a slow joint-0 sweep
    targets = {h: rng.normal(0.3, 0.05, size=(H, 3)).tolist()
               for h in layout.HANDS}

    events = []
    infer = {"t": T0, "kind": "infer_result", "latency": 0.2, "horizon": H,
             "mode": "sync", "actions": actions.tolist()}
    if with_targets:
        infer["flange_targets"] = targets
    events.append(infer)
    for k in range(H):
        t = T0 + DT * (k + 1)
        obs = {"t": t - 0.001, "kind": "obs", "step": k, "state_age": 0.0}
        if with_measured:
            obs["arm_q"] = (actions[k, :_actions.ARM_DOF] * 0.9).tolist()
        events.append(obs)
        events.append({"t": t, "kind": "action", "step": k,
                       "row": actions[k].tolist()})
    events.append({"t": T0 + 0.35, "kind": "tracking", "worst_m": 0.02})
    events.append({"t": T0 + 0.45, "kind": "tracking", "worst_m": 0.07})
    (d / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n")
    (d / "frames.jsonl").write_text("")
    return d, actions, targets


def test_frame_reconstruction(tmp_path):
    d, actions, targets = make_session(tmp_path)
    r = _Replayer(d)

    t = T0 + DT * 3 + 0.02                     # rows 0..2 popped; row 2 executing
    fr = r.frame(t)
    np.testing.assert_allclose(fr["measured_q"],
                               actions[2, :_actions.ARM_DOF] * 0.9)
    np.testing.assert_allclose(fr["commanded_q"],
                               actions[2, :_actions.ARM_DOF])
    # the RED marker is the logged slot-2 target, pelvis -> world
    for h in layout.HANDS:
        want = (r.base @ np.append(targets[h][2], 1.0))[:3]
        np.testing.assert_allclose(fr["targets"][h], want)
        assert np.isfinite(fr["commanded_flange"][h]).all()
        assert fr["tracking"][h] >= 0.0
    assert fr["ready"] and fr["phase"] == "executing"


def test_worst_tracking_time_reads_tracking_events(tmp_path):
    d, _, _ = make_session(tmp_path)
    r = _Replayer(d)
    wt, wv = r.worst_tracking_time()
    assert wv == pytest.approx(0.07)
    assert wt == pytest.approx(T0 + 0.45)


def test_fallbacks_for_old_recordings(tmp_path):
    """Pre-arm_q, pre-flange_targets sessions still replay: body falls back to
    the commanded joints and there are simply no RED markers."""
    d, actions, _ = make_session(tmp_path, with_measured=False,
                                 with_targets=False)
    r = _Replayer(d)
    fr = r.frame(T0 + DT * 2 + 0.02)
    assert fr["measured_q"] is None
    np.testing.assert_allclose(fr["commanded_q"],
                               actions[1, :_actions.ARM_DOF])
    assert fr["targets"] == {} and fr["tracking"] == {}
    assert {h for h in fr["commanded_flange"]} == set(layout.HANDS)


def test_check_runs_headless(tmp_path, capsys):
    d, _, _ = make_session(tmp_path)
    check(d, 20)
    out = capsys.readouterr().out
    assert "checked 20 frames" in out
    assert "reconstruction OK" in out
