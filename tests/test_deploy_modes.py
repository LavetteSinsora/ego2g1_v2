"""The mode registry (deploy/modes/): the extensibility claim, executable
(docs/deploy_refactor_plan.md §2, §9 task 9).

The load-bearing test here is the FIFTH-MODE smoke test: register a brand-new
policy family in-test — its own observation shape, its own hand-state
bookkeeping, its own telemetry panel and recorder drain — and run a full
MockExecutor rollout through the UNMODIFIED DeployRunner. If adding a mode
ever again requires editing runner.py, this test is where that shows up.
"""

import numpy as np
import pytest

from ego2g1.core import layout
from ego2g1.deploy import actions as _actions
from ego2g1.deploy import modes as _modes
from ego2g1.deploy.executor import MockExecutor
from ego2g1.deploy.runner import DeployRunner
from ego2g1.deploy.strategies import SynchronousStrategy

FPS = 200
NO_WAIT = lambda t_end, **kw: None  # noqa: E731


def test_builtin_modes_registered():
    assert set(_modes.MODES) >= {"joint", "relative_eef", "relation_eef"}
    for name, mode in _modes.MODES.items():
        assert mode.name == name


def test_get_unknown_mode_fails_loud():
    with pytest.raises(ValueError, match="unknown action mode"):
        _modes.get("teleop_magic")


def test_mode_capability_flags():
    assert _modes.get("relative_eef").supports_rtc
    assert not _modes.get("relation_eef").supports_rtc
    assert not _modes.get("relation_eef").supports_reset_to_episode


# --------------------------------------------------------------------------
# the fifth-mode smoke test
# --------------------------------------------------------------------------


class _EchoMode(_modes.DeployMode):
    """A deliberately weird new family: scalar per-hand 'effort' hand state,
    an observation carrying a custom key, per-tick recorder drain."""

    name = "echo_test_mode"
    supports_rtc = False
    supports_reset_to_episode = False

    def build_adapter(self, client, args, fps):   # pragma: no cover - unused
        raise NotImplementedError

    def build_observation(self, executor, camera, last_hands, prompt,
                          adapter=None):
        # `adapter` is part of the DeployMode contract (base.py): it is the
        # only hook called at CONTROL rate, so a mode that accumulates
        # per-tick state (umi_eef's pose history) needs it here. Ignored by
        # every mode that does not.
        return {"arm_q": executor.arm_q(), "efforts": dict(last_hands),
                "image": None, "prompt": prompt}

    def initial_hand_state(self):
        return {h: 0.0 for h in layout.HANDS}

    def hand_state_from_row(self, row, adapter):
        return {h: float(row[_actions.HAND[h]].mean()) for h in layout.HANDS}

    def telemetry_extras(self, adapter):
        return {"echo": True}

    def record_tick(self, adapter, recorder, step, since_t):
        recorder.log("percept", step=step, echo=True)   # reuse a declared kind
        return since_t


class _EchoPolicy:
    """Adapter for _EchoMode: consumes its custom observation shape."""

    def __init__(self, horizon=4):
        self.horizon = horizon
        self.prompt = "echo"
        self.calls = 0
        self.mode = "echo_test_mode"

    def infer(self, request):
        assert "efforts" in request, "runner must use the mode's observation"
        self.calls += 1
        rows = np.zeros((self.horizon, _actions.ROBOT_DIM))
        rows[:, _actions.ARM] = np.asarray(request["arm_q"])
        for h in layout.HANDS:
            rows[:, _actions.HAND[h]] = 0.25
        return {"actions": rows}


class _LogRecorder:
    def __init__(self):
        self.events = []

    def log(self, kind, **fields):
        self.events.append({"kind": kind, **fields})


@pytest.fixture
def echo_mode():
    _modes.register(_EchoMode())
    yield _modes.get("echo_test_mode")
    _modes.MODES.pop("echo_test_mode", None)


def test_fifth_mode_full_rollout_through_unmodified_runner(echo_mode):
    policy = _EchoPolicy()
    executor = MockExecutor(fps=FPS)
    executor.connect()
    rec = _LogRecorder()
    strategy = SynchronousStrategy(policy, chunk_size=policy.horizon)
    runner = DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                          recorder=rec, fps=FPS, wait=NO_WAIT, max_steps=8)
    runner.run()

    assert not runner.watchdog.tripped
    assert runner.steps_executed == 8
    assert policy.calls >= 2
    # the mode's hand bookkeeping ran: scalar mean of the 0.25 hand block
    for h in layout.HANDS:
        assert runner.last_hands[h] == pytest.approx(0.25)
    # the mode's telemetry panel and recorder drain both flowed through
    assert runner.telemetry()["relation"] == {"echo": True}
    assert sum(1 for e in rec.events
               if e["kind"] == "percept" and e.get("echo")) == 8


def test_runner_defaults_mode_from_adapter_attribute(echo_mode):
    policy = _EchoPolicy()
    executor = MockExecutor(fps=FPS)
    executor.connect()
    strategy = SynchronousStrategy(policy, chunk_size=policy.horizon)
    # no mode= passed: the adapter's own .mode attribute selects it
    runner = DeployRunner(adapter=policy, strategy=strategy, executor=executor,
                          fps=FPS, wait=NO_WAIT, max_steps=4)
    assert runner.mode is echo_mode
