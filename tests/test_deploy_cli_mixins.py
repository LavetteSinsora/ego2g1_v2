"""tools/cli.py mixins (docs/deploy_refactor_plan.md §6.3): the shared CLI
defaults live in one place, the tools compose them, and they stay equal to
the runner's own values (runner.Args deliberately does not inherit them —
core must not depend on tools/)."""

import dataclasses

from ego2g1.deploy.core.runner import Args as RunnerArgs
from ego2g1.deploy.tools import cli, replay_dataset, replay_diag
from ego2g1.deploy.tools import replay_relation_openloop as openloop


def _defaults(cls) -> dict:
    out = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
    return out


def test_tools_compose_the_mixins():
    for args_cls in (replay_dataset.Args, replay_diag.Args, openloop.Args):
        assert issubclass(args_cls, cli.RobotArgs)
        assert issubclass(args_cls, cli.RunArgs)
    assert issubclass(openloop.Args, cli.IKArgs)


def test_mixin_defaults_match_runner_args():
    runner = _defaults(RunnerArgs)
    for mixin in (cli.RobotArgs, cli.RunArgs, cli.IKArgs):
        for name, value in _defaults(mixin).items():
            if name not in runner:      # e.g. `yes` — the runner gates via
                continue                # Enter/--dashboard, not a flag
            assert runner[name] == value, (
                f"{mixin.__name__}.{name}={value!r} drifted from "
                f"runner.Args' {runner[name]!r}")


def test_deliberate_ik_iters_divergence_is_documented():
    """replay_dataset's ik_iters=40 (offline re-solve, no tick budget) is the
    one allowed divergence — pin it as deliberate, with its comment, so a
    future 'unification' doesn't silently flatten it either way."""
    assert _defaults(replay_dataset.Args)["ik_iters"] == 40
    import inspect

    src = inspect.getsource(replay_dataset)
    assert "deliberate divergence" in src


def test_args_still_construct_with_required_dataset():
    a = replay_dataset.Args(dataset="d")
    assert a.network_interface is None and a.dry_run is False
    b = openloop.Args(dataset="d")
    assert b.ik_iters == 25 and b.posture_cost == 0.05
