"""Shared CLI fragments for the deploy tools (docs/deploy_refactor_plan.md
§6.3).

Before this module every tool redeclared `network_interface`, `dry_run`,
`yes`, `ik_iters`, ... with drifting defaults and, in check.py's case,
drifting NAMES (`iface` vs `network_interface`). The mixins put each default
in exactly one place; a tool that genuinely needs a different value overrides
the field explicitly with a comment saying why (see replay_dataset's
`ik_iters=40` — an offline re-solve has no 33 ms budget to honor).

All mixins are `kw_only` dataclasses so a tool's own required fields (e.g.
`dataset: str`) can follow them without the classic "non-default argument
follows default argument" ordering trap; tyro maps every field to a `--flag`
either way, so the CLI surface is unchanged.

`core/runner.py`'s Args deliberately does NOT inherit these (core must not
depend on tools/); tests/test_deploy_cli_mixins.py pins that the shared
defaults stay equal to the runner's instead.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class RobotArgs:
    """DDS + executor knobs every hardware-touching tool takes."""

    network_interface: str | None = None   # DDS iface; None joins the default domain
    max_pos_speed: float | None = None     # soften the interpolator cap for bring-up


@dataclasses.dataclass(kw_only=True)
class RunArgs:
    """The dry-run/confirmation pair every replay tool takes."""

    dry_run: bool = False                  # MockExecutor, no robot
    yes: bool = False                      # skip the real-robot confirmation prompt


@dataclasses.dataclass(kw_only=True)
class IKArgs:
    """The deploy IK's knobs, matching Kinematics' own signature defaults."""

    ik_iters: int = 25
    posture_cost: float = 0.05             # the measured smoothness knob
    collision_min_dist: float = 0.005
