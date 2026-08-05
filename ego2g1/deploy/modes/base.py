"""DeployMode: everything that differs between policy families, as one object
(docs/deploy_refactor_plan.md §2).

Before this package, "the modes differ only in actions.py/policy_adapter.py"
had drifted into `if relation_mode:` branches across ~8 files — a parallel
`_observe_relation`, a mode-conditional hand-state bookkeeping, a
mode-conditional recorder drain, a mode-conditional telemetry block, and a
mode-aware probe builder, all inside runner.py alone. Now `DeployRunner` (and
`runner.main`) ask the mode object; adding a policy family means one file in
this package plus a `register()` call, and nothing else
(tests/test_deploy_modes.py's fifth-mode smoke test is the executable proof).

The registry is keyed by the same names the server handshake's
`control_mode` uses. `resolve()` implements the CLI's `--action-mode auto`.
"""

from __future__ import annotations

import numpy as np

from ...core import layout


class DeployMode:
    """One policy family's contract with the runner. Stateless singleton —
    all per-rollout state lives on the adapter/runner; the mode object only
    knows HOW, never WHAT happened."""

    name: str = "?"
    supports_rtc: bool = False
    supports_reset_to_episode: bool = False

    # --- wiring (runner.main) ------------------------------------------------

    def build_adapter(self, client, args, fps: int):
        """Adapter + converter (+ perception, where applicable), including
        any fail-loud required-config checks. `args` is runner.Args (or a
        duck-typed stub carrying the fields this mode reads)."""
        raise NotImplementedError

    # --- the observation (runner loop step 1, and the startup probe) ---------

    def build_observation(self, executor, camera, last_hands: dict,
                          prompt: str) -> dict:
        """The request dict `adapter.infer` consumes. Also used for the
        startup latency probe (with `initial_hand_state()` as last_hands) —
        one definition, no separate probe builder."""
        raise NotImplementedError

    # --- hand-command bookkeeping (runner loop step 3) ------------------------

    def initial_hand_state(self) -> dict:
        """last_hands at rollout start (hands OPEN). The model's hand block
        is the command stream, never encoders — docs/deploy.md."""
        raise NotImplementedError

    def hand_state_from_row(self, row: np.ndarray, adapter) -> dict:
        """last_hands for the NEXT tick, recovered from the just-EXECUTED
        (26,) row."""
        raise NotImplementedError

    # --- optional per-mode extras (base = the no-op every simple mode wants) --

    def discard_probe_state(self, adapter) -> None:
        """After the startup latency self-check: the probe inference polluted
        the causal filters (and, for perception modes, seeded live state) —
        discard it all before the operator can press Start."""
        reset = getattr(adapter, "reset", None)
        if callable(reset):
            reset()

    def telemetry_extras(self, adapter) -> dict | None:
        """The dashboard's per-mode panel data (the `relation` key), or None
        ("n/a" on the page) for modes without one."""
        return None

    def record_tick(self, adapter, recorder, step: int, since_t: float) -> float:
        """Per-tick recorder drain of mode-specific events (runner loop step
        4b). Returns the new high-water mark for `since_t`; the base drains
        nothing."""
        return since_t


# --- shared base for the two proprioceptive modes ------------------------------


class ProprioModeBase(DeployMode):
    """joint / relative_eef: single-eye observation, (6,)-vector hand
    commands read straight off the executed row."""

    def build_observation(self, executor, camera, last_hands, prompt) -> dict:
        return {"arm_q": executor.arm_q(),
                "hand_cmds": {h: np.asarray(last_hands[h]).copy()
                              for h in layout.HANDS},
                "image": camera.read() if camera is not None else None,
                "prompt": prompt}

    def initial_hand_state(self) -> dict:
        return {h: np.zeros(layout.HAND_DIM) for h in layout.HANDS}

    def hand_state_from_row(self, row, adapter) -> dict:
        from .. import actions as _actions
        return {h: row[_actions.HAND[h]].copy() for h in layout.HANDS}


# --- registry -------------------------------------------------------------------

MODES: dict[str, DeployMode] = {}


def register(mode: DeployMode) -> DeployMode:
    MODES[mode.name] = mode
    return mode


def get(name: str) -> DeployMode:
    try:
        return MODES[name]
    except KeyError:
        raise ValueError(f"unknown action mode {name!r} "
                         f"(registered: {sorted(MODES)})") from None


def resolve_action_mode(action_mode: str, control_mode: str) -> str:
    """"auto" reads the checkpoint's control_mode off the server handshake
    (`client.PolicyClient.control_mode`) and picks the matching mode name;
    any other value passes through unchanged (useful only to deliberately
    test a mismatched pairing on purpose). An unknown/future control_mode
    falls back to relative_eef — the original two-way ternary's behavior."""
    if action_mode != "auto":
        return action_mode
    if control_mode in MODES:
        return control_mode
    return "relative_eef"


def resolve(action_mode: str, control_mode: str) -> DeployMode:
    return get(resolve_action_mode(action_mode, control_mode))
