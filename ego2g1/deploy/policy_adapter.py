"""The policy⇄execution boundary: everything downstream sees JOINT chunks.

Pattern from zh_deploy_inference/examples/unitree_inference/policy_adapter.py
(their EEFPolicyAdapter), rebuilt for ego2g1's model contract: the adapter
owns the model-facing observation (FK state), the action-mode conversion,
and the RTC prefix translation — so strategies.py and runner.py are
byte-identical whether the checkpoint speaks joints or anchor-relative EEF.

The adapter classes live with their modes now (docs/deploy_refactor_plan.md
§1: one file per policy family — modes/joint.py, modes/relative_eef.py,
modes/relation_eef.py; the shared reply tail is modes/eef.py's
`convert_with_diagnostics`). This module re-exports them under their
historical names, lazily (an eager import would be circular through
actions.py's row-layout constants), and keeps `make_adapter` as the factory
callers know.

Runner-side request dict (built by each mode's `build_observation`):

    {"arm_q":     (14,) measured arm joints at the observation tick,
     "hand_cmds": {hand: (6,)} LAST COMMANDED hand values (not encoders),
     "image":     (H, W, 3) uint8 RGB or None,
     "prompt":    str,
     # relation_eef only:
     "rgb_left"/"rgb_right": the stereo pair, "hand_cmds_last": {hand: float},
     # attached by AsyncStrategy when rtc=True:
     "enable_rtc": bool, "inference_delay": int ticks,
     "prev_action_chunk": (K, 26) JOINT rows — the leftover plan}

Adapter reply: {"actions": (H, 26) joint rows, ...server extras}.
"""

from __future__ import annotations

_MOVED = {
    "JointPolicyAdapter": ("joint", "JointPolicyAdapter"),
    "RelativeEEFPolicyAdapter": ("relative_eef", "RelativeEEFPolicyAdapter"),
    "RelationPolicyAdapter": ("relation_eef", "RelationPolicyAdapter"),
}


def __getattr__(name):
    if name in _MOVED:
        import importlib

        mod, cls = _MOVED[name]
        return getattr(importlib.import_module(f"{__package__}.modes.{mod}"), cls)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_adapter(action_mode: str, client, prompt: str = "", **kwargs):
    if action_mode == "joint":
        return __getattr__("JointPolicyAdapter")(client, prompt)
    if action_mode == "relative_eef":
        return __getattr__("RelativeEEFPolicyAdapter")(client, prompt, **kwargs)
    if action_mode == "relation_eef":
        return __getattr__("RelationPolicyAdapter")(client, prompt, **kwargs)
    raise ValueError(f"unknown action mode {action_mode!r}")
