"""Moved to ego2g1/deploy/tools/replay_mujoco.py (docs/deploy_refactor_plan.md
§1). This shim keeps the documented import path and `python -m`
entrypoint working; new code should import the real location."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("ego2g1.deploy.tools.replay_mujoco",
                     run_name="__main__", alter_sys=True)
else:
    import sys as _sys

    from .tools import replay_mujoco as _mod

    globals().update(
        {k: v for k, v in vars(_mod).items() if not k.startswith("__")})
    _sys.modules[__name__] = _mod
