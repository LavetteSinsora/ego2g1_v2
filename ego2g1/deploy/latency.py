"""Moved to ego2g1/deploy/core/latency.py (docs/deploy_refactor_plan.md
§1). This shim keeps the documented import path and `python -m`
entrypoint working; new code should import the real location."""

import sys as _sys

from .core import latency as _mod

globals().update(
    {k: v for k, v in vars(_mod).items() if not k.startswith("__")})
_sys.modules[__name__] = _mod
