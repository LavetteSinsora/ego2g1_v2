"""Skip the serve-side tests when the train group isn't installed.

Mirrors tests/train/conftest.py: ego2g1.serve.policy pulls in openpi/jax, which
only come with `uv sync --group train`.
"""

import pytest

pytest.importorskip("openpi", reason="train group not installed — run: uv sync --group train")
pytest.importorskip("jax", reason="train group not installed — run: uv sync --group train")
