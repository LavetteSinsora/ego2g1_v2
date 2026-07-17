"""Skip the train-side tests when the train group isn't installed.

Plain `uv sync` gives the default set (core+kin+data); openpi/jax/flax come
with `uv sync --group train`. On a machine without them these tests should
skip with a pointer, not break collection.
"""

import pytest

pytest.importorskip("openpi", reason="train group not installed — run: uv sync --group train")
pytest.importorskip("jax", reason="train group not installed — run: uv sync --group train")
