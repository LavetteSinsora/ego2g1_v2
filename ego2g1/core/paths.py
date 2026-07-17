"""Single anchor for every on-disk location.

All path resolution routes through here so nothing else in the package ever
hardcodes a location — the repo can live anywhere, and machine-specific
overrides are env vars, not code edits.
"""

import os
from pathlib import Path

# ego2g1/core/paths.py -> ego2g1/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def assets_dir() -> Path:
    """Robot models (G1 MJCF + Revo2). The single copy in the repo."""
    return Path(os.environ.get("EGO2G1_ASSETS", REPO_ROOT / "assets"))


def data_dir() -> Path:
    """Raw recordings + generated datasets (git-ignored; see docs/datasets.md)."""
    return Path(os.environ.get("EGO2G1_DATA", REPO_ROOT / "data"))


def work_dir() -> Path:
    """Extraction pipeline recompute cache (content-addressed, regenerable)."""
    return Path(os.environ.get("EGO2G1_WORK", REPO_ROOT / "ego2g1" / "data" / "work"))
