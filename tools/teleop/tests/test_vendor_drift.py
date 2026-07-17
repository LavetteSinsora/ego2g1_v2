"""The vendored copies in `_vendor/` must not silently drift from their sources.

This is what lets the teleop package be self-contained WITHOUT the retarget quietly
diverging from the code that produced the training labels. Two checks per file:

  integrity  the vendored file matches its recorded hash — runs everywhere, incl. the
             robot PC, which has only the copy.
  drift      the SOURCE file matches its recorded hash — runs only where data_extraction
             and ego2g1 are present. A mismatch means someone changed the source and must
             re-run `python -m tools.teleop._vendor._build`.

The eg/ files are rewritten (absolute ego2g1 imports -> relative), so their vendored hash
differs from the source hash; the manifest records both, and re-running _build reproduces
the vendored hash deterministically from the source.
"""

import hashlib
import json
import pathlib

import pytest

VENDOR = pathlib.Path(__file__).resolve().parents[1] / "_vendor"
REPO = pathlib.Path(__file__).resolve().parents[2]


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _manifest():
    m = json.loads((VENDOR / "MANIFEST.json").read_text())["files"]
    assert m, "empty manifest — _vendor/_build never ran"
    return m


def test_vendored_files_are_intact():
    """Every vendored file matches the hash recorded when it was built."""
    for e in _manifest():
        v = VENDOR / e["vendor"]
        assert v.exists(), f"vendored file missing: {e['vendor']}"
        assert _sha(v) == e["dst_sha"], f"vendored {e['vendor']} corrupt vs MANIFEST"


def test_sources_have_not_drifted():
    """Every source file still matches the hash it had when vendored."""
    if not (REPO / "data_extraction").exists() or not (REPO / "third_party" / "openpi" / "ego2g1").exists():
        pytest.skip("source trees absent (e.g. robot PC) — integrity checked, drift cannot be")
    drifted = [e["source"] for e in _manifest() if _sha(REPO / e["source"]) != e["src_sha"]]
    assert not drifted, ("source changed since vendoring; re-run "
                         "`python -m tools.teleop._vendor._build`. Stale: "
                         + ", ".join(sorted(drifted)))
