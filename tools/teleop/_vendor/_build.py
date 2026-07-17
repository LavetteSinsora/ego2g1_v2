"""Build `human_hand_teleoperate/_vendor/` — a self-contained copy of the retarget +
deploy code this package needs, so teleop imports NOTHING outside its own directory.

Two source trees, two styles (same idea as `ego2g1/deploy/vendor_g1_sim.py`):

  de/                <- data_extraction. RELATIVE imports (`from . import frames`), so a
                        byte-identical copy resolves within de/ unchanged.
  eg/                <- ego2g1. ABSOLUTE imports (`from ego2g1.common import layout`), so
                        each `ego2g1.` is rewritten to the right number of dots to become
                        relative within eg/. The rewrite is deterministic.
  eg/deploy/_g1_sim/ <- data_extraction sim/g1 + common/frames + G1 assets, built FRESH
                        from data_extraction (exactly as deploy/_g1_sim), because
                        kinematics.py imports it and it must match the label-time sim.

MANIFEST.json records, per file, the sha256 of the SOURCE (drift) and of the VENDORED
copy (integrity). tests/test_vendor_drift.py fails if the source changed (re-run this) or
the copy was corrupted.

    .venv/bin/python -m human_hand_teleoperate._vendor._build     # source = this repo
"""

import argparse
import hashlib
import json
import pathlib
import re
import shutil

HERE = pathlib.Path(__file__).resolve().parent            # _vendor/
REPO = HERE.parents[1]                                     # repo root

# byte-identical from data_extraction -> de/
DE_FILES = ["common/frames.py", "common/episode.py",
            "hand/constants.py", "hand/fk_tables.py", "hand/retarget.py"]
DE_TREES = ["assets/revo2"]
# ego2g1 (imports rewritten) -> eg/
EG_FILES = ["chunk_math.py", "common/layout.py", "common/se3.py",
            "deploy/safety.py", "deploy/trajectory.py", "deploy/ramp.py",
            "deploy/dds.py", "deploy/kinematics.py"]
# _g1_sim, built from data_extraction like deploy/vendor_g1_sim.py -> eg/deploy/_g1_sim/
G1SIM_FILES = ["sim/__init__.py", "sim/g1.py", "common/__init__.py", "common/frames.py"]
G1SIM_TREES = ["assets/unitree_g1"]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _iter_tree(root: pathlib.Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            yield p


def _rewrite_ego2g1(text: str, rel: str) -> str:
    """`from ego2g1.X import Y` -> relative, for a file at eg/<rel>.

    A file at eg/<d1>/.../<file>.py is `depth` dirs below eg's root, so eg's root is
    `.`*(depth+1) away. Replacing the literal `ego2g1` with that many dots turns every
    absolute ego2g1 import into the matching relative one (`ego2g1.common` -> `..common`
    at depth 1). Prose mentioning `ego2g1/` is untouched (no `from ego2g1.`).
    """
    depth = len(pathlib.PurePosixPath(rel).parent.parts)
    dots = "." * (depth + 1)
    if re.search(r"\bimport ego2g1\b(?!\.)", text):
        raise SystemExit(f"{rel}: bare `import ego2g1` — the rewrite only handles `from`")
    return re.sub(r"\bfrom ego2g1\.", f"from {dots}", text)


def _write_pkg(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    init = path / "__init__.py"
    if not init.exists():
        init.write_text("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=pathlib.Path, default=REPO,
                    help="repo root holding data_extraction/ and third_party/openpi/ego2g1/")
    args = ap.parse_args()
    de_src = args.source / "data_extraction"
    eg_src = args.source / "third_party" / "openpi" / "ego2g1"
    if not (de_src / "hand" / "retarget.py").exists() or not (eg_src / "deploy" / "dds.py").exists():
        raise SystemExit(f"data_extraction / ego2g1 not found under {args.source}")

    for sub in ("de", "eg"):
        if (HERE / sub).exists():
            shutil.rmtree(HERE / sub)

    manifest: list[dict] = []

    def take(vendor_rel: str, source_path: pathlib.Path, transform=None) -> None:
        raw = source_path.read_bytes()
        out = raw if transform is None else transform(raw.decode()).encode()
        dst = HERE / vendor_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(out)
        manifest.append({
            "vendor": vendor_rel,
            "source": source_path.relative_to(args.source).as_posix(),
            "src_sha": _sha(raw), "dst_sha": _sha(out)})

    # --- de/ (byte-identical) ---
    _write_pkg(HERE / "de")
    for rel in DE_FILES:
        _write_pkg((HERE / "de" / rel).parent)
        take(f"de/{rel}", de_src / rel)
    for tree in DE_TREES:
        for src in _iter_tree(de_src / tree):
            take(f"de/{src.relative_to(de_src).as_posix()}", src)

    # --- eg/ (imports rewritten) ---
    _write_pkg(HERE / "eg")
    for rel in EG_FILES:
        _write_pkg((HERE / "eg" / rel).parent)
        take(f"eg/{rel}", eg_src / rel, transform=lambda t, r=rel: _rewrite_ego2g1(t, r))

    # --- eg/deploy/_g1_sim/ (fresh from data_extraction, byte-identical) ---
    g1 = HERE / "eg" / "deploy" / "_g1_sim"
    for rel in G1SIM_FILES:
        (g1 / rel).parent.mkdir(parents=True, exist_ok=True)
        take(f"eg/deploy/_g1_sim/{rel}", de_src / rel)
    for tree in G1SIM_TREES:
        for src in _iter_tree(de_src / tree):
            take(f"eg/deploy/_g1_sim/{src.relative_to(de_src).as_posix()}", src)
    (g1 / "__init__.py").write_text(
        '"""Vendored G1 arm sim (data_extraction sim/g1 + common/frames + assets).\n'
        'Do NOT edit by hand — re-run human_hand_teleoperate._vendor._build."""\n')

    (HERE / "MANIFEST.json").write_text(
        json.dumps({"files": manifest}, indent=2, sort_keys=True) + "\n")
    print(f"vendored {len(manifest)} files into {HERE}")


if __name__ == "__main__":
    main()
