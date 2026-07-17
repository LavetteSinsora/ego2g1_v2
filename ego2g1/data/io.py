"""Stage output storage + freshness checks for the orchestrator.

Layout: work_dir/<episode>/<stage>.npz + <stage>.manifest.json per episode
stage, work_dir/_global/<stage>.* for global stages. A stage output is fresh
when its manifest records the current config stage-hash (dependency-closure
hash, see config.STAGE_FIELDS) and the same source-file signature.
"""

import json
from pathlib import Path

import numpy as np

GLOBAL_KEY = "_global"


def _dir(cfg, ep_name):
    d = Path(cfg.work_dir) / (ep_name or GLOBAL_KEY)
    d.mkdir(parents=True, exist_ok=True)
    return d


def npz_path(cfg, ep_name, stage):
    return _dir(cfg, ep_name) / f"{stage}.npz"


def manifest_path(cfg, ep_name, stage):
    return _dir(cfg, ep_name) / f"{stage}.manifest.json"


def source_signature(path):
    st = Path(path).stat()
    return f"{Path(path).resolve()}:{st.st_size}:{int(st.st_mtime)}"


def is_fresh(cfg, ep_name, stage, source_sig):
    mp = manifest_path(cfg, ep_name, stage)
    if not mp.exists() or not npz_path(cfg, ep_name, stage).exists():
        return False
    try:
        m = json.loads(mp.read_text())
    except json.JSONDecodeError:
        return False
    return (m.get("stage_hash") == cfg.stage_hash(stage)
            and m.get("source_sig") == source_sig)


def save_stage(cfg, ep_name, stage, arrays, meta, source_sig):
    np.savez_compressed(npz_path(cfg, ep_name, stage), **arrays)
    manifest_path(cfg, ep_name, stage).write_text(json.dumps({
        "stage_hash": cfg.stage_hash(stage),
        "source_sig": source_sig,
        "meta": meta,
    }, indent=2, default=str))


def load_stage(cfg, ep_name, stage):
    """-> (dict of arrays, meta dict). Raises if the stage never ran."""
    mp = manifest_path(cfg, ep_name, stage)
    if not mp.exists():
        raise FileNotFoundError(
            f"stage {stage} has no output for {ep_name or GLOBAL_KEY} - run it first")
    with np.load(npz_path(cfg, ep_name, stage), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(mp.read_text())["meta"]
