"""Pipeline orchestrator: raw Pico episodes -> LeRobot training dataset.

Usage (always from the repo root, repo venv):
    uv run python -m ego2g1.data.run_pipeline                 # all stages, all episodes
    uv run python -m ego2g1.data.run_pipeline --through s004 --limit 3
    uv run python -m ego2g1.data.run_pipeline --stages s001,s002_02 --force
    uv run python -m ego2g1.data.run_pipeline --set control_hz=15 --set repo_id=ego-pi/test

Stages (config.STAGE_ORDER):
    s001            resample streams onto the uniform control grid
    s003_placement  per-episode rigid placement S (reach + ready-pose)
    b_calib         [global] fixed wrist->flange alignment B
    s003_state      IK pass -> proprioception state + tracking-error signals
    s002_01         canonical flange poses (the action-label source) + gates
    s002_02         BrainCo Revo2 hand commands
    s004            filters -> sub-episodes with episode_real_end markers
    s004b_smooth    smooth action labels (fingers + EEF pose) within each good run
    s005            [global] write the LeRobot dataset + sidecar

Caching: a stage is skipped when its config dependency-closure hash and its
input signature both match the stored manifest (see common/io.py). --force
re-runs requested stages regardless.
"""

import argparse
import hashlib
import importlib
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import io
from .config import GLOBAL_STAGES, STAGE_ORDER, load_config

STAGE_MODULES = {
    "s001": "ego2g1.data.s001_uniformalize",
    "s003_placement": "ego2g1.data.s003_proprioception.placement_stage",
    "b_calib": "ego2g1.data.s002_action_label.b_alignment",
    "s003_state": "ego2g1.data.s003_proprioception.state_stage",
    "s002_01": "ego2g1.data.s002_action_label.eef_label",
    "s002_02": "ego2g1.data.s002_action_label.hand_label",
    "s004": "ego2g1.data.s004_filter_split",
    "s004b_smooth": "ego2g1.data.s004b_smooth",
    "s004c_resolve": "ego2g1.data.s004c_resolve",
    "s005": "ego2g1.data.s005_write_lerobot",
}

# stages whose inputs include the global B (their signature must change when
# b_calib's output changes, e.g. after adding episodes). s004b_smooth reads the
# B-dependent s002_01/s004 arrays, so it must re-run when B does.
B_DEPENDENT = {"s003_state", "s002_01", "s004", "s004b_smooth", "s004c_resolve"}


def find_episodes(cfg, paths, limit):
    if paths:
        out = []
        for p in map(Path, paths):
            out += sorted(p.glob("*.hdf5")) if p.is_dir() else [p]
    else:
        out = sorted(Path(cfg.episodes_dir).glob("*.hdf5"),
                     key=lambda p: (len(p.stem), p.stem))   # episode_2 < episode_10
    if not out:
        sys.exit(f"no episodes found (looked in {paths or cfg.episodes_dir})")
    return out[:limit] if limit else out


def b_digest(cfg):
    try:
        arrays, _ = io.load_stage(cfg, None, "b_calib")
    except FileNotFoundError:
        return "no-b"
    h = hashlib.sha256()
    # only the alignment itself: auxiliary b_calib outputs (e.g. the render
    # mount rotations) don't affect labels and must not invalidate stages
    for k in ("B_left", "B_right"):
        h.update(arrays[k].tobytes())
    return h.hexdigest()[:12]


def episode_sig(cfg, stage, ep_path):
    sig = io.source_signature(ep_path)
    if stage in B_DEPENDENT:
        sig += "|B:" + b_digest(cfg)
    return sig


def global_sig(cfg, stage, ep_paths):
    parts = [io.source_signature(p) for p in ep_paths]
    if stage in B_DEPENDENT or stage == "s005":
        parts.append("B:" + b_digest(cfg))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _run_episode_stage(args):
    """Top-level for ProcessPoolExecutor pickling."""
    module_name, stage, cfg, ep_path, sig = args
    mod = importlib.import_module(module_name)
    try:
        arrays, meta = mod.run_episode(cfg, ep_path)
    except Exception:
        return ep_path.stem, traceback.format_exc()
    io.save_stage(cfg, ep_path.stem, stage, arrays, meta, sig)
    return ep_path.stem, None


def run_stage(cfg, stage, ep_paths, jobs, force):
    mod_name = STAGE_MODULES[stage]
    if stage in GLOBAL_STAGES:
        sig = global_sig(cfg, stage, ep_paths)
        if not force and io.is_fresh(cfg, None, stage, sig):
            print(f"[{stage}] fresh, skipping")
            return
        print(f"[{stage}] running (global, {len(ep_paths)} episodes)")
        mod = importlib.import_module(mod_name)
        arrays, meta = mod.run_global(cfg, ep_paths)
        io.save_stage(cfg, None, stage, arrays, meta, sig)
        return

    todo = []
    for p in ep_paths:
        sig = episode_sig(cfg, stage, p)
        if force or not io.is_fresh(cfg, p.stem, stage, sig):
            todo.append((mod_name, stage, cfg, p, sig))
    print(f"[{stage}] {len(todo)}/{len(ep_paths)} episodes to run")
    failures = []
    if jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for name, err in ex.map(_run_episode_stage, todo):
                print(f"  [{stage}] {name}: {'FAILED' if err else 'ok'}")
                if err:
                    failures.append((name, err))
    else:
        for args in todo:
            name, err = _run_episode_stage(args)
            print(f"  [{stage}] {name}: {'FAILED' if err else 'ok'}")
            if err:
                failures.append((name, err))
    if failures:
        for name, err in failures:
            print(f"\n--- {stage} failure: {name} ---\n{err}")
        sys.exit(f"{stage}: {len(failures)} episode(s) failed")


def parse_set(kvs):
    out = {}
    for kv in kvs or []:
        k, _, v = kv.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="episode files/dirs (default: cfg.episodes_dir)")
    ap.add_argument("--stages", default=None,
                    help="comma-separated subset (default: all)")
    ap.add_argument("--through", default=None,
                    help="run all stages up to and including this one")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=None, help="JSON file of config overrides")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="single config override (repeatable)")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_set(args.set))
    print(f"config hash: {cfg.config_hash}")

    if args.stages and args.through:
        sys.exit("use --stages or --through, not both")
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",")]
        unknown = set(stages) - set(STAGE_ORDER)
        if unknown:
            sys.exit(f"unknown stages: {sorted(unknown)}")
        stages = [s for s in STAGE_ORDER if s in stages]   # execution order
    elif args.through:
        if args.through not in STAGE_ORDER:
            sys.exit(f"unknown stage: {args.through}")
        stages = STAGE_ORDER[:STAGE_ORDER.index(args.through) + 1]
    else:
        stages = list(STAGE_ORDER)

    ep_paths = find_episodes(cfg, args.episodes, args.limit)
    print(f"{len(ep_paths)} episodes, stages: {', '.join(stages)}")
    for stage in stages:
        run_stage(cfg, stage, ep_paths, args.jobs, args.force)
    print("done")


if __name__ == "__main__":
    main()
