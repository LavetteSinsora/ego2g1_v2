"""Raw-material extraction: episode HDF5 -> per-frame masks, boxes, orientations.

    uv run --group perception-v2 python -m data_extraction.extract \
        --episodes data/raw_hdf5/ego2g1/red_block_in_pen_holder_ego/episode_2.hdf5 \
        --prompts "red block,yellow block,black pen holder" \
        --out-dir data_extraction/out

One HDF5 in, one HDF5 out (plus a `.meta.json` sidecar). Point `--episodes` at
a directory to do the whole run; the models load once.

THE PIPELINE

    decode left JPEGs
      -> SAM 3 forward pass over the whole video   (hotstart removal ON)
      -> SAM 3 reverse pass over the whole video   (fills pre-detection frames)
      -> merge, retract hotstart-removed tracklets globally
      -> replay the deploy visibility gates (recorded, NOT enforced)
      -> Orient Anything V2 on EVERY mask, batched across frames
      -> write

Everything offline-specific and why it is worth having is documented in
`sam3_offline.py`'s module docstring; the orientation gating decision is in
`orient_offline.py`'s. Read those before changing a flag here.

RUN IT ON THE PPU BOX. `envs/` has the profiles. A 610-frame episode is two
full SAM 3 passes plus ~1800 orientation crops — minutes, not seconds, and the
staged video sits in host RAM at ~6 MB/frame.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

_DEFAULT_OUT = "data_extraction/out"


# ---------------------------------------------------------------------------
# roster
# ---------------------------------------------------------------------------

def _objects(prompts: str | None, episode, task_config: str | None):
    """The roster: three slots, in the order the packing depends on.

    Priority is explicit `--prompts`, then a real task config, then whatever
    the recorder wrote into the episode. The recorded prompts are a fallback
    and usually a coarser roster than the task (this dataset's episodes name
    two objects in `object_prompts_json` and three in the instruction), so
    using them silently would quietly change what is being measured — hence
    the loud print.
    """
    from ego2g1.deploy.perception.task_config import ObjectSpec, load_task_config

    if prompts:
        items = [p.strip() for p in prompts.split(",") if p.strip()]
        source = "--prompts"
    elif task_config:
        cfg = load_task_config(task_config)
        print(f"[roster] from task config {task_config}")
        return cfg.objects
    else:
        recorded = episode.recorded_prompts()
        if not recorded:
            raise SystemExit(
                f"{episode.name} has no object_prompts_json and no --prompts "
                f"was given. Nothing to detect.\n"
                f"  task instruction: {episode.task_instruction!r}")
        items = list(recorded.values())
        source = f"episode attrs ({list(recorded)})"

    objects = tuple(ObjectSpec(instance_id=f"obj{i}", category=p,
                               detector_prompt=p, graspable=True)
                    for i, p in enumerate(items))
    print(f"[roster] {len(objects)} slot(s) from {source}: "
          f"{[(o.instance_id, o.detector_prompt) for o in objects]}")
    print(f"[roster] anchor = {objects[0].instance_id} (roster order is "
          f"load-bearing: training gives obj_keys[0] the raw rotation)")
    return objects


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _report(episode, tracks, orientation, depth=None) -> None:
    import numpy as np

    from data_extraction.orient_offline import SKIP_NAMES, SKIP_NONE
    from data_extraction.sam3_offline import SOURCE_REVERSE

    print(f"\n--- {episode.name}: {tracks.n_frames} frames ---")
    print("  slot     mask%   det%  orient%  depth%   rev-only  presence  "
          "alpha   gate:crop%")
    for slot in tracks.slot_ids:
        rows = tracks.frames[slot]
        rev_only = sum(sf.source == SOURCE_REVERSE for sf in rows)
        crop_ok = sum(sf.crop_usable for sf in rows) / max(1, len(rows))
        pres = np.asarray([sf.presence for sf in rows], dtype=float)
        pres_med = float(np.nanmedian(pres)) if np.isfinite(pres).any() else float("nan")
        sym = (orientation.stats.get("symmetry") or {}).get(slot) or {}
        mode, agree = sym.get("mode"), sym.get("agreement", 0.0)
        a_txt = "--" if mode is None else f"{mode}({agree:.0%})"
        p_txt = "--" if pres_med != pres_med else f"{pres_med:.3f}"
        d_pct = 100 * depth.rate(slot) if depth is not None else float("nan")
        print(f"  {slot:8s} {100 * tracks.coverage(slot):5.1f}  "
              f"{100 * tracks.detection_rate(slot):5.1f}  "
              f"{100 * orientation.rate(slot):6.1f}  "
              f"{d_pct:6.1f}  {rev_only:8d}  {p_txt:>8s}  {a_txt:>7s}  "
              f"{100 * crop_ok:9.1f}")

    empty = [s for s in tracks.slot_ids if tracks.coverage(s) == 0.0]
    if empty:
        print(f"  [FAIL] slot(s) {empty} never produced a mask in EITHER "
              f"direction. That is the plan's §2.4 defect, and no downstream "
              f"analysis of this episode is meaningful.")
        # Presence separates the two diagnoses, which have opposite fixes.
        for slot in empty:
            pres = np.asarray([sf.presence for sf in tracks.frames[slot]],
                              dtype=float)
            if not np.isfinite(pres).any():
                continue
            hi = float(np.nanmax(pres))
            print(f"    {slot}: max presence {hi:.3f} — "
                  + ("SAM 3 says the concept IS in the video, so this is a "
                     "localisation/threshold problem, not a prompt problem."
                     if hi > 0.5 else
                     "SAM 3 never sees the concept at all; the PROMPT is "
                     "wrong for this scene, not the thresholds."))

    # Reverse-only frames are the direct, quantified answer to "was the second
    # pass worth it" — they are frames the online loop provably cannot have.
    total_rev = sum(sum(sf.source == SOURCE_REVERSE
                        for sf in tracks.frames[s]) for s in tracks.slot_ids)
    if "reverse" in tracks.stats.get("passes", []):
        print(f"  [reverse] contributed {total_rev} (frame, slot) masks that "
              f"the forward pass alone did not have.")

    skips: dict[str, int] = {}
    for slot in tracks.slot_ids:
        for code, name in SKIP_NAMES.items():
            if code == SKIP_NONE:
                continue
            n = int((orientation.skip[slot] == code).sum())
            if n:
                skips[name] = skips.get(name, 0) + n
    if skips:
        print(f"  [orient] not run: {skips}")


# ---------------------------------------------------------------------------

def main(
    *,
    episodes: str,
    out_dir: str = _DEFAULT_OUT,
    prompts: str | None = None,
    task_config: str | None = None,
    eye: str = "left",
    passes: str = "forward,reverse",
    device: str | None = None,
    dtype: str = "bfloat16",
    prune: bool = True,
    max_frames: int | None = None,
    orient: bool = True,
    orient_size: int = 518,
    orient_cast_weights: bool = False,
    orient_batch: int = 24,
    fit_symmetry: bool = True,
    depth: bool = True,
    save_depth_map: bool = False,
    perception_config: str | None = None,
    auto_download: bool = True,
    overwrite: bool = False,
    shard: str | None = None,
    reindex_only: bool = False,
    progress: bool = True,
):
    """Extract per-frame masks, boxes and orientations from raw episodes.

    episodes: one .hdf5 file, or a directory of them.
    prompts: comma-separated SAM 3 text prompts, one per object, IN ROSTER
        ORDER. Overrides both --task-config and the episode's own attrs.
    passes: which SAM 3 directions to run, comma-separated. "forward" alone
        reproduces what the deploy loop can see (minus hotstart); the default
        adds the reverse pass, which is the point of doing this offline.
    prune: bound the memory bank during each pass. Lossless at the corrected
        horizon (see `Sam3Source.prune`) — turn it off only to measure what it
        costs.
    orient_size: Orient Anything V2 input side. Deploy considers dropping this
        to 336 or 252 for latency (R2); offline there is no reason not to run
        the full 518, so that is the default and the comparison baseline.
    orient_batch: crops per forward. Offline-only lever — batches span frames,
        not just the three roster slots.
    fit_symmetry: run upstream's `val_fit_alpha` on every crop's azimuth
        distribution to get the rotational symmetry order (0/1/2/4). A scipy
        curve fit per crop, so it is the slowest cheap thing here; deploy
        defers it entirely (plan Q8) and this is what answers it.
    depth: stereo SGBM from the episode's OWN recorded per-eye intrinsics and
        extrinsics — nothing external to calibrate. Gives every object a
        median depth over its mask and a 3D point in the camera frame.
    save_depth_map: also store the full per-frame depth map (uint16 mm). Adds
        roughly 50-150 MB per episode; the per-object depths are stored
        either way, and those are what the depth is FOR.
    max_frames: truncate each episode. For a quick shape-check, not a result.
    shard: "k/n" — take every n-th episode starting at k, so n processes can
        split one directory across n accelerators. Each shard loads the models
        once and keeps them, which is the whole point: launching one process
        per EPISODE would pay the ~60 s model load 50 times over. Pin the
        device per shard from outside, e.g.
        `CUDA_VISIBLE_DEVICES=$k python -m data_extraction.extract
         --shard $k/16 ...`. Interleaved rather than contiguous so a shard
        does not inherit a run of unusually long episodes.
    """
    if reindex_only:
        # Rebuild index.json from whatever is on disk. Deliberately before any
        # model import, so it costs nothing and runs in a plain shell after a
        # sharded run has finished.
        out = Path(out_dir)
        found = sorted(out.glob("*.h5"))
        (out / "index.json").write_text(json.dumps(
            {"schema": "ego2g1.data_extraction.index/1",
             "files": [p.name for p in found]}, indent=2))
        print(f"index   : {out / 'index.json'} ({len(found)} file(s))")
        return

    import numpy as np  # noqa: F401  (imported early so a bad env fails fast)

    from data_extraction import orient_offline, stereo
    from data_extraction.episode import find_episodes, load_episode
    from data_extraction.sam3_offline import OfflineSam3
    from data_extraction.store import write_extraction
    from ego2g1.deploy.perception.v2.config import PerceptionV2Config
    from ego2g1.deploy.perception.v2.sam3_source import Sam3Source
    from ego2g1.deploy.perception_v2_latency import _preflight

    if not _preflight(auto=auto_download):
        return

    import torch

    cfg = PerceptionV2Config.load(perception_config)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tdtype = getattr(torch, dtype)
    pass_list = tuple(p.strip() for p in passes.split(",") if p.strip())
    files = find_episodes(episodes)
    out_dir = Path(out_dir)

    shard_label = ""
    if shard:
        try:
            k_str, n_str = shard.split("/")
            k, n_shards = int(k_str), int(n_str)
        except ValueError:
            raise SystemExit(f"--shard must look like 'k/n', got {shard!r}")
        if not 0 <= k < n_shards:
            raise SystemExit(f"--shard {shard}: need 0 <= k < n")
        files = files[k::n_shards]
        shard_label = f" [shard {k}/{n_shards}]"
        if not files:
            print(f"[shard {k}/{n_shards}] no episodes fall in this shard — "
                  f"fewer episodes than shards. Nothing to do.")
            return

    print("=" * 72)
    if dev.startswith("cuda"):
        pr = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"device : {pr.name} ({pr.total_memory / 1024 ** 3:.1f} GB)")
    else:
        print(f"device : {dev}  -- this will be extremely slow off the PPU box")
    print(f"torch  : {torch.__version__}")
    print(f"episodes: {len(files)} file(s) from {episodes}{shard_label}")
    print(f"passes  : {list(pass_list)}")
    print("=" * 72)

    # Roster comes from the FIRST episode when it is being read off attrs, so
    # every file in a directory run lands in the same slot order. A directory
    # whose episodes disagree about their objects is a mixed dataset and
    # should not be extracted under one roster anyway.
    first = load_episode(files[0], eye=eye)
    objects = _objects(prompts, first, task_config)

    t0 = time.perf_counter()
    sam3 = Sam3Source(objects, device=dev, dtype=tdtype, repo=cfg.sam3.repo,
                      prune=False,               # pruning is driven per pass
                      visibility=cfg.visibility)
    print(f"[setup] SAM 3 loaded in {time.perf_counter() - t0:.1f} s "
          f"(num_maskmem={sam3.num_maskmem}, "
          f"max_object_pointers={sam3.max_object_pointers}, "
          f"prune horizon={sam3.memory_horizon})")

    orient_model = None
    if orient:
        orient_model = _build_orient(cfg, dev, auto_download, orient_size,
                                     orient_cast_weights)
    if dev.startswith("cuda"):
        print(f"[setup] VRAM after models: "
              f"{torch.cuda.memory_allocated() / 1024 ** 2:.0f} MB")

    driver = OfflineSam3(sam3, passes=pass_list, prune=prune,
                         visibility=cfg.visibility, progress=progress)

    written, failed = [], []
    for path in files:
        out_path = out_dir / f"{Path(path).stem}.h5"
        if out_path.exists() and not overwrite:
            print(f"\n[skip] {out_path} exists (--overwrite to replace)")
            continue
        print(f"\n{'=' * 72}\n[episode] {path}")
        try:
            ep = load_episode(path, eye=eye, stereo=depth)
            if max_frames is not None and max_frames < ep.n_frames:
                ep._jpegs = ep._jpegs[:max_frames]
                if ep._jpegs_other is not None:
                    ep._jpegs_other = ep._jpegs_other[:max_frames]
                print(f"[episode] truncated to {max_frames} frames")
            print(f"[episode] {ep.n_frames} frames, {ep.width}x{ep.height}, "
                  f"eye={ep.eye}, stereo={'yes' if ep.has_stereo else 'no'}")

            t_ep = time.perf_counter()
            tracks = driver.run(ep)
            if orient_model is not None:
                orientation = orient_offline.estimate_over_episode(
                    orient_model, ep, tracks, batch_size=orient_batch,
                    fit_symmetry=fit_symmetry, progress=progress)
            else:
                orientation = orient_offline.empty_result(tracks)

            depth_result = stereo.empty_depth(tracks)
            if depth:
                rig = stereo.rig_from_episode(ep)
                if rig is not None:
                    depth_result = stereo.depth_over_episode(
                        rig, ep, tracks, sgbm=cfg.sgbm,
                        save_map=save_depth_map, progress=progress)

            meta = {
                "prompt_to_slot": dict(sam3.prompt_to_slot),
                "sam3": {"repo": cfg.sam3.repo, "dtype": dtype,
                         "device": dev, "passes": list(pass_list),
                         "prune": prune,
                         "num_maskmem": sam3.num_maskmem,
                         "memory_horizon": sam3.memory_horizon},
                "visibility": cfg.visibility.__dict__,
                "convention": cfg.convention.__dict__,
                "orientation": orientation.stats,
                "depth": depth_result.stats,
                "sam3_stats": tracks.stats,
                "wall_s": round(time.perf_counter() - t_ep, 2),
                "torch": torch.__version__,
            }
            write_extraction(out_path, episode=ep, tracks=tracks,
                             orientation=orientation, depth=depth_result,
                             meta=meta)
            _report(ep, tracks, orientation, depth_result)
            size_mb = out_path.stat().st_size / 1024 ** 2
            print(f"  -> {out_path} ({size_mb:.1f} MB, "
                  f"{meta['wall_s']:.0f} s)")
            written.append(out_path)
        except Exception as exc:                         # noqa: BLE001
            # One bad episode must not abandon a directory run that has
            # already paid for the model load.
            failed.append((path, repr(exc)))
            print(f"  [ERROR] {path}: {exc!r}")
            if progress:
                import traceback
                traceback.print_exc()
        finally:
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()

    print(f"\n{'=' * 72}\nwrote {len(written)} file(s) to {out_dir}")
    if failed:
        print(f"FAILED {len(failed)}:")
        for path, err in failed:
            print(f"  {path}: {err}")
    if written and shard:
        # Every shard globs the SAME directory, so concurrent shards would race
        # and whichever finishes first would publish an index missing the rest.
        # One rebuild after `wait` is correct by construction.
        print(f"index   : not written under --shard. After all shards finish:\n"
              f"          python -m data_extraction.extract --episodes {episodes} "
              f"--out-dir {out_dir} --reindex-only")
    elif written:
        index = out_dir / "index.json"
        index.write_text(json.dumps(
            {"schema": "ego2g1.data_extraction.index/1",
             "files": [p.name for p in sorted(out_dir.glob('*.h5'))]}, indent=2))
        print(f"index   : {index}")


def _build_orient(cfg, device: str, auto_download: bool, size: int,
                  cast_weights: bool):
    from ego2g1.deploy.perception.v2.orientation_v2 import OrientAnythingV2
    from ego2g1.deploy.perception_v2_latency import _ensure_oriany, _repo_root

    repo = _ensure_oriany(_repo_root() / cfg.orient.repo_dir, auto=auto_download)
    if repo is None:
        print("[warn] Orient Anything V2 unavailable — masks and boxes will be "
              "extracted, every rotation will be NaN.")
        return None
    t0 = time.perf_counter()
    model = OrientAnythingV2(repo, device=device,
                             checkpoint=cfg.orient.checkpoint,
                             cast_weights=cast_weights, size=size,
                             crop_pad=cfg.orient.crop_pad,
                             background=cfg.orient.background,
                             convention=cfg.convention)
    print(f"[setup] Orient Anything V2 loaded in {time.perf_counter() - t0:.1f} s "
          f"(size={model.size}, cast_weights={model.cast_weights}, "
          f"background={model.background})")
    return model


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
