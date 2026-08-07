"""End-to-end perception v2 bench: SAM 3 + Orient Anything V2 co-resident.

    uv run --group perception-v2 python -m ego2g1.deploy.perception_v2_e2e \
        --prompts "red cube,yellow cube,black pen holder"

Runs the REAL pipeline — the same `PerceptionRound` the deploy loop will use,
not a reimplementation — from stereo capture through to the 56-dim relation
vector, with both models resident on one GPU. No robot is needed: a synthetic
30 Hz control loop publishes identity flange poses into a `ControlTickLog`, so
the tick binding (T4) and the single-instant composition (T2) run exactly as
they will on hardware.

THE HEADLINE QUESTION
    Does round time degrade as the memory bank grows?

Latency and memory are usually treated as separate budgets. They are coupled
here: SAM 3's memory bank grows ~11.5 MB/frame, and a torch allocator under
pressure fragments, falls back to slower paths, and eventually syncs. So a
pipeline that is fast for 30 s can be slow at 5 minutes with nothing else
changed. `--compare-prune` answers it as a DELTA — the same workload run with
pruning off and on — because an absolute number cannot distinguish "the loop
is slow" from "the loop got slow".

WHAT ELSE IT REPORTS
    * per-round latency deciles, so a drift is visible as a trend not a mean
    * VRAM (current, not peak — a high-water mark can never show a prune
      working) and stored memory-bank entries, sampled together with latency
      so the correlation is readable off one table
    * per-slot detection / gate outcomes, so a roster slot that never fills
      (the plan's §2.4 defect) shows up immediately
    * the 56-dim vector's own health

Latching is deliberately NOT exercised: the synthetic hand never closes, so
every latch stays UNLATCHED and object poses pass through from perception.
That stage needs real hardware work and is being left until last.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

_DEFAULT_PROMPTS = "red cube,yellow cube,black pen holder"


# ============================================================================
# a robot that is not there
# ============================================================================

class SyntheticControlLoop:
    """Publishes control ticks at `fps` so perception has something to bind to.

    The real control loop is the only thing that ever writes here, and the
    perception thread only ever reads — so this stands in for it exactly, and
    the tick binding under test is the real one. Flange poses are fixed:
    nothing in the latency or memory question depends on the arm moving, and a
    static arm makes a drifting object pose obviously a perception artefact
    rather than a kinematics one.
    """

    def __init__(self, tick_log, hands=("left", "right"), fps: int = 30):
        self.tick_log = tick_log
        self.hands = tuple(hands)
        self.dt = 1.0 / float(fps)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.ticks = 0

    def _pose(self, i: int) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = [0.3, 0.2 if i == 0 else -0.2, 0.1]
        return T

    def start(self):
        def loop():
            n = 0
            flange = {h: self._pose(i) for i, h in enumerate(self.hands)}
            hand_frac = {h: 0.0 for h in self.hands}      # never closes
            while not self._stop.is_set():
                self.tick_log.record(n, time.monotonic(), flange, hand_frac)
                self.ticks = n
                n += 1
                time.sleep(self.dt)

        self._thread = threading.Thread(target=loop, daemon=True,
                                        name="synthetic-control")
        self._thread.start()
        time.sleep(0.1)                                   # let a few ticks land

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)


# ============================================================================
# build
# ============================================================================

def _task_config(prompts: list[str], path: str | None):
    """Load the real task config, or synthesise one from --prompts.

    Synthesising keeps this runnable before `task_config.yaml` exists (plan
    Q17) — the bench does not need the checkpoint cross-check, and demanding
    it would block the measurement on an unrelated prerequisite.
    """
    from ego2g1.deploy.perception.task_config import (
        DeployTaskConfig, ObjectSpec, load_task_config,
    )
    if path:
        cfg = load_task_config(path)
        print(f"[setup] task config {path}: "
              f"{[o.instance_id for o in cfg.objects]}")
        return cfg
    objects = tuple(ObjectSpec(instance_id=f"obj{i}", category=p,
                               detector_prompt=p, graspable=True)
                    for i, p in enumerate(prompts))
    print(f"[setup] no --task-config; synthesised roster from --prompts: "
          f"{[o.instance_id for o in objects]} (anchor = {objects[0].instance_id})")
    return DeployTaskConfig(objects=objects)


def _calib(path: str | None, shape):
    from ego2g1.deploy.perception.depth import StereoCalibration
    if path:
        return StereoCalibration.load(path)
    h, w = shape[:2]
    print(f"[warn] no stereo calibration — PLACEHOLDER sized to {w}x{h}. SGBM's "
          f"COST is right; depth VALUES are meaningless, so object positions "
          f"are not.")
    K = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1.0]])
    return StereoCalibration(K_left=K, K_right=K.copy(), dist_left=np.zeros(5),
                             dist_right=np.zeros(5), R=np.eye(3),
                             T=np.array([0.06, 0, 0]), image_size=(w, h))


def _extrinsic(path: str | None) -> np.ndarray:
    if path:
        return np.load(path)["T_pelvis_camera"]
    print("[warn] no --camera-extrinsic — using identity. Object poses are "
          "then in the CAMERA frame, which is fine for latency/memory and "
          "wrong for anything geometric.")
    return np.eye(4)


def _camera(fake: bool, host: str, w: int, h: int):
    if fake:
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        # Textured AND shifted so SGBM has a real disparity to find; a blank
        # pair under-reports its cost.
        return (lambda: (base, np.roll(base, -8, axis=1))), (lambda: None)
    from ego2g1.deploy.camera import HeadCamera
    cam = HeadCamera(host=host, eye="left")
    cam.connect()

    def read():
        left, right = cam.read_stereo()
        if left is None or right is None:
            raise RuntimeError("camera has no stereo frame yet")
        return left, right

    return read, cam.close


def _build_round(*, task_config, calib, T_pelvis_camera, read, device, dtype,
                 prune, orient, cfg):
    from ego2g1.deploy.perception.depth import StereoSGBMDepthSource
    from ego2g1.deploy.perception.v2.async_perception import PerceptionRound
    from ego2g1.deploy.perception.v2.sam3_source import Sam3Source
    from ego2g1.deploy.perception.v2.snapshot import ControlTickLog

    t0 = time.perf_counter()
    sam3 = Sam3Source(task_config.objects, device=device, dtype=dtype,
                      repo=cfg.sam3.repo, prune=prune,
                      visibility=cfg.visibility)
    print(f"[setup] SAM 3 loaded in {time.perf_counter() - t0:.1f} s "
          f"(num_maskmem={sam3.num_maskmem}, prune={'ON' if prune else 'OFF'})")

    tick_log = ControlTickLog(maxlen=120)
    control = SyntheticControlLoop(tick_log, task_config.hands)
    round_ = PerceptionRound(
        read_stereo=read, tick_log=tick_log, sam3=sam3,
        depth_source=StereoSGBMDepthSource(calib, **cfg.sgbm), calib=calib,
        T_pelvis_camera=T_pelvis_camera, objects=task_config.objects,
        orientation=orient, tracker_kwargs=cfg.tracker,
        anchor_id=cfg.orient.anchor_id)
    return round_, sam3, control


def _build_orient(cfg, device: str, auto_download: bool):
    from ego2g1.deploy.perception.v2.orientation_v2 import OrientAnythingV2
    from ego2g1.deploy.perception_v2_latency import _ensure_oriany, _repo_root

    repo = _ensure_oriany(_repo_root() / cfg.orient.repo_dir,
                          auto=auto_download)
    if repo is None:
        print("[warn] Orient Anything V2 unavailable — running position-only. "
              "Every rotation stays at its nominal value.")
        return None
    t0 = time.perf_counter()
    orient = OrientAnythingV2(repo, device=device,
                              checkpoint=cfg.orient.checkpoint,
                              cast_weights=cfg.orient.cast_weights,
                              size=cfg.orient.size,
                              crop_pad=cfg.orient.crop_pad,
                              background=cfg.orient.background,
                              convention=cfg.convention)
    print(f"[setup] Orient Anything V2 loaded in {time.perf_counter() - t0:.1f} s "
          f"(size={orient.size}, cast_weights={orient.cast_weights}, "
          f"background={orient.background})")
    return orient


# ============================================================================
# run
# ============================================================================

def _vram_mb(device: str) -> float | None:
    """CURRENT allocation. Not `max_memory_allocated`, which is a high-water
    mark that only ever rises and so can never show a prune working."""
    if not str(device).startswith("cuda"):
        return None
    import torch
    return torch.cuda.memory_allocated() / 1024 ** 2


def _run(round_, sam3, builder, rounds: int, device: str, label: str,
         sample_every: int = 25) -> dict:
    print(f"\n--- {label}: {rounds} rounds ---")
    times, trace, misses, state_err = [], [], 0, None

    for i in range(rounds):
        t0 = time.perf_counter()
        snapshot = round_.step()
        times.append(time.perf_counter() - t0)
        if snapshot is None:
            misses += 1
            continue
        if builder is not None:
            builder.on_snapshot(snapshot)
            builder.on_control_tick(t=time.monotonic(),
                                    flange_poses=snapshot.flange_pelvis,
                                    hand_frac=snapshot.hand_frac)
            try:
                builder.state_for(snapshot)
            except RuntimeError as exc:
                state_err = str(exc)
        if i % sample_every == 0:
            non_cond, cond = sam3.stored_frames()
            trace.append((i, _vram_mb(device), non_cond, cond,
                          sum(snapshot.crop_usable.values())))

    t = np.asarray(times) * 1e3
    d = max(1, len(t) // 10)
    print("  deciles (ms): " + "  ".join(f"{t[i * d:(i + 1) * d].mean():.0f}"
                                         for i in range(10)))
    p50, p95 = np.percentile(t, [50, 95])
    print(f"  mean {t.mean():.1f}  p50 {p50:.1f}  p95 {p95:.1f}  "
          f"max {t.max():.1f} ms   -> {1000 / max(p95, 1e-9):.1f} Hz")

    if trace and trace[0][1] is not None:
        print("  round   vram    non-cond  cond   usable")
        for i, vram, nc, cd, usable in trace[:12]:
            print(f"  {i:5d}  {vram:7.0f} {nc:9d} {cd:6d} {usable:7d}")
        if len(trace) > 12:
            print(f"  ... ({len(trace) - 12} more samples)")

    first, last = t[:d].mean(), t[-d:].mean()
    drift = last - first
    print(f"  [drift] first decile {first:.1f} ms -> last {last:.1f} ms "
          f"({drift:+.1f} ms, {100 * drift / max(first, 1e-9):+.1f}%)")
    if misses:
        print(f"  [warn] {misses} round(s) dropped — no control tick to bind to")
    if state_err:
        print(f"  [warn] relation state unavailable: {state_err}")
    return {"p95": float(p95), "mean": float(t.mean()), "drift_ms": float(drift),
            "vram_end": trace[-1][1] if trace else None,
            "non_cond_end": trace[-1][2] if trace else None}


def _report_slots(snapshot, task_config) -> None:
    print("\n--- per-slot outcome (last round) ---")
    print("  slot        det    trk   area   mask  crop   depth   pose")
    for obj in task_config.objects:
        oid = obj.instance_id
        det = snapshot.det_score.get(oid)
        depth = snapshot.object_depth_m.get(oid)
        pose = snapshot.object_pose_pelvis.get(oid)
        print(f"  {oid:10s} {'--' if det is None else f'{det:.2f}':>5s} "
              f"{snapshot.tracker_score.get(oid, 0):6.2f} "
              f"{snapshot.mask_area_px.get(oid, 0):6d} "
              f"{str(snapshot.mask_usable.get(oid))[:5]:>6s} "
              f"{str(snapshot.crop_usable.get(oid))[:5]:>5s} "
              f"{'--' if depth is None else f'{depth:.3f}':>7s} "
              f"{'--' if pose is None else 'ok':>6s}")
    empty = [o.instance_id for o in task_config.objects
             if snapshot.object_pose_pelvis.get(o.instance_id) is None]
    if empty:
        print(f"  [FAIL] slot(s) {empty} never filled. A roster slot empty for "
              f"the whole episode blocks everything downstream (plan §2.4).")


# ============================================================================

def main(
    *,
    prompts: str = _DEFAULT_PROMPTS,
    task_config: str | None = None,
    perception_config: str | None = None,
    stereo_calib: str | None = None,
    camera_extrinsic: str | None = None,
    camera_host: str | None = None,
    fake_camera: bool = False,
    device: str | None = None,
    dtype: str = "bfloat16",
    rounds: int = 300,
    prune: bool = True,
    compare_prune: bool = False,
    skip_orient: bool = False,
    orient_size: int | None = None,
    orient_cast_weights: bool | None = None,
    auto_download: bool = True,
):
    """Run the v2 pipeline end to end and report latency vs memory growth.

    rounds: perception rounds per phase. 300 is ~70 s of wall clock at the
        measured 4.5 Hz; use 2000+ to chase a multi-minute drift.
    compare_prune: run the whole workload twice, pruning off then on, and
        print the delta. This is the direct answer to "does memory growth slow
        computation" — an absolute number cannot distinguish a slow loop from
        a loop that GOT slow.
    """
    import os

    from ego2g1.deploy.perception.v2.config import PerceptionV2Config
    from ego2g1.deploy.perception.v2.relation import RelationStateBuilder
    from ego2g1.deploy.perception_v2_latency import _preflight, _repo_root

    if not _preflight(auto=auto_download):
        return

    cfg = PerceptionV2Config.load(perception_config)
    if orient_size is not None:
        cfg = dataclasses_replace(cfg, "orient", size=orient_size)
    if orient_cast_weights is not None:
        cfg = dataclasses_replace(cfg, "orient", cast_weights=orient_cast_weights)

    root = _repo_root()
    if stereo_calib is None:
        p = root / "stereo_calib.npz"
        stereo_calib = str(p) if p.is_file() else None
    camera_host = camera_host or os.environ.get("EGO2G1_CAMERA_HOST",
                                                "192.168.123.164")

    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tdtype = getattr(torch, dtype)
    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]

    print("=" * 72)
    if dev.startswith("cuda"):
        pr = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"device : {pr.name} ({pr.total_memory / 1024 ** 3:.1f} GB)")
    else:
        print(f"device : {dev}  -- CPU numbers do not represent the deploy box")
    print(f"torch  : {torch.__version__}")
    print("=" * 72)

    tcfg = _task_config(prompt_list, task_config)
    read, close = _camera(fake_camera, camera_host, 640, 480)
    try:
        left, _ = read()
        print(f"frame  : {left.shape[1]}x{left.shape[0]}")
        calib = _calib(stereo_calib, left.shape)
        T_pelvis_camera = _extrinsic(camera_extrinsic)

        orient = None if skip_orient else _build_orient(cfg, dev, auto_download)
        if dev.startswith("cuda"):
            print(f"[setup] VRAM after both models: {_vram_mb(dev):.0f} MB")

        builder = None
        try:
            builder = RelationStateBuilder(tcfg, latch_config=cfg.latch)
        except ValueError as exc:
            print(f"[note] relation state disabled: {exc}")

        phases = [False, True] if compare_prune else [prune]
        results = {}
        last_snapshot = None
        for do_prune in phases:
            round_, sam3, control = _build_round(
                task_config=tcfg, calib=calib, T_pelvis_camera=T_pelvis_camera,
                read=read, device=dev, dtype=tdtype, prune=do_prune,
                orient=orient, cfg=cfg)
            control.start()
            try:
                round_.step()                              # warm up, untimed
                results[do_prune] = _run(
                    round_, sam3, builder, rounds, dev,
                    f"prune={'ON' if do_prune else 'OFF'}")
                last_snapshot = round_.step()
            finally:
                control.stop()
                round_.close()
            if builder is not None:
                builder.reset()

        if last_snapshot is not None:
            _report_slots(last_snapshot, tcfg)
        _verdict(results, compare_prune, orient is not None)
    finally:
        close()


def dataclasses_replace(cfg, section: str, **kw):
    import dataclasses
    return dataclasses.replace(
        cfg, **{section: dataclasses.replace(getattr(cfg, section), **kw)})


def _verdict(results: dict, compared: bool, had_orient: bool) -> None:
    print("\n" + "=" * 72)
    print("VERDICT")

    for do_prune, r in results.items():
        tag = "ON " if do_prune else "OFF"
        print(f"\n  [prune {tag}] p95 {r['p95']:.0f} ms "
              f"({1000 / max(r['p95'], 1e-9):.1f} Hz), "
              f"drift {r['drift_ms']:+.1f} ms across the run")
        if r["vram_end"] is not None:
            print(f"             VRAM {r['vram_end']:.0f} MB, "
                  f"{r['non_cond_end']} non-cond entries at the end")

    if compared and False in results and True in results:
        off, on = results[False], results[True]
        d_lat = off["drift_ms"] - on["drift_ms"]
        print(f"\n  [answer] latency drift without pruning {off['drift_ms']:+.1f} ms, "
              f"with pruning {on['drift_ms']:+.1f} ms.")
        if off["drift_ms"] > 10 and d_lat > 5:
            print("  -> Memory growth DOES slow computation, and pruning fixes "
                  "it. R1 is load-bearing for latency, not just for VRAM.")
        elif off["drift_ms"] > 10:
            print("  -> Latency degrades and pruning does NOT fix it. Something "
                  "other than the memory bank is growing — profile before "
                  "trusting a long rollout.")
        else:
            print("  -> Latency is flat either way. Pruning is a pure VRAM "
                  "measure here; the headroom it buys is memory, not speed.")
        if off["vram_end"] and on["vram_end"]:
            print(f"  [vram]   {off['vram_end']:.0f} MB unpruned vs "
                  f"{on['vram_end']:.0f} MB pruned at the same round count.")
    elif not compared:
        print("\n  Re-run with --compare-prune for the direct answer to whether "
              "memory growth slows computation. A single run cannot separate "
              "'slow' from 'got slow'.")

    if not had_orient:
        print("\n  [note] orientation was NOT loaded — this did not test the "
              "co-residency question it was written for.")
    print("\n  Latching is not exercised: the synthetic hand never closes.")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
