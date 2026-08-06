"""Stage-by-stage latency + VRAM benchmark for the perception v2 pipeline
(docs/perception_v2_pipeline.md §8, "numbers to measure before building").

This is the sibling of `perception_latency.py` (which measures the CURRENT
GroundingDINO+SAM2 cascade). Same reporting conventions -- warmup separated
from steady state, p50/p95/p99, tyro CLI, `--fake-camera` so it runs without
the robot. What it measures is different: the four v2 stages, each on its
own, plus the composites the schedule in §5 of that doc actually depends on.

    S1  sam3_track        SAM 3 tracker streaming step, N objects   (GPU, 10 Hz)
    S1' sam3_track_pcs    SAM 3 FULL video model streaming step     (GPU, 10 Hz)
                          -- detector runs every frame; upper bound
    S5  sam3_detect       SAM 3 image detector, N text prompts      (GPU,  1 Hz)
    S6  orient            Orient Anything V2 over N crops           (GPU,  1 Hz)
    S2  sgbm              StereoSGBM depth                          (CPU, 10 Hz)
    S3  join              mask centroid + median depth + backproject (CPU)

and the composites:

    critical_path         S1 || S2, then S3      -- what a policy tick waits on
    critical_path_serial  S1 then S2, then S3    -- the same without the overlap
    critical_path_orient  S1 || S2, S3, then S6  -- orientation on the hot path

THREE MEASUREMENT TRAPS THIS SCRIPT AVOIDS, because getting any of them wrong
produces numbers that look fine and are wrong by 2-10x:

  1. CUDA is asynchronous. `t0; model(...); t1-t0` measures kernel LAUNCH
     time, not execution. Every timed GPU call here is wrapped in
     `torch.cuda.synchronize()`. `perception_latency.py` got away without it
     because its stages end by moving results to CPU (an implicit sync); do
     not rely on that here.

  2. The SAM 3 tracker's memory bank GROWS as you push frames. A benchmark
     that runs 20 steps from a fresh session measures the cheap early frames
     and under-reports steady state. This script runs `--track-frames`
     (default 300 = 30 s of 10 Hz operation) and reports the timing binned by
     frame index, so you can SEE where it plateaus -- or fails to. It reports
     VRAM the same way, because the one public streaming adaptation of SAM 3
     OOMs after ~5 minutes and we need to know if we inherit that.

  3. First call is not steady state. CUDA context init, cuDNN autotune, weight
     upload and autocast compilation all land on call #1 and can be 10-50x the
     steady cost. Warmup samples are timed and printed SEPARATELY, never
     averaged in.

--------------------------------------------------------------------------
RUNNING IT
--------------------------------------------------------------------------

    uv run python -m ego2g1.deploy.perception_v2_latency

That is the whole invocation. No flags, nothing to clone or download by hand.
Everything else in this section is what the script does FOR you, documented so
that when a step fails you know which one it was.

Defaults are baked in from the repo, not invented:

    --stereo-calib   <repo>/stereo_calib.npz      if present, else a sized
                                                  placeholder (compute cost is
                                                  right, depth VALUES are not)
    --oriany-repo    <repo>/third_party/Orient-Anything-V2   (auto-cloned)
    --oriany-ckpt    auto-fetched from the Hub    (~5 GB, cached)
    --camera-host    EGO2G1_CAMERA_HOST or the lab default

Auto-setup, all skipped when already satisfied, all disable-able with
--no-auto-download:

  * Orient Anything V2 is a GitHub repo, not a pip package -- it is imported
    by PATH. `git clone --depth 1` into third_party/ on first run. Its vendored
    `vggt/` package comes with it, which is why cloning is enough.
  * Its checkpoint lives at `demo_ckpts/rotmod_realrotaug_best.pt` inside
    `Viglong/OriAnyV2_ckpt` (NOT at the repo root -- that path 404s). Public,
    so no auth needed. ~5 GB, downloaded once into the HF cache.
  * `utils/app_utils.py` does `import rembg` at module scope, so importing it
    at all requires rembg installed even though we never want to run it. If
    rembg is absent, a stub is injected into sys.modules that raises only if
    something actually calls it. `--measure-rembg` needs the real one.
  * SAM 3 weights are pre-fetched via `snapshot_download` during preflight,
    rather than lazily inside a timed stage, so a slow network never lands in
    a latency number.

We deliberately do NOT `pip install -r` Orient Anything V2's requirements.txt.
It pins `gradio==5.9.0`, `pydantic==2.10.6`, `lightning==2.2` and `bpy==4.2`
(Blender, hundreds of MB) -- installing that into this project's environment
would be destructive and none of it is needed for a forward pass.

THE ONE STEP THAT CANNOT BE AUTOMATED: `facebook/sam3` is `gated: manual` on
the Hub -- a human has to approve your access request. Preflight detects this
and prints the exact two steps. Everything else is automatic.

If the orientation stage cannot be set up, it is SKIPPED and every other stage
still reports. The SAM 3 and SGBM numbers are what gate the 10 Hz loop, and
you should be able to get them without first resolving a 5 GB third-party
checkpoint.

--------------------------------------------------------------------------
A NOTE ON VRAM, AND ON WHICH GPU THIS IS
--------------------------------------------------------------------------

Report the GPU this ran on alongside the numbers -- they do not transfer.
`docs/4090_serve_deploy.md` describes a 4090 for the POLICY SERVER; if the
deploy PC driving the camera is a different (smaller) card, that is the one
these numbers must come from, because that is where S1/S2/S5/S6 run.

Rough resident cost if everything is loaded at once:

    SAM 3            848M params   ~1.7 GB bf16   + activations @ 1008px
    Orient Any V2    5.05 GB ckpt (VGGT-based)    -- this is NOT a small model

On a 8 GB card those two co-resident plus activations is tight to infeasible.
`--sequential-load` loads/frees each model around its own measurement so you
can still get per-stage numbers on a small card; the composite stages need
them co-resident and will be skipped if `--sequential-load` is set. Peak VRAM
is reported per stage either way -- that number decides whether the real
deploy can hold both at once or has to page one in and out.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

_DEFAULT_PROMPTS = "cup,bowl,plate"

# Native SAM 3 resolution. The model is trained for this; 560 is the documented
# speed lever and is measured alongside it by --also-560, at an accuracy cost
# nothing in this script quantifies.
_NATIVE_IMAGE_SIZE = 1008


_ORIANY_GIT = "https://github.com/SpatialVision/Orient-Anything-V2"
_ORIANY_HF_REPO = "Viglong/OriAnyV2_ckpt"
# NOTE the subdirectory. The checkpoint is NOT at the repo root; asking for a
# bare "rotmod_realrotaug_best.pt" 404s.
_ORIANY_HF_FILE = "demo_ckpts/rotmod_realrotaug_best.pt"
_SAM3_REPO = "facebook/sam3"


# ----------------------------------------------------------------------------
# auto-setup: everything needed to run with zero arguments
# ----------------------------------------------------------------------------

def _repo_root() -> Path:
    """<root>/ego2g1/deploy/perception_v2_latency.py -> <root>."""
    return Path(__file__).resolve().parents[2]


def _default_stereo_calib() -> str | None:
    p = _repo_root() / "stereo_calib.npz"
    return str(p) if p.is_file() else None


def _default_oriany_repo() -> Path:
    return _repo_root() / "third_party" / "Orient-Anything-V2"


def _ensure_oriany_repo(path: Path, *, auto: bool) -> Path | None:
    """Clone Orient Anything V2 if absent. Returns the path, or None if the
    orientation stage cannot run (which is survivable -- every other stage
    still reports)."""
    import subprocess
    path = Path(path).expanduser()
    if (path / "vision_tower.py").is_file():
        return path
    if path.exists() and any(path.iterdir()):
        print(f"[setup] {path} exists but has no vision_tower.py -- not "
              f"touching it. Remove it or pass --oriany-repo elsewhere.")
        return None
    if not auto:
        print(f"[setup] Orient Anything V2 not at {path} and --no-auto-download "
              f"is set; skipping the orientation stage.")
        return None
    print(f"[setup] cloning Orient Anything V2 -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", _ORIANY_GIT, str(path)],
                       check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[setup] clone failed ({exc}); skipping the orientation stage. "
              f"Clone it by hand and re-run:\n    git clone --depth 1 "
              f"{_ORIANY_GIT} {path}")
        return None
    return path if (path / "vision_tower.py").is_file() else None


def _ensure_oriany_ckpt(explicit: str | None, *, auto: bool) -> str | None:
    """Fetch the ~5 GB checkpoint into the HF cache. Public repo, no auth."""
    if explicit:
        return explicit
    if not auto:
        return None
    from huggingface_hub import hf_hub_download
    print(f"[setup] resolving {_ORIANY_HF_REPO}/{_ORIANY_HF_FILE} "
          f"(~5 GB, cached after the first run)")
    try:
        return hf_hub_download(_ORIANY_HF_REPO, _ORIANY_HF_FILE)
    except Exception as exc:                       # noqa: BLE001
        print(f"[setup] checkpoint download failed: {type(exc).__name__}: {exc}")
        return None


def _stub_rembg_if_missing(*, want_rembg: bool) -> None:
    """`utils/app_utils.py` does `import rembg` at module scope, so we cannot
    import ANY of its helpers without rembg present -- even though the whole
    point is that we never run it (we have a SAM 3 mask, which is better
    information than rembg's guess).

    Install a stub that satisfies the import and raises only if something
    actually calls it. Real rembg is left alone when present, and is required
    for --measure-rembg.
    """
    import types
    try:
        import rembg                                # noqa: F401
        return
    except ImportError:
        pass
    if want_rembg:
        print("[setup] --measure-rembg needs the real rembg: "
              "`uv pip install rembg onnxruntime`. Continuing without it.")

    def _unavailable(*_a, **_k):
        raise RuntimeError(
            "rembg is not installed. It is deliberately unnecessary here -- we "
            "crop from the SAM 3 mask instead. Install it only for "
            "--measure-rembg.")

    stub = types.ModuleType("rembg")
    stub.remove = _unavailable
    stub.new_session = _unavailable
    sys.modules["rembg"] = stub
    print("[setup] rembg absent -> stub installed (import-only; never called)")


def _preflight(*, sam3_repo: str, auto: bool) -> bool:
    """Verify and pre-fetch everything the SAM 3 stages need. Returns False if they
    cannot run, having already printed what to do about it."""
    ok = True
    try:
        import transformers
    except ImportError:
        print("[setup] transformers is not installed:\n"
              "    uv pip install -U 'transformers>=5'")
        return False
    if not hasattr(transformers, "Sam3Model"):
        print(f"[setup] transformers {transformers.__version__} has no "
              f"Sam3Model -- SAM 3 landed in v5.0.0:\n"
              f"    uv pip install -U 'transformers>=5'")
        ok = False

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError
    try:
        if auto:
            print(f"[setup] pre-fetching {sam3_repo} (cached after first run; "
                  f"done here so a slow network never lands in a timed stage)")
            snapshot_download(sam3_repo)
    except GatedRepoError:
        print(f"\n[setup] {sam3_repo} is GATED and your account has no access.\n"
              f"        This is the ONE step that cannot be automated:\n"
              f"          1. open https://huggingface.co/{sam3_repo} and click "
              f"'Request access'\n"
              f"          2. once approved:  hf auth login\n"
              f"        Then re-run this script.")
        return False
    except Exception as exc:                       # noqa: BLE001
        name = type(exc).__name__
        if "401" in str(exc) or "Unauthorized" in name or "Token" in name:
            print(f"\n[setup] not authenticated to the Hub. Run:  hf auth login"
                  f"\n        (and make sure you have access to {sam3_repo})")
            return False
        print(f"[setup] pre-fetch of {sam3_repo} failed ({name}: {exc}); "
              f"continuing -- from_pretrained will retry.")
    return ok


# ----------------------------------------------------------------------------
# timing primitives
# ----------------------------------------------------------------------------

def _sync(device: str) -> None:
    """Block until queued CUDA work is done. See trap #1 in the module
    docstring -- without this every GPU stage reports launch time."""
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def _time_calls(fn, *, n: int, warmup: int, device: str,
                clock=time.perf_counter):
    """(warmup_samples_s, steady_samples_s, last_error).

    Mirrors `perception_latency._time_calls`, including its "time it even when
    it raises" behaviour -- a --fake-camera run, or a real frame that simply
    doesn't show the configured objects, still exercises and should still time
    every compute stage before the "nothing found" branch. A benchmark that
    aborts on that tells you nothing.

    Difference from that function: `_sync(device)` around every call.
    """
    def once() -> float:
        _sync(device)                 # drain anything still in flight
        t0 = clock()
        err = None
        try:
            fn()
        except Exception as e:        # noqa: BLE001 -- see docstring
            err = e
        _sync(device)                 # wait for THIS call's kernels
        return clock() - t0, err

    warmup_samples, last_error = [], None
    for _ in range(max(0, warmup)):
        dt, err = once()
        last_error = err or last_error
        warmup_samples.append(dt)

    samples = []
    for _ in range(n):
        dt, err = once()
        last_error = err or last_error
        samples.append(dt)
    return warmup_samples, samples, last_error


def _quiet(fn, *args, **kwargs):
    """Call `fn` with stdout swallowed.

    Orient Anything V2's `inf_single_batch` has a bare `print(ans_dict)` in it.
    Left alone that prints once per crop per iteration -- hundreds of lines of
    noise, and terminal I/O charged to the stage being timed.
    """
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _report_stage(name: str, warmup_s, samples_s, error, *,
                  vram_mb: float | None = None) -> float:
    """Print one stage's numbers, return its p95 in ms. Same layout as
    `perception_latency._report_stage` so the two tools' output can sit
    side by side in a commit message."""
    w_ms = np.asarray(warmup_s) * 1000.0
    s_ms = np.asarray(samples_s) * 1000.0
    print(f"\n--- {name} ---")
    if error is not None:
        print(f"  raised {type(error).__name__}: {error}")
        print("  (timing is still the real compute cost incurred before the "
              "error -- expected with --fake-camera or a frame that doesn't "
              "show the configured objects)")
    if len(w_ms):
        print(f"  warmup   ({len(w_ms)} call(s), includes CUDA ctx/autotune/"
              f"weight upload): {', '.join(f'{x:.0f}' for x in w_ms)} ms")
    if not len(s_ms):
        print("  steady   (n=0): stage skipped")
        return float("nan")
    p50, p95, p99 = np.percentile(s_ms, [50, 95, 99])
    print(f"  steady   (n={len(s_ms)}): mean {s_ms.mean():.1f}  p50 {p50:.1f}  "
          f"p95 {p95:.1f}  p99 {p99:.1f}  max {s_ms.max():.1f}  ms")
    if vram_mb is not None:
        print(f"  vram     peak {vram_mb:.0f} MB allocated during this stage")
    return float(p95)


def _vram_reset(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.reset_peak_memory_stats()


def _vram_peak_mb(device: str) -> float | None:
    if not str(device).startswith("cuda"):
        return None
    import torch
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _gpu_banner(device: str) -> None:
    import torch
    print("=" * 74)
    if str(device).startswith("cuda"):
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"device   : {props.name}  ({props.total_memory / 1024**3:.1f} GB, "
              f"sm_{props.major}{props.minor})")
    else:
        print(f"device   : {device}  -- CPU timings are NOT representative of "
              "the deploy machine")
    print(f"torch    : {torch.__version__}   cuda {torch.version.cuda}")
    try:
        import transformers
        print(f"transformers: {transformers.__version__}")
    except ImportError:
        print("transformers: NOT INSTALLED")
    print("=" * 74)


# ----------------------------------------------------------------------------
# frame source
# ----------------------------------------------------------------------------

def _open_camera(*, fake: bool, host: str, width: int, height: int):
    """Returns (read_stereo() -> (rgb_left, rgb_right), close()).

    --fake-camera synthesises a textured pair rather than a flat one: SGBM's
    cost depends on how much of the image it can actually match, so a blank
    frame under-reports it. This is still not a substitute for real data --
    it exercises the code path and gives a compute-cost floor.
    """
    if fake:
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)

        def read_stereo():
            # ~8 px horizontal shift so SGBM has a real disparity to find
            return base, np.roll(base, -8, axis=1)

        return read_stereo, (lambda: None)

    from ego2g1.deploy.camera import HeadCamera
    cam = HeadCamera(host=host, eye="left")
    cam.connect()

    def read_stereo():
        left, right = cam.read_stereo()
        if left is None or right is None:
            raise RuntimeError("camera returned no stereo frame yet")
        return left, right

    return read_stereo, cam.close


def _load_stereo_calib(path: str | None, rgb_shape):
    from ego2g1.deploy.perception.depth import StereoCalibration
    if path:
        return StereoCalibration.load(path)
    h, w = rgb_shape[:2]
    print(f"\n[warn] --stereo-calib not given: PLACEHOLDER calibration sized to "
          f"the real {w}x{h} frame. Fine for a COMPUTE-cost measurement "
          f"(SGBM's cost depends on image size/params, not calibration "
          f"accuracy); the depth VALUES are meaningless.")
    K = np.array([[600.0, 0.0, w / 2.0], [0.0, 600.0, h / 2.0], [0.0, 0.0, 1.0]])
    return StereoCalibration(
        K_left=K, K_right=K.copy(), dist_left=np.zeros(5), dist_right=np.zeros(5),
        R=np.eye(3), T=np.array([0.06, 0.0, 0.0]), image_size=(w, h))


# ----------------------------------------------------------------------------
# S3 join -- mask centroid + median depth + back-projection
# ----------------------------------------------------------------------------

def join_masks_to_3d(masks, depth_m, K, *, min_mask_px: int = 64,
                     min_valid_depth_px: int = 16):
    """Mirrors §3 S3 of docs/perception_v2_pipeline.md, including its guards.

    masks:   (N, H, W) bool     depth_m: (H, W) float32 metres (0/NaN = invalid)
    Returns list[(X, Y, Z) | None] in camera frame; None means "no measurement",
    which is the correct output for an occluded or textureless object and is
    NOT the same as a bad measurement.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    out = []
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    for m in masks:
        if m.sum() < min_mask_px:
            out.append(None)                  # occluded or lost
            continue
        ys, xs = np.nonzero(m)
        u, v = xs.mean(), ys.mean()           # mask MEAN, per Detection.centroid_uv
        sel = m & valid_depth
        if sel.sum() < min_valid_depth_px:
            out.append(None)                  # SGBM holes: textureless object
            continue
        z = float(np.median(depth_m[sel]))    # MEDIAN depth, robust to edges
        out.append(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return out


# ----------------------------------------------------------------------------
# model wrappers
# ----------------------------------------------------------------------------

class Sam3Detect:
    """S5 -- SAM 3 image detector over N text prompts.

    One `Sam3Model` call PER PROMPT. The backbone is recomputed per call here;
    the real pipeline should reuse vision features across prompts the way
    `Sam3VideoModel.run_detection` does (it loops prompts but reuses
    `vision_embeds`). So this is a mild OVER-estimate of the reseed cost --
    noted rather than silently corrected, because the correction requires
    reaching past the public API.
    """

    def __init__(self, repo: str, device: str, dtype, prompts: list[str],
                 image_size: int | None = None):
        import torch
        from transformers import Sam3Model, Sam3Processor
        kwargs = {}
        if image_size and image_size != _NATIVE_IMAGE_SIZE:
            from transformers import Sam3Config
            cfg = Sam3Config.from_pretrained(repo)
            cfg.image_size = image_size
            kwargs["config"] = cfg
        self.model = Sam3Model.from_pretrained(repo, dtype=dtype, **kwargs).to(device).eval()
        size = {"height": image_size, "width": image_size} if image_size else None
        self.processor = Sam3Processor.from_pretrained(repo, size=size) if size \
            else Sam3Processor.from_pretrained(repo)
        self.prompts, self.device, self.torch = prompts, device, torch

    def __call__(self, rgb):
        from PIL import Image
        img = Image.fromarray(rgb)
        results = []
        for p in self.prompts:
            inputs = self.processor(images=img, text=p, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model(**inputs)
            r = self.processor.post_process_instance_segmentation(
                out, threshold=0.5, target_sizes=inputs["original_sizes"].tolist())[0]
            results.append(r)
        return results

    def best_boxes(self, rgb):
        """Highest-scoring box per prompt -- used to seed the tracker, which is
        exactly how the real cold start (§6) works."""
        boxes = []
        for r in self(rgb):
            if len(r.get("scores", [])) == 0:
                boxes.append(None)
                continue
            i = int(np.argmax(r["scores"].float().cpu().numpy()))
            boxes.append(r["boxes"][i].float().cpu().numpy().tolist())
        return boxes

    def free(self):
        del self.model
        gc.collect()


class Sam3Track:
    """S1 -- tracker-only streaming step. THE number the 10 Hz loop depends on.

    Seeded once from boxes (the detector's, in the real pipeline), then each
    call pushes ONE frame. No mask is passed back in: the memory bank is
    internal, and self-conditioning on our own last output drifts
    (docs/perception_v2_pipeline.md I2).
    """

    def __init__(self, repo: str, device: str, dtype, image_size: int | None = None):
        import torch
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        self.model = Sam3TrackerVideoModel.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(repo)
        self.device, self.torch, self.session = device, torch, None

    def seed(self, rgb, boxes):
        from PIL import Image
        self.session = self.processor.init_video_session(inference_device=self.device)
        inputs = self.processor(images=Image.fromarray(rgb), device=self.device,
                                return_tensors="pt")
        for obj_id, box in enumerate(boxes, start=1):
            if box is None:
                continue
            self.processor.add_inputs_to_inference_session(
                inference_session=self.session, frame_idx=0, obj_ids=obj_id,
                input_boxes=[[box]], original_size=inputs.original_sizes[0])
        with self.torch.no_grad():
            self.model(inference_session=self.session, frame=inputs.pixel_values[0])

    def step(self, rgb):
        from PIL import Image
        inputs = self.processor(images=Image.fromarray(rgb), device=self.device,
                                return_tensors="pt")
        with self.torch.no_grad():
            return self.model(inference_session=self.session,
                              frame=inputs.pixel_values[0])

    def reset(self):
        if self.session is not None:
            self.session.reset_inference_session()

    def free(self):
        del self.model
        gc.collect()


class Sam3TrackPCS:
    """S1' -- the FULL video model in streaming mode: detector on EVERY frame
    plus association. Not what the design uses (I3), measured because the gap
    between this and Sam3Track is the entire quantitative argument for not
    using native PCS fusion in the 10 Hz loop."""

    def __init__(self, repo: str, device: str, dtype, prompts: list[str]):
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        self.model = Sam3VideoModel.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(repo)
        self.device, self.torch = device, torch
        self.session = self.processor.init_video_session(inference_device=device)
        self.processor.add_text_prompt(self.session, prompts)

    def step(self, rgb):
        from PIL import Image
        inputs = self.processor(images=Image.fromarray(rgb), device=self.device,
                                return_tensors="pt")
        with self.torch.no_grad():
            return self.model(inference_session=self.session,
                              frame=inputs.pixel_values[0])

    def free(self):
        del self.model
        gc.collect()


class OrientAnythingV2:
    """S6 -- orientation from a masked crop. VGGT-based; 5.05 GB checkpoint.

    Imported BY PATH, not by name: upstream is a research repo, not a pip
    package. There is no `inference.py` at the root -- the real entry points
    live in `utils/app_utils.py` and are driven by `app.py` (a Gradio demo).

    FOUR THINGS THE STOCK PATH DOES THAT WE SHOULD NOT PAY FOR AT 1 Hz, each
    measured separately below so the saving is a number and not a claim:

      1. `inf_single_case(model, ref, tgt)` stacks [ref, tgt] into an S=2
         sequence. Passing the SAME crop twice (the obvious way to ask for
         absolute orientation) pushes two images through VGGT for one answer
         -- about 2x the work for nothing. `tgt=None` gives the S=1 path.
         Use `mode="absolute"`.

      2. `inf_single_batch` takes `(B, S, C, H, W)` and its forward handles
         B > 1 fine -- only the output unpacking is hardcoded to B=1 (the
         `S == 1` branch indexes `[0]` and discards the rest of the batch).
         So true batching over our N crops IS available; it just needs ~10
         lines of unpacking rather than the public helper. `mode="batched"`.
         (This corrects an earlier reading of the docs, which document no
         batched entry point. The capability is there; only the wrapper is
         single-case.)

      3. `val_fit_alpha` runs a scipy distribution fit over the 360-bin
         azimuth histogram on EVERY call, to recover the symmetry parameter.
         We need alpha once, at seed, to choose the symmetry group for the
         episode -- not at 1 Hz forever. `--orient-skip-alpha` measures the
         path without it.

      4. `background_preprocess` runs **rembg** (a separate U2-Net matting
         model) and calls `rembg.new_session()` INSIDE the function, i.e.
         re-creates the session on every invocation. We already have a SAM 3
         mask, which is strictly better information than rembg's guess, so
         this should never run in the deploy loop. Measured only to show what
         it would have cost.

    Preprocessing is 518x518 pad mode (DINOv2 patch-14 grid), so crop size
    barely matters to cost -- everything is resized to the same tensor.
    """

    TARGET = 518

    def __init__(self, repo_dir: str, ckpt: str | None, device: str,
                 dtype=None):
        import torch
        repo = Path(repo_dir).expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"--oriany-repo not a directory: {repo}")
        sys.path.insert(0, str(repo))
        from vision_tower import VGGT_OriAny_Ref                     # noqa: E402
        from utils.app_utils import (background_preprocess,          # noqa: E402
                                     inf_single_case, preprocess_images)
        try:
            from utils.app_utils import val_fit_alpha                # noqa: E402
        except ImportError:
            val_fit_alpha = None

        # Upstream's own dtype rule (app.py): bf16 on sm_80+, else fp16.
        # Ada (4060/4090, sm_89) takes the bf16 branch.
        if dtype is None:
            dtype = (torch.bfloat16
                     if torch.cuda.is_available()
                     and torch.cuda.get_device_capability()[0] >= 8
                     else torch.float16)
        if ckpt is None:
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download(_ORIANY_HF_REPO, _ORIANY_HF_FILE)
        self.model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.to(device).eval()

        self.torch = torch
        self.device = device
        self._inf_single = inf_single_case
        self._preprocess = preprocess_images
        self._bkg = background_preprocess
        self._val_fit_alpha = val_fit_alpha

    # -- crop ---------------------------------------------------------------

    @staticmethod
    def crop(rgb, mask, *, pad: float = 0.15):
        """Square, padded crop around the mask bbox. Square because a
        non-square resize would shear the object's apparent orientation --
        the one thing this stage exists to measure."""
        from PIL import Image
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) * (0.5 + pad)
        H, W = mask.shape
        x0 = int(max(0, cx - half)); x1 = int(min(W, cx + half))
        y0 = int(max(0, cy - half)); y1 = int(min(H, cy + half))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return Image.fromarray(rgb[y0:y1, x0:x1]).convert("RGB")

    # -- inference modes ----------------------------------------------------

    def absolute_sequential(self, crops):
        """Stock path, one crop at a time, S=1 (tgt=None -- NOT the same crop
        twice). This is the honest baseline for 'what does the public API
        cost'."""
        return [self._inf_single(self.model, c, None) for c in crops]

    def absolute_batched(self, crops, *, with_alpha: bool = False):
        """All N crops in ONE forward. (B, S=1, C, 518, 518).

        Reimplements `inf_single_batch`'s unpacking because that function's
        S==1 branch keeps only element [0]. The angle decode below is
        line-for-line the same arithmetic, minus the B=1 assumption.
        """
        torch = self.torch
        # preprocess_images returns (N, C, H, W) for a list of N images;
        # unsqueeze(1) makes each its own S=1 sequence -> (N, 1, C, H, W).
        t = self._preprocess(list(crops), mode="pad").to(self.model.get_device())
        batch = t.unsqueeze(1)
        with torch.no_grad():
            pose_enc = self.model(batch)                    # (B, S, D)
        pose_enc = pose_enc.view(pose_enc.shape[0] * pose_enc.shape[1], -1)
        az = torch.argmax(pose_enc[:, 0:360], dim=-1)
        el = torch.argmax(pose_enc[:, 360:540], dim=-1) - 90
        ro = torch.argmax(pose_enc[:, 540:900], dim=-1) - 180
        out = {"az": az, "el": el, "ro": ro}
        if with_alpha and self._val_fit_alpha is not None:
            import torch.nn.functional as F
            dist = F.sigmoid(pose_enc[:, 0:360]).cpu().float().numpy()
            out["alpha"] = self._val_fit_alpha(distribute=dist)
        return out

    def relative(self, ref_crops, tgt_crops):
        """S=2 relative-rotation mode: rotation of tgt w.r.t. ref. The
        alternative to re-estimating absolute orientation every tick --
        estimate once at seed, then track relative rotation. Costs ~2 images
        per case, so it is only a win if it lets you skip something else."""
        return [self._inf_single(self.model, r, t)
                for r, t in zip(ref_crops, tgt_crops)]

    def rembg_preprocess(self, crops):
        """Measured to justify NOT using it -- see class docstring item 4."""
        return [self._bkg(c, True) for c in crops]

    def free(self):
        del self.model
        gc.collect()


# ----------------------------------------------------------------------------
# composites
# ----------------------------------------------------------------------------

def _parallel(gpu_fn, cpu_fn):
    """Run a GPU stage and a CPU stage concurrently, return when both are done.

    This is real parallelism, not a GIL illusion: OpenCV's SGBM and torch's
    CUDA launches both release the GIL in their C extensions. It is the ONLY
    true overlap in the pipeline (docs/perception_v2_pipeline.md §4), so the
    gap between `critical_path` and `critical_path_serial` below is the exact
    value of arranging the deploy loop to exploit it.
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_cpu = pool.submit(cpu_fn)
        r_gpu = gpu_fn()
        return r_gpu, f_cpu.result()


# ----------------------------------------------------------------------------

def main(
    *,
    camera_host: str | None = None,
    fake_camera: bool = False,
    fake_width: int = 640,
    fake_height: int = 480,
    prompts: str = _DEFAULT_PROMPTS,
    stereo_calib: str | None = None,
    sam3_repo: str = _SAM3_REPO,
    oriany_repo: str | None = None,
    oriany_ckpt: str | None = None,
    device: str | None = None,
    dtype: str = "bfloat16",
    n: int = 30,
    measure_rembg: bool = False,
    warmup: int = 5,
    track_frames: int = 300,
    also_560: bool = False,
    skip_pcs: bool = False,
    sequential_load: bool = False,
    policy_period_ms: float = 1000.0,
    tracker_hz: float = 10.0,
    auto_download: bool = True,
):
    """Measure every v2 perception stage on THIS machine.

    track_frames: how many streaming steps to push before reporting, so the
        memory-bank growth in trap #2 is visible. 300 = 30 s at 10 Hz. Raise
        to 3000 (5 min) if you specifically want to chase the OOM the public
        streaming fork reports.
    sequential_load: load and free each model around its own stage. Lets a
        small card produce per-stage numbers; disables the composites, which
        need the models co-resident.
    """
    # ---- auto-setup, before any model touches the GPU ----------------------
    _stub_rembg_if_missing(want_rembg=measure_rembg)
    if not _preflight(sam3_repo=sam3_repo, auto=auto_download):
        print("\nAborting: the SAM 3 stages cannot run. See above.")
        return
    if oriany_repo is None:
        oriany_repo = _default_oriany_repo()
    resolved_oriany = _ensure_oriany_repo(Path(oriany_repo), auto=auto_download)
    if resolved_oriany is not None and oriany_ckpt is None:
        oriany_ckpt = _ensure_oriany_ckpt(None, auto=auto_download)
        if oriany_ckpt is None:
            resolved_oriany = None
    oriany_repo = str(resolved_oriany) if resolved_oriany else None

    if stereo_calib is None:
        stereo_calib = _default_stereo_calib()
        if stereo_calib:
            print(f"[setup] stereo calibration: {stereo_calib}")
    if camera_host is None:
        import os
        camera_host = os.environ.get("EGO2G1_CAMERA_HOST", "192.168.123.164")

    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, dtype)
    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]

    _gpu_banner(dev)
    print(f"prompts  : {prompt_list}  ({len(prompt_list)} object slots)")
    print(f"schedule : policy every {policy_period_ms:.0f} ms, "
          f"tracker at {tracker_hz:.0f} Hz "
          f"({1000.0 / tracker_hz:.0f} ms budget per tracker slot)")

    read_stereo, close_cam = _open_camera(
        fake=fake_camera, host=camera_host, width=fake_width, height=fake_height)
    rgb_left, rgb_right = read_stereo()
    print(f"frame    : {rgb_left.shape[1]}x{rgb_left.shape[0]}")

    calib = _load_stereo_calib(stereo_calib, rgb_left.shape)
    results: dict[str, float] = {}

    # ---- S2 sgbm (CPU) -----------------------------------------------------
    from ego2g1.deploy.perception.depth import StereoSGBMDepthSource
    sgbm = StereoSGBMDepthSource(calib)
    w, s, e = _time_calls(lambda: sgbm.estimate(*read_stereo()),
                          n=n, warmup=warmup, device="cpu")
    results["sgbm"] = _report_stage("S2 sgbm  (CPU, 10 Hz)", w, s, e)
    depth_m = sgbm.estimate(rgb_left, rgb_right)

    # ---- S5 detect (GPU) ---------------------------------------------------
    _vram_reset(dev)
    detect = Sam3Detect(sam3_repo, dev, torch_dtype, prompt_list)
    w, s, e = _time_calls(lambda: detect(rgb_left), n=max(5, n // 3),
                          warmup=warmup, device=dev)
    results["detect"] = _report_stage(
        f"S5 detect  (GPU, 1 Hz, {len(prompt_list)} prompts)", w, s, e,
        vram_mb=_vram_peak_mb(dev))
    print("  note: one backbone pass PER PROMPT here; the real reseed should "
          "share vision features across prompts, so this over-estimates.")
    boxes = detect.best_boxes(rgb_left)
    found = [i for i, b in enumerate(boxes) if b is not None]
    print(f"  seeded {len(found)}/{len(prompt_list)} slots from this frame: "
          f"{[prompt_list[i] for i in found]}")
    if not found:
        print("  [warn] NO objects detected -- tracker timings below are for an "
              "EMPTY session and are not representative. Point the camera at "
              "the objects named by --prompts.")
    if sequential_load:
        detect.free()

    if also_560:
        _vram_reset(dev)
        d560 = Sam3Detect(sam3_repo, dev, torch_dtype, prompt_list, image_size=560)
        w, s, e = _time_calls(lambda: d560(rgb_left), n=max(5, n // 3),
                              warmup=warmup, device=dev)
        _report_stage("S5 detect @560px  (accuracy cost NOT measured here)",
                      w, s, e, vram_mb=_vram_peak_mb(dev))
        d560.free()

    # ---- S1 track (GPU) ----------------------------------------------------
    _vram_reset(dev)
    track = Sam3Track(sam3_repo, dev, torch_dtype)
    track.seed(rgb_left, boxes)

    # Long run first, so the binned trace shows memory-bank growth (trap #2).
    print(f"\n--- S1 track  (GPU, {tracker_hz:.0f} Hz) ---")
    print(f"  pushing {track_frames} frames to expose memory-bank growth...")
    per_frame, vram_trace = [], []
    for i in range(track_frames):
        rgb_l, _ = read_stereo()
        _sync(dev); t0 = time.perf_counter()
        track.step(rgb_l)
        _sync(dev); per_frame.append(time.perf_counter() - t0)
        if i % 25 == 0:
            vram_trace.append((i, _vram_peak_mb(dev)))
    pf = np.asarray(per_frame) * 1000.0
    print(f"  first 10 frames : mean {pf[:10].mean():.1f} ms")
    bins = max(1, len(pf) // 10)
    print("  decile means    : " +
          "  ".join(f"{pf[i * bins:(i + 1) * bins].mean():.0f}" for i in range(10)) + " ms")
    steady = pf[len(pf) // 2:]                      # second half = plateau
    p50, p95, p99 = np.percentile(steady, [50, 95, 99])
    print(f"  steady (2nd half, n={len(steady)}): mean {steady.mean():.1f}  "
          f"p50 {p50:.1f}  p95 {p95:.1f}  p99 {p99:.1f}  max {steady.max():.1f} ms")
    if vram_trace:
        print("  vram by frame   : " +
              "  ".join(f"{i}:{v:.0f}MB" for i, v in vram_trace[:8]) +
              (" ..." if len(vram_trace) > 8 else ""))
        first, last = vram_trace[0][1], vram_trace[-1][1]
        if last > first * 1.5:
            print(f"  [warn] VRAM grew {first:.0f} -> {last:.0f} MB over "
                  f"{track_frames} frames. If this does not plateau, long "
                  "episodes will OOM -- the failure the public streaming fork "
                  "reports. Re-run with --track-frames 3000 before trusting "
                  "a multi-minute episode.")
    results["track"] = float(p95)
    budget = 1000.0 / tracker_hz
    print(f"  -> {100 * p95 / budget:.0f}% of the {budget:.0f} ms tracker slot")

    # ---- S1' full PCS streaming (GPU) --------------------------------------
    if not skip_pcs and not sequential_load:
        _vram_reset(dev)
        pcs = Sam3TrackPCS(sam3_repo, dev, torch_dtype, prompt_list)
        w, s, e = _time_calls(lambda: pcs.step(read_stereo()[0]),
                              n=n, warmup=warmup, device=dev)
        results["track_pcs"] = _report_stage(
            "S1' track_pcs  (full video model: detector EVERY frame)", w, s, e,
            vram_mb=_vram_peak_mb(dev))
        if not np.isnan(results.get("track_pcs", np.nan)):
            print(f"  -> {results['track_pcs'] / results['track']:.1f}x the "
                  "tracker-only step. This ratio is the cost of using native "
                  "PCS fusion in the 10 Hz loop instead of per-slot IoU (I3).")
        pcs.free()

    # ---- S3 join (CPU) -----------------------------------------------------
    out = track.step(rgb_left)
    masks = _masks_from_tracker(out, rgb_left.shape[:2])
    w, s, e = _time_calls(lambda: join_masks_to_3d(masks, depth_m, calib.K_left),
                          n=n * 5, warmup=warmup, device="cpu")
    results["join"] = _report_stage("S3 join  (CPU)", w, s, e)

    # ---- S6 orientation (GPU) ---------------------------------------------
    # Five variants, because "the orientation model's latency" is not one
    # number -- the stock path does several things we can skip, and each
    # skip needs a measurement to justify it (see OrientAnythingV2's docstring).
    orient = None
    if oriany_repo:
        _vram_reset(dev)
        try:
            t_load = time.perf_counter()
            orient = OrientAnythingV2(oriany_repo, oriany_ckpt, dev)
            _sync(dev)
            load_s = time.perf_counter() - t_load
            print(f"\n--- S6 orient: model load ---")
            print(f"  {load_s:.1f} s to load + upload (5.05 GB checkpoint). "
                  f"Cold-start cost, paid once.")
            print(f"  vram after load: {_vram_peak_mb(dev):.0f} MB")

            crops = [c for c in (OrientAnythingV2.crop(rgb_left, m) for m in masks)
                     if c is not None]
            nc = len(crops)
            if not crops:
                print("\n--- S6 orient --- skipped: no usable crops (no masks). "
                      "Point the camera at the --prompts objects.")
            else:
                nn = max(5, n // 3)

                # (a) stock public path, one case at a time, S=1
                _vram_reset(dev)
                w, s, e = _time_calls(
                    lambda: _quiet(orient.absolute_sequential, crops),
                    n=nn, warmup=warmup, device=dev)
                seq = _report_stage(
                    f"S6a orient  sequential, {nc} crops, S=1  [stock API]",
                    w, s, e, vram_mb=_vram_peak_mb(dev))
                results["orient"] = seq

                # (b) one crop, to expose per-crop scaling
                w, s, e = _time_calls(
                    lambda: _quiet(orient.absolute_sequential, crops[:1]),
                    n=nn, warmup=2, device=dev)
                one = _report_stage("S6b orient  sequential, 1 crop", w, s, e)

                # (c) all crops in ONE forward -- the finding this stage exists
                #     to confirm or kill
                _vram_reset(dev)
                w, s, e = _time_calls(lambda: orient.absolute_batched(crops),
                                      n=nn, warmup=warmup, device=dev)
                bat = _report_stage(
                    f"S6c orient  BATCHED, {nc} crops in one forward",
                    w, s, e, vram_mb=_vram_peak_mb(dev))
                results["orient_batched"] = bat

                if not (np.isnan(seq) or np.isnan(bat)):
                    print(f"\n  scaling: 1 crop {one:.0f} ms -> {nc} sequential "
                          f"{seq:.0f} ms ({seq / max(one, 1e-9):.1f}x)  vs  "
                          f"{nc} batched {bat:.0f} ms "
                          f"({bat / max(one, 1e-9):.1f}x)")
                    if bat < 0.8 * seq:
                        print(f"  -> batching saves {seq - bat:.0f} ms "
                              f"({100 * (seq - bat) / seq:.0f}%). Worth the ~10 "
                              f"lines of custom unpacking; use absolute_batched.")
                    else:
                        print("  -> batching does NOT help here (likely "
                              "compute-bound, not launch-bound at this size). "
                              "Stay on the stock API and keep the code simpler.")
                    results["orient"] = min(seq, bat)

                # (d) cost of val_fit_alpha, which we only need at seed
                w, s, e = _time_calls(
                    lambda: orient.absolute_batched(crops, with_alpha=True),
                    n=nn, warmup=2, device=dev)
                alpha = _report_stage(
                    "S6d orient  batched + val_fit_alpha (scipy, CPU)", w, s, e)
                if not (np.isnan(alpha) or np.isnan(bat)):
                    print(f"  -> alpha costs {alpha - bat:.0f} ms per call. We "
                          f"need it ONCE at seed to fix the episode's symmetry "
                          f"group, never at 1 Hz. Skip it in the loop.")

                # (e) relative (S=2) mode -- the seed-once-then-track option
                w, s, e = _time_calls(
                    lambda: _quiet(orient.relative, crops, crops),
                    n=nn, warmup=2, device=dev)
                rel = _report_stage(
                    f"S6e orient  RELATIVE mode, {nc} pairs, S=2", w, s, e)
                if not (np.isnan(rel) or np.isnan(seq)):
                    print(f"  -> {rel / max(seq, 1e-9):.1f}x the absolute S=1 "
                          f"path (two images per case). Only worth it if "
                          f"tracking relative rotation lets you drop something "
                          f"else -- it is not a speed win by itself.")

                # (f) what rembg would have cost, had we not had SAM 3 masks
                if measure_rembg:
                    w, s, e = _time_calls(
                        lambda: orient.rembg_preprocess(crops),
                        n=3, warmup=1, device="cpu")
                    rb = _report_stage(
                        "S6f orient  background_preprocess (rembg) -- NOT USED",
                        w, s, e)
                    print(f"  -> {rb:.0f} ms of U2-Net matting per call, plus a "
                          f"fresh rembg session per call. We already have a SAM 3 "
                          f"mask, which is better information. Never enable this "
                          f"in the deploy loop.")
        except Exception as exc:                    # noqa: BLE001
            import traceback
            print(f"\n--- S6 orient --- FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print("  every other stage above is still valid.")
            orient = None
    else:
        print("\n--- S6 orient --- skipped: pass --oriany-repo to measure it")

    # ---- composites --------------------------------------------------------
    if not sequential_load:
        print("\n" + "=" * 74)
        print("COMPOSITES -- what a policy tick actually waits on")

        def gpu_step():
            return track.step(read_stereo()[0])

        def cpu_step():
            return sgbm.estimate(*read_stereo())

        w, s, e = _time_calls(lambda: _parallel(gpu_step, cpu_step),
                              n=n, warmup=warmup, device=dev)
        results["cp_par"] = _report_stage("critical_path  (S1 || S2)", w, s, e)

        def serial():
            gpu_step(); cpu_step()

        w, s, e = _time_calls(serial, n=n, warmup=warmup, device=dev)
        results["cp_ser"] = _report_stage("critical_path_serial  (S1 then S2)",
                                          w, s, e)

        if orient is not None and "orient" in results:
            def with_orient():
                o = gpu_step()
                m = _masks_from_tracker(o, rgb_left.shape[:2])
                c = [x for x in (OrientAnythingV2.crop(rgb_left, mm) for mm in m)
                     if x is not None]
                if c:
                    orient.absolute_batched(c)

            w, s, e = _time_calls(lambda: _parallel(with_orient, cpu_step),
                                  n=max(5, n // 3), warmup=3, device=dev)
            results["cp_orient"] = _report_stage(
                "critical_path_orient  (S1 || S2, then S6 on the hot path)",
                w, s, e)

    # ---- verdict -----------------------------------------------------------
    _verdict(results, policy_period_ms=policy_period_ms, tracker_hz=tracker_hz,
             n_objects=len(prompt_list))
    close_cam()


def _masks_from_tracker(out, hw) -> np.ndarray:
    """Tracker output -> (N, H, W) bool at full frame resolution. Tolerant of
    both the low-res dict form and an already-upsampled tensor, because the
    exact field layout differs between the tracker and video models."""
    import torch
    import torch.nn.functional as F
    m = getattr(out, "pred_masks", None)
    if m is None:
        d = getattr(out, "obj_id_to_mask", None) or {}
        if not d:
            return np.zeros((0, *hw), dtype=bool)
        m = torch.cat(list(d.values()), dim=0)
    m = m.float()
    if m.ndim == 4:
        m = m.squeeze(1)
    m = F.interpolate(m.unsqueeze(1), size=hw, mode="bilinear",
                      align_corners=False).squeeze(1)
    return (m > 0).cpu().numpy()


def _verdict(r: dict, *, policy_period_ms: float, tracker_hz: float,
             n_objects: int) -> None:
    print("\n" + "=" * 74)
    print("VERDICT")
    budget = 1000.0 / tracker_hz
    track = r.get("track", float("nan"))
    sgbm = r.get("sgbm", float("nan"))
    cp = r.get("cp_par", float("nan"))

    if not np.isnan(track):
        if track > budget:
            print(f"  [BLOCKER] tracker step p95 {track:.0f} ms > {budget:.0f} ms "
                  f"slot. {tracker_hz:.0f} Hz tracking is NOT viable on this "
                  f"machine. Options, in order: drop to "
                  f"{1000.0 / track:.0f} Hz; --also-560 to see what the "
                  f"resolution lever buys; or a bigger GPU.")
        elif track > 0.7 * budget:
            print(f"  [tight] tracker step p95 {track:.0f} ms is "
                  f"{100 * track / budget:.0f}% of the {budget:.0f} ms slot. "
                  f"No room for the reseed to displace a step cleanly.")
        else:
            print(f"  [ok] tracker step p95 {track:.0f} ms fits the "
                  f"{budget:.0f} ms slot with {budget - track:.0f} ms spare.")

    if not (np.isnan(sgbm) or np.isnan(track)):
        if sgbm > track:
            print(f"  [note] SGBM ({sgbm:.0f} ms) now EXCEEDS the tracker "
                  f"({track:.0f} ms): the CPU is the bottleneck and the "
                  f"GPU||CPU overlap has stopped helping. Cut SGBM cost "
                  f"(resolution, disparity range) before optimising SAM 3.")

    if not (np.isnan(r.get("cp_par", np.nan)) or np.isnan(r.get("cp_ser", np.nan))):
        saved = r["cp_ser"] - r["cp_par"]
        print(f"  [overlap] running S1||S2 saves {saved:.0f} ms per tick "
              f"({r['cp_ser']:.0f} -> {r['cp_par']:.0f}). That is the entire "
              f"value of threading the deploy loop.")

    if not np.isnan(cp):
        print(f"  [critical path] {cp:.0f} ms p95, "
              f"{100 * cp / policy_period_ms:.1f}% of the "
              f"{policy_period_ms:.0f} ms policy period, BEFORE the policy "
              f"server's own latency.")

    if "cp_orient" in r and not np.isnan(r["cp_orient"]):
        delta = r["cp_orient"] - cp
        print(f"  [orientation placement] putting S6 on the hot path adds "
              f"{delta:.0f} ms ({100 * delta / cp:.0f}%).")
        if delta < 0.2 * cp:
            print("    -> cheap. Put it on the critical path and stop thinking "
                  "about it; freshness is free here.")
        else:
            print("    -> expensive. Keep it mid-window with the reseed and let "
                  "orientation.py's stabiliser absorb the staleness. Rotation "
                  "is slow-moving, symmetry-snapped, and held during latch.")

    if "detect" in r and not np.isnan(r["detect"]):
        print(f"  [reseed] {r['detect']:.0f} ms displaces "
              f"{np.ceil(r['detect'] / budget):.0f} tracker slot(s) mid-window. "
              f"Free, because the reseed emits masks of its own.")

    print(f"\n  measured with {n_objects} object slot(s). Re-run with more "
          "--prompts to see the per-object scaling directly.")
    print("  This script only measures. It changes no deploy default.")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
