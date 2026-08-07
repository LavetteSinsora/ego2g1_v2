"""Latency + VRAM benchmark for the perception v2 pipeline.

Run it with no arguments:

    uv run --group perception-v2 python -m ego2g1.deploy.perception_v2_latency

It measures four stages and the composite they add up to:

    sam3     SAM 3, ONE session, all prompts, detection AND tracking   GPU
    sgbm     StereoSGBM depth                                          CPU
    join     mask centroid + median depth + back-projection            CPU
    orient   Orient Anything V2 over the masked crops                  GPU

    perception_step   (sam3 || sgbm) -> join -> orient
                      = one full iteration of the free-running async loop.
                      Its period sets how stale every state vector the policy
                      sees will be, so this is the headline number.

WHY THERE IS ONE SAM 3 STAGE AND NOT THREE
------------------------------------------
`Sam3VideoModel.forward` computes the vision backbone ONCE per frame and
shares it:

    vision_embeds  = self.detector_model.get_vision_features(pixel_values)
    all_detections = self.run_detection(..., vision_embeds=vision_embeds)
    vision_feats   = self.get_vision_features_for_tracker(vision_embeds)

`run_detection` loops over prompts, but the loop body is only the small
text-conditioned detector head -- the backbone is already computed. The
tracker then reuses the same embeddings again and propagates all objects in
one batched call.

So N prompts cost ONE backbone + N cheap heads, NOT N forward passes. You
only pay N backbones by putting each prompt in its own session, and that buys
nothing: a single session already gives every object its own memory bank, and
cross-prompt association is explicitly blocked -- `_associate_det_trk` zeroes
IoU between different prompt ids, so a "cup" detection can never bind to a
"plate" track. One session, all prompts.

THREE MEASUREMENT TRAPS THIS AVOIDS
-----------------------------------
1. CUDA is asynchronous. Timing without `torch.cuda.synchronize()` measures
   kernel LAUNCH, not execution -- wrong by 2-10x. Every GPU stage here is
   synchronised on both sides.
2. The tracker's memory bank GROWS with frames pushed. Twenty steps from a
   fresh session measures only the cheap early frames. This pushes `--frames`
   (default 300 = 30 s at 10 Hz) and prints per-decile means so you can see
   the plateau -- and traces VRAM, because the one public streaming adaptation
   of SAM 3 OOMs after ~5 minutes.
3. First call is not steady state. Warmup is timed and printed separately,
   never averaged in.

SETUP -- all automatic except one step
--------------------------------------
Auto-cloned/downloaded on first run: Orient Anything V2 (git, into
third_party/), its checkpoint (`demo_ckpts/rotmod_realrotaug_best.pt` inside
`Viglong/OriAnyV2_ckpt` -- note the subdirectory; the bare filename 404s), and
the SAM 3 weights. Stereo calibration defaults to <repo>/stereo_calib.npz;
camera host to $EGO2G1_CAMERA_HOST.

`facebook/sam3` is `gated: manual` -- a human approves access. That is the one
step nothing can automate; preflight prints the two commands if you lack it.

Needs `transformers>=5` (SAM 3 landed in 5.0.0), which conflicts with the
`perception` group (GroundingDINO pins <5) and `train` (openpi pins ==4.53.2).
Hence `--group perception-v2` on both `uv sync` and `uv run`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_DEFAULT_PROMPTS = "cup,bowl,plate"
_SAM3_REPO = "facebook/sam3"
_ORIANY_GIT = "https://github.com/SpatialVision/Orient-Anything-V2"
_ORIANY_HF_REPO = "Viglong/OriAnyV2_ckpt"
_ORIANY_HF_FILE = "demo_ckpts/rotmod_realrotaug_best.pt"   # NOT at repo root


# ============================================================================
# setup
# ============================================================================

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_oriany(path: Path, *, auto: bool) -> Path | None:
    """Clone Orient Anything V2 if absent. Returns None if the orientation
    stage cannot run -- survivable, every other stage still reports."""
    import subprocess
    if (path / "vision_tower.py").is_file():
        return path
    if path.exists() and any(path.iterdir()):
        print(f"[setup] {path} exists without vision_tower.py; not touching it.")
        return None
    if not auto:
        return None
    print(f"[setup] cloning Orient Anything V2 -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", _ORIANY_GIT, str(path)],
                       check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[setup] clone failed ({exc}); skipping orientation.")
        return None
    return path if (path / "vision_tower.py").is_file() else None


def _stub_rembg() -> None:
    """`utils/app_utils.py` does `import rembg` at module scope, so we cannot
    import its helpers without rembg -- even though we never want to run it
    (we crop from the SAM 3 mask, which beats rembg's matting guess). Stub it
    so the import succeeds and only a real call would raise."""
    import types
    try:
        import rembg  # noqa: F401
        return
    except ImportError:
        pass

    def _unavailable(*_a, **_k):
        raise RuntimeError("rembg deliberately not installed; we use SAM 3 masks")

    stub = types.ModuleType("rembg")
    stub.remove = stub.new_session = _unavailable
    sys.modules["rembg"] = stub
    print("[setup] rembg absent -> import stub installed (never called)")


def _preflight(*, auto: bool) -> bool:
    try:
        import transformers
    except ImportError:
        print("[setup] no transformers:  uv sync --group perception-v2")
        return False
    if not hasattr(transformers, "Sam3VideoModel"):
        print(f"[setup] transformers {transformers.__version__} has no "
              f"Sam3VideoModel (SAM 3 landed in 5.0.0).\n"
              f"        uv sync --group perception-v2\n"
              f"        ...and pass --group perception-v2 to `uv run` too.")
        return False
    try:
        import kernels  # noqa: F401
    except ImportError:
        print("\n[setup] the `kernels` package is MISSING. transformers then "
              "silently skips NMS post-processing, hole filling and sprinkle\n"
              "        removal inside SAM 3. That is not cosmetic for a "
              "benchmark: it removes real per-frame work, so the numbers\n"
              "        UNDER-report, and it degrades masks (duplicate "
              "detections survive, masks keep holes).\n"
              "        Fix:  uv sync --group perception-v2   "
              "(or: uv pip install kernels)\n")

    if not auto:
        return True
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError
    try:
        print(f"[setup] pre-fetching {_SAM3_REPO} (kept out of the timed stages)")
        snapshot_download(_SAM3_REPO)
    except GatedRepoError:
        print(f"\n[setup] {_SAM3_REPO} is GATED and you lack access. This is "
              f"the one manual step:\n"
              f"    1. https://huggingface.co/{_SAM3_REPO} -> 'Request access'\n"
              f"    2. once approved:  hf auth login")
        return False
    except Exception as exc:                                   # noqa: BLE001
        if "401" in str(exc) or "Unauthorized" in type(exc).__name__:
            print("\n[setup] not authenticated to the Hub:  hf auth login")
            return False
        print(f"[setup] pre-fetch failed ({type(exc).__name__}); continuing.")
    return True


# ============================================================================
# timing
# ============================================================================

def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def _quiet(fn, *args, **kwargs):
    """Orient Anything's `inf_single_batch` has a bare `print(ans_dict)`.
    Unmuted it floods the log and charges terminal I/O to the timed stage."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _time(fn, *, n: int, warmup: int, device: str):
    """(warmup_s, steady_s, last_error). Timed even when it raises, so a frame
    that happens not to show the objects still reports its real compute cost
    instead of aborting the whole run."""
    def once():
        _sync(device)
        t0 = time.perf_counter()
        err = None
        try:
            fn()
        except Exception as e:                                  # noqa: BLE001
            err = e
        _sync(device)
        return time.perf_counter() - t0, err

    warm, last = [], None
    for _ in range(max(0, warmup)):
        dt, e = once()
        last = e or last
        warm.append(dt)
    steady = []
    for _ in range(n):
        dt, e = once()
        last = e or last
        steady.append(dt)
    return warm, steady, last


def _report(name: str, warm, steady, err, *, vram: float | None = None) -> float:
    """Print one stage, return its p95 in ms."""
    print(f"\n--- {name} ---")
    if err is not None:
        print(f"  raised {type(err).__name__}: {err}")
    if warm:
        w = np.asarray(warm) * 1e3
        print(f"  warmup ({len(w)}): {', '.join(f'{x:.0f}' for x in w)} ms "
              f"(CUDA ctx / autotune / weight upload)")
    if not steady:
        return float("nan")
    s = np.asarray(steady) * 1e3
    p50, p95, p99 = np.percentile(s, [50, 95, 99])
    print(f"  steady (n={len(s)}): mean {s.mean():.1f}  p50 {p50:.1f}  "
          f"p95 {p95:.1f}  p99 {p99:.1f}  max {s.max():.1f} ms")
    if vram is not None:
        print(f"  vram peak {vram:.0f} MB")
    return float(p95)


def _vram_reset(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.reset_peak_memory_stats()


def _vram(device: str) -> float | None:
    """PEAK allocated since the last reset."""
    if not str(device).startswith("cuda"):
        return None
    import torch
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def _vram_now(device: str) -> float | None:
    """CURRENTLY allocated. This is the one that matters for a leak test --
    `max_memory_allocated` is a high-water mark and only ever rises, so it can
    never show a prune working."""
    if not str(device).startswith("cuda"):
        return None
    import torch
    return torch.cuda.memory_allocated() / 1024 ** 2


def _parallel(gpu_fn, cpu_fn):
    """Genuine overlap, not a GIL illusion: OpenCV's SGBM and torch's CUDA
    launches both release the GIL. This is the pipeline's only real
    parallelism."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut = pool.submit(cpu_fn)
        return gpu_fn(), fut.result()


# ============================================================================
# stages
# ============================================================================

class Sam3:
    """ONE session, all prompts, detection + tracking on every frame.

    Text prompts mean there is no manual seeding step: the detector finds the
    objects itself on the first frame and every frame after, and the tracker
    carries identity between them.
    """

    def __init__(self, repo: str, device: str, dtype, prompts: list[str],
                 prune: bool = True):
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        self.model = Sam3VideoModel.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(repo)
        self.device, self.torch = device, torch
        self.dtype = dtype
        self._prune = bool(prune)
        self._frame_idx = -1
        self._pruned_total = 0
        # Memory attention reads non-conditioning outputs ONLY for frames
        # t-1 .. t-(num_maskmem-1). Read the real value rather than assuming 7.
        trk = getattr(self.model, "tracker_model", None)
        self.num_maskmem = int(getattr(trk, "num_maskmem", 7) or 7)
        # The session carries a dtype for everything it stores; leaving it at
        # the float32 default while the weights are bf16 is what produces
        # "input and bias type should be the same" on the first conv.
        self.session = self.processor.init_video_session(
            inference_device=device, dtype=dtype)
        self.processor.add_text_prompt(self.session, prompts)   # all at once

    def step(self, rgb):
        """One streaming frame in, per-object masks out at full resolution."""
        from PIL import Image
        inputs = self.processor(images=Image.fromarray(rgb), device=self.device,
                                return_tensors="pt")
        # The processor always emits float32 pixel_values. Cast to the weight
        # dtype explicitly rather than relying on autocast, so the benchmark
        # times the same arithmetic the deploy loop will run.
        frame = inputs.pixel_values[0].to(dtype=self.dtype)
        with self.torch.no_grad():
            out = self.model(inference_session=self.session, frame=frame)
        # The raw output knows which frame this was; postprocess_outputs drops it.
        fi = getattr(out, "frame_idx", None)
        self._frame_idx = int(fi) if fi is not None else self._frame_idx + 1
        if self._prune:
            self.prune()
        return self.processor.postprocess_outputs(
            self.session, out, original_sizes=inputs.original_sizes)

    def prune(self) -> int:
        """Drop non-conditioning memory entries the model can no longer read.

        `_get_memory_frames` only ever looks up
        `non_cond_frame_outputs[frame_idx - k]` for k in 1..num_maskmem-1, and
        conditioning frames separately. Anything older than that window is
        written once and never read again -- it is the entire ~11.5 MB/frame
        growth. Deleting it is provably lossless.

        Conditioning frames are LEFT ALONE: they are the long-lived anchors,
        and they accumulate 16x slower (recondition_every_nth_frame=16).
        """
        per_obj = getattr(self.session, "output_dict_per_obj", None)
        if not per_obj:
            return 0
        cutoff = self._frame_idx - self.num_maskmem
        if cutoff < 0:
            return 0
        freed = 0
        for obj in per_obj.values():
            nc = obj.get("non_cond_frame_outputs") if isinstance(obj, dict) else None
            if not nc:
                continue
            for fidx in [k for k in nc if isinstance(k, int) and k < cutoff]:
                del nc[fidx]
                freed += 1
        self._pruned_total += freed
        return freed

    def stored_frames(self) -> tuple[int, int]:
        """(non_cond entries, cond entries) summed over objects -- the thing
        that must stop growing."""
        per_obj = getattr(self.session, "output_dict_per_obj", None) or {}
        nc = cd = 0
        for obj in per_obj.values():
            if not isinstance(obj, dict):
                continue
            nc += len(obj.get("non_cond_frame_outputs") or {})
            cd += len(obj.get("cond_frame_outputs") or {})
        return nc, cd

    @staticmethod
    def masks(res) -> np.ndarray:
        m = res.get("masks")
        if m is None or len(m) == 0:
            return np.zeros((0, 1, 1), dtype=bool)
        return (m > 0).cpu().numpy() if hasattr(m, "cpu") else np.asarray(m) > 0


class Orient:
    """Orient Anything V2 over masked crops.

    Three things the stock path does that we skip, each deliberate:
      * `inf_single_case(m, ref, tgt)` builds an S=2 sequence; passing the same
        crop twice pushes two images through VGGT for one answer. tgt=None is
        the S=1 absolute path.
      * `inf_single_batch` takes (B, S, C, H, W) and its FORWARD handles B>1 --
        only the output unpacking hardcodes [0]. So all crops go in one pass,
        which is why __call__ below re-implements the (identical) angle decode.
      * `val_fit_alpha` runs a scipy fit on every call; the symmetry parameter
        is needed once at seed to fix the episode's symmetry group, not per
        iteration. Not called here.
    Preprocessing is 518x518 pad, so crop size barely affects cost.
    """

    def __init__(self, repo_dir: Path, ckpt: str | None, device: str,
                 cast_weights: bool = False, size: int = 518):
        import torch
        sys.path.insert(0, str(repo_dir))
        from vision_tower import VGGT_OriAny_Ref                # noqa: E402
        from utils.app_utils import preprocess_images           # noqa: E402
        dtype = (torch.bfloat16
                 if torch.cuda.is_available()
                 and torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)                            # upstream's rule
        if ckpt is None:
            from huggingface_hub import hf_hub_download
            print(f"[setup] resolving {_ORIANY_HF_REPO}/{_ORIANY_HF_FILE} (~5 GB)")
            ckpt = hf_hub_download(_ORIANY_HF_REPO, _ORIANY_HF_FILE)
        self.model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        # Upstream keeps fp32 PARAMETERS and leans on internal autocast, which
        # is why this model sits at ~10 GB resident -- by far the largest
        # consumer on the card. Casting the weights should roughly halve that.
        # Off by default because it deviates from upstream and the accuracy
        # impact is unmeasured; --orient-cast-weights turns it on so the VRAM
        # and latency deltas can be observed directly.
        self.model = (self.model.to(device=device, dtype=dtype) if cast_weights
                      else self.model.to(device)).eval()
        self.torch, self._pre, self.dtype = torch, preprocess_images, dtype
        # VGGT is patch-14, so the input side must be a multiple of 14.
        self.size = max(14, int(round(size / 14)) * 14)

    @staticmethod
    def crop(rgb, mask, *, pad: float = 0.15):
        """Square padded crop around the mask bbox -- square because a
        non-square resize shears apparent orientation, the one thing this
        stage exists to measure."""
        from PIL import Image
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (0.5 + pad)
        H, W = mask.shape
        x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
        y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return Image.fromarray(rgb[y0:y1, x0:x1]).convert("RGB")

    def _preprocess(self, crops, target: int):
        """Upstream `preprocess_images(mode="pad")` with target_size made a
        parameter -- it hardcodes 518.

        Faithful to the original: NO mean/std normalisation (just ToTensor,
        so 0..1), bicubic resize so the LONGEST side is `target` with both
        sides divisible by 14, then pad to a square with white (value=1.0).
        Cost scales with token count = (target/14)^2, so 518 -> 336 is about
        0.42x the work and 518 -> 252 about 0.24x.
        """
        from PIL import Image
        from torchvision import transforms as TF
        to_tensor = TF.ToTensor()
        out = []
        for img in crops:
            img = img.convert("RGB")
            w, h = img.size
            if w >= h:
                nw = target
                nh = max(14, round(h * (nw / w) / 14) * 14)
            else:
                nh = target
                nw = max(14, round(w * (nh / h) / 14) * 14)
            t = to_tensor(img.resize((nw, nh), Image.Resampling.BICUBIC))
            ph, pw = target - t.shape[1], target - t.shape[2]
            if ph > 0 or pw > 0:
                t = self.torch.nn.functional.pad(
                    t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2),
                    mode="constant", value=1.0)
            out.append(t)
        return self.torch.stack(out)

    def __call__(self, crops):
        """All crops in one forward.

        `VGGT_OriAny_Ref.forward` returns DIFFERENT RANKS for the two paths:

            S >  1:  pose_enc = cat([ref_feat.unsqueeze(1),
                                     tgt_feat.view(B, S-1, -1)], dim=1)   -> (B, S, D)
            S == 1:  pose_enc = self.ref_sampler(pose_tokens.view(B*S, C)) -> (B, D)

        Assuming 3-D unconditionally is what produced
        "argmax(): Expected reduction dim 1 to have non-zero size" -- the
        reshape collapsed a 2-D result into a degenerate shape. Handle both.
        """
        t = (self._pre(list(crops), mode="pad") if self.size == 518
             else self._preprocess(crops, self.size)).to(
            device=self.model.get_device(), dtype=self.dtype)
        with self.torch.no_grad():
            pose = self.model(t.unsqueeze(1))          # (B, D) here, since S=1
        if pose.ndim == 3:                             # (B, S, D) -> (B*S, D)
            pose = pose.reshape(pose.shape[0] * pose.shape[1], -1)
        elif pose.ndim != 2:
            raise RuntimeError(f"unexpected pose_enc rank {pose.ndim}, "
                               f"shape {tuple(pose.shape)}")
        if pose.shape[-1] < 900:
            raise RuntimeError(
                f"pose_enc last dim is {pose.shape[-1]}, expected >=900 "
                f"(360 az + 180 el + 360 ro). Model built with the wrong "
                f"out_dim?")
        return {"az": self.torch.argmax(pose[:, 0:360], -1),
                "el": self.torch.argmax(pose[:, 360:540], -1) - 90,
                "ro": self.torch.argmax(pose[:, 540:900], -1) - 180}


def join_to_3d(masks, depth_m, K, *, min_mask_px=64, min_depth_px=16):
    """Mask centroid + median depth -> camera-frame 3D, with the guards that
    matter. `None` means "no measurement", which is the right answer for an
    occluded or textureless object and is NOT the same as a bad measurement."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    valid = np.isfinite(depth_m) & (depth_m > 0)
    out = []
    for m in masks:
        if m.shape != depth_m.shape or m.sum() < min_mask_px:
            out.append(None)                       # occluded / lost / shape skew
            continue
        ys, xs = np.nonzero(m)
        u, v = xs.mean(), ys.mean()                # mask MEAN centroid
        sel = m & valid
        if sel.sum() < min_depth_px:
            out.append(None)                       # SGBM holes: textureless
            continue
        z = float(np.median(depth_m[sel]))         # MEDIAN depth, edge-robust
        out.append(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return out


# ============================================================================
# io
# ============================================================================

def _camera(*, fake: bool, host: str, w: int, h: int):
    if fake:
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        # textured AND shifted, so SGBM has a real disparity to find; a blank
        # pair under-reports its cost
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


def _calib(path: str | None, shape):
    from ego2g1.deploy.perception.depth import StereoCalibration
    if path:
        return StereoCalibration.load(path)
    h, w = shape[:2]
    print(f"[warn] no stereo_calib.npz -- PLACEHOLDER sized to {w}x{h}. SGBM's "
          f"COST is right; the depth VALUES are meaningless.")
    K = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1.0]])
    return StereoCalibration(K_left=K, K_right=K.copy(), dist_left=np.zeros(5),
                             dist_right=np.zeros(5), R=np.eye(3),
                             T=np.array([0.06, 0, 0]), image_size=(w, h))


# ============================================================================

def main(
    *,
    camera_host: str | None = None,
    fake_camera: bool = False,
    prompts: str = _DEFAULT_PROMPTS,
    stereo_calib: str | None = None,
    device: str | None = None,
    dtype: str = "bfloat16",
    n: int = 30,
    warmup: int = 5,
    frames: int = 300,
    prune: bool = True,
    skip_orient: bool = False,
    orient_cast_weights: bool = False,
    orient_size: int = 518,
    auto_download: bool = True,
    policy_period_ms: float = 1000.0,
):
    """Measure the v2 perception stages on this machine.

    frames: streaming steps pushed before reporting, so memory-bank growth is
        visible. 300 = 30 s at 10 Hz; use 3000 to chase a 5-minute leak.
    """
    import os

    _stub_rembg()
    if not _preflight(auto=auto_download):
        return
    root = _repo_root()
    oriany = None if skip_orient else _ensure_oriany(
        root / "third_party" / "Orient-Anything-V2", auto=auto_download)
    if stereo_calib is None:
        p = root / "stereo_calib.npz"
        stereo_calib = str(p) if p.is_file() else None
        if stereo_calib:
            print(f"[setup] stereo calibration: {stereo_calib}")
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
    print(f"torch  : {torch.__version__}   prompts: {prompt_list}")
    print("=" * 72)

    read, close = _camera(fake=fake_camera, host=camera_host, w=640, h=480)
    rgb_l, rgb_r = read()
    print(f"frame  : {rgb_l.shape[1]}x{rgb_l.shape[0]}")
    calib = _calib(stereo_calib, rgb_l.shape)
    R: dict[str, float] = {}

    # ---- sgbm (CPU) --------------------------------------------------------
    from ego2g1.deploy.perception.depth import StereoSGBMDepthSource
    sgbm = StereoSGBMDepthSource(calib)
    w, s, e = _time(lambda: sgbm.estimate(*read()), n=n, warmup=warmup,
                    device="cpu")
    R["sgbm"] = _report("sgbm  (CPU)", w, s, e)
    depth = sgbm.estimate(rgb_l, rgb_r)

    # ---- sam3 (GPU) --------------------------------------------------------
    _vram_reset(dev)
    sam3 = Sam3(_SAM3_REPO, dev, tdtype, prompt_list, prune=prune)
    res = sam3.step(rgb_l)
    found = len(res.get("object_ids", []))
    print(f"\n[detect] {found} object(s) on the first frame"
          + (f" -- {res.get('prompt_to_obj_ids')}" if found else ""))
    if found == 0:
        print("[warn] nothing detected: every number below is for an EMPTY "
              "session and is not representative. Point the camera at the "
              "--prompts objects.")

    print(f"\n--- sam3  (GPU, one session, {len(prompt_list)} prompts, "
          f"detect+track, prune={'ON' if prune else 'OFF'}) ---")
    print(f"  memory window: num_maskmem={sam3.num_maskmem} "
          f"(non-cond entries older than this are unreachable)")
    print(f"  pushing {frames} frames...")
    per, vtrace = [], []
    for i in range(frames):
        f, _ = read()
        _sync(dev)
        t0 = time.perf_counter()
        sam3.step(f)
        _sync(dev)
        per.append(time.perf_counter() - t0)
        if i % 25 == 0:
            nc, cd = sam3.stored_frames()
            vtrace.append((i, _vram_now(dev), nc, cd))
    pf = np.asarray(per) * 1e3
    b = max(1, len(pf) // 10)
    print(f"  first 10 : mean {pf[:10].mean():.1f} ms")
    print("  deciles  : " + "  ".join(f"{pf[i * b:(i + 1) * b].mean():.0f}"
                                      for i in range(10)) + " ms")
    st = pf[len(pf) // 2:]                          # 2nd half = plateau
    p50, p95, p99 = np.percentile(st, [50, 95, 99])
    print(f"  steady   (n={len(st)}): mean {st.mean():.1f}  p50 {p50:.1f}  "
          f"p95 {p95:.1f}  p99 {p99:.1f}  max {st.max():.1f} ms")
    R["sam3"] = float(p95)
    if vtrace and vtrace[0][1] is not None:
        print("  vram(now): " + "  ".join(f"{i}:{v:.0f}MB" for i, v, _, _ in vtrace[:8])
              + (" ..." if len(vtrace) > 8 else ""))
        print("  stored   : " + "  ".join(f"{i}:{nc}nc/{cd}cd"
                                          for i, _, nc, cd in vtrace[:8])
              + (" ..." if len(vtrace) > 8 else ""))
        first, last = vtrace[0][1], vtrace[-1][1]
        # Compare the last two samples, not first-vs-last: startup allocation
        # always rises, and what matters is whether it is STILL rising at the end.
        tail_growth = (vtrace[-1][1] - vtrace[-2][1]) if len(vtrace) >= 2 else 0.0
        span = vtrace[-1][0] - vtrace[-2][0] if len(vtrace) >= 2 else 1
        per_frame = tail_growth / max(span, 1)
        print(f"  growth   : {first:.0f} -> {last:.0f} MB overall; "
              f"tail {per_frame:+.2f} MB/frame over the last {span} frames")
        R["vram_per_frame"] = per_frame
        if per_frame > 0.5:
            print(f"  [FAIL] still growing at the tail. At {per_frame:.1f} MB/frame "
                  f"the card fills in ~{(23500 - last) / max(per_frame, 1e-6):.0f} "
                  f"more frames. Pruning did not bound it -- investigate what "
                  f"else the session retains before trusting a long rollout.")
        elif prune:
            print("  [OK] flat at the tail -- pruning bounds the session. "
                  "Confirm with --frames 3000 before signing this off.")

    # ---- join (CPU) --------------------------------------------------------
    masks = Sam3.masks(sam3.step(rgb_l))
    w, s, e = _time(lambda: join_to_3d(masks, depth, calib.K_left),
                    n=n * 5, warmup=warmup, device="cpu")
    R["join"] = _report("join  (CPU)", w, s, e)

    # ---- orient (GPU) ------------------------------------------------------
    orient = None
    if oriany is not None:
        try:
            t0 = time.perf_counter()
            orient = Orient(oriany, None, dev,
                            cast_weights=orient_cast_weights,
                            size=orient_size)
            _sync(dev)
            vm = _vram(dev)
            print(f"\n[setup] Orient Anything V2 loaded in "
                  f"{time.perf_counter() - t0:.1f} s"
                  + (f", vram {vm:.0f} MB" if vm else ""))
            crops = [c for c in (Orient.crop(rgb_l, m) for m in masks) if c]
            if crops:
                _vram_reset(dev)
                w, s, e = _time(lambda: _quiet(orient, crops),
                                n=max(5, n // 3), warmup=warmup, device=dev)
                R["orient"] = _report(f"orient  (GPU, {len(crops)} crops @ "
                                      f"{orient.size}px, one batched forward)", w, s, e,
                                      vram=_vram(dev))
                w, s, e = _time(lambda: _quiet(orient, crops[:1]),
                                n=max(5, n // 3), warmup=2, device=dev)
                one = _report("orient  (1 crop, for scaling)", w, s, e)
                if not (np.isnan(R["orient"]) or np.isnan(one)):
                    print(f"  -> {R['orient'] / one:.2f}x for {len(crops)} "
                          f"crops vs 1. Near 1.0 = batching works; near "
                          f"{len(crops)}.0 = compute-bound, batching buys "
                          f"nothing.")
            else:
                print("\n--- orient --- skipped: no masks to crop")
        except Exception as exc:                                # noqa: BLE001
            import traceback
            print(f"\n--- orient --- FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            orient = None
    else:
        print("\n--- orient --- skipped")

    # ---- perception_step: one iteration of the async loop -------------------
    print("\n" + "=" * 72)

    def gpu_side(frame):
        out = sam3.step(frame)
        if orient is not None and "orient" in R:
            cs = [c for c in (Orient.crop(frame, m) for m in Sam3.masks(out)) if c]
            if cs:
                _quiet(orient, cs)
        return out

    def step_all():
        # ONE camera read, handed to both arms. Reading separately in each
        # arm serialises them behind HeadCamera's lock, which measures lock
        # contention rather than GPU||CPU overlap.
        left, right = read()
        out, d = _parallel(lambda: gpu_side(left),
                           lambda: sgbm.estimate(left, right))
        join_to_3d(Sam3.masks(out), d, calib.K_left)

    w, s, e = _time(step_all, n=max(5, n // 2), warmup=3, device=dev)
    R["step"] = _report("perception_step  (sam3||sgbm -> join"
                        + (" -> orient)" if orient else ")"), w, s, e,
                        vram=_vram(dev))

    def serial():                                   # prices the GPU||CPU overlap
        left, right = read()
        out = gpu_side(left)
        d = sgbm.estimate(left, right)
        join_to_3d(Sam3.masks(out), d, calib.K_left)

    w, s, e = _time(serial, n=max(5, n // 2), warmup=3, device=dev)
    R["step_serial"] = _report("perception_step  (same, SERIAL)", w, s, e)

    _verdict(R, policy_period_ms=policy_period_ms, n_obj=len(prompt_list))
    close()


def _verdict(r: dict, *, policy_period_ms: float, n_obj: int) -> None:
    nan = float("nan")
    step, sam3, sgbm = (r.get(k, nan) for k in ("step", "sam3", "sgbm"))
    print("\n" + "=" * 72)
    print("VERDICT")

    if not np.isnan(step):
        print(f"\n  [async loop] one full perception iteration = {step:.0f} ms "
              f"p95 -> free-running at {1000 / step:.1f} Hz.")
        print(f"  A policy tick consumes the newest COMPLETED perception, so "
              f"state age spans {step:.0f} ms (just finished) to "
              f"~{2 * step:.0f} ms (one had just started). Budget the worst "
              f"case, not the mean.")
        print(f"  That is {100 * step / policy_period_ms:.0f}% of the "
              f"{policy_period_ms:.0f} ms policy period.")

    if not (np.isnan(step) or np.isnan(r.get("step_serial", nan))):
        d = r["step_serial"] - step
        verdict = ("Worth threading the deploy loop."
                   if d > 5 else "Not worth the threading complexity.")
        print(f"\n  [overlap] running SAM 3 and SGBM concurrently saves "
              f"{d:.0f} ms/iteration ({r['step_serial']:.0f} -> {step:.0f}). "
              f"{verdict}")

    if not (np.isnan(sam3) or np.isnan(sgbm)):
        if sgbm > sam3:
            print(f"\n  [bottleneck] SGBM ({sgbm:.0f} ms) EXCEEDS SAM 3 "
                  f"({sam3:.0f} ms): the CPU is the limit and the overlap has "
                  f"stopped helping. Cut depth cost (resolution, disparity "
                  f"range) before optimising SAM 3.")
        else:
            print(f"\n  [bottleneck] SAM 3 ({sam3:.0f} ms) dominates SGBM "
                  f"({sgbm:.0f} ms), so depth hides inside the GPU stage and "
                  f"is effectively free.")

    if not (np.isnan(step) or np.isnan(r.get("orient", nan))):
        o = r["orient"]
        print(f"\n  [orientation] {o:.0f} ms = {100 * o / step:.0f}% of the "
              f"iteration. Moving it to a slower loop of its own would cut "
              f"position-state age to ~{step - o:.0f}-{2 * (step - o):.0f} ms.")

    print(f"\n  {n_obj} prompt(s), ONE session, ONE backbone pass per frame "
          f"shared by every prompt head and the tracker. Re-run with more "
          f"--prompts to see the (small) per-prompt head cost directly.")
    print("  This script only measures. It changes no deploy default.")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
