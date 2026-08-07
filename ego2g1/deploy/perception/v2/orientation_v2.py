"""Orient Anything V2: object rotation from a masked crop (plan R2, S1, §5.2).

This is the stage that finally fills the honest gap `../relation_perception.py`
documents at length — v1 had no orientation estimator at all and held every
object at its nominal rotation forever.

Two halves, deliberately separated so only one of them needs a GPU:

    pure      `crop_from_mask` (crop geometry), `angles_to_matrix` (the
              az/el/ro -> SO(3) convention), `preprocess_crops` (upstream's
              resize/pad, with the size made a parameter). Testable anywhere.
    torch     `OrientAnythingV2`, which loads a 5 GB checkpoint.

MATCHING THE TRAINING PIPELINE
Training labels come from the SAME model, via
`data_extraction_zh/third_party/humanego_runtime/preprocess/OrientAnything.py`
(`pose_method: vlm`, configs/default.yaml). The absolute canonical frame
therefore does not matter — what matters is that this file reproduces THAT
file's decode exactly, because a rotation that is self-consistent but differs
from training by a fixed remapping is still a valid rotation and nothing
downstream can detect it.

Four things had to be matched, and three of them were initially wrong here:

  1. `angles_to_rot_matrix`: `R = Rz(ro) @ Rx(el) @ Ry(+az)`. This file's
     first draft used `Ry(-az)` — a mirrored azimuth, invisible downstream.
     `TRAINING_ANGLES_TO_MATRIX` below is a verbatim port and a test pins the
     two together; if either moves, the test fails.
  2. Background removal is ON in training (`do_rm_bkg=True` ->
     `background_preprocess`). The model sees a matted foreground, not a raw
     crop. Reproduced by compositing the SAM 3 mask onto a flat background —
     a better segmentation than rembg's matting guess, feeding the model the
     same KIND of image it was trained on.
  3. Only the ANCHOR object gets the model's rotation. Every other object is
     "context" and its rotation is CONSTRUCTED: x-axis toward the anchor,
     y-axis from the model, z = x x y. See `compose_relational_rotation`.
  4. `tgt=None` (S=1) and `ref_alpha_pred` read but never applied — both
     already matched.

Still unmatched, deliberately: the CROP. Training crops the bbox of CoTracker
2D keypoints with a 40 px pad; this crops the SAM 3 mask bbox with a 15%
fractional pad, squared. Both are white-padded to square by upstream
preprocessing so neither shears, but they frame the object differently. If
that shows up as an accuracy gap, the cheaper fix is to re-extract with this
crop rather than to reproduce a keypoint-bbox that has no analogue at deploy.

Everything here is a deliberate deviation from upstream's demo path, each one
costing real debugging time to find and none of it inferable from the docs:

  * Import BY PATH. Orient Anything V2 is a git repo, not a package, and
    there is no `inference.py` — the entry points live in `utils/app_utils.py`.
  * Stub `rembg` if absent. `app_utils` imports it at module scope, but we
    crop from the SAM 3 mask, which beats its matting guess. Never called.
  * `tgt=None`. `inf_single_case(m, ref, tgt)` builds an S=2 sequence;
    passing the same crop twice pushes two images through VGGT for one answer.
  * Batch all crops in ONE forward. `inf_single_batch` takes (B,S,C,H,W) and
    its forward handles B>1 — only the output unpacking hardcodes [0], which
    is why the decode is re-implemented here rather than called.
  * Handle BOTH output ranks. S=1 returns (B, D); S>1 returns (B, S, D).
    Assuming rank 3 produces "argmax(): Expected reduction dim 1 to have
    non-zero size".
  * Skip `val_fit_alpha` in the loop. It runs a scipy fit every call. The
    symmetry parameter is needed ONCE at seed to fix the episode's symmetry
    group; a group that changes per frame destroys the reference tracking the
    snap depends on.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["OrientationConvention", "CAMERA_OPENCV", "angles_to_matrix",
           "TRAINING_ANGLES_TO_MATRIX", "compose_relational_rotation",
           "crop_from_mask", "preprocess_crops", "OrientAnythingV2",
           "ORIANY_GIT", "ORIANY_HF_REPO", "ORIANY_HF_FILE"]

ORIANY_GIT = "https://github.com/SpatialVision/Orient-Anything-V2"
ORIANY_HF_REPO = "Viglong/OriAnyV2_ckpt"
ORIANY_HF_FILE = "demo_ckpts/rotmod_realrotaug_best.pt"   # NOT at repo root


# ---------------------------------------------------------------------------
# pure: angles -> rotation
# ---------------------------------------------------------------------------

def _rot(axis: int, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    R = np.eye(3, dtype=np.float64)
    a, b = [(1, 2), (2, 0), (0, 1)][axis]
    R[a, a] = R[b, b] = c
    R[a, b], R[b, a] = -s, s
    return R


def TRAINING_ANGLES_TO_MATRIX(az_deg, el_deg, ro_deg) -> np.ndarray:
    """Verbatim port of the TRAINING pipeline's decode. Do not "improve" it.

    Source: `data_extraction_zh/third_party/humanego_runtime/preprocess/
    OrientAnything.py::angles_to_rot_matrix`, which produced every object
    rotation the connected checkpoint was trained on. Its own comment reads
    "Rotation order: Yaw (Y) -> Pitch (X) -> Roll (Z)".

    This exists ONLY as the reference `angles_to_matrix` is pinned against
    (see tests). Keeping a second, independent expression of the same
    arithmetic is the point: if someone edits the parameterised version and
    silently changes a sign, the test catches it, which is the one failure
    mode nothing downstream can.
    """
    az, el, ro = np.radians(az_deg), np.radians(el_deg), np.radians(ro_deg)
    R_y = np.array([[np.cos(az), 0, np.sin(az)],
                    [0, 1, 0],
                    [-np.sin(az), 0, np.cos(az)]])
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(el), -np.sin(el)],
                    [0, np.sin(el), np.cos(el)]])
    R_z = np.array([[np.cos(ro), -np.sin(ro), 0],
                    [np.sin(ro), np.cos(ro), 0],
                    [0, 0, 1]])
    return R_z @ R_x @ R_y


@dataclasses.dataclass(frozen=True)
class OrientationConvention:
    """Which camera axis each predicted angle turns about, and in which sense.

    **The defaults reproduce the training pipeline exactly** — they are not a
    guess. `R = Rz(roll) @ Rx(elevation) @ Ry(azimuth)`, matching
    `TRAINING_ANGLES_TO_MATRIX` above.

    Parameterised anyway, for two reasons. If the extraction pipeline is ever
    re-run with a different decode, this is a config change rather than a code
    change; and it lands in `meta.json` via `PerceptionV2Config.as_dict`, so a
    recording made under one convention can be reinterpreted later. That
    matters more here than almost anywhere else in the codebase: getting a
    sign wrong mirrors every object about a plane, the state vector still
    looks entirely plausible, and there is no downstream check that would
    catch it.

    Axis indices are 0=X, 1=Y, 2=Z in the CAMERA frame — OpenCV's, X right,
    Y down, Z forward, the same frame `join_to_camera` back-projects into.
    """

    azimuth_axis: int = 1        # camera Y
    azimuth_sign: float = 1.0    # matches training's Ry(+az)
    elevation_axis: int = 0      # camera X
    elevation_sign: float = 1.0
    roll_axis: int = 2           # camera Z
    roll_sign: float = 1.0


CAMERA_OPENCV = OrientationConvention()


def angles_to_matrix(azimuth_deg, elevation_deg, roll_deg, *,
                     convention: OrientationConvention = CAMERA_OPENCV
                     ) -> np.ndarray:
    """(azimuth, elevation, roll) in degrees -> (3, 3) rotation, camera frame.

    Vectorised: scalars give (3, 3); arrays of length N give (N, 3, 3).

    Orient Anything V2's head emits three classification distributions —
    360 azimuth bins, 180 elevation bins offset by -90, 360 roll bins offset
    by -180 — describing the camera's viewpoint relative to the object's
    canonical front. This turns that triple into the rotation matrix the rest
    of the pipeline speaks in.

    ⚠ UNVALIDATED. The axis assignment and the three signs in `convention`
    are a reasoned default, not a measurement. Two independent things must be
    confirmed before any state vector built from this is trusted:

      1. that this matches Orient Anything V2's own canonical frame (check
         against its renderer on a synthetic object at known angles), and
      2. that THAT frame matches the frame the training labels used, which
         came from a VLM and not from this model at all (plan Q6).

    (2) is the one that actually decides whether the policy sees what it was
    trained on, and it cannot be resolved by reading either codebase — it
    needs a recorded episode with known object poses. Until then this returns
    a self-consistent rotation in an unconfirmed frame.
    """
    return _angles_to_matrix(azimuth_deg, elevation_deg, roll_deg, convention)


def compose_relational_rotation(R_model: np.ndarray, t_cam: np.ndarray,
                                anchor_center_cam: np.ndarray) -> np.ndarray:
    """A CONTEXT object's rotation: x toward the anchor, y from the model.

    Verbatim reproduction of the training pipeline's non-anchor branch
    (`OrientAnything.py::estimate_frame_vlm`, `is_anchor=False`). Only the
    FIRST object in the roster is the anchor there (`CamTriangulator.py:197`,
    `anchor_key = obj_keys[0]`) and takes the model's rotation unchanged;
    every other object's rotation is CONSTRUCTED:

        y = model's y-axis                       (the up/down the model is
                                                  actually reliable about)
        x = (anchor_center - t) projected perpendicular to y, normalised
        z = x cross y

    Feeding the raw model rotation for context objects would be structurally
    different from what the checkpoint was trained on for every slot but one —
    not noisier, DIFFERENT, and in a way no downstream check detects.

    Frame-covariant: doing this with pelvis-frame vectors gives exactly
    `R_pelvis_camera @ (the camera-frame result)`. It is done in the camera
    frame here to match training term for term.

    Degenerate case — the object sits directly above or below the anchor, so
    the projection vanishes — falls back to the model's own x-axis, which is
    what training does (`vlm_context_stacked_fallback`).
    """
    R_model = np.asarray(R_model, dtype=np.float64)
    y_axis = R_model[:, 1].copy()
    to_anchor = np.asarray(anchor_center_cam, dtype=np.float64) - np.asarray(
        t_cam, dtype=np.float64)
    x_proj = to_anchor - np.dot(to_anchor, y_axis) * y_axis
    norm = float(np.linalg.norm(x_proj))
    x_axis = x_proj / norm if norm > 1e-4 else R_model[:, 0].copy()
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _angles_to_matrix(azimuth_deg, elevation_deg, roll_deg, convention):
    az = np.atleast_1d(np.asarray(azimuth_deg, dtype=np.float64))
    el = np.atleast_1d(np.asarray(elevation_deg, dtype=np.float64))
    ro = np.atleast_1d(np.asarray(roll_deg, dtype=np.float64))
    if not (az.shape == el.shape == ro.shape):
        raise ValueError(f"angle shapes disagree: {az.shape} {el.shape} "
                         f"{ro.shape}")
    c = convention
    out = np.stack([
        _rot(c.roll_axis, c.roll_sign * r)
        @ _rot(c.elevation_axis, c.elevation_sign * e)
        @ _rot(c.azimuth_axis, c.azimuth_sign * a)
        for a, e, r in zip(az, el, ro)
    ])
    return out[0] if np.ndim(azimuth_deg) == 0 else out


# ---------------------------------------------------------------------------
# pure: crop geometry and preprocessing
# ---------------------------------------------------------------------------

def crop_from_mask(rgb: np.ndarray, mask: np.ndarray, *, pad: float = 0.15,
                   min_side_px: int = 8, background: str = "white"):
    """Square, padded crop around the mask's bounding box, or None.

    `background`:
        "white"  zero out everything outside the mask, to white.
        "gray"   ... to mid-gray (128).
        "none"   leave the raw crop.

    Background removal is NOT optional decoration — training runs
    `background_preprocess(crop, True)` (rembg), so the checkpoint's labels
    came from a model that saw a matted foreground. A raw crop at deploy is a
    domain shift on the model's input, and orientation is exactly the sort of
    global-shape judgement a cluttered background moves.

    Using the SAM 3 mask instead of rembg is a deliberate upgrade: it is a
    better segmentation of the object we actually care about, and it costs
    nothing because the mask already exists. What must be confirmed on
    hardware is the FILL COLOUR — `background_preprocess` composites onto a
    specific background, and it should be the same one. White is the default
    because upstream `preprocess_images` also pads to square with white
    (value 1.0), so the fill and the pad agree and the object sits on one
    uniform field. See docs/perception_v2_notes.md.

    SQUARE is not cosmetic. Preprocessing resizes the longest side and pads
    the rest, so a non-square crop of a non-square object arrives with a
    different aspect ratio than the model saw in training — which shears
    apparent orientation, the one quantity this stage exists to measure.
    Padding the crop to square in PIXEL space instead makes the later resize
    isotropic.

    `pad` adds context around the bbox: the model reads orientation partly
    from how the object sits against its surroundings, and a crop clipped
    exactly to the silhouette removes that.
    """
    from PIL import Image

    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    cx = (float(xs.min()) + float(xs.max())) / 2.0
    cy = (float(ys.min()) + float(ys.max())) / 2.0
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (0.5 + float(pad))
    H, W = mask.shape
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    if x1 - x0 < min_side_px or y1 - y0 < min_side_px:
        return None

    patch = np.asarray(rgb)[y0:y1, x0:x1]
    if background != "none":
        fill = {"white": 255, "gray": 128}.get(background)
        if fill is None:
            raise ValueError(f"background must be 'white', 'gray' or 'none', "
                             f"got {background!r}")
        patch = np.where(mask[y0:y1, x0:x1, None], patch,
                         np.uint8(fill)).astype(np.uint8)
    return Image.fromarray(patch).convert("RGB")


def preprocess_crops(crops, target: int):
    """Upstream `preprocess_images(mode="pad")` with `target_size` made a
    parameter — it hardcodes 518.

    Faithful to the original, and every one of these details matters because
    the model was trained on exactly this: NO mean/std normalisation (plain
    ToTensor, so 0..1), bicubic resize so the LONGEST side is `target` with
    both sides divisible by 14 (VGGT is patch-14), then pad to square with
    WHITE (value 1.0), not black or edge-replicate.

    Cost scales with the token count `(target/14)^2`, which is the R2 lever:
    518 -> 336 is ~0.42x the work, 518 -> 252 is ~0.24x. Rough orientation is
    sufficient for this pipeline (plan §2.3), so this is the cheapest
    remaining latency available.
    """
    import torch
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
            t = torch.nn.functional.pad(
                t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2),
                mode="constant", value=1.0)
        out.append(t)
    return torch.stack(out)


def stub_rembg() -> None:
    """`utils/app_utils.py` does `import rembg` at module scope, so its
    helpers cannot be imported without it — even though we never want to run
    it, because we crop from the SAM 3 mask and that beats rembg's matting
    guess. Stub it so the import succeeds and only a real CALL would raise."""
    import types
    try:
        import rembg  # noqa: F401
        return
    except ImportError:
        pass

    def _unavailable(*_a, **_k):
        raise RuntimeError("rembg is deliberately not installed — perception "
                           "v2 crops from SAM 3 masks (orientation_v2.py)")

    stub = types.ModuleType("rembg")
    stub.remove = stub.new_session = _unavailable
    sys.modules["rembg"] = stub
    logger.info("rembg absent -> import stub installed (never called)")


# ---------------------------------------------------------------------------
# torch
# ---------------------------------------------------------------------------

class OrientAnythingV2:
    """Batched orientation inference over masked crops.

    `estimate(rgb, observations, crop_usable)` returns
    {instance_id: (3, 3) rotation or None}. `None` means "do not update" —
    never a fabricated identity. The caller holds the last usable rotation,
    which is S1's rule and the reason this returns None rather than guessing:
    a sliver crop yields a confident, wrong answer, and a wrong rotation
    poisons the symmetry-snap reference for every later frame.
    """

    def __init__(self, repo_dir, *, device: str, checkpoint: str | None = None,
                 cast_weights: bool = False, size: int = 518,
                 crop_pad: float = 0.15, background: str = "white",
                 convention: OrientationConvention = CAMERA_OPENCV):
        import torch

        stub_rembg()
        repo_dir = Path(repo_dir)
        if not (repo_dir / "vision_tower.py").is_file():
            raise FileNotFoundError(
                f"{repo_dir} does not look like an Orient-Anything-V2 "
                f"checkout (no vision_tower.py). Clone it:\n"
                f"    git clone --depth 1 {ORIANY_GIT} {repo_dir}")
        sys.path.insert(0, str(repo_dir))
        from vision_tower import VGGT_OriAny_Ref                  # noqa: E402

        self._torch = torch
        self.device = device
        self.convention = convention
        self.crop_pad = float(crop_pad)
        self.background = str(background)
        # VGGT is patch-14, so the input side must be a multiple of 14.
        self.size = max(14, int(round(size / 14)) * 14)
        # Upstream's own rule: bf16 on Ampere and later, fp16 below.
        self.dtype = (torch.bfloat16
                      if torch.cuda.is_available()
                      and torch.cuda.get_device_capability()[0] >= 8
                      else torch.float16)

        if checkpoint is None:
            from huggingface_hub import hf_hub_download
            logger.info("resolving %s/%s (~5 GB)", ORIANY_HF_REPO, ORIANY_HF_FILE)
            checkpoint = hf_hub_download(ORIANY_HF_REPO, ORIANY_HF_FILE)

        model = VGGT_OriAny_Ref(out_dim=900, dtype=self.dtype, nopretrain=True)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        # Upstream keeps fp32 PARAMETERS and leans on internal autocast, which
        # is why a 5 GB checkpoint occupies ~10.2 GB resident — by far the
        # largest consumer on the card, and the reason the combined footprint
        # does not fit (plan §2.2). Casting should roughly halve it. Off by
        # default because it deviates from upstream and the accuracy cost is
        # unmeasured (plan §8 step 3).
        self.model = (model.to(device=device, dtype=self.dtype) if cast_weights
                      else model.to(device)).eval()
        self.cast_weights = bool(cast_weights)

    def estimate(self, rgb: np.ndarray, observations: dict,
                 crop_usable: dict[str, bool],
                 *, skip: frozenset[str] = frozenset(),
                 anchor_id: str | None = None,
                 points_cam: dict | None = None
                 ) -> dict[str, np.ndarray | None]:
        """One batched forward over every slot worth looking at.

        `skip` is for latched objects: while an object is rigidly held its
        pose comes from FK, so inferring its orientation is both unnecessary
        and unreliable (the hand is covering it). Dropping one crop is ~30 ms
        off the round — the third R2 lever, and the only one that costs
        nothing at all.

        `anchor_id` / `points_cam` reproduce training's anchor-vs-context
        split. The anchor keeps the model's rotation; every other object gets
        the relational construction (`compose_relational_rotation`), which
        needs camera-frame translations for itself and the anchor. Passing
        neither falls back to raw model rotations for everything — correct
        only if the checkpoint was trained with a single-object roster.
        """
        wanted = [oid for oid in observations
                  if crop_usable.get(oid) and oid not in skip]
        crops, keep = [], []
        for oid in wanted:
            crop = crop_from_mask(rgb, observations[oid].mask,
                                  pad=self.crop_pad, background=self.background)
            if crop is not None:
                crops.append(crop)
                keep.append(oid)

        out: dict[str, np.ndarray | None] = {oid: None for oid in observations}
        if not crops:
            return out

        az, el, ro = self.angles_for_crops(crops)
        mats = angles_to_matrix(az, el, ro, convention=self.convention)
        raw = {oid: mats[i] for i, oid in enumerate(keep)}

        if anchor_id is None or points_cam is None:
            return {**out, **raw}

        anchor_t = points_cam.get(anchor_id)
        for oid, R in raw.items():
            t = points_cam.get(oid)
            if oid == anchor_id or anchor_t is None or t is None:
                # No anchor position this round means the relational x-axis is
                # undefined. Falling back to the raw rotation keeps the slot
                # updating; holding instead would freeze it whenever the
                # anchor blinks, which is more often than it is worth.
                out[oid] = R
            else:
                out[oid] = compose_relational_rotation(R, t, anchor_t)
        return out

    def angles_for_crops(self, crops, *, return_azimuth_distribution=False):
        """(azimuth, elevation, roll) in degrees for a list of PIL crops.

        Public because the batch is the only lever that scales: online there
        are at most three crops (one per roster slot) and `estimate` is the
        whole story, but an offline extraction has thousands and wants them
        batched across FRAMES, not just across slots. Same forward, same
        decode — sharing it is what keeps an offline extraction measuring the
        model the deploy loop actually runs.

        `return_azimuth_distribution` additionally returns the (B, 360)
        SIGMOID of the azimuth logits. The argmax alone throws away the shape
        of that distribution, and the shape is where the symmetry lives: a
        two-fold symmetric object produces two equal peaks 180 deg apart, and
        upstream's `val_fit_alpha` recovers the symmetry order by fitting
        von Mises curves to exactly this array. Sigmoid, not softmax — the
        head is trained with BCE, so the bins are independent probabilities
        and softmax would distort the peak ratios the fit reads.
        """
        torch = self._torch
        batch = preprocess_crops(crops, self.size).to(
            device=self.model.get_device(), dtype=self.dtype)
        with torch.no_grad():
            pose = self.model(batch.unsqueeze(1))     # S=1 -> (B, D)

        # VGGT_OriAny_Ref.forward returns DIFFERENT RANKS for the two paths:
        #   S >  1:  cat([ref.unsqueeze(1), tgt.view(B, S-1, -1)], 1) -> (B, S, D)
        #   S == 1:  ref_sampler(pose_tokens.view(B*S, C))            -> (B, D)
        # Assuming rank 3 unconditionally is what produced "argmax():
        # Expected reduction dim 1 to have non-zero size".
        if pose.ndim == 3:
            pose = pose.reshape(pose.shape[0] * pose.shape[1], -1)
        elif pose.ndim != 2:
            raise RuntimeError(f"unexpected pose_enc rank {pose.ndim}, shape "
                               f"{tuple(pose.shape)}")
        if pose.shape[-1] < 900:
            raise RuntimeError(
                f"pose_enc last dim is {pose.shape[-1]}, expected >= 900 "
                f"(360 azimuth + 180 elevation + 360 roll). Model built with "
                f"the wrong out_dim?")

        az = torch.argmax(pose[:, 0:360], -1).float().cpu().numpy()
        el = (torch.argmax(pose[:, 360:540], -1) - 90).float().cpu().numpy()
        ro = (torch.argmax(pose[:, 540:900], -1) - 180).float().cpu().numpy()
        if return_azimuth_distribution:
            dist = torch.sigmoid(pose[:, 0:360].float()).cpu().numpy()
            return az, el, ro, dist
        return az, el, ro
