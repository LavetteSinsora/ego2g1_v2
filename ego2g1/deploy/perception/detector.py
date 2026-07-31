"""The ~2 Hz detector stage of the perception cascade (docs/relation_deploy_plan.md §5.3).

`ObjectDetector` is the boundary: given an RGB frame and the set of objects
to look for, return per-object 2D geometry (mask and/or box) + a confidence.
Two implementations:

  GroundingDinoSam2Detector   The real thing -- GroundingDINO (HF zero-shot
                              text->box) + SAM2 (box->mask), mirroring
                              `data_extraction_zh`'s own
                              `third_party/humanego_runtime/preprocess/
                              DINOSAM.py::DINOSAMEngine` call shape exactly
                              (read-only reference; not imported from here --
                              that repo is its own uv project). Heavy deps
                              (torch/transformers/sam2/huggingface_hub) are
                              imported lazily, inside `__init__`, exactly like
                              `deploy/kinematics.py` does for mujoco/mink and
                              `deploy/executor.py` does for unitree_deploy --
                              constructing this class is the only place a
                              joint/relative_eef-mode deploy would ever pay
                              for those imports, and it never constructs it.
  FakeDetector                Deterministic test double: returns whatever was
                              pre-programmed per instance_id, ignores the
                              actual image. Used by this module's own tests
                              and by whoever wires the full cascade
                              (tracker.py, orientation.py, the not-yet-built
                              `RelationPerception.observe`) without a camera
                              or a GPU.

Object query shape: the plan's `ego2g1/deploy/perception/task_config.py`
(§5.1) defines `ObjectSpec(instance_id, category, detector_prompt,
graspable)`, but that module was not present in this worktree at the time
this was written (parallel task, see docs/relation_deploy_plan.md §9 task 6).
`ObjectQuery` below is the minimal (`instance_id`, `detector_prompt`) subset
this module actually needs -- any `ObjectSpec` instance satisfies it too by
duck typing (both are plain attributes), so integration is a drop-in swap:
reconcile the two at wiring time rather than importing a module that may not
exist yet.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import NamedTuple, Sequence

import numpy as np


class ObjectQuery(NamedTuple):
    """What the detector needs per object. Subset of (the not-yet-merged)
    `task_config.ObjectSpec` -- reconcile field-for-field at integration
    time; this class exists so `detector.py` has zero import dependency on
    that module's merge timing."""

    instance_id: str
    detector_prompt: str


@dataclasses.dataclass(frozen=True)
class Detection:
    """One object's 2D detection result for one frame.

    Exactly one of `mask`/`box_xyxy` may be omitted, not both -- a detector
    that only found a box (no segmentation) still gives the downstream
    depth-lift stage something to sample; a detector that only has a mask
    can derive a tight box from it if needed, but this module does not do
    that automatically (keep the two representations honest about what was
    actually measured).
    """

    instance_id: str
    confidence: float
    mask: np.ndarray | None = None       # (H, W) bool, True == object pixel
    box_xyxy: np.ndarray | None = None   # (4,) pixel-space [x0, y0, x1, y1]

    def __post_init__(self):
        if self.mask is None and self.box_xyxy is None:
            raise ValueError(
                f"Detection({self.instance_id!r}) needs a mask, a box, or both"
            )
        if self.mask is not None and np.asarray(self.mask).ndim != 2:
            raise ValueError(
                f"Detection({self.instance_id!r}).mask must be (H, W), "
                f"got shape {np.asarray(self.mask).shape}"
            )

    def centroid_uv(self) -> np.ndarray:
        """(2,) pixel centroid: mask centroid if a mask is present (robust to
        box padding around a non-rectangular object), else the box center."""
        if self.mask is not None:
            mask = np.asarray(self.mask)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                raise ValueError(f"Detection({self.instance_id!r}).mask is empty")
            return np.array([float(np.mean(xs)), float(np.mean(ys))])
        x0, y0, x1, y1 = np.asarray(self.box_xyxy, dtype=np.float64)
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])


class ObjectDetector(abc.ABC):
    """The cascade's detector-stage interface. One call = one frame's worth
    of per-object 2D geometry. Missing `instance_id`s in the returned dict
    mean "not found this frame" -- the caller (tracker / latch state
    machine) decides how to treat a miss, this interface does not guess."""

    @abc.abstractmethod
    def detect(
        self, image: np.ndarray, queries: Sequence[ObjectQuery]
    ) -> dict[str, Detection]:
        """image: (H, W, 3) RGB uint8. queries: what to look for this call.
        Returns {instance_id: Detection} for every query that was found."""
        raise NotImplementedError


class GroundingDinoSam2Detector(ObjectDetector):
    """GroundingDINO (text->box) + SAM2 (box->mask), one prompt per object.

    Assumes the model API surface `transformers.AutoModelForZeroShotObjectDetection`
    (GroundingDINO via HF) and `sam2.build_sam.build_sam2` /
    `sam2.sam2_image_predictor.SAM2ImagePredictor` expose today -- verified
    against `DINOSAMEngine.__init__`/`predict_frame_internal` in
    `data_extraction_zh/third_party/humanego_runtime/preprocess/DINOSAM.py`
    (read-only reference). Picks the single highest-confidence box per
    prompt (this cascade expects exactly one instance per `ObjectQuery`,
    unlike that reference's multi-instance-per-category handling).

    Does NOT download/run anything at import time -- only when constructed.
    Add the `perception` dependency group (pyproject.toml) before
    instantiating this on real hardware; see the __init__ docstring below
    for what happens if you don't.
    """

    def __init__(
        self,
        *,
        dino_model_id: str = "IDEA-Research/grounding-dino-tiny",
        sam2_config: str = "sam2_hiera_l.yaml",
        sam2_repo_id: str = "facebook/sam2-hiera-large",
        sam2_checkpoint_name: str = "sam2_hiera_large.pt",
        box_threshold: float = 0.3,
        device: str | None = None,
    ):
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GroundingDinoSam2Detector needs the optional 'perception' "
                "dependency group (torch, transformers, sam2, huggingface-hub, "
                "and friends). Install it with:\n"
                "  uv sync --group perception\n"
                "(see pyproject.toml's `perception` group; do NOT combine with "
                "--group train on macOS, see docs/environments.md). This class "
                "is only ever constructed for `relation_eef`-mode deploy on "
                "hardware with those weights available -- joint/relative_eef "
                "deploys never hit this import."
            ) from exc

        self._torch = torch
        self.box_threshold = float(box_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._processor = AutoProcessor.from_pretrained(dino_model_id)
        self._dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model_id
        ).to(self.device)
        checkpoint_path = hf_hub_download(
            repo_id=sam2_repo_id, filename=sam2_checkpoint_name
        )
        self._predictor = SAM2ImagePredictor(
            build_sam2(sam2_config, checkpoint_path, device=self.device)
        )

    def detect(
        self, image: np.ndarray, queries: Sequence[ObjectQuery]
    ) -> dict[str, Detection]:
        from PIL import Image as PILImage

        image = np.ascontiguousarray(image)
        image_pil = PILImage.fromarray(image)
        width, height = image_pil.size
        self._predictor.set_image(image)

        out: dict[str, Detection] = {}
        for query in queries:
            inputs = self._processor(
                images=image_pil, text=query.detector_prompt, return_tensors="pt"
            ).to(self.device)
            with self._torch.no_grad():
                outputs = self._dino(**inputs)

            logits = outputs.logits.sigmoid()[0]
            boxes = outputs.pred_boxes[0]
            keep = logits.max(-1)[0] > self.box_threshold
            filtered_logits = logits[keep]
            filtered_boxes = boxes[keep]
            if len(filtered_boxes) == 0:
                continue

            confidences = filtered_logits.max(-1)[0].detach().cpu().numpy()
            best = int(np.argmax(confidences))
            scale = self._torch.tensor(
                [width, height, width, height], device=self.device
            )
            cx, cy, w, h = (filtered_boxes[best] * scale).tolist()
            box_xyxy = np.array(
                [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                dtype=np.float64,
            )

            masks, _, _ = self._predictor.predict(
                box=box_xyxy[None], multimask_output=False
            )
            mask = np.asarray(masks[0], dtype=bool)
            out[query.instance_id] = Detection(
                instance_id=query.instance_id,
                confidence=float(confidences[best]),
                mask=mask,
                box_xyxy=box_xyxy,
            )
        return out


class FakeDetector(ObjectDetector):
    """Deterministic test double. Returns exactly what was programmed for
    each `instance_id` via `set_detection`/the constructor, regardless of
    the image content -- and records every call so a test (or later
    integration code) can assert on cadence/what was asked for."""

    def __init__(self, detections: dict[str, Detection] | None = None):
        self._programmed: dict[str, Detection] = dict(detections or {})
        self.calls: list[tuple[np.ndarray, tuple[str, ...]]] = []

    def set_detection(self, instance_id: str, detection: Detection) -> None:
        if detection.instance_id != instance_id:
            raise ValueError(
                f"detection.instance_id {detection.instance_id!r} != "
                f"key {instance_id!r}"
            )
        self._programmed[instance_id] = detection

    def clear_detection(self, instance_id: str) -> None:
        """Simulate a miss: the next `detect()` call won't return this id."""
        self._programmed.pop(instance_id, None)

    def detect(
        self, image: np.ndarray, queries: Sequence[ObjectQuery]
    ) -> dict[str, Detection]:
        self.calls.append((image, tuple(q.instance_id for q in queries)))
        return {
            q.instance_id: self._programmed[q.instance_id]
            for q in queries
            if q.instance_id in self._programmed
        }
