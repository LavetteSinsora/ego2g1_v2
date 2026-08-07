"""SAM 3: one session, all prompts, detect+track every frame (plan M1/M3/R1/S1).

Replaces the GroundingDINO->SAM2 cascade of `../detector.py` outright. The
shape of the win is in `Sam3VideoModel.forward`, which computes the vision
backbone ONCE per frame and shares it:

    vision_embeds  = self.detector_model.get_vision_features(pixel_values)
    all_detections = self.run_detection(..., vision_embeds=vision_embeds)
    vision_feats   = self.get_vision_features_for_tracker(vision_embeds)

`run_detection` loops prompts, but the loop body is only the small
text-conditioned head. N prompts cost one backbone + N cheap heads, not N
forward passes (M1) — which is why detection can run on EVERY frame alongside
tracking instead of on a slower reseed tier (M3). One session for all prompts
is also the only correct choice: cross-prompt association is blocked outright
(`_associate_det_trk` zeroes IoU between different prompt ids), so a "cup"
detection can never bind to a "plate" track, and separate sessions would buy
nothing for N times the backbone cost.

Three things in here are not obvious from the upstream API:

  * `postprocess_outputs` DROPS the tracker score. It returns only
    object_ids/scores/boxes/masks/prompt_to_obj_ids. The tracker score is half
    of the S1 visibility signal, so it is read off the RAW output first. Miss
    this and all three S1 gates silently disable — they do not fail, they just
    stop discriminating.
  * The processor always emits float32 `pixel_values`. Weights are bf16. Both
    the session dtype AND the frame need casting, or the first conv raises
    "input and bias type should be the same".
  * The memory bank grows ~11.5 MB per frame forever. `prune()` bounds it, and
    it is provably lossless — see that method.

The `kernels` package is deliberately absent (M4), so SAM 3 skips NMS
post-processing, hole filling and sprinkle removal. Masks keep holes and
duplicate detections survive. `VisibilityGate`'s area test is doing more work
than it otherwise would because of this.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SlotObservation", "VisibilityConfig", "VisibilityGate",
           "normalize_prompt", "Sam3Source"]

SAM3_REPO = "facebook/sam3"


# ---------------------------------------------------------------------------
# pure: what one roster slot looks like on one frame, and whether to trust it
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SlotObservation:
    """One roster slot's SAM 3 result for one frame.

    `det_score is None` is the load-bearing distinction, not a missing value:
    it means the DETECTOR did not re-find this object this frame, so whatever
    mask is present came from memory propagation. Present detection ⇒
    independently re-detected ⇒ genuinely visible ⇒ the crop shows the object.
    Absent detection with a high tracker score ⇒ occluded, and the crop is a
    guess that will produce a confident, wrong orientation.
    """

    instance_id: str
    mask: np.ndarray | None            # (H, W) bool
    box_xyxy: np.ndarray | None        # (4,) float
    det_score: float | None            # None == not re-detected this frame
    tracker_score: float
    mask_area_px: int
    occluded: bool                     # session's own last-occluded flag
    # SAM 3's own tracklet id behind this slot, when there was one. Nothing in
    # the online loop needs it — a slot IS the identity there. Offline it is
    # load-bearing: hotstart removes a tracklet at the frame it becomes
    # confident, so an id that appears in `hotstart_removed_obj_ids` at the END
    # of a pass invalidates every EARLIER frame that used it, and that
    # retraction can only be applied by something holding the id.
    obj_id: int | None = None

    @property
    def detected(self) -> bool:
        return self.det_score is not None

    def centroid_uv(self) -> np.ndarray | None:
        """Mask MEAN centroid. Mean, not median: it is the projection of the
        visible region's centre of area, which is what pairs with a median
        depth over the same region in `join_to_camera`. (The two are
        deliberately different statistics — see that function.)"""
        if self.mask is None:
            return None
        ys, xs = np.nonzero(self.mask)
        if xs.size == 0:
            return None
        return np.array([float(xs.mean()), float(ys.mean())])


@dataclasses.dataclass(frozen=True)
class Visibility:
    """Two gates, not one — see `VisibilityGate` for why.

    `mask_usable`  the mask is a real region of this object: position may be
                   measured and depth may be sampled from it.
    `crop_usable`  the mask is a re-detected, substantially complete view:
                   orientation may be inferred from the crop and the latch
                   divergence test may run against it.

    `crop_usable` implies `mask_usable`. `reason` names the first failing
    test, which is what makes a slot that silently stops updating debuggable
    from a recording instead of from a rerun.
    """

    mask_usable: bool
    crop_usable: bool
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class VisibilityConfig:
    """Thresholds for the two visibility gates (plan S1, Q10).

    These are BRING-UP DEFAULTS, not measured constants. Every one of them
    should be set from a recorded episode before a real rollout — see
    docs/perception_v2_notes.md.

    `min_area_fraction` is a fraction of that object's own running maximum
    mask area, not an absolute pixel count, because objects legitimately
    differ in size by an order of magnitude and an absolute floor would either
    pass a sliver of the big one or reject the small one entirely.

    `area_max_decay` is applied to that running maximum every round. Without
    it the maximum is a permanent high-water mark, so an object that was once
    close to the camera (huge mask) and has since been placed further away
    (small mask, perfectly visible) would be marked unusable forever. At the
    default and ~4.5 Hz the maximum halves in about 30 s — slow enough that a
    brief occlusion cannot drag it down, fast enough that a genuine scale
    change is forgiven within an episode.
    """

    min_det_score: float = 0.5
    min_tracker_score: float = 0.5
    min_area_fraction: float = 0.35
    min_area_px: int = 64
    area_max_decay: float = 0.995


class VisibilityGate:
    """The visibility decision for one roster slot. One instance per slot.

    Plan S1 lists three consumers of one signal — orientation, latch
    divergence, depth sampling — and then states the governing asymmetry:

        "position survives occlusion, orientation does not. A visible-sliver
         centroid is biased but bounded by the object's extent; an orientation
         estimate from the same sliver can be off by 180 degrees. So position
         keeps updating from perception while orientation holds."

    Those two statements CONFLICT for the depth consumer. If one gate drives
    all three, then rejecting the depth sample also stops position updating,
    which is exactly what the asymmetry says must not happen. Resolved by
    splitting the gate in two, along the line the asymmetry itself draws:

        mask_usable   a real region of this object exists — enough pixels,
                      not flagged occluded. Position and depth may be
                      measured. PERMISSIVE, because a biased centroid is
                      still bounded by the object, and the tracker's causal
                      MAD gate downstream exists precisely to catch the
                      occasional bad depth sample that gets through (S2).
        crop_usable   additionally re-detected this frame, with a
                      substantially complete mask and a confident tracker.
                      Orientation and latch divergence may run. STRICT,
                      because both of those failure modes are unbounded:
                      a sliver orientation can be wrong by 180 degrees and
                      poisons the symmetry-snap reference for every later
                      frame, and a spurious divergence drops a latch that
                      was carrying correctly.

    This is a deliberate deviation from the plan's single-gate wording. See
    docs/perception_v2_notes.md.
    """

    def __init__(self, config: VisibilityConfig | None = None):
        self.config = config or VisibilityConfig()
        self._area_max: float = 0.0

    def reset(self) -> None:
        self._area_max = 0.0

    @property
    def area_max_px(self) -> float:
        return self._area_max

    def update(self, obs: SlotObservation) -> Visibility:
        """Fold one frame in and return both gates.

        Order matters: the running maximum decays and then absorbs THIS
        frame's area before the test, so an object that is genuinely growing
        in the frame is never judged against a stale, smaller maximum.

        Stateful — call exactly once per round per slot. Calling it twice on
        one frame decays the area maximum twice.
        """
        cfg = self.config
        self._area_max *= cfg.area_max_decay
        area = int(obs.mask_area_px)
        # Only a re-detected mask may raise the maximum. A memory-propagated
        # mask can drift larger while the object is hidden behind the hand,
        # and letting that set the bar would make every later real detection
        # look like a collapse.
        if obs.detected:
            self._area_max = max(self._area_max, float(area))

        if obs.mask is None or area <= 0:
            return Visibility(False, False, "no mask")
        if area < cfg.min_area_px:
            return Visibility(False, False, f"area {area} < {cfg.min_area_px} px")
        if obs.occluded:
            return Visibility(False, False, "session reports occluded")

        # mask_usable from here on. Everything below only gates the crop.
        if not obs.detected:
            return Visibility(True, False, "not re-detected (memory propagation)")
        if obs.det_score < cfg.min_det_score:
            return Visibility(True, False,
                              f"det {obs.det_score:.2f} < {cfg.min_det_score}")
        if obs.tracker_score < cfg.min_tracker_score:
            return Visibility(True, False,
                              f"tracker {obs.tracker_score:.2f} < "
                              f"{cfg.min_tracker_score}")
        if self._area_max > 0.0 and area < cfg.min_area_fraction * self._area_max:
            return Visibility(True, False,
                              f"area {area} < {cfg.min_area_fraction:.0%} of "
                              f"running max {self._area_max:.0f}")
        return Visibility(True, True, "")


def normalize_prompt(prompt: str) -> str:
    """Strip GroundingDINO's phrase separator from a SAM 3 text prompt.

    `perception/task_config.py`'s documented YAML shape has prompts like
    `"a red cube ."` — that trailing " ." is GroundingDINO's way of separating
    phrases in a single concatenated prompt string. SAM 3 takes ONE concept
    per prompt and has no such convention, so the token is just noise fed to
    the text encoder.

    This is a live suspect for the plan's §2.4 defect (one of three prompts
    never detects). Stripping is the right call rather than rejecting, because
    the failure it causes is silent — a roster slot that is simply empty for
    the whole episode — and a config copied from the v1 example is exactly how
    it gets in. The warning is loud so the config still gets fixed.
    """
    cleaned = prompt.strip()
    if cleaned.endswith(" ."):
        cleaned = cleaned[:-2].strip()
        logger.warning(
            "detector_prompt %r ends in ' .' — that is GroundingDINO's phrase "
            "separator, not SAM 3 syntax. Using %r. Fix the task config.",
            prompt, cleaned)
    elif cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
        logger.warning("detector_prompt %r ends in '.' — stripped to %r for "
                       "SAM 3.", prompt, cleaned)
    if not cleaned:
        raise ValueError(f"detector_prompt {prompt!r} is empty after "
                         "normalisation")
    return cleaned


def build_prompt_map(objects) -> dict[str, str]:
    """{normalised prompt: instance_id}, rejecting collisions.

    `objects` is a sequence of `task_config.ObjectSpec` (anything with
    `.instance_id` / `.detector_prompt`).

    Two roster slots sharing a prompt is refused rather than resolved. SAM 3
    returns detections keyed by prompt, so two slots on one prompt makes the
    slot assignment genuinely ambiguous — and guessing wrong feeds the model
    one object's geometry under another's slot, which degrades quietly instead
    of crashing. That is the same failure `task_config
    .validate_against_server_metadata` exists to prevent, so it gets the same
    treatment.
    """
    mapping: dict[str, str] = {}
    for obj in objects:
        prompt = normalize_prompt(obj.detector_prompt)
        if prompt in mapping:
            raise ValueError(
                f"objects {mapping[prompt]!r} and {obj.instance_id!r} share "
                f"the SAM 3 prompt {prompt!r}. SAM 3 keys detections by "
                "prompt, so the two slots cannot be told apart — give them "
                "distinguishing detector_prompts.")
        mapping[prompt] = obj.instance_id
    return mapping


# ---------------------------------------------------------------------------
# torch: the session
# ---------------------------------------------------------------------------

class Sam3Source:
    """One streaming SAM 3 video session covering the whole object roster.

    Construction loads weights and opens the session; `step(rgb)` pushes one
    frame and returns one `SlotObservation` per roster slot, ALWAYS for every
    slot (a slot with nothing found gets `mask=None`, `det_score=None`,
    `tracker_score=0.0`) so downstream code never has to distinguish "key
    absent" from "not seen".

    torch/transformers are imported in `__init__`, never at module scope —
    same discipline as the rest of this package, so a joint/relative_eef
    deploy never pays for them.
    """

    def __init__(self, objects, *, device: str, dtype=None,
                 repo: str = SAM3_REPO, prune: bool = True,
                 visibility: VisibilityConfig | None = None):
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self._torch = torch
        self.device = device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.repo = repo
        self._prune_enabled = bool(prune)

        self.prompt_to_slot = build_prompt_map(objects)
        self.slot_ids = tuple(obj.instance_id for obj in objects)
        self._prompts = list(self.prompt_to_slot)

        self.model = (Sam3VideoModel.from_pretrained(repo, dtype=self.dtype)
                      .to(device).eval())
        self.processor = Sam3VideoProcessor.from_pretrained(repo)

        # Memory attention reads non-conditioning outputs ONLY for frames
        # t-1 .. t-(num_maskmem-1). Read the real value off the config rather
        # than assuming the documented 7 — prune() correctness depends on it,
        # and a wrong value here deletes entries the model still reads.
        tracker = getattr(self.model, "tracker_model", None)
        self.num_maskmem = int(getattr(tracker, "num_maskmem", 7) or 7)
        # ...and memory attention is NOT the only reader. `_get_object_pointers`
        # walks `non_cond_frame_outputs` back a further
        # `max_object_pointers_in_encoder` frames (default 16 > num_maskmem 7)
        # for object pointers, with `.get(idx, None)`. So pruning at
        # `frame_idx - num_maskmem` does not crash — it silently drops
        # 9 frames' worth of pointers, which is exactly the failure mode this
        # class exists to avoid. The horizon is the MAX of the two readers.
        tcfg = getattr(tracker, "config", None)
        self.max_object_pointers = int(
            getattr(tcfg, "max_object_pointers_in_encoder", 16) or 16)
        self.memory_horizon = max(self.num_maskmem, self.max_object_pointers)

        self._visibility_config = visibility or VisibilityConfig()
        self._gates = {oid: VisibilityGate(self._visibility_config)
                       for oid in self.slot_ids}
        self._frame_idx = -1
        self._pruned_total = 0
        self._warned_missing_scores = False
        self._session = None
        self._open_session()

    # -- session lifecycle --------------------------------------------------

    def _open_session(self) -> None:
        # The session carries a dtype for everything it stores. Leaving it at
        # the float32 default while the weights are bf16 is what produces
        # "input and bias type should be the same" on the first conv.
        self._session = self.processor.init_video_session(
            inference_device=self.device, dtype=self.dtype)
        self.processor.add_text_prompt(self._session, self._prompts)
        self._frame_idx = -1

    def reset(self) -> None:
        """Throw the session away and open a fresh one.

        This is an EPISODE boundary operation, not a memory-management one —
        R1 is explicit that pruning strictly dominates any reset for bounding
        memory, because a reset costs identity and a re-acquisition gap.
        Between episodes the scene has genuinely changed and that gap is what
        you want.
        """
        self._session = None
        self._open_session()
        for gate in self._gates.values():
            gate.reset()

    @property
    def session(self):
        return self._session

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    @property
    def pruned_total(self) -> int:
        return self._pruned_total

    # -- one frame ----------------------------------------------------------

    def preprocess_frame(self, rgb: np.ndarray, *, device: str | None = None):
        """One RGB frame -> the (3, S, S) tensor the session stores.

        Split out of `step` so an offline driver can prepare a whole video
        ONCE and replay it through several sessions (forward, then reverse)
        without paying the resize twice. `device` defaults to this source's
        inference device; pass "cpu" when staging a long video, since the
        preprocessed video is ~6 MB/frame and a few thousand frames of it will
        not fit alongside the weights.
        """
        from PIL import Image

        inputs = self.processor(images=Image.fromarray(np.asarray(rgb)),
                                device=device or self.device, return_tensors="pt")
        return inputs.pixel_values[0].to(dtype=self.dtype)

    def step(self, rgb: np.ndarray) -> dict[str, SlotObservation]:
        """Push one frame; return one observation per roster slot."""
        from PIL import Image

        inputs = self.processor(images=Image.fromarray(np.asarray(rgb)),
                                device=self.device, return_tensors="pt")
        frame = inputs.pixel_values[0].to(dtype=self.dtype)
        with self._torch.no_grad():
            raw = self.model(inference_session=self._session, frame=frame)

        fi = getattr(raw, "frame_idx", None)
        self._frame_idx = int(fi) if fi is not None else self._frame_idx + 1

        # Read the tracker score off the RAW output. postprocess_outputs drops
        # it, and it is half of the S1 signal.
        det_scores = self.raw_map(raw, "obj_id_to_score")
        trk_scores = self.raw_map(raw, "obj_id_to_tracker_score")
        occluded = self.raw_map(raw, "obj_id_to_last_occluded")
        if not trk_scores and not self._warned_missing_scores:
            self._warned_missing_scores = True
            logger.error(
                "SAM 3 output exposes no obj_id_to_tracker_score (looked on "
                "the raw output and the session). The S1 visibility gate will "
                "run on detection score and mask area alone — it will not "
                "fail, it will just discriminate less. Check the transformers "
                "version against docs/perception_v2_pipeline.md §5.1.")

        res = self.processor.postprocess_outputs(
            self._session, raw, original_sizes=inputs.original_sizes)

        if self._prune_enabled:
            self.prune()

        return self.to_slots(res, det_scores, trk_scores, occluded)

    def raw_map(self, raw, name: str) -> dict:
        """Per-object side-channel dict, from the output or the session.

        transformers has moved these between the two across versions, so look
        in both rather than pinning a version. An empty dict is a survivable
        degradation (the gate loses one input), not a crash — the alternative
        is a rollout that refuses to start over a diagnostic field.
        """
        for holder in (raw, self._session):
            value = getattr(holder, name, None)
            if isinstance(value, dict) and value:
                return {int(k): v for k, v in value.items()}
        return {}

    def to_slots(self, res: dict, det_scores: dict, trk_scores: dict,
                 occluded: dict) -> dict[str, SlotObservation]:
        object_ids = [int(i) for i in _to_list(res.get("object_ids"))]
        masks = res.get("masks")
        boxes = res.get("boxes")
        scores = _to_list(res.get("scores"))
        index = {oid: i for i, oid in enumerate(object_ids)}

        # {prompt: [obj_id, ...]} -> {slot: [obj_id, ...]}
        per_slot: dict[str, list[int]] = {oid: [] for oid in self.slot_ids}
        for prompt, ids in (res.get("prompt_to_obj_ids") or {}).items():
            slot = self.prompt_to_slot.get(normalize_prompt(str(prompt)))
            if slot is None:
                continue
            per_slot[slot].extend(int(i) for i in _to_list(ids))

        out: dict[str, SlotObservation] = {}
        for slot in self.slot_ids:
            candidates = [i for i in per_slot[slot] if i in index]
            chosen = self._pick(slot, candidates, det_scores, trk_scores, scores,
                                index)
            if chosen is None:
                out[slot] = SlotObservation(
                    instance_id=slot, mask=None, box_xyxy=None, det_score=None,
                    tracker_score=0.0, mask_area_px=0, occluded=True,
                    obj_id=None)
                continue
            obj_id = chosen
            i = index[obj_id]
            mask = _mask_at(masks, i)
            box = _row_at(boxes, i)
            det = det_scores.get(obj_id)
            if det is None and not det_scores and i < len(scores):
                # Fallback only when the raw per-object map is unavailable:
                # postprocess's `scores` is the detection score for objects
                # that were detected this frame. When the raw map IS present,
                # a missing key genuinely means "not re-detected" and must
                # stay None — that absence is the S1 signal.
                det = scores[i]
            out[slot] = SlotObservation(
                instance_id=slot,
                mask=mask,
                box_xyxy=None if box is None else np.asarray(box, dtype=np.float64),
                det_score=None if det is None else float(det),
                tracker_score=float(trk_scores.get(obj_id, 0.0)),
                mask_area_px=0 if mask is None else int(mask.sum()),
                occluded=bool(occluded.get(obj_id, False)),
                obj_id=int(obj_id),
            )
        return out

    def _pick(self, slot: str, candidates: list[int], det_scores: dict,
              trk_scores: dict, scores: list, index: dict) -> int | None:
        """One instance per prompt is assumed (plan §5.1, Q5). On a collision,
        take the highest-scoring instance and say so — silently taking the
        first would make the slot's identity flip between frames, which the
        tracker's outlier gate would then read as the object teleporting."""
        if not candidates:
            return None
        if len(candidates) > 1:
            logger.warning(
                "slot %s matched %d SAM 3 instances %s this frame; taking the "
                "highest-scoring one. The one-instance-per-prompt assumption "
                "in §5.1 does not hold for this scene (plan Q5).",
                slot, len(candidates), candidates)

        def key(obj_id: int) -> float:
            if obj_id in det_scores:
                return float(det_scores[obj_id])
            i = index[obj_id]
            if i < len(scores):
                return float(scores[i])
            return float(trk_scores.get(obj_id, 0.0))

        return max(candidates, key=key)

    # -- visibility ---------------------------------------------------------

    def visibility(self, observations: dict[str, SlotObservation]
                   ) -> dict[str, Visibility]:
        """Fold one frame's observations through the per-slot gates (S1).

        Stateful — call it exactly once per round, in order. Calling it twice
        on the same frame decays the area maximum twice.
        """
        return {oid: self._gates[oid].update(obs)
                for oid, obs in observations.items()}

    # -- memory -------------------------------------------------------------

    def prune(self) -> int:
        """Drop non-conditioning memory entries the model can no longer read.

        Memory attention reads exactly two things:

            conditioning_outputs, unselected = self._select_closest_cond_frames(...)
            for relative_temporal_offset in range(self.num_maskmem - 1, 0, -1):
                previous_frame_idx = frame_idx - relative_temporal_offset
                output_data = ...["non_cond_frame_outputs"].get(previous_frame_idx, ...)

        ...but memory attention is NOT the only reader, and the plan's R1 is
        wrong on this point. `_get_object_pointers` also walks the same dict:

            for t_diff_offset in range(1, max_object_pointers_to_use):
                ref_frame_idx = frame_idx - t_diff_offset
                out_data = ...["non_cond_frame_outputs"].get(ref_frame_idx, None)

        with `max_object_pointers_to_use = min(num_frames,
        max_object_pointers_in_encoder)` — default 16, more than DOUBLE
        num_maskmem. Cutting at `frame_idx - num_maskmem` therefore deletes
        entries frames t-8 .. t-15 whose object pointers are still read. It
        does not crash, because the read is `.get(..., None)` and a missing
        pointer is simply omitted; it silently weakens re-identification of an
        object that has been occluded for a few frames. That is precisely the
        class of failure this file is written to prevent, so the cutoff is
        `memory_horizon` = max of BOTH readers, not num_maskmem.

        With the real horizon it is again PROVABLY lossless — the deleted
        entries are unreachable by either reader — at the cost of a memory
        bank ~2.3x the size R1 assumed. Bounded is still bounded.

        Conditioning frames are left alone: they are the long-lived anchors,
        and they accumulate ~16x slower (recondition_every_nth_frame=16).

        Strictly better than any reset, partial or total — no identity loss,
        no re-acquisition gap, no discontinuity in the state vector.
        """
        per_obj = getattr(self._session, "output_dict_per_obj", None)
        if not per_obj:
            return 0
        cutoff = self._frame_idx - self.memory_horizon
        if cutoff < 0:
            return 0
        freed = 0
        for obj in per_obj.values():
            non_cond = obj.get("non_cond_frame_outputs") if isinstance(obj, dict) else None
            if not non_cond:
                continue
            for fidx in [k for k in non_cond if isinstance(k, int) and k < cutoff]:
                del non_cond[fidx]
                freed += 1
        self._pruned_total += freed
        return freed

    def stored_frames(self) -> tuple[int, int]:
        """(non-conditioning entries, conditioning entries) summed over
        objects. The first number is the one that must plateau; expect it at
        roughly `memory_horizon x n_objects` once pruning is steady (NOT
        `num_maskmem x n_objects` — see `prune`)."""
        per_obj = getattr(self._session, "output_dict_per_obj", None) or {}
        non_cond = cond = 0
        for obj in per_obj.values():
            if not isinstance(obj, dict):
                continue
            non_cond += len(obj.get("non_cond_frame_outputs") or {})
            cond += len(obj.get("cond_frame_outputs") or {})
        return non_cond, cond


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def _to_list(value) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return list(value.tolist())
    return list(value)


def _mask_at(masks, i: int) -> np.ndarray | None:
    if masks is None or len(masks) <= i:
        return None
    m = masks[i]
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    m = np.asarray(m)
    if m.ndim == 3 and m.shape[0] == 1:      # (1, H, W) -> (H, W)
        m = m[0]
    return m > 0 if m.dtype != bool else m


def _row_at(rows, i: int):
    if rows is None or len(rows) <= i:
        return None
    r = rows[i]
    return r.cpu().numpy() if hasattr(r, "cpu") else r


def join_to_camera(observations: dict[str, SlotObservation],
                   depth_m: np.ndarray, K: np.ndarray, *,
                   visibility: dict[str, Visibility] | None = None,
                   min_mask_px: int = 64, min_depth_px: int = 16
                   ) -> dict[str, tuple[np.ndarray, float] | None]:
    """Mask centroid + median depth -> camera-frame 3D point, per slot.

    Returns {instance_id: ((3,) point, depth_m) or None}. `None` means "no
    measurement", which is the right answer for an occluded or textureless
    object and is NOT the same as a bad measurement — the caller holds instead
    of updating.

    Deliberate asymmetry: MEAN pixel centroid, MEDIAN depth. For a non-convex
    mask these describe slightly different physical points, which is
    acceptable at centroid-of-object precision — but do not call both
    "median" and do not "fix" one to match the other. The mean centroid is the
    visible area's centre; the median depth is the robust one, because a mask
    boundary that bleeds onto the background produces depth outliers a mean
    would swallow.

    Gated on `mask_usable`, NOT `crop_usable` — position survives occlusion
    (see `VisibilityGate`). The residual risk that a partly-occluded mask
    reads the gripper's depth rather than the object's is handled downstream
    by the tracker's causal MAD gate, which is the layer designed for exactly
    one bad sample among good ones.
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    depth_m = np.asarray(depth_m)
    valid = np.isfinite(depth_m) & (depth_m > 0)

    out: dict[str, tuple[np.ndarray, float] | None] = {}
    for oid, obs in observations.items():
        if visibility is not None and not visibility[oid].mask_usable:
            out[oid] = None
            continue
        mask = obs.mask
        if mask is None or mask.shape != depth_m.shape or mask.sum() < min_mask_px:
            out[oid] = None
            continue
        sel = mask & valid
        if sel.sum() < min_depth_px:
            out[oid] = None                      # SGBM holes: textureless
            continue
        uv = obs.centroid_uv()
        if uv is None:
            out[oid] = None
            continue
        u, v = uv
        z = float(np.median(depth_m[sel]))
        out[oid] = (np.array([(u - cx) * z / fx, (v - cy) * z / fy, z],
                             dtype=np.float64), z)
    return out
