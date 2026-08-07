"""SAM 3 over a whole recorded video, using what only offline can use.

The deploy loop (`ego2g1/deploy/perception/v2/sam3_source.py`) drives SAM 3 in
STREAMING mode: one frame in, one result out, no future. That is forced — the
robot has no future frames. This module drives the same weights, the same
session type and the same slot mapping through the OFFLINE path instead, and
the differences are not cosmetic.

WHAT OFFLINE BUYS, and where each one is in the transformers source
(`models/sam3_video/modeling_sam3_video.py`)

1.  HOTSTART TRACKLET REMOVAL — the big one.
    `_process_hotstart` guards its two removal rules with `if not streaming:`.
    Streaming keeps every tracklet it ever births; offline deletes
      * tracklets unmatched by the detector for `hotstart_unmatch_thresh`
        frames (default 8) — spurious births, and
      * duplicate tracklets that overlap an earlier one for
        `hotstart_dup_thresh` frames (default 8).
    `streaming` is set from `frame is not None` in `forward`, so it is the
    ARGUMENT SHAPE that selects the behaviour: pass `frame=` and you get the
    streaming path, pass `frame_idx=` against a session that already holds the
    video and you get the offline one. The plan's M3 names this cost
    ("duplicates and unmatched tracklets are pruned less aggressively than
    offline"); here we simply do not pay it.

2.  A GLOBAL, NOT DELAYED, HOTSTART FILTER.
    `propagate_in_video_iterator` buffers only `hotstart_delay` (15) frames
    before yielding, and `postprocess_outputs` filters against whatever
    `hotstart_removed_obj_ids` holds AT CALL TIME. So a tracklet retracted at
    frame 400 still appears in frames 0..384 of the iterator's own output.
    Having the whole video in hand, we keep each frame's chosen tracklet id
    and apply the FINAL removed set to every frame afterwards (`_retract`).
    This is strictly more filtering than the upstream iterator performs, and
    it is only possible offline.

3.  REVERSE PROPAGATION.
    `forward(..., reverse=True)` runs memory attention forward in time from
    the END of the video. This is the fix for the structural weakness of a
    causal tracker: an object is invisible until the detector first fires on
    it, so every frame before that is empty — and in an egocentric recording
    the interesting object is very often occluded, off-frame or motion-blurred
    at frame 0. A reverse pass propagates the confident late-video mask
    BACKWARD into exactly those frames. Merging the two passes is what makes
    "a mask on every frame" achievable rather than aspirational.

4.  NO LATENCY BUDGET.
    Detection runs on every frame in both directions, orientation runs on
    every mask regardless of the S1 crop gate, and crops batch across FRAMES
    rather than across the three roster slots (see `orient_offline.py`).

WHAT IS DELIBERATELY THE SAME
    Slot mapping, the one-instance-per-prompt collision rule, the raw-output
    tracker-score extraction, and the two visibility gates all come from
    `Sam3Source`. An experiment that re-derived them would be measuring a
    pipeline the robot does not run.

WHAT IS NOT DONE HERE
    The visibility gates are EVALUATED and recorded, but nothing is dropped on
    account of them. Offline the question is not "is this crop safe to use" —
    it is "what would the online gate have thrown away, and was it right?".
    Answering that needs the rejected frames present in the output.
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from ego2g1.deploy.perception.v2.sam3_source import (
    SlotObservation, VisibilityConfig, VisibilityGate,
)

logger = logging.getLogger(__name__)

__all__ = ["SOURCE_NAMES", "SlotFrame", "Tracks", "OfflineSam3", "PresenceProbe"]


class PresenceProbe:
    """Captures SAM 3's per-PROMPT presence score, which is otherwise thrown away.

    SAM 3's detector carries a dedicated presence token whose head answers a
    different question from any per-instance score: *does the queried concept
    appear in this image at all?* `Sam3DetectorOutput` documents it as

        final_scores = pred_logits.sigmoid() * presence_logits.sigmoid()

    and `run_detection` computes exactly that:

        pred_probs = pred_logits.sigmoid()
        presence_scores = presence_logits.sigmoid()
        pred_probs = pred_probs * presence_scores          # <- then discarded

    Only the PRODUCT survives into `det_out["scores"]`, and that is what
    reaches `obj_id_to_score` and every threshold downstream. The presence
    factor itself never leaves the function. That makes a genuinely useful
    signal unavailable: a low product cannot distinguish "the concept is
    absent from this frame" from "the concept is here but no query localised
    it well", and those have opposite fixes — the first is a prompt problem,
    the second a threshold problem. For the plan's §2.4 defect (a roster slot
    that never fills) it is the first question you would want to ask.

    Captured with a forward hook on the detector module rather than by
    patching `run_detection`, so no upstream code is reimplemented. The hook
    fires once per prompt per frame, in the order `run_detection` iterates
    (`list(inference_session.prompts.keys())`), which is the order prompts
    were added — so call k maps to prompt k.

    Presence is PER PROMPT, not per instance: every slot fed by one prompt
    gets one number for the frame, and a slot with no detection still gets
    one. That is the point — it is the only score that exists when nothing
    was found.
    """

    def __init__(self, model, prompt_order: list[str], prompt_to_slot: dict):
        self.slots_in_order = [prompt_to_slot[p] for p in prompt_order]
        self._calls: list[float] = []
        self._handle = None
        self._model = model
        self.available = False
        detector = getattr(model, "detector_model", None)
        if detector is None:
            logger.warning("SAM 3 model exposes no detector_model — presence "
                           "score unavailable.")
            return
        self._handle = detector.register_forward_hook(self._hook)
        self.available = True

    def _hook(self, _module, _inputs, output):
        logits = getattr(output, "presence_logits", None)
        if logits is None:
            self._calls.append(float("nan"))
            return
        import torch

        with torch.no_grad():
            self._calls.append(float(torch.sigmoid(logits.float()).max()))

    def take(self) -> dict[str, float]:
        """Drain one frame's calls -> {slot: presence}. Call once per frame."""
        calls, self._calls = self._calls, []
        if not calls:
            return {}
        if len(calls) != len(self.slots_in_order):
            # Fires if upstream ever stops calling the detector once per
            # prompt. Reporting nothing beats reporting a misaligned mapping,
            # which would silently attribute one object's presence to another.
            logger.warning("presence hook saw %d detector calls for %d prompts "
                           "— dropping this frame's presence scores.",
                           len(calls), len(self.slots_in_order))
            return {}
        return dict(zip(self.slots_in_order, calls))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

# Provenance of the merged observation for one (frame, slot). Written into the
# output as a uint8 so the dashboard can colour a timeline by it — which is
# the direct read-out of what the reverse pass was worth.
SOURCE_NONE = 0        # neither pass produced a mask
SOURCE_FORWARD = 1     # forward only
SOURCE_REVERSE = 2     # reverse only  <- frames the online loop cannot have
SOURCE_BOTH_FORWARD = 3
SOURCE_BOTH_REVERSE = 4
SOURCE_NAMES = {
    SOURCE_NONE: "none",
    SOURCE_FORWARD: "forward",
    SOURCE_REVERSE: "reverse",
    SOURCE_BOTH_FORWARD: "both->forward",
    SOURCE_BOTH_REVERSE: "both->reverse",
}


# ---------------------------------------------------------------------------
# per-slot per-frame record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SlotFrame:
    """One roster slot on one frame, after merging the passes.

    The mask is stored PACKED (`np.packbits`) because it is the only field
    with any size: a 610-frame episode at 640x480 with three slots is 560 MB
    of bool and 70 MB of bits, held for two passes at once during the merge.
    Unpack with `mask(height, width)`.
    """

    mask_bits: np.ndarray | None = None
    box_xyxy: np.ndarray | None = None
    det_score: float | None = None
    tracker_score: float = 0.0
    mask_area_px: int = 0
    occluded: bool = True
    obj_id: int | None = None
    source: int = SOURCE_NONE
    # SAM 3's presence score for this slot's PROMPT on this frame — "does this
    # concept appear at all?", independent of whether anything was localised.
    # NaN when unavailable. Survives an empty slot on purpose: it is the only
    # score that exists when nothing was found.
    presence: float = float("nan")
    # filled by the visibility replay, after the merge
    mask_usable: bool = False
    crop_usable: bool = False
    gate_reason: str = "no mask"

    @property
    def has_mask(self) -> bool:
        return self.mask_bits is not None and self.mask_area_px > 0

    @property
    def detected(self) -> bool:
        return self.det_score is not None

    def mask(self, height: int, width: int) -> np.ndarray | None:
        if self.mask_bits is None:
            return None
        flat = np.unpackbits(self.mask_bits, count=height * width)
        return flat.reshape(height, width).astype(bool)

    @classmethod
    def from_observation(cls, obs: SlotObservation, source: int,
                         presence: float = float("nan")) -> "SlotFrame":
        return cls(
            presence=float(presence),
            mask_bits=None if obs.mask is None else np.packbits(obs.mask),
            box_xyxy=None if obs.box_xyxy is None
            else np.asarray(obs.box_xyxy, dtype=np.float32),
            det_score=obs.det_score,
            tracker_score=float(obs.tracker_score),
            mask_area_px=int(obs.mask_area_px),
            occluded=bool(obs.occluded),
            obj_id=obs.obj_id,
            source=source,
        )

    def to_observation(self, instance_id: str, height: int,
                       width: int) -> SlotObservation:
        """Back to the deploy type, so the deploy gates can score it."""
        return SlotObservation(
            instance_id=instance_id,
            mask=self.mask(height, width),
            box_xyxy=self.box_xyxy,
            det_score=self.det_score,
            tracker_score=self.tracker_score,
            mask_area_px=self.mask_area_px,
            occluded=self.occluded,
            obj_id=self.obj_id,
        )


@dataclasses.dataclass
class Tracks:
    """The merged result: `[slot][frame] -> SlotFrame`, plus provenance."""

    slot_ids: tuple[str, ...]
    n_frames: int
    height: int
    width: int
    frames: dict[str, list[SlotFrame]]
    stats: dict

    def coverage(self, slot: str) -> float:
        rows = self.frames[slot]
        return sum(f.has_mask for f in rows) / max(1, len(rows))

    def detection_rate(self, slot: str) -> float:
        rows = self.frames[slot]
        return sum(f.detected for f in rows) / max(1, len(rows))

    def source_counts(self, slot: str) -> dict[str, int]:
        counts = {name: 0 for name in SOURCE_NAMES.values()}
        for f in self.frames[slot]:
            counts[SOURCE_NAMES[f.source]] += 1
        return counts


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

class OfflineSam3:
    """Runs the offline passes over one episode using a live `Sam3Source`.

    Takes the `Sam3Source` rather than building one: loading SAM 3 costs tens
    of seconds and an extraction over a directory of episodes must pay it
    once. `run()` opens a fresh session per pass and leaves the source's own
    streaming session untouched.
    """

    def __init__(self, sam3, *, passes=("forward", "reverse"),
                 prune: bool = True, visibility: VisibilityConfig | None = None,
                 progress: bool = True):
        unknown = set(passes) - {"forward", "reverse"}
        if unknown:
            raise ValueError(f"unknown pass(es) {sorted(unknown)}; "
                             f"choose from 'forward', 'reverse'")
        if not passes:
            raise ValueError("at least one pass is required")
        self.sam3 = sam3
        self.passes = tuple(passes)
        self.prune = bool(prune)
        self.progress = bool(progress)
        self.visibility_config = visibility or VisibilityConfig()

    # -- entry point --------------------------------------------------------

    def run(self, episode) -> Tracks:
        import torch

        H, W = episode.height, episode.width
        n = episode.n_frames

        # Preprocess ONCE and stage on the CPU. Each pass needs a fresh
        # session (a session that has already run has a full memory bank), but
        # the resized frames are identical, and at ~6 MB/frame on the GPU a
        # 610-frame video would not fit beside the weights.
        t0 = time.perf_counter()
        pixels = []
        for i, rgb in episode.frames():
            pixels.append(self.sam3.preprocess_frame(rgb, device="cpu"))
            if self.progress and (i + 1) % 100 == 0:
                print(f"  [prep] {i + 1}/{n} frames", flush=True)
        prep_s = time.perf_counter() - t0
        staged_gb = sum(p.numel() * p.element_size() for p in pixels) / 1024 ** 3
        if self.progress:
            print(f"  [prep] {n} frames in {prep_s:.1f} s "
                  f"({staged_gb:.2f} GB staged on CPU)")

        results, timings = {}, {}
        for direction in self.passes:
            t0 = time.perf_counter()
            results[direction] = self._propagate(
                pixels, height=H, width=W, reverse=(direction == "reverse"))
            timings[direction] = time.perf_counter() - t0
            if self.progress:
                print(f"  [{direction}] {n} frames in {timings[direction]:.1f} s "
                      f"({n / max(timings[direction], 1e-9):.1f} fps)")
            # The next pass wants a clean card; the session and its memory
            # bank are the largest torch allocation in this loop.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tracks = self._merge(results, n_frames=n, height=H, width=W)
        self._replay_gates(tracks)
        tracks.stats.update(
            preprocess_s=prep_s, staged_gb=staged_gb,
            pass_seconds={k: round(v, 2) for k, v in timings.items()},
            passes=list(self.passes),
            prune=self.prune,
            memory_horizon=getattr(self.sam3, "memory_horizon", None),
        )
        return tracks

    # -- one pass -----------------------------------------------------------

    def _propagate(self, pixels, *, height: int, width: int, reverse: bool):
        """One full-video pass. Returns (per-frame slot dict, removed ids)."""
        sam3 = self.sam3
        session = sam3.processor.init_video_session(
            inference_device=sam3.device,
            video_storage_device="cpu",     # NOT the default (= inference dev)
            dtype=sam3.dtype)
        sam3.processor.add_text_prompt(session, list(sam3.prompt_to_slot))
        # Frames added by hand have no original size attached, and
        # postprocess_outputs needs one to interpolate the low-res masks up.
        session.video_height, session.video_width = int(height), int(width)
        for i, px in enumerate(pixels):
            session.add_new_frame(px, frame_idx=i)

        n = session.num_frames
        per_frame: list[dict[str, SlotFrame]] = [dict() for _ in range(n)]
        source = SOURCE_REVERSE if reverse else SOURCE_FORWARD
        probe = PresenceProbe(sam3.model, list(sam3.prompt_to_slot),
                              sam3.prompt_to_slot)
        label = "reverse" if reverse else "forward"

        # The loop is driven HERE rather than through
        # `propagate_in_video_iterator`, for one decisive reason: that
        # generator buffers `hotstart_delay` (15) frames before yielding, so
        # the frame it hands you is 15 behind the frame the model just ran.
        # The presence hook fires on the model's schedule, so draining it at
        # yield time would attribute every frame's presence score to a
        # different frame — a silent 15-frame shift in the one signal that
        # says whether the object is there at all.
        #
        # Nothing is lost by driving it directly. The buffer exists only to
        # delay OUTPUT so late hotstart removals can be applied before a
        # frame is emitted, and `_retract` already applies the FINAL removed
        # set to every frame, which strictly dominates. The one piece of
        # bookkeeping the generator does — accumulating `removed_obj_ids`
        # into the session, which `postprocess_outputs` reads — is replicated
        # below.
        order = range(n - 1, -1, -1) if reverse else range(n)
        try:
          import torch
          with torch.inference_mode():      # the generator's own decorator
            for done, frame_idx in enumerate(order, 1):
                raw = sam3.model(inference_session=session,
                                 frame_idx=frame_idx, reverse=reverse)
                presence = probe.take()     # exactly THIS frame's calls
                session.hotstart_removed_obj_ids.update(raw.removed_obj_ids)

                # Same three side-channels the streaming path reads, and for
                # the same reason: postprocess_outputs drops the tracker
                # score, which is half of the S1 visibility signal.
                det = sam3.raw_map(raw, "obj_id_to_score")
                trk = sam3.raw_map(raw, "obj_id_to_tracker_score")
                occ = sam3.raw_map(raw, "obj_id_to_last_occluded")
                res = sam3.processor.postprocess_outputs(session, raw)
                slots = sam3.to_slots(res, det, trk, occ)

                per_frame[frame_idx] = {
                    oid: SlotFrame.from_observation(
                        obs, source, presence.get(oid, float("nan")))
                    for oid, obs in slots.items()}

                if self.prune:
                    self._prune(session, frame_idx, reverse=reverse)
                if self.progress and done % 100 == 0:
                    print(f"  [{label}] {done}/{n} frames", flush=True)
        finally:
            probe.close()

        removed = {int(i) for i in getattr(session, "hotstart_removed_obj_ids", ())}
        if removed:
            logger.info("%s pass: hotstart retracted tracklets %s",
                        "reverse" if reverse else "forward", sorted(removed))
        self._retract(per_frame, removed)

        # Drop the session before the next pass builds one.
        del session
        return per_frame, removed

    def _prune(self, session, frame_idx: int, *, reverse: bool) -> int:
        """R1's prune, mirrored for direction and using the REAL horizon.

        Two corrections over a naive reading:

        * The horizon is `max(num_maskmem, max_object_pointers_in_encoder)`,
          not `num_maskmem`. `_get_object_pointers` reads the same dict up to
          16 frames back with `.get(idx, None)`, so a shorter cutoff loses
          object pointers silently rather than crashing. See
          `Sam3Source.prune`.
        * Reverse propagation reads FORWARD in time
          (`previous_frame_idx = frame_idx + relative_temporal_offset`), so
          the dead half of the dict is above the cursor, not below it.
        """
        per_obj = getattr(session, "output_dict_per_obj", None)
        if not per_obj:
            return 0
        horizon = getattr(self.sam3, "memory_horizon", None) or 16
        freed = 0
        for obj in per_obj.values():
            non_cond = obj.get("non_cond_frame_outputs") if isinstance(obj, dict) else None
            if not non_cond:
                continue
            if reverse:
                dead = [k for k in non_cond
                        if isinstance(k, int) and k > frame_idx + horizon]
            else:
                dead = [k for k in non_cond
                        if isinstance(k, int) and k < frame_idx - horizon]
            for k in dead:
                del non_cond[k]
                freed += 1
        return freed

    @staticmethod
    def _retract(per_frame, removed: set[int]) -> int:
        """Apply the FINAL hotstart removals to every frame of the pass.

        `postprocess_outputs` filters against `hotstart_removed_obj_ids` as it
        stands when it is called, and the iterator only buffers 15 frames — so
        a tracklet retracted late still survives in early frames. With the
        whole pass in hand the retraction is applied everywhere, which is the
        offline-only half of hotstart.

        A retracted slot becomes EMPTY rather than falling back to a
        second-choice tracklet. There is no second choice to fall back to: the
        losing candidates were never postprocessed, and a slot with two live
        instances is already the collision `Sam3Source._pick` warns about.
        The other pass, or the merge, is what fills the hole.
        """
        if not removed:
            return 0
        wiped = 0
        for slots in per_frame:
            for oid, sf in slots.items():
                if sf.obj_id is not None and sf.obj_id in removed:
                    # Presence survives: it is a per-PROMPT score about the
                    # frame, not a property of the tracklet being retracted.
                    # "The concept was visibly there and the only tracklet for
                    # it was spurious" is exactly the diagnosis worth keeping.
                    slots[oid] = SlotFrame(source=SOURCE_NONE,
                                           presence=sf.presence)
                    wiped += 1
        return wiped

    # -- merge --------------------------------------------------------------

    def _merge(self, results: dict, *, n_frames: int, height: int,
               width: int) -> Tracks:
        """One observation per (frame, slot) from up to two passes.

        The rule, in order:

        1.  A frame the DETECTOR fired on beats one carried by memory. This is
            the S1 distinction and it is the whole reason `det_score is None`
            is kept as a value rather than collapsed to 0.0: a re-detected
            mask shows the object, a propagated one is a guess.
        2.  Between two detected (or two propagated) candidates, the higher
            score wins — detection score first, tracker score as the
            tiebreak.
        3.  A mask of any kind beats no mask.

        Note what is NOT done: the two masks are never averaged or unioned.
        They come from independently seeded tracklets and a union would
        manufacture a silhouette neither model produced, which is exactly the
        kind of fabricated evidence the orientation stage cannot survive.
        """
        forward = results.get("forward", (None, set()))[0]
        reverse = results.get("reverse", (None, set()))[0]
        slot_ids = tuple(self.sam3.slot_ids)

        frames = {oid: [] for oid in slot_ids}
        for i in range(n_frames):
            f_slots = forward[i] if forward is not None else {}
            r_slots = reverse[i] if reverse is not None else {}
            for oid in slot_ids:
                a = f_slots.get(oid)
                b = r_slots.get(oid)
                frames[oid].append(self._pick(a, b))

        return Tracks(slot_ids=slot_ids, n_frames=n_frames, height=height,
                      width=width, frames=frames,
                      stats={"hotstart_removed": {
                          k: sorted(v[1]) for k, v in results.items()}})

    @staticmethod
    def _pick(a: SlotFrame | None, b: SlotFrame | None) -> SlotFrame:
        a_ok = a is not None and a.has_mask
        b_ok = b is not None and b.has_mask
        # Presence is per-prompt and per-frame, so both passes should agree on
        # it up to detector nondeterminism. Keep whichever is a real number,
        # INCLUDING when no mask survives anywhere — a frame where the concept
        # is clearly present but nothing was tracked is the single most useful
        # row in the file for diagnosing an empty roster slot.
        pres = next((f.presence for f in (a, b)
                     if f is not None and f.presence == f.presence),
                    float("nan"))
        if not a_ok and not b_ok:
            return SlotFrame(source=SOURCE_NONE, presence=pres)
        if a_ok and not b_ok:
            return dataclasses.replace(a, source=SOURCE_FORWARD, presence=pres)
        if b_ok and not a_ok:
            return dataclasses.replace(b, source=SOURCE_REVERSE, presence=pres)

        def rank(sf: SlotFrame):
            return (sf.detected,
                    sf.det_score if sf.det_score is not None else -1.0,
                    sf.tracker_score)

        if rank(a) >= rank(b):
            return dataclasses.replace(a, source=SOURCE_BOTH_FORWARD)
        return dataclasses.replace(b, source=SOURCE_BOTH_REVERSE)

    # -- gate replay --------------------------------------------------------

    def _replay_gates(self, tracks: Tracks) -> None:
        """Score the merged track with the deploy visibility gates.

        Recorded, never enforced. `VisibilityGate` is stateful — it carries a
        decaying running maximum of each slot's mask area — so it has to see
        the merged sequence in frame order, once, exactly as the online loop
        would. What that buys is the experiment's real question: run
        orientation on every mask, then use `crop_usable` to split the results
        into "what deploy would have used" and "what deploy would have
        discarded", and check whether the discarded ones were actually bad.
        """
        for oid in tracks.slot_ids:
            gate = VisibilityGate(self.visibility_config)
            for sf in tracks.frames[oid]:
                obs = sf.to_observation(oid, tracks.height, tracks.width)
                vis = gate.update(obs)
                sf.mask_usable = vis.mask_usable
                sf.crop_usable = vis.crop_usable
                sf.gate_reason = vis.reason
