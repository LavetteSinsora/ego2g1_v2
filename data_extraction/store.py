"""The extraction file: one HDF5 per episode, everything a dashboard needs.

Layout — F frames, one group per roster slot:

    /                         attrs: schema, episode, prompts, config, stats
    frame_index               (F,)  int64
    timestamp_ns              (F,)  int64   camera clock, from the raw episode
    K                         (3,3) float64 intrinsics of the eye used
    objects                   (O,)  str     roster slot ids, in roster order
    prompts                   (O,)  str     the normalised SAM 3 prompt each used

    depth_mm                  (F,H,W) uint16   OPTIONAL, --save-depth-map only

    obj/<slot>/mask           (F,H,W) uint8  0/1, gzip, chunked one frame deep
    obj/<slot>/box_xyxy       (F,4)   float32  NaN where no mask
    obj/<slot>/det_score      (F,)    float32  NaN == not re-detected (S1!)
    obj/<slot>/presence       (F,)    float32  is the CONCEPT in this frame?
    obj/<slot>/tracker_score  (F,)    float32
    obj/<slot>/mask_area_px   (F,)    int32
    obj/<slot>/occluded       (F,)    bool
    obj/<slot>/source         (F,)    uint8    which pass won (SOURCE_NAMES)
    obj/<slot>/mask_usable    (F,)    bool     deploy gate, recorded not applied
    obj/<slot>/crop_usable    (F,)    bool     deploy gate, recorded not applied
    obj/<slot>/gate_reason    (F,)    str      first failing test
    obj/<slot>/azimuth_deg    (F,)    float32  NaN where no orientation
    obj/<slot>/elevation_deg  (F,)    float32
    obj/<slot>/roll_deg       (F,)    float32
    obj/<slot>/R_cam          (F,3,3) float32  NaN where no orientation
    obj/<slot>/orient_skip    (F,)    uint8    why not (SKIP_NAMES)
    obj/<slot>/alpha          (F,)    int8     symmetry order; -1 = not measured
    obj/<slot>/depth_m        (F,)    float32  median depth over the mask
    obj/<slot>/point_cam      (F,3)   float32  raw-left camera frame, metres
    obj/<slot>/depth_px       (F,)    int32    valid depth pixels behind depth_m

`presence` and `alpha` each have their own "absent" convention, and they are
NOT the same as the neighbouring fields':

    presence  NaN only when the probe could not read it. It is a PER-PROMPT
              score and survives a frame with no mask — that is the case it
              exists for ("the concept is visibly there, nothing tracked it").
    alpha     -1 means not measured; 0 is a REAL answer meaning "no confident
              symmetry call". Collapsing the two loses the distinction between
              a fit that was never run and one that ran and declined.

Three choices worth stating:

MASKS ARE STORED, NOT RLE'D.
    (610, 480, 640) uint8 is 187 MB raw per slot and gzips to a couple of MB,
    because a binary mask is almost all runs. gzip with a one-frame chunk
    keeps a dashboard's `mask[i]` a single chunk read, which RLE in an
    attribute would not. h5py does this transparently; a custom encoding
    would need a decoder in every consumer.

NaN IS THE MISSING VALUE, AND `det_score` IS WHY.
    `det_score = NaN` means the DETECTOR did not re-find the object on that
    frame — the mask came from memory propagation. It is not a missing
    measurement, it is the S1 signal, and writing 0.0 there would erase the
    single most useful distinction in the file. Same for the orientation
    fields, which is what `orient_skip` disambiguates.

METADATA IS DUPLICATED INTO A SIDECAR JSON.
    Everything in the root attrs is also written to `<out>.meta.json`. A
    dashboard, a notebook or a shell script can then see what a run was
    without opening HDF5, and a run whose extraction crashed still leaves a
    readable record of what it was trying to do.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["write_extraction", "SCHEMA"]

SCHEMA = "ego2g1.data_extraction/1"

_STR = None  # h5py.string_dtype(), resolved lazily


def _strings(values):
    import h5py

    global _STR
    if _STR is None:
        _STR = h5py.string_dtype(encoding="utf-8")
    return np.array([str(v) for v in values], dtype=object), _STR


def write_extraction(out_path, *, episode, tracks, orientation, meta: dict,
                     depth=None,
                     compression: str = "gzip", compression_opts: int = 4):
    """Write one episode's extraction. Returns the path written."""
    import h5py

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    F, H, W = tracks.n_frames, tracks.height, tracks.width
    slots = tracks.slot_ids

    # HDF5 attributes take scalars and strings, so every structured piece of
    # metadata is JSON. The sidecar gets the SAME dict unencoded — encoding it
    # twice is how a meta file ends up full of quoted strings that read back
    # as text instead of objects.
    plain = {
        "schema": SCHEMA,
        "episode_path": str(episode.path),
        "episode_name": episode.name,
        "eye": episode.eye,
        "n_frames": F,
        "height": H,
        "width": W,
        "task_instruction": episode.task_instruction,
        "recorded_anchor_object": episode.anchor_object,
    }
    root_attrs = {**plain,
                  **{k: json.dumps(v, default=str) for k, v in meta.items()}}

    with h5py.File(out_path, "w") as f:
        for k, v in root_attrs.items():
            f.attrs[k] = v

        f.create_dataset("frame_index", data=np.arange(F, dtype=np.int64))
        f.create_dataset("timestamp_ns",
                         data=np.asarray(episode.timestamps_ns[:F],
                                         dtype=np.int64))
        f.create_dataset("K", data=np.asarray(episode.K, dtype=np.float64))
        names, sdt = _strings(slots)
        f.create_dataset("objects", data=names, dtype=sdt)
        prompts = {v: k for k, v in meta.get("prompt_to_slot", {}).items()}
        pnames, _ = _strings([prompts.get(s, "") for s in slots])
        f.create_dataset("prompts", data=pnames, dtype=sdt)

        for slot in slots:
            g = f.create_group(f"obj/{slot}")
            rows = tracks.frames[slot]

            masks = g.create_dataset(
                "mask", shape=(F, H, W), dtype=np.uint8,
                chunks=(1, H, W), compression=compression,
                compression_opts=compression_opts)
            box = np.full((F, 4), np.nan, dtype=np.float32)
            det = np.full(F, np.nan, dtype=np.float32)
            trk = np.zeros(F, dtype=np.float32)
            area = np.zeros(F, dtype=np.int32)
            occ = np.zeros(F, dtype=bool)
            src = np.zeros(F, dtype=np.uint8)
            m_use = np.zeros(F, dtype=bool)
            c_use = np.zeros(F, dtype=bool)
            pres = np.full(F, np.nan, dtype=np.float32)
            reasons = []

            for i, sf in enumerate(rows):
                m = sf.mask(H, W)
                if m is not None:
                    masks[i] = m.astype(np.uint8)
                if sf.box_xyxy is not None:
                    box[i] = sf.box_xyxy
                if sf.det_score is not None:
                    det[i] = sf.det_score
                trk[i] = sf.tracker_score
                area[i] = sf.mask_area_px
                occ[i] = sf.occluded
                src[i] = sf.source
                m_use[i] = sf.mask_usable
                c_use[i] = sf.crop_usable
                pres[i] = sf.presence
                reasons.append(sf.gate_reason)

            g.create_dataset("box_xyxy", data=box)
            g.create_dataset("det_score", data=det)
            g.create_dataset("presence", data=pres)
            g.create_dataset("tracker_score", data=trk)
            g.create_dataset("mask_area_px", data=area)
            g.create_dataset("occluded", data=occ)
            g.create_dataset("source", data=src)
            g.create_dataset("mask_usable", data=m_use)
            g.create_dataset("crop_usable", data=c_use)
            rdata, _ = _strings(reasons)
            g.create_dataset("gate_reason", data=rdata, dtype=sdt)

            g.create_dataset("azimuth_deg", data=orientation.azimuth_deg[slot])
            g.create_dataset("elevation_deg", data=orientation.elevation_deg[slot])
            g.create_dataset("roll_deg", data=orientation.roll_deg[slot])
            g.create_dataset("R_cam", data=orientation.R_cam[slot])
            g.create_dataset("orient_skip", data=orientation.skip[slot])
            if orientation.alpha:
                g.create_dataset("alpha", data=orientation.alpha[slot])

            if depth is not None:
                g.create_dataset("depth_m", data=depth.depth_m[slot])
                g.create_dataset("point_cam", data=depth.point_cam[slot])
                g.create_dataset("depth_px", data=depth.depth_px[slot])
                g.attrs["depth_rate"] = float(depth.rate(slot))

            g.attrs["coverage"] = float(tracks.coverage(slot))
            g.attrs["detection_rate"] = float(tracks.detection_rate(slot))
            g.attrs["orientation_rate"] = float(orientation.rate(slot))
            g.attrs["source_counts"] = json.dumps(tracks.source_counts(slot))
            sym = (orientation.stats.get("symmetry") or {}).get(slot)
            if sym:
                g.attrs["symmetry"] = json.dumps(sym)

        if depth is not None and depth.depth_map is not None:
            # uint16 MILLIMETRES, not float32 metres: a float depth map is
            # 750 MB for one episode and gzips badly (the mantissa is noise),
            # while millimetre integers are 2 bytes, quantise well below SGBM's
            # own error, and compress. 0 means "no valid depth", which is what
            # `estimate` already returns for a failed match.
            f.create_dataset("depth_mm", data=depth.depth_map, dtype=np.uint16,
                             chunks=(1, H, W), compression=compression,
                             compression_opts=compression_opts)

    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    sidecar.write_text(json.dumps(
        {**plain, **meta,
         "per_object": {
             s: {"coverage": tracks.coverage(s),
                 "detection_rate": tracks.detection_rate(s),
                 "orientation_rate": orientation.rate(s),
                 "depth_rate": (depth.rate(s) if depth is not None else None),
                 "symmetry": (orientation.stats.get("symmetry") or {}).get(s),
                 "source_counts": tracks.source_counts(s)}
             for s in slots}},
        indent=2, default=str))
    return out_path
