"""Single-episode self-contained HTML report.

Everything is embedded: human egocentric frames (deduped by camera frame),
G1 renders for the deployment-faithful anchor mode, per-tick hand renders,
error curves for BOTH anchor modes, the s004 filter timeline and
sub-episode boundaries. Frames are WebP-encoded, downscaled to ~360 px wide
to keep the file a few MB.
"""

import base64
import io as _io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_W = 360
WEBP_Q = 55


def encode_webp(rgb_or_jpeg, is_jpeg=False, max_w=IMG_W, quality=WEBP_Q):
    from PIL import Image
    img = (Image.open(_io.BytesIO(rgb_or_jpeg)) if is_jpeg
           else Image.fromarray(np.asarray(rgb_or_jpeg)))
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)),
                         Image.LANCZOS)
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _nanlist(a, nd=3):
    """float array -> list with None for NaN (JSON has no NaN)."""
    a = np.asarray(a, dtype=float)
    out = np.round(a, nd).tolist()
    return [None if (v is None or v != v) else v for v in out]


def _human_slots(rec):
    """Dedup per-tick human frames by camera key -> (slots b64, slot_of)."""
    slot_of, slots, seen = [], [], {}
    for t in range(rec.T):
        key = rec.cam_key(t)
        if key not in seen:
            img = rec.tick_image(t)
            seen[key] = len(slots)
            slots.append(encode_webp(img, is_jpeg=isinstance(img, bytes)))
        slot_of.append(seen[key])
    return slots, slot_of


def build_data(rec, result):
    """Assemble the JSON payload the template consumes."""
    slots, slot_of = _human_slots(rec)
    render_mode = result["render_mode"]
    chunk = result["modes"][render_mode]["chunk"]

    hands = {}
    for side in ("left", "right"):
        h = result["hands"][side]
        hands[side] = {
            "frames": [None if f is None else encode_webp(f)
                       for f in h["frames"]],
            "err_deg": _nanlist(h["err_deg"], 2),
            "blocked": h["blocked"].astype(int).tolist(),
            "contact": h["contact"].astype(int).tolist(),
            "pair_counts": h["pair_counts"],
        }

    return {
        "episode": rec.name,
        "source": rec.source,
        "fps": rec.fps,
        "H": rec.horizon,
        "T": rec.T,
        "config_hash": rec.config_hash,
        "b_mode": rec.b_mode,
        "ticks_kept": rec.ticks_kept,
        "ticks_total": rec.ticks_total,
        "tick_orig": rec.tick_orig.tolist(),
        "subeps": [[se.start, se.end, bool(se.real_end)] for se in rec.subeps],
        "subep_of": result["subep_of"].tolist(),
        "chunk": [None if c < 0 else int(c) for c in chunk],
        "human": {"slots": slots, "slot_of": slot_of},
        "g1": {"mode": render_mode,
               "frames": [None if f is None else encode_webp(f)
                          for f in result["g1_frames"]]},
        "hands": hands,
        "arm": {m: {s: {"pos": _nanlist(result["modes"][m][s]["pos_cm"]),
                        "ori": _nanlist(result["modes"][m][s]["ori_deg"])}
                    for s in ("left", "right")}
                for m in result["modes"]},
        "masks": {k: v.astype(int).tolist() for k, v in rec.masks.items()},
        "filter_stats": {k: int(v) for k, v in rec.filter_stats.items()},
        "S": np.round(rec.S, 4).tolist(),
        "B": {s: np.round(np.asarray(rec.B[s])[:3, :3], 4).tolist()
              for s in rec.B},
        "meta": {"source_file": str(rec.meta.get("source_file", "?")),
                 "source_episode": str(rec.meta.get("source_episode", "?"))},
    }


def write_report(out_path, rec, result):
    data = build_data(rec, result)
    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as fh:
        html = fh.read()
    token = "/*__DATA_JSON__*/"
    assert token in html, "template is missing the data placeholder"
    html = html.replace(token, json.dumps(data), 1)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    return out_path
