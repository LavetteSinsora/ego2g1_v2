"""Extraction -> a self-contained HTML player with mask, centroid and axis overlays.

    uv run python -m data_extraction.dashboard \
        --extraction data_extraction/out/episode_2.h5

Writes `<name>_dashboard.html` next to it: one file, no server, no network.
Needs only numpy/h5py/Pillow — NOT the perception-v2 group, so it runs on a
laptop against files the PPU box produced.

WHAT IT DRAWS, AND WHY EACH IS THE THING IT IS

The centre pixel — a PLAIN UNWEIGHTED MEAN.
    `SlotObservation.centroid_uv` is `xs.mean(), ys.mean()` over the mask's
    non-zero pixels. Not the bbox centre, not a median, not a distance
    transform: the centre of area of the VISIBLE region. This module builds a
    real `SlotObservation` and calls that method rather than reimplementing
    it, so the dot on screen is the pixel the deploy loop back-projects.

    The bbox centre is drawn too, as a hollow marker. The gap between the two
    IS the occlusion bias `latch.py` warns about — under partial occlusion the
    mean is pulled toward whatever is still uncovered, and that pull moves as
    the hand rotates. One is a measurement, the other is a reference; seeing
    them separate is how you tell an occluded frame from a moving object.

The rotation — COLUMNS of R, drawn from the centroid.
    Orient Anything V2 emits azimuth (360 bins), elevation (180, offset -90)
    and roll (360, offset -180). `angles_to_matrix` composes them as
    `R = Rz(roll) @ Rx(elevation) @ Ry(azimuth)` about CAMERA axes (OpenCV:
    X right, Y down, Z forward).

    R is `R_camera<-object`: its COLUMNS are the object's canonical basis
    vectors expressed in camera coordinates. Two independent confirmations in
    the deploy code — `compose_relational_rotation` reads `R_model[:, 1]` as a
    camera-frame direction and returns `stack([x, y, z], axis=1)`, and the
    snapshot uses R as the rotation block of `T_camera_object`. So drawing
    column i from the centroid draws axis i of the object's own frame.

    ...AND THE CONVENTION IS UNVALIDATED. `angles_to_matrix`'s own docstring
    flags the axis assignment and all three signs as a reasoned default that
    has never been measured, and plan Q6 asks whether it even matches the
    frame the training labels used. This page is the instrument for settling
    that: transpose and per-angle sign flips are LIVE controls. Flip until the
    triad stays glued to the object as it turns, and you have measured the
    convention. The page prints the resulting settings for pasting into
    `perception_v2.yaml`'s `convention:` block.

    Axes are ORTHOGRAPHIC — endpoint = centroid + L * (d_x, d_y) for the
    camera-frame unit axis d. This extraction is monocular, so there is no
    depth and no true perspective projection. Direction and foreshortening are
    right (an axis pointing at the camera collapses to a dot); apparent length
    is not metric. `_project_axis` is where a real depth + K would slot in.

    Red/green/blue for X/Y/Z is the domain convention and is kept — but red vs
    green is the single worst pair for deuteranopia, so every axis is ALSO
    labelled with its letter at the tip. Identity is never colour alone.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["build_payload", "render_html", "main"]

# Categorical slots 1-3, light / dark (references/palette.md). Validated
# all-pairs in both modes: worst CVD dE 9.2 light / 9.4 dark, normal-vision
# 24.0 / 20.9. Aqua is below 3:1 on the light surface, so the relief rule
# applies — every object carries a visible text label, on canvas and in the
# panel, never colour alone.
OBJECT_COLORS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
OBJECT_COLORS_DARK = ["#3987e5", "#d95926", "#199e70"]


def _fit(values, n):
    """Cycle-free slot assignment: colour follows the entity, never its rank.

    Past the palette's validated depth there is no 4th hue to invent, so extra
    slots reuse the last one and lean entirely on the text label. A roster
    that big is outside what this pipeline is for anyway (three objects).
    """
    return [values[min(i, len(values) - 1)] for i in range(n)]


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------

def _rle(mask: np.ndarray) -> list[int]:
    """Row-major run lengths, starting with a background run (possibly 0).

    A binary mask is almost all runs, so this is ~2 KB per frame per object
    against 77 KB raw — the difference between a 14 MB page and a 150 MB one.
    The browser decodes it back to a flat Uint8Array in one pass.
    """
    flat = mask.reshape(-1).astype(np.uint8)
    if flat.size == 0:
        return []
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    return runs if flat[0] == 0 else [0] + runs


def _jpeg_uri(rgb: np.ndarray, scale: float, quality: int) -> str:
    from PIL import Image

    img = Image.fromarray(rgb)
    if scale != 1.0:
        img = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))),
                         Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_payload(extraction, episode, *, scale: float = 0.5,
                  quality: int = 78, max_frames: int | None = None,
                  progress: bool = True) -> dict:
    """Everything the page needs, as one JSON-able dict."""
    import h5py

    from ego2g1.deploy.perception.v2.sam3_source import SlotObservation

    with h5py.File(extraction, "r") as f:
        slots = [s.decode() if isinstance(s, bytes) else str(s)
                 for s in f["objects"][:]]
        prompts = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in f["prompts"][:]]
        H, W = int(f.attrs["height"]), int(f.attrs["width"])
        F = int(f.attrs["n_frames"])
        if max_frames is not None:
            F = min(F, max_frames)
        if F > episode.n_frames:
            raise ValueError(
                f"extraction has {F} frames but {episode.name} has "
                f"{episode.n_frames}. Wrong episode for this extraction?")

        meta = {k: v for k, v in f.attrs.items() if isinstance(v, (str, bytes))}
        meta = {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in meta.items()}

        # Mask display resolution. The CENTROID is computed at FULL resolution
        # and scaled afterwards — downscaling first would move it, and the
        # whole point is that this dot is the deploy loop's dot.
        step = max(1, int(round(1 / scale))) if scale < 1.0 else 1
        mh, mw = len(range(0, H, step)), len(range(0, W, step))
        dw, dh = max(1, round(W * scale)), max(1, round(H * scale))

        objects = []
        for slot, prompt in zip(slots, prompts):
            g = f[f"obj/{slot}"]
            masks = g["mask"]
            det = g["det_score"][:F]
            trk = g["tracker_score"][:F]
            area = g["mask_area_px"][:F]
            box = g["box_xyxy"][:F]
            src = g["source"][:F]
            mu = g["mask_usable"][:F]
            cu = g["crop_usable"][:F]
            reason = [r.decode() if isinstance(r, bytes) else str(r)
                      for r in g["gate_reason"][:F]]
            az, el, ro = (g["azimuth_deg"][:F], g["elevation_deg"][:F],
                          g["roll_deg"][:F])
            R = g["R_cam"][:F]
            skip = g["orient_skip"][:F]
            # Added later than the rest of the schema, so read defensively —
            # an extraction made before they existed should still open.
            pres = g["presence"][:F] if "presence" in g else np.full(F, np.nan)
            alpha = g["alpha"][:F] if "alpha" in g else np.full(F, -1, np.int8)
            dep = g["depth_m"][:F] if "depth_m" in g else np.full(F, np.nan)
            pt = (g["point_cam"][:F] if "point_cam" in g
                  else np.full((F, 3), np.nan))

            cx, cy, rles = [], [], []
            for i in range(F):
                m = masks[i].astype(bool)
                if not m.any():
                    cx.append(None)
                    cy.append(None)
                    rles.append(None)
                    continue
                # The real thing, not a copy of it.
                obs = SlotObservation(
                    instance_id=slot, mask=m, box_xyxy=None, det_score=None,
                    tracker_score=0.0, mask_area_px=int(m.sum()), occluded=False)
                uv = obs.centroid_uv()
                cx.append(round(float(uv[0]) * scale, 2))
                cy.append(round(float(uv[1]) * scale, 2))
                rles.append(_rle(m[::step, ::step]))
                if progress and (i + 1) % 200 == 0:
                    print(f"  [{slot}] {i + 1}/{F}", flush=True)

            objects.append({
                "id": slot, "prompt": prompt,
                "cx": cx, "cy": cy,
                "box": [None if not np.isfinite(b).all()
                        else [round(float(v) * scale, 2) for v in b]
                        for b in box],
                "det": [None if not np.isfinite(v) else round(float(v), 3)
                        for v in det],
                "trk": [round(float(v), 3) for v in trk],
                "area": [int(v) for v in area],
                "src": [int(v) for v in src],
                "mu": [bool(v) for v in mu],
                "cu": [bool(v) for v in cu],
                "reason": reason,
                "az": [None if not np.isfinite(v) else round(float(v), 1) for v in az],
                "el": [None if not np.isfinite(v) else round(float(v), 1) for v in el],
                "ro": [None if not np.isfinite(v) else round(float(v), 1) for v in ro],
                "R": [None if not np.isfinite(m).all()
                      else [round(float(v), 5) for v in m.reshape(-1)]
                      for m in R],
                "skip": [int(v) for v in skip],
                "pres": [None if not np.isfinite(v) else round(float(v), 4)
                         for v in pres],
                "alpha": [int(v) for v in alpha],
                "depth": [None if not np.isfinite(v) else round(float(v), 4)
                          for v in dep],
                "pt": [None if not np.isfinite(p).all()
                       else [round(float(v), 4) for v in p] for p in pt],
                "rle": rles,
            })

    if progress:
        print("  [frames] encoding JPEGs...", flush=True)
    frames = [_jpeg_uri(episode.frame(i), scale, quality) for i in range(F)]

    return {
        "meta": meta,
        "episode": episode.name,
        "n": F, "W": dw, "H": dh, "mw": mw, "mh": mh,
        "fullW": W, "fullH": H,
        "timestamp_ns": [int(v) for v in episode.timestamps_ns[:F]],
        "colors_light": _fit(OBJECT_COLORS_LIGHT, len(objects)),
        "colors_dark": _fit(OBJECT_COLORS_DARK, len(objects)),
        "objects": objects,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
*,*::before,*::after{box-sizing:border-box}
:root{
  color-scheme:light dark;
  --surface-0:#ffffff; --surface-1:#fcfcfb; --surface-2:#f0efec;
  --line:#d9d8d3;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#82817c;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --ord-0:#e6e5e1; --ord-1:#86b6ef; --ord-2:#3987e5; --ord-3:#1c5cab;
  --accent:#2a78d6;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#242423;
  --line:#3a3a37;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --ord-0:#2e2e2c; --ord-1:#86b6ef; --ord-2:#3987e5; --ord-3:#1c5cab;
  --accent:#3987e5;
}}
:root[data-theme=dark]{
  --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#242423;
  --line:#3a3a37;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  --ord-0:#2e2e2c; --ord-1:#86b6ef; --ord-2:#3987e5; --ord-3:#1c5cab;
  --accent:#3987e5;
}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1{font-size:16px;margin:0;font-weight:600}
h2{font-size:12px;margin:0 0 8px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;color:var(--text-muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums}
header{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:baseline;
  padding:14px 20px;border-bottom:1px solid var(--line);background:var(--surface-1)}
header .sub{color:var(--text-secondary);font-size:12.5px}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:18px;
  padding:18px 20px;max-width:1500px;margin:0 auto;align-items:start}
@media (max-width:1000px){.wrap{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--surface-1);border:1px solid var(--line);
  border-radius:10px;padding:14px}
.stage{background:#000;border-radius:8px;overflow:hidden;line-height:0;
  position:relative}
canvas#view{width:100%;height:auto;display:block;image-rendering:auto}
.transport{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
button{font:inherit;background:var(--surface-2);color:var(--text-primary);
  border:1px solid var(--line);border-radius:7px;padding:5px 11px;cursor:pointer}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);color:#fff;border-color:transparent;
  min-width:74px;font-weight:600}
input[type=range]{accent-color:var(--accent)}
#scrub{flex:1;min-width:160px}
label.chk{display:flex;gap:7px;align-items:center;cursor:pointer;
  font-size:12.5px;color:var(--text-secondary);white-space:nowrap}
.controls{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:10px;
  padding-top:10px;border-top:1px solid var(--line)}
.strips{margin-top:16px}
.strip-row{display:grid;grid-template-columns:132px minmax(0,1fr);
  gap:10px;align-items:center;margin-bottom:5px}
.chip{display:inline-block;width:9px;height:9px;border-radius:2px;
  margin-right:6px;vertical-align:baseline}
/* Strip rows only. The "which pass" mode fills the bars with categorical
   hues, and a solid square chip beside them would read as another data
   block in the same encoding. A vertical rule is unmistakably a row
   marker, so object identity and per-frame state never share a mark. */
.strip-row .chip{width:4px;height:13px;border-radius:1px;vertical-align:-2px}
.strip-label{font-size:12px;color:var(--text-secondary);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
canvas.strip{width:100%;height:19px;display:block;border-radius:3px;cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:5px 14px;margin-top:9px;font-size:12px;
  color:var(--text-secondary)}
.legend span.sw{display:inline-block;width:11px;height:11px;border-radius:2px;
  margin-right:5px;vertical-align:-1px}
table.readout{width:100%;border-collapse:collapse;font-size:12.5px}
table.readout th{text-align:left;font-weight:600;color:var(--text-muted);
  font-size:11px;text-transform:uppercase;letter-spacing:.05em;
  padding:4px 6px 4px 0}
table.readout td{padding:3px 6px 3px 0;border-top:1px solid var(--line);
  color:var(--text-secondary)}
table.readout td.k{color:var(--text-muted);white-space:nowrap}
table.readout td.v{color:var(--text-primary);text-align:right}
.obj-block{margin-bottom:14px;padding-bottom:2px}
.obj-title{font-weight:600;font-size:13px;margin-bottom:5px;
  display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.pill{font-size:10.5px;padding:1px 7px;border-radius:99px;
  border:1px solid var(--line);color:var(--text-secondary);white-space:nowrap}
.pill.ok{border-color:transparent;background:var(--ord-3);color:#fff}
.pill.warn{border-color:transparent;background:var(--ord-1);color:#0b0b0b}
.pill.off{opacity:.65}
details{margin-top:10px}
summary{cursor:pointer;font-size:12.5px;color:var(--text-secondary)}
pre.conv{background:var(--surface-2);border:1px solid var(--line);
  border-radius:6px;padding:9px;font-size:11.5px;overflow-x:auto;margin:8px 0 0}
.note{font-size:12px;color:var(--text-muted);margin-top:9px;line-height:1.45}
.axisctl{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;
  align-items:center;font-size:12.5px;color:var(--text-secondary)}
.tblwrap{max-height:340px;overflow:auto}
kbd{font:11px ui-monospace,monospace;background:var(--surface-2);
  border:1px solid var(--line);border-radius:4px;padding:1px 5px}
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <span class="sub mono" id="hdr-sub"></span>
  <span style="flex:1"></span>
  <button id="theme">Theme</button>
</header>

<div class="wrap">
  <div>
    <div class="card">
      <div class="stage"><canvas id="view"></canvas></div>

      <div class="transport">
        <button class="primary" id="play">Play</button>
        <button id="prev">&#8592;</button>
        <button id="next">&#8594;</button>
        <input type="range" id="scrub" min="0" value="0" step="1">
        <span class="mono" id="fcount" style="min-width:104px;text-align:right"></span>
        <label class="chk">speed
          <input type="range" id="speed" min="1" max="60" value="20" style="width:78px">
          <span class="mono" id="speedv" style="width:34px"></span>
        </label>
        <label class="chk"><input type="checkbox" id="loop" checked> loop</label>
      </div>

      <div class="controls">
        <label class="chk"><input type="checkbox" id="t-mask" checked> mask</label>
        <label class="chk"><input type="checkbox" id="t-fill"> fill mask</label>
        <label class="chk"><input type="checkbox" id="t-box" checked> bbox</label>
        <label class="chk"><input type="checkbox" id="t-cent" checked> centroid (mean)</label>
        <label class="chk"><input type="checkbox" id="t-bcent" checked> bbox centre</label>
        <label class="chk"><input type="checkbox" id="t-axes" checked> rotation axes</label>
        <label class="chk"><input type="checkbox" id="t-label" checked> labels</label>
        <label class="chk"><input type="checkbox" id="t-gate"> only crop_usable</label>
        <label class="chk">axis length
          <input type="range" id="axlen" min="20" max="220" value="100" style="width:78px">
        </label>
      </div>

      <div class="strips">
        <h2 id="strip-title">Per-frame state</h2>
        <div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:9px">
          <label class="chk"><input type="radio" name="stripmode" value="quality" checked> gate quality</label>
          <label class="chk"><input type="radio" name="stripmode" value="source"> which SAM 3 pass</label>
          <label class="chk"><input type="radio" name="stripmode" value="presence"> presence</label>
        </div>
        <div id="strip-rows"></div>
        <div class="legend" id="strip-legend"></div>
        <div class="note" id="strip-note"></div>
      </div>
    </div>
  </div>

  <div>
    <div class="card">
      <h2>Frame</h2>
      <div id="panel"></div>
    </div>

    <div class="card" style="margin-top:14px">
      <h2>Rotation convention</h2>
      <div class="note" style="margin-top:0">
        Columns of <span class="mono">R</span> are the object's axes in camera
        coordinates (<span class="mono">R&nbsp;=&nbsp;R_camera&#8592;object</span>).
        The axis assignment and signs below are <strong>unvalidated defaults</strong>
        &mdash; flip them until the triad stays glued to the object as it turns.
      </div>
      <div class="axisctl" style="margin-top:10px">
        <label class="chk"><input type="checkbox" id="c-transpose"> transpose R</label>
        <span class="mono" style="font-size:11.5px;color:var(--text-muted)">draw rows instead of columns</span>
        <label class="chk"><input type="checkbox" id="c-az"> flip azimuth</label>
        <span class="mono" style="font-size:11.5px;color:var(--text-muted)">Ry sign</span>
        <label class="chk"><input type="checkbox" id="c-el"> flip elevation</label>
        <span class="mono" style="font-size:11.5px;color:var(--text-muted)">Rx sign</span>
        <label class="chk"><input type="checkbox" id="c-ro"> flip roll</label>
        <span class="mono" style="font-size:11.5px;color:var(--text-muted)">Rz sign</span>
      </div>
      <pre class="conv mono" id="conv-out"></pre>
      <div class="note" id="conv-check"></div>
    </div>

    <div class="card" style="margin-top:14px">
      <h2>Episode</h2>
      <div class="tblwrap"><table class="readout"><tbody id="meta-body"></tbody></table></div>
      <div class="note">
        <kbd>space</kbd> play &middot; <kbd>&#8592;</kbd><kbd>&#8594;</kbd> step
        &middot; <kbd>shift</kbd>+arrow &times;10 &middot; <kbd>home</kbd>/<kbd>end</kbd>
      </div>
    </div>
  </div>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const D = JSON.parse(document.getElementById("payload").textContent);
const N = D.n, OBJ = D.objects, NO = OBJ.length;

/* ---------------------------------------------------------------- theme */
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function isDark(){
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t === "dark";
  return matchMedia("(prefers-color-scheme: dark)").matches;
}
let COLORS = [];
function refreshColors(){ COLORS = isDark() ? D.colors_dark : D.colors_light; }
document.getElementById("theme").onclick = () => {
  document.documentElement.setAttribute("data-theme", isDark() ? "light" : "dark");
  refreshColors(); buildStrips(); draw();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  refreshColors(); buildStrips(); draw();
});
refreshColors();

/* ------------------------------------------------- rotation, in the browser
   A faithful port of orientation_v2._rot / _angles_to_matrix. It exists so
   the sign toggles are LIVE: R is also shipped from the file, and at load we
   recompute it under the default convention and compare. A mismatch means
   this port drifted from the Python and every axis on screen is suspect, so
   it is reported rather than swallowed.                                     */
const AXPAIR = [[1,2],[2,0],[0,1]];
function rot(axis, deg){
  const c = Math.cos(deg*Math.PI/180), s = Math.sin(deg*Math.PI/180);
  const R = [1,0,0, 0,1,0, 0,0,1];
  const [a,b] = AXPAIR[axis];
  R[a*3+a] = c; R[b*3+b] = c; R[a*3+b] = -s; R[b*3+a] = s;
  return R;
}
function mul(A,B){
  const C = new Array(9).fill(0);
  for (let i=0;i<3;i++) for (let j=0;j<3;j++){
    let s=0; for (let k=0;k<3;k++) s += A[i*3+k]*B[k*3+j];
    C[i*3+j]=s;
  }
  return C;
}
const CONV = {azAxis:1, azSign:1, elAxis:0, elSign:1, roAxis:2, roSign:1};
function anglesToMatrix(az, el, ro, conv){
  return mul(rot(conv.roAxis, conv.roSign*ro),
             mul(rot(conv.elAxis, conv.elSign*el), rot(conv.azAxis, conv.azSign*az)));
}
function currentConv(){
  return {azAxis:1, azSign: ui("c-az") ? -1 : 1,
          elAxis:0, elSign: ui("c-el") ? -1 : 1,
          roAxis:2, roSign: ui("c-ro") ? -1 : 1};
}
const defaultConv = () => ({azAxis:1,azSign:1,elAxis:0,elSign:1,roAxis:2,roSign:1});

/* Self-check against the shipped matrices. */
(function verify(){
  let worst = 0, checked = 0;
  for (const o of OBJ) for (let i=0;i<N;i++){
    if (o.R[i]==null || o.az[i]==null) continue;
    const M = anglesToMatrix(o.az[i], o.el[i], o.ro[i], defaultConv());
    for (let k=0;k<9;k++) worst = Math.max(worst, Math.abs(M[k]-o.R[i][k]));
    if (++checked >= 200) break;
  }
  const el = document.getElementById("conv-check");
  if (!checked){ el.textContent = "No rotations in this extraction to verify."; return; }
  if (worst < 1e-3){
    el.textContent = `Decode self-check OK (${checked} matrices, max |diff| ${worst.toExponential(1)}).`;
  } else {
    el.innerHTML = `<strong>Decode MISMATCH</strong> vs the stored matrices (max |diff| ${worst.toFixed(4)}). ` +
      `This page's port of angles_to_matrix has drifted from the Python &mdash; treat every axis as suspect.`;
    el.style.color = "var(--s2)";
  }
})();

/* ------------------------------------------------------------------ state */
let frame = 0, playing = false, lastT = 0;
const ui = id => document.getElementById(id).checked;
const view = document.getElementById("view");
const ctx = view.getContext("2d");
view.width = D.W; view.height = D.H;

const imgs = D.frames.map(src => { const i = new Image(); i.src = src; return i; });

/* RLE -> flat Uint8Array, cached for the frame on screen only. */
const maskCache = new Map();
function maskOf(oi, fi){
  const key = oi + ":" + fi;
  if (maskCache.has(key)) return maskCache.get(key);
  const runs = OBJ[oi].rle[fi];
  let m = null;
  if (runs){
    m = new Uint8Array(D.mw * D.mh);
    let p = 0, v = 0;
    for (const r of runs){ if (v) m.fill(1, p, p+r); p += r; v ^= 1; }
  }
  if (maskCache.size > 64) maskCache.clear();
  maskCache.set(key, m);
  return m;
}

function hexToRgb(h){
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}

/* Mask overlay: outline by default so the object stays visible under the
   axes; optional translucent fill. Edge = a set pixel with an unset
   4-neighbour, computed here rather than shipped as contours. */
const layer = document.createElement("canvas");
function drawMask(m, color, fill){
  layer.width = D.mw; layer.height = D.mh;
  const lc = layer.getContext("2d");
  const img = lc.createImageData(D.mw, D.mh);
  const [r,g,b] = hexToRgb(color);
  const px = img.data;

  // Two passes so the contour is 2px, not 1px. A single-pixel outline over
  // video is close to invisible — it competes with JPEG noise at exactly its
  // own frequency — and the marks spec asks for 2px lines regardless.
  const W_ = D.mw, H_ = D.mh;
  const edge = new Uint8Array(W_*H_);
  for (let y=0;y<H_;y++) for (let x=0;x<W_;x++){
    const i = y*W_+x;
    if (!m[i]) continue;
    if ((x===0||!m[i-1]) || (x===W_-1||!m[i+1]) ||
        (y===0||!m[i-W_]) || (y===H_-1||!m[i+W_])) edge[i] = 1;
  }
  for (let y=0;y<H_;y++) for (let x=0;x<W_;x++){
    const i = y*W_+x;
    if (!m[i]) continue;
    const on = edge[i] ||
      (x>0 && edge[i-1]) || (x<W_-1 && edge[i+1]) ||
      (y>0 && edge[i-W_]) || (y<H_-1 && edge[i+W_]);
    const a = on ? 255 : (fill ? 90 : 0);
    if (!a) continue;
    px[i*4] = r; px[i*4+1] = g; px[i*4+2] = b; px[i*4+3] = a;
  }
  lc.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(layer, 0, 0, D.W, D.H);
  ctx.imageSmoothingEnabled = true;
}

/* ------------------------------------------------------- the axis overlay
   Orthographic: the camera-frame unit axis d is drawn as (d.x, d.y) scaled.
   Camera Y is down and image Y is down, so no flip. d.z only shades: > 0 is
   away from the camera (dashed), < 0 is toward it (solid). Foreshortening
   falls out for free -- an axis aimed at the lens collapses to a dot.
   A real depth + K would replace the two lines marked below.               */
const AXIS_COLOR = ["#ff3b30", "#00c853", "#2196f3"];   // X, Y, Z -- convention
const AXIS_NAME  = ["X", "Y", "Z"];
// Where to put the letter when the axis is too head-on to have a direction of
// its own. Without this the label lands exactly on the centroid and the axis
// silently disappears -- which is the COMMON case, not an edge case: an
// object facing the camera puts a whole axis along the view ray.
const HEADON_LABEL = [[1,0], [0,1], [0.72,-0.72]];
const HEADON_PX = 7;

function drawAxes(R, cx, cy, len){
  const cols = [];
  for (let a=0;a<3;a++){
    // column a of R (or row a when transposed)
    cols.push(ui("c-transpose") ? [R[a*3], R[a*3+1], R[a*3+2]]
                                : [R[a], R[a+3], R[a+6]]);
  }
  // Farthest-first so the axis nearest the camera is drawn on top.
  const order = [0,1,2].sort((p,q) => cols[q][2] - cols[p][2]);
  for (const a of order){
    const d = cols[a];
    const ex = cx + len * d[0];      // <- perspective would go here
    const ey = cy + len * d[1];      // <-
    const away = d[2] > 0;
    const plen = Math.hypot(ex-cx, ey-cy);
    const headOn = plen < HEADON_PX;

    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (const pass of [0,1]){
      ctx.strokeStyle = pass ? AXIS_COLOR[a] : "rgba(0,0,0,.75)";
      ctx.lineWidth = pass ? 2 : 4.5;
      if (headOn){
        // Along the view ray: draw the surveyor's glyph instead of a stub.
        // (+) away from the camera, (o) toward it -- the ring keeps the axis
        // visible and the fill says which way it points.
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(cx, cy, 8, 0, 6.2832); ctx.stroke();
        ctx.beginPath();
        if (away){                                   // crossed ring
          ctx.moveTo(cx-5.6, cy-5.6); ctx.lineTo(cx+5.6, cy+5.6);
          ctx.moveTo(cx+5.6, cy-5.6); ctx.lineTo(cx-5.6, cy+5.6);
        } else {                                     // dotted ring
          ctx.arc(cx, cy, 1.8, 0, 6.2832);
        }
        ctx.stroke();
        continue;
      }
      ctx.setLineDash(away && pass ? [4,3] : []);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.setLineDash([]);
      if (plen > 9){                        // arrowhead
        const ang = Math.atan2(ey-cy, ex-cx), s = 6;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - s*Math.cos(ang-0.42), ey - s*Math.sin(ang-0.42));
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - s*Math.cos(ang+0.42), ey - s*Math.sin(ang+0.42));
        ctx.stroke();
      }
    }
    // Letter label -- red/green is the worst CVD pair, so identity is never
    // colour alone. Always drawn, including for a head-on axis.
    const u = headOn ? HEADON_LABEL[a] : [d[0], d[1]];
    const off = headOn ? 15 : len + 9;
    const lx = cx + off * u[0], ly = cy + off * u[1];
    ctx.font = "700 11px ui-monospace,monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.8)";
    ctx.strokeText(AXIS_NAME[a], lx, ly);
    ctx.fillStyle = AXIS_COLOR[a];
    ctx.fillText(AXIS_NAME[a], lx, ly);
  }
}

function marker(x, y, color, filled){
  ctx.lineWidth = 2.5; ctx.strokeStyle = "rgba(0,0,0,.75)";
  if (filled){
    ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 6.2832); ctx.stroke();
    ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(255,255,255,.9)";
    ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 6.2832); ctx.stroke();
  } else {
    ctx.setLineDash([3,2]);
    ctx.strokeRect(x-4.5, y-4.5, 9, 9);
    ctx.lineWidth = 1.4; ctx.strokeStyle = color;
    ctx.strokeRect(x-4.5, y-4.5, 9, 9);
    ctx.setLineDash([]);
  }
}

function draw(){
  const im = imgs[frame];
  ctx.fillStyle = "#000"; ctx.fillRect(0,0,D.W,D.H);
  if (im && im.complete) ctx.drawImage(im, 0, 0, D.W, D.H);

  const gateOnly = ui("t-gate");
  for (let oi=0; oi<NO; oi++){
    const o = OBJ[oi], col = COLORS[oi];
    if (gateOnly && !o.cu[frame]) continue;
    const m = maskOf(oi, frame);
    if (m && ui("t-mask")) drawMask(m, col, ui("t-fill"));

    const bb = o.box[frame];
    if (bb && ui("t-box")){
      ctx.setLineDash([5,3]);
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.6)";
      ctx.strokeRect(bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]);
      ctx.lineWidth = 1.5; ctx.strokeStyle = col;
      ctx.strokeRect(bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1]);
      ctx.setLineDash([]);
    }
    if (bb && ui("t-bcent")) marker((bb[0]+bb[2])/2, (bb[1]+bb[3])/2, col, false);

    const cx = o.cx[frame], cy = o.cy[frame];
    if (cx == null) continue;

    if (ui("t-axes") && o.az[frame] != null){
      const R = anglesToMatrix(o.az[frame], o.el[frame], o.ro[frame], currentConv());
      const size = bb ? Math.max(bb[2]-bb[0], bb[3]-bb[1]) : 40;
      const len = Math.max(16, Math.min(0.5*size, 110)) *
                  (+document.getElementById("axlen").value / 100);
      drawAxes(R, cx, cy, len);
    }
    if (ui("t-cent")) marker(cx, cy, col, true);

    if (ui("t-label")){
      // Words, not glyphs: "prop" means the mask is memory propagation and
      // the detector did not re-find the object this frame (S1) -- the single
      // most load-bearing distinction on screen, so it is spelled out.
      const t = o.id + (o.det[frame] == null ? " · prop" : " · det");
      ctx.font = "600 11px ui-sans-serif,system-ui,sans-serif";
      ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      const ty = (bb ? bb[1] : cy) - 5;
      ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.8)";
      ctx.strokeText(t, (bb ? bb[0] : cx), ty);
      ctx.fillStyle = col; ctx.fillText(t, (bb ? bb[0] : cx), ty);
    }
  }
  paintStrips();
  panel();
  document.getElementById("fcount").textContent =
    `${String(frame+1).padStart(String(N).length," ")} / ${N}`;
  document.getElementById("scrub").value = frame;
}

/* ------------------------------------------------------------- the strips
   One row per object. The row's LABEL carries identity (chip + name); the
   FILL carries the per-frame state, so there is no colour collision between
   "which object" and "what happened". The mode radio swaps which state.    */
const QUALITY = [
  ["no mask",              "--ord-0"],
  ["memory propagation",   "--ord-1"],
  ["detected, crop rejected", "--ord-2"],
  ["crop_usable",          "--ord-3"],
];
const SOURCE = [
  ["none",    "--ord-0"],
  ["forward", "--s1"],
  ["reverse", "--s2"],
  ["both",    "--s3"],
];
// Presence is continuous, so it gets a SEQUENTIAL ramp (one hue, light->dark)
// rather than the categorical set — magnitude, not identity. Binned to the
// same four steps so one legend shape serves every mode.
const PRESENCE = [
  ["not measured", "--ord-0"],
  ["< 0.33",       "--ord-1"],
  ["0.33 – 0.66",  "--ord-2"],
  ["> 0.66",       "--ord-3"],
];
function stripMode(){
  return document.querySelector('input[name=stripmode]:checked').value;
}
function stateAt(o, i){
  const mode = stripMode();
  if (mode === "source"){
    const s = o.src[i];
    return s === 0 ? 0 : s === 1 ? 1 : s === 2 ? 2 : 3;
  }
  if (mode === "presence"){
    const p = (o.pres || [])[i];
    if (p == null) return 0;
    return p > 0.66 ? 3 : p > 0.33 ? 2 : 1;
  }
  if (o.cx[i] == null) return 0;
  if (o.det[i] == null) return 1;
  return o.cu[i] ? 3 : 2;
}
function stripSpec(){
  const m = stripMode();
  return m === "source" ? SOURCE : m === "presence" ? PRESENCE : QUALITY;
}
function buildStrips(){
  const host = document.getElementById("strip-rows");
  host.innerHTML = "";
  OBJ.forEach((o, oi) => {
    const row = document.createElement("div");
    row.className = "strip-row";
    row.innerHTML =
      `<div class="strip-label" title="${o.id} — ${o.prompt}">` +
      `<span class="chip" style="background:${COLORS[oi]}"></span>${o.id}</div>` +
      `<canvas class="strip" id="strip${oi}"></canvas>`;
    host.appendChild(row);
  });
  const spec = stripSpec();
  document.getElementById("strip-legend").innerHTML = spec.map(
    ([name, v]) => `<span><span class="sw" style="background:${css(v)}"></span>${name}</span>`
  ).join("");
  document.getElementById("strip-note").textContent =
    stripMode() === "source"
      ? "Frames marked “reverse” came only from the backward pass — the online loop provably cannot have them."
    : stripMode() === "presence"
      ? "SAM 3’s per-prompt presence: does this concept appear at all? It is scored even where nothing was tracked — a bright row with an empty quality row means the object IS there and localisation failed."
      : "Recorded, not enforced: every mask here got an orientation regardless of the gate.";
  OBJ.forEach((o, oi) => {
    const c = document.getElementById("strip" + oi);
    c.addEventListener("pointerdown", e => seekFromStrip(e, c));
    c.addEventListener("pointermove", e => { if (e.buttons) seekFromStrip(e, c); });
  });
  bakeStrips();
  paintStrips();
}
function seekFromStrip(e, c){
  const r = c.getBoundingClientRect();
  frame = Math.max(0, Math.min(N-1, Math.floor((e.clientX-r.left)/r.width*N)));
  draw();
}
/* The strip fill only changes with mode, theme or width — never with the
   frame. Bake it once into an offscreen bitmap so playback costs one
   drawImage plus a playhead per row instead of ~600 fillRects. */
const stripBg = [];
function bakeStrips(){
  const spec = stripSpec();
  const cols = spec.map(([,v]) => css(v));
  OBJ.forEach((o, oi) => {
    const c = document.getElementById("strip" + oi);
    if (!c) return;
    const w = Math.max(1, c.clientWidth || 600), h = 19;
    c.width = w; c.height = h;
    const b = stripBg[oi] || (stripBg[oi] = document.createElement("canvas"));
    b.width = w; b.height = h;
    const g = b.getContext("2d");
    for (let x=0; x<w; x++){
      const i = Math.min(N-1, Math.floor(x/w*N));
      g.fillStyle = cols[stateAt(o, i)];
      g.fillRect(x, 0, 1, h);
    }
  });
}
function paintStrips(){
  OBJ.forEach((o, oi) => {
    const c = document.getElementById("strip" + oi), b = stripBg[oi];
    if (!c || !b) return;
    if (c.width !== (c.clientWidth || c.width)) { bakeStrips(); }
    const g = c.getContext("2d");
    g.clearRect(0, 0, c.width, c.height);
    g.drawImage(b, 0, 0);
    const px = Math.round(frame / N * c.width);
    g.fillStyle = css("--text-primary");
    g.fillRect(Math.min(c.width - 2, px), 0, 2, c.height);
  });
}

/* ------------------------------------------------------------------ panel */
const SKIP = ["ok", "no mask", "crop too small"];
// alpha: rotational symmetry order. 0 is a REAL answer ("the fit ran and
// declined"); -1 means it never ran. Never merge the two.
const ALPHA = {"-1":"— (not measured)", "0":"0 — no confident call",
               "1":"1 — one front", "2":"2 — 180° ambiguous",
               "4":"4 — 90° ambiguous"};
function fmt(v, d){ return v == null ? "—" : (+v).toFixed(d); }
function panel(){
  const out = [];
  for (let oi=0; oi<NO; oi++){
    const o = OBJ[oi], i = frame;
    const det = o.det[i], has = o.cx[i] != null;
    const pill = !has ? '<span class="pill off">no mask</span>'
      : o.cu[i] ? '<span class="pill ok">crop_usable</span>'
      : det == null ? '<span class="pill warn">propagated</span>'
      : '<span class="pill">detected</span>';
    const pt = (o.pt || [])[i];
    const rows = [
      ["prompt", o.prompt],
      // Presence is per-PROMPT: it is present even on a frame with no mask,
      // and that pairing ("concept visible, nothing tracked") is the whole
      // reason it is worth capturing.
      ["presence", (o.pres||[])[i] == null ? "—" : fmt(o.pres[i],3)],
      ["det score", det == null ? "— (not re-detected)" : fmt(det,3)],
      ["tracker score", fmt(o.trk[i],3)],
      ["mask area px", has ? o.area[i] : "—"],
      ["centroid u,v", has ? `${fmt(o.cx[i]/ (D.W/D.fullW),1)}, ${fmt(o.cy[i]/(D.H/D.fullH),1)}` : "—"],
      ["depth (median)", (o.depth||[])[i] == null ? "—" : fmt(o.depth[i],3)+" m"],
      ["point xyz cam", pt == null ? "—"
        : `${fmt(pt[0],3)}, ${fmt(pt[1],3)}, ${fmt(pt[2],3)}`],
      ["pass", ["none","forward","reverse","both→fwd","both→rev"][o.src[i]]],
      ["mask_usable", has ? String(o.mu[i]) : "—"],
      ["gate", o.reason[i] || "—"],
      ["azimuth", o.az[i]==null ? SKIP[o.skip[i]] : fmt(o.az[i],1)+"°"],
      ["elevation", o.el[i]==null ? "—" : fmt(o.el[i],1)+"°"],
      ["roll", o.ro[i]==null ? "—" : fmt(o.ro[i],1)+"°"],
      ["symmetry α", ALPHA[String((o.alpha||[])[i] ?? -1)] || "—"],
    ];
    out.push(
      `<div class="obj-block"><div class="obj-title">` +
      `<span><span class="chip" style="background:${COLORS[oi]}"></span>${o.id}</span>${pill}</div>` +
      `<table class="readout"><tbody>` +
      rows.map(([k,v]) => `<tr><td class="k">${k}</td><td class="v mono">${v}</td></tr>`).join("") +
      `</tbody></table></div>`);
  }
  document.getElementById("panel").innerHTML = out.join("");

  const c = currentConv();
  document.getElementById("conv-out").textContent =
    `convention:\n  azimuth_axis: ${c.azAxis}\n  azimuth_sign: ${c.azSign.toFixed(1)}\n` +
    `  elevation_axis: ${c.elAxis}\n  elevation_sign: ${c.elSign.toFixed(1)}\n` +
    `  roll_axis: ${c.roAxis}\n  roll_sign: ${c.roSign.toFixed(1)}` +
    (ui("c-transpose") ? "\n\n# transpose is ON: R is being read as R_object<-camera.\n# That is NOT a config knob -- if this is the one that\n# looks right, the decode itself needs inverting." : "");
}

/* ------------------------------------------------------------- transport */
function step(d){ frame = Math.max(0, Math.min(N-1, frame+d)); draw(); }
document.getElementById("play").onclick = () => {
  playing = !playing;
  document.getElementById("play").textContent = playing ? "Pause" : "Play";
  lastT = 0;
  if (playing) requestAnimationFrame(tick);
};
function tick(t){
  if (!playing) return;
  const fps = +document.getElementById("speed").value;
  if (!lastT || t - lastT >= 1000/fps){
    lastT = t;
    if (frame >= N-1){
      if (document.getElementById("loop").checked) frame = 0;
      else { playing = false; document.getElementById("play").textContent = "Play"; draw(); return; }
    } else frame++;
    draw();
  }
  requestAnimationFrame(tick);
}
document.getElementById("prev").onclick = () => step(-1);
document.getElementById("next").onclick = () => step(1);
document.getElementById("scrub").oninput = e => { frame = +e.target.value; draw(); };
document.getElementById("speed").oninput = e => {
  document.getElementById("speedv").textContent = e.target.value + "fps";
};
addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
  const k = e.key, big = e.shiftKey ? 10 : 1;
  if (k === " "){ e.preventDefault(); document.getElementById("play").click(); }
  else if (k === "ArrowLeft"){ e.preventDefault(); step(-big); }
  else if (k === "ArrowRight"){ e.preventDefault(); step(big); }
  else if (k === "Home"){ frame = 0; draw(); }
  else if (k === "End"){ frame = N-1; draw(); }
});
for (const id of ["t-mask","t-fill","t-box","t-cent","t-bcent","t-axes","t-label",
                  "t-gate","c-transpose","c-az","c-el","c-ro"])
  document.getElementById(id).onchange = draw;
document.getElementById("axlen").oninput = draw;
for (const r of document.querySelectorAll('input[name=stripmode]'))
  r.onchange = () => { buildStrips(); draw(); };
addEventListener("resize", () => { bakeStrips(); paintStrips(); });

/* --------------------------------------------------------------- chrome */
document.getElementById("scrub").max = N-1;
document.getElementById("speedv").textContent =
  document.getElementById("speed").value + "fps";
document.getElementById("hdr-sub").textContent =
  `${N} frames · ${D.fullW}×${D.fullH} · ${NO} objects`;
document.getElementById("meta-body").innerHTML = Object.entries(D.meta)
  .filter(([k]) => k !== "schema")
  .map(([k,v]) => `<tr><td class="k">${k}</td><td class="v mono" style="word-break:break-all">${
     String(v).length > 260 ? String(v).slice(0,260)+"…" : v}</td></tr>`).join("");

buildStrips();
let ready = 0;
imgs.forEach(i => { const go = () => { if (++ready === 1 || ready === N) draw(); };
                    i.complete ? go() : (i.onload = go); });
draw();
</script></body></html>
"""


def render_html(payload: dict, title: str) -> str:
    return (_HTML
            .replace("__TITLE__", title)
            .replace("__PAYLOAD__",
                     json.dumps(payload, separators=(",", ":"))
                     .replace("</", "<\\/")))


# ---------------------------------------------------------------------------

def main(
    *,
    extraction: str,
    episode: str | None = None,
    out: str | None = None,
    scale: float = 0.5,
    quality: int = 78,
    max_frames: int | None = None,
    progress: bool = True,
):
    """Build a self-contained HTML dashboard from one extraction file.

    extraction: a .h5 written by `data_extraction.extract`.
    episode: the raw episode HDF5. Defaults to the `episode_path` recorded in
        the extraction, which is right unless the tree has moved.
    scale: display scale for frames AND masks. 0.5 keeps a 610-frame episode
        near 14 MB; 1.0 is roughly four times that. The CENTROID is always
        computed at full resolution and scaled afterwards, so this never moves
        the dot being validated.
    quality: JPEG quality for the embedded frames.
    """
    import h5py

    from data_extraction.episode import load_episode

    ex = Path(extraction)
    with h5py.File(ex, "r") as f:
        recorded = f.attrs.get("episode_path")
        eye = str(f.attrs.get("eye", "left"))
    ep_path = Path(episode) if episode else Path(str(recorded))
    if not ep_path.is_file():
        raise SystemExit(
            f"raw episode not found: {ep_path}\n"
            f"  (the extraction recorded {recorded!r}) — pass --episode")

    print(f"[dashboard] extraction {ex}")
    print(f"[dashboard] episode    {ep_path} (eye={eye})")
    ep = load_episode(ep_path, eye=eye)

    payload = build_payload(ex, ep, scale=scale, quality=quality,
                            max_frames=max_frames, progress=progress)
    html = render_html(payload, f"{ep.name} — perception v2 extraction")

    out_path = Path(out) if out else ex.with_name(ex.stem + "_dashboard.html")
    out_path.write_text(html, encoding="utf-8")
    mb = out_path.stat().st_size / 1024 ** 2
    print(f"[dashboard] wrote {out_path} ({mb:.1f} MB, {payload['n']} frames)")
    if mb > 60:
        print("[dashboard] that is large for one page — lower --scale or "
              "--quality, or cap --max-frames.")
    return out_path


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
