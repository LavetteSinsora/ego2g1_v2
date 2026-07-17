"""Headless replay over many episodes -> one self-contained summary HTML.

Runs the measured-anchor (deployment-faithful) closed loop with no frame
rendering, collects per-episode aggregates, and emits a sortable table plus
inline-SVG histograms. Red-flag rows (max measured pos err > 2 cm, or any
hand-blocked tick) sort first.
"""

import html as _html
import json
import time

import numpy as np

from .replay import replay_record

POS_FLAG_CM = 2.0


def run_batch(reader, episodes, ik_iters=None, verbose=True):
    rows = []
    for ep in episodes:
        t0 = time.time()
        rec = reader.load(ep)
        try:
            res = replay_record(rec, modes=("measured",), render=False,
                                render_hands=False, ik_iters=ik_iters,
                                verbose=verbose)
        finally:
            rec.close()
        row = {"episode": ep, "source": rec.source, "T": rec.T,
               "ticks_total": rec.ticks_total, "ticks_kept": rec.ticks_kept,
               "n_subeps": len(rec.subeps),
               "config_hash": rec.config_hash}
        m = res["modes"]["measured"]
        for s in ("left", "right"):
            pos, ori = m[s]["pos_cm"], m[s]["ori_deg"]
            row[f"pos_mean_{s[0]}"] = float(np.nanmean(pos))
            row[f"pos_max_{s[0]}"] = float(np.nanmax(pos))
            row[f"ori_mean_{s[0]}"] = float(np.nanmean(ori))
            row[f"ori_max_{s[0]}"] = float(np.nanmax(ori))
            h = res["hands"][s]
            row[f"blocked_{s[0]}"] = int(h["blocked"].sum())
            row[f"contact_{s[0]}"] = int(h["contact"].sum())
        tot = max(1, rec.ticks_total)
        row["drop_frac"] = {k: v / tot for k, v in rec.filter_stats.items()
                            if v and k != "bad_any"}
        row["drop_total"] = 1.0 - rec.ticks_kept / tot
        row["flag"] = (max(row["pos_max_l"], row["pos_max_r"]) > POS_FLAG_CM
                       or row["blocked_l"] + row["blocked_r"] > 0)
        row["secs"] = time.time() - t0
        rows.append(row)
        print(f"[batch] {ep}: pos max L {row['pos_max_l']:.2f} / "
              f"R {row['pos_max_r']:.2f} cm, blocked "
              f"{row['blocked_l']}+{row['blocked_r']}, "
              f"{'FLAG' if row['flag'] else 'ok'} ({row['secs']:.0f}s)")
    return rows


def write_batch_rows_json(rows):
    """Rows for the single-page site manifest: flagged first, then by max
    position error (same order as the HTML table)."""
    return sorted(rows, key=lambda r: (not r["flag"],
                                       -max(r["pos_max_l"], r["pos_max_r"])))


# ------------------------------------------------------------------- html

def _svg_hist(values, title, unit, width=340, height=120, bins=14,
              flag_at=None):
    values = [v for v in values if v == v]
    if not values:
        return f"<div class='hist'><h3>{title}</h3><p>no data</p></div>"
    lo, hi = 0.0, max(max(values), 1e-6) * 1.02
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    cmax = max(1, counts.max())
    pad_l, pad_b, pad_t = 8, 16, 6
    bw = (width - 2 * pad_l) / bins
    parts = []
    for i, c in enumerate(counts):
        h = (height - pad_b - pad_t) * c / cmax
        x = pad_l + i * bw
        bad = flag_at is not None and edges[i] >= flag_at
        parts.append(
            f"<rect x='{x:.1f}' y='{height - pad_b - h:.1f}' "
            f"width='{max(bw - 1.5, 1):.1f}' height='{h:.1f}' "
            f"fill='{'var(--bad)' if bad else 'var(--mode-meas)'}' "
            f"opacity='0.85'/>")
        if c:
            parts.append(f"<text x='{x + bw / 2:.1f}' "
                         f"y='{height - pad_b - h - 3:.1f}' font-size='8' "
                         f"text-anchor='middle' fill='var(--muted)'>{c}</text>")
    for frac in (0, 0.5, 1.0):
        x = pad_l + (width - 2 * pad_l) * frac
        parts.append(f"<text x='{x:.1f}' y='{height - 4}' font-size='9' "
                     f"text-anchor='middle' fill='var(--muted)'>"
                     f"{lo + (hi - lo) * frac:.2f}</text>")
    if flag_at is not None and flag_at < hi:
        x = pad_l + (width - 2 * pad_l) * (flag_at - lo) / (hi - lo)
        parts.append(f"<line x1='{x:.1f}' y1='{pad_t}' x2='{x:.1f}' "
                     f"y2='{height - pad_b}' stroke='var(--bad)' "
                     f"stroke-dasharray='3 3'/>")
    return (f"<div class='hist'><h3>{title} ({unit})</h3>"
            f"<svg viewBox='0 0 {width} {height}'>{''.join(parts)}</svg></div>")


_CSS = """
:root { --surface-1:#fcfcfb; --page:#f9f9f7; --ink-1:#0b0b0b; --ink-2:#52514e;
  --muted:#898781; --ring:rgba(11,11,11,0.10); --bad:#d5493f;
  --mode-meas:#1baf7a; --flagbg:rgba(213,73,63,0.08); }
@media (prefers-color-scheme: dark) {
  :root { --surface-1:#1a1a19; --page:#0d0d0d; --ink-1:#fff; --ink-2:#c3c2b7;
    --ring:rgba(255,255,255,0.10); --flagbg:rgba(213,73,63,0.14); } }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--ink-1);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1240px; margin:0 auto; padding:20px 16px 60px; }
h1 { font-size:19px; margin:0 0 4px; } .sub { color:var(--ink-2); margin-bottom:16px; }
table { border-collapse:collapse; width:100%; background:var(--surface-1);
  border:1px solid var(--ring); border-radius:10px; font-size:12.5px;
  font-variant-numeric:tabular-nums; }
th,td { padding:5px 8px; text-align:right; border-bottom:1px solid var(--ring); }
th { color:var(--ink-2); font-weight:600; position:sticky; top:0;
  background:var(--surface-1); }
td:first-child, th:first-child { text-align:left; }
tr.flag { background:var(--flagbg); }
tr.flag td:first-child::before { content:"⚑ "; color:var(--bad); }
.warn { color:var(--bad); font-weight:600; }
.hists { display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
  gap:10px; margin-top:16px; }
.hist { background:var(--surface-1); border:1px solid var(--ring);
  border-radius:10px; padding:10px; }
.hist h3 { font-size:12.5px; font-weight:600; color:var(--ink-2); margin:0 0 4px; }
.hist svg { width:100%; display:block; }
.small { color:var(--muted); font-size:12px; margin-top:10px; }
"""


def write_batch_report(out_path, rows, source):
    rows = sorted(rows, key=lambda r: (not r["flag"],
                                       -max(r["pos_max_l"], r["pos_max_r"])))
    n_flag = sum(r["flag"] for r in rows)

    def num(v, nd=2):
        return f"{v:.{nd}f}"

    def warn(v, thr, nd=2):
        s = num(v, nd)
        return f"<span class='warn'>{s}</span>" if v > thr else s

    trs = []
    for r in rows:
        drops = ", ".join(f"{k.replace('bad_', '')} {v * 100:.1f}%"
                          for k, v in sorted(r["drop_frac"].items())) or "—"
        trs.append(
            f"<tr class='{'flag' if r['flag'] else ''}'>"
            f"<td>{_html.escape(r['episode'])}</td>"
            f"<td>{r['ticks_kept']}/{r['ticks_total']}</td>"
            f"<td>{r['n_subeps']}</td>"
            f"<td>{num(r['pos_mean_l'])}</td><td>{warn(r['pos_max_l'], POS_FLAG_CM)}</td>"
            f"<td>{num(r['pos_mean_r'])}</td><td>{warn(r['pos_max_r'], POS_FLAG_CM)}</td>"
            f"<td>{num(r['ori_mean_l'], 1)}</td><td>{num(r['ori_max_l'], 1)}</td>"
            f"<td>{num(r['ori_mean_r'], 1)}</td><td>{num(r['ori_max_r'], 1)}</td>"
            f"<td>{warn(r['blocked_l'] + r['blocked_r'], 0, 0)}</td>"
            f"<td>{warn(r['contact_l'] + r['contact_r'], 0, 0)}</td>"
            f"<td>{num(r['drop_total'] * 100, 1)}%</td>"
            f"<td style='text-align:left'>{drops}</td></tr>")

    hists = "".join([
        _svg_hist([max(r["pos_max_l"], r["pos_max_r"]) for r in rows],
                  "measured-anchor MAX position error per episode", "cm",
                  flag_at=POS_FLAG_CM),
        _svg_hist([(r["pos_mean_l"] + r["pos_mean_r"]) / 2 for r in rows],
                  "measured-anchor MEAN position error per episode", "cm"),
        _svg_hist([max(r["ori_max_l"], r["ori_max_r"]) for r in rows],
                  "measured-anchor MAX orientation error per episode", "deg"),
        _svg_hist([r["drop_total"] * 100 for r in rows],
                  "ticks dropped by filters per episode", "%"),
    ])

    cfg_hashes = sorted({r["config_hash"] for r in rows})
    html_doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ego2g1.data batch report</title><style>{_CSS}</style></head><body>
<div class="wrap">
<h1>ego2g1.data batch report — {len(rows)} episode(s), source: {source}</h1>
<div class="sub">Measured-anchor (deployment-faithful) closed-loop replay,
headless. {n_flag} flagged (max pos err &gt; {POS_FLAG_CM} cm or any blocked
hand tick) — flagged rows first. config_hash: {', '.join(cfg_hashes)}</div>
<table><thead><tr>
<th>episode</th><th>ticks kept</th><th>subeps</th>
<th>L pos mean (cm)</th><th>L pos max</th><th>R pos mean</th><th>R pos max</th>
<th>L ori mean (°)</th><th>L ori max</th><th>R ori mean</th><th>R ori max</th>
<th>blocked</th><th>contact</th><th>dropped</th><th>drop fractions per filter</th>
</tr></thead><tbody>{''.join(trs)}</tbody></table>
<div class="hists">{hists}</div>
<div class="small">Generated by ego2g1.data.dashboard --batch.
Raw rows: <details><summary>json</summary><pre>{_html.escape(
    json.dumps(rows, indent=1, default=str))}</pre></details></div>
</div></body></html>"""
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)
    print(f"wrote {out_path} ({len(rows)} rows, {n_flag} flagged)")
    return rows
