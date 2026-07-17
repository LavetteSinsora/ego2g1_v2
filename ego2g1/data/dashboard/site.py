"""Site generator: ONE dashboard.html with an episode dropdown; each
episode's full payload (frames, curves, masks) lives in a sidecar
episodes/<ep>.json fetched on selection. The batch summary table is embedded
inline. Fully offline, but browsers block fetch() on file:// - so serve the
directory:

    uv run python -m ego2g1.data.dashboard --site [-o dashboard_site]
        [--limit N] [--source auto|dataset|workdir] [--force] [--serve PORT]

or afterwards:  python3 -m http.server -d dashboard_site 8123

Per-episode JSONs are only regenerated when missing or --force.
"""

import json
import os
from pathlib import Path


def build_site(reader, source, out_dir, episodes, ik_iters=None, force=False):
    from .batch import run_batch, write_batch_rows_json
    from .replay import replay_record
    from .report import build_data

    out = Path(out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)

    done, failed = [], []
    for ep in episodes:
        target = out / "episodes" / f"{ep}.json"
        if target.exists() and not force:
            done.append(ep)
            continue
        try:
            rec = reader.load(ep)
        except FileNotFoundError as e:
            print(f"  [site] {ep}: skipped ({e})")
            failed.append(ep)
            continue
        try:
            result = replay_record(rec, modes=("measured", "ground-truth"),
                                   ik_iters=ik_iters, verbose=False)
            payload = build_data(rec, result)
            target.write_text(json.dumps(payload))
            print(f"  [site] {ep}: {target.stat().st_size / 1e6:.1f} MB json")
            done.append(ep)
        except Exception as e:
            print(f"  [site] {ep}: FAILED ({e})")
            failed.append(ep)
        finally:
            rec.close()

    rows = run_batch(reader, done, ik_iters=ik_iters)
    manifest = {"source": source, "episodes": done, "failed": failed,
                "batch": write_batch_rows_json(rows)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    template = Path(__file__).parent / "site_template.html"
    html = template.read_text()
    (out / "dashboard.html").write_text(html)
    print(f"[site] {out / 'dashboard.html'}: {len(done)} episode(s)"
          + (f", {len(failed)} failed" if failed else "")
          + f"\n[site] serve it:  python3 -m http.server -d '{out}' 8123"
          + "  ->  http://localhost:8123/dashboard.html")
    return done, failed


def serve(out_dir, port):
    import http.server
    import socketserver
    os.chdir(out_dir)
    with socketserver.TCPServer(("", port),
                                http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"serving {out_dir} at http://localhost:{port}/dashboard.html "
              f"(Ctrl-C to stop)")
        httpd.serve_forever()
