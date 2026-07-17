"""CLI for the verification dashboard.

Single-episode report (scrubbable timeline, frames, curves):
    uv run python -m ego2g1.data.dashboard episode_1 -o report.html \
        [--source auto|dataset|workdir] [--anchor both|measured|ground-truth] \
        [--dataset-root PATH]

Batch summary over many episodes (headless, table + histograms):
    uv run python -m ego2g1.data.dashboard --batch \
        [--out batch_report.html] [--limit N]
"""

import argparse
import sys

from .reader import open_reader


def main():
    ap = argparse.ArgumentParser(
        prog="python -m ego2g1.data.dashboard",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", nargs="?",
                    help="episode name, e.g. episode_1 (omit with --batch)")
    ap.add_argument("-o", "--out", default=None,
                    help="output HTML path (default <episode>_dashboard.html "
                         "or batch_report.html)")
    ap.add_argument("--source", default="auto",
                    choices=("auto", "dataset", "workdir"),
                    help="read the final LeRobot dataset or the work-dir "
                         "stage outputs (auto: dataset if it exists)")
    ap.add_argument("--anchor", default="both",
                    choices=("both", "measured", "ground-truth"))
    ap.add_argument("--dataset-root", default=None,
                    help="LeRobot dataset root (default cfg.output_root/repo_id)")
    ap.add_argument("--batch", action="store_true",
                    help="summary over many episodes instead of one report")
    ap.add_argument("--site", action="store_true",
                    help="build ONE dashboard.html (episode dropdown, "
                         "lazy-loaded per-episode json) + manifest")
    ap.add_argument("--serve", type=int, default=None, metavar="PORT",
                    help="after --site, serve the site directory (fetch is "
                         "blocked on file://)")
    ap.add_argument("--force", action="store_true",
                    help="site: regenerate reports that already exist")
    ap.add_argument("--limit", type=int, default=None,
                    help="batch/site: only the first N episodes")
    ap.add_argument("--ik-iters", type=int, default=None,
                    help="override cfg.ik_iters (default 5)")
    args = ap.parse_args()

    try:
        reader, source = open_reader(args.source, args.dataset_root)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")

    if args.site:
        from .site import build_site, serve
        episodes = reader.episodes()
        if args.episode:
            episodes = [e for e in episodes if e == args.episode]
        if args.limit:
            episodes = episodes[:args.limit]
        if not episodes:
            sys.exit(f"no episodes found in {source} source")
        out_dir = args.out or "dashboard_site"
        build_site(reader, source, out_dir, episodes,
                   ik_iters=args.ik_iters, force=args.force)
        if args.serve:
            serve(out_dir, args.serve)
        return

    if args.batch:
        from .batch import run_batch, write_batch_report
        episodes = reader.episodes()
        if args.episode:
            episodes = [e for e in episodes if e == args.episode]
        if args.limit:
            episodes = episodes[:args.limit]
        if not episodes:
            sys.exit(f"no episodes found in {source} source")
        print(f"batch over {len(episodes)} episode(s) from {source}")
        rows = run_batch(reader, episodes, ik_iters=args.ik_iters)
        write_batch_report(args.out or "batch_report.html", rows, source)
        return

    if not args.episode:
        ap.error("episode name required unless --batch is given")
    modes = (("measured", "ground-truth") if args.anchor == "both"
             else (args.anchor,))

    from .replay import replay_record
    from .report import write_report

    print(f"loading {args.episode} from {source} ...")
    try:
        rec = reader.load(args.episode)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    print(f"  {rec.T} ticks @ {rec.fps:.0f} Hz, H={rec.horizon}, "
          f"{len(rec.subeps)} sub-episode(s), config {rec.config_hash}")
    try:
        result = replay_record(rec, modes=modes, ik_iters=args.ik_iters)
        write_report(args.out or f"{args.episode}_dashboard.html", rec, result)
    finally:
        rec.close()


if __name__ == "__main__":
    main()
