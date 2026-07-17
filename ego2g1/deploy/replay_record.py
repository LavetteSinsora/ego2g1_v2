"""Reconstruct a recorded deploy session at any time tick.

Pure reader — no JAX, no DDS, no mujoco, no torch. Adapted from the old
deploy's replay_record.py (third_party/openpi/ego2g1/deploy) to the NEW
recorder schema (deploy/recorder.py). Same pattern: nothing was snapshotted at
record time; the loop's core data structures are a pure function of the event
stream, so we replay events.jsonl up to a queried monotonic time `t`:

    * the strategy's chunk state. `sync`: the active (H, 26) joint chunk is the
      last `infer_result` at or before t (its `actions` field), and the pointer
      advances by one per `action` event since — one pop == one send == one
      logged row. Async modes: the REAL buffer class from strategies.py is
      re-fed the recorded add_chunk (`infer_result`) / pop (`action`) sequence,
      so the splice/blend math cannot drift from the live emitter's.
    * the executor's command: the last `action` row at or before t (post-clamp,
      exactly what was sent).

Usage::

    from ego2g1.deploy.replay_record import Session
    s = Session("recordings/put_bottle_in_box_20260717T142530")
    snap = s.at(12.34)          # everything true at t=12.34 (monotonic)
    print(snap["index"], snap["action_row"])
    s.timeline()                # the absolute event timeline, for a report

`t` is in the recording's monotonic clock. `meta.json` carries `t0_monotonic` /
`t0_wall` if you need to convert to or from wall time. The CLI's `--at` is in
seconds since session start, matching the timeline column.
"""

import bisect
import json
import pathlib

import numpy as np


class Session:
    def __init__(self, session_dir):
        self.dir = pathlib.Path(session_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.mode = self.meta.get("mode", "sync")
        self.fps = int(self.meta.get("fps", 30))

        self.events = sorted(self._read_jsonl(self.dir / "events.jsonl"),
                             key=lambda e: e["t"])
        self.frames = self._read_jsonl(self.dir / "frames.jsonl")

        # Split by kind once; each list stays in time order.
        self._by_kind: dict[str, list] = {}
        for e in self.events:
            self._by_kind.setdefault(e["kind"], []).append(e)

        self._infer = self._by_kind.get("infer_result", [])
        self._infer_t = [e["t"] for e in self._infer]
        self._acts = self._by_kind.get("action", [])
        self._act_t = [e["t"] for e in self._acts]

        # Frame times per camera, for time->frame_id lookup.
        self._frame_t: dict[str, list] = {}
        self._frame_id: dict[str, list] = {}
        for fr in self.frames:
            self._frame_t.setdefault(fr["cam"], []).append(fr["t"])
            self._frame_id.setdefault(fr["cam"], []).append(fr["frame_id"])

    @staticmethod
    def _read_jsonl(path):
        if not path.exists():
            return []
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    # --- reconstruction -----------------------------------------------------

    def _make_buffer(self):
        """The production buffer class for this session's mode, with the
        recorded constructor params (meta.json) — reusing strategies.py so the
        reconstruction math is the emitter's, not a copy of it."""
        from . import strategies as _strategies

        if self.mode == "async":
            return _strategies.NaiveAsyncBuffer()
        if self.mode == "temporal_ensembling":
            return _strategies.TemporalEnsemblingBuffer(
                float(self.meta.get("exp_weight_m", 0.01)))
        if self.mode in ("temporal_smoothing", "rtc"):
            return _strategies.TemporalSmoothingBuffer(
                int(self.meta.get("max_latency_steps", 8)),
                int(self.meta.get("min_smooth_steps", 10)))
        raise ValueError(f"no buffer for mode {self.mode!r}")

    def _replay_buffer(self, t: float):
        """(buffer, rows popped since the last install) after replaying every
        add_chunk/pop up to t, in log-time order — the live interleaving."""
        buf = self._make_buffer()
        popped: list[np.ndarray] = []
        for e in self.events:
            if e["t"] > t:
                break
            kind = e["kind"]
            if kind == "infer_result" and "actions" in e:
                buf.add_chunk(np.asarray(e["actions"], dtype=np.float64),
                              int(e.get("start_timestep", 0)))
                popped = []
            elif kind == "action":
                row = buf.pop_action()
                # the buffer's own value; the logged row is post-clamp
                popped.append(np.asarray(e["row"], dtype=np.float64)
                              if row is None else np.asarray(row))
        return buf, popped

    def chunk_at(self, t: float):
        """(chunk (H, 26) | None, index) — the strategy's chunk state at t.
        chunk[index] is the next row to pop when index < len(chunk).

        sync                the active chunk as inferred.
        async               the newest installed chunk; index includes the
                            skipped rows (naive-async skip arithmetic).
        temporal_smoothing/ the current COMBINED (blended) chunk: rows already
        rtc                 popped since its install + the queued remainder.
        temporal_ensembling no single chunk exists (every live chunk votes);
                            returns (None, 0) — use at()['votes'] instead.
        """
        if self.mode == "sync":
            i = bisect.bisect_right(self._infer_t, t) - 1
            if i < 0:
                return None, 0
            e = self._infer[i]
            if "actions" not in e:
                return None, 0
            chunk = np.asarray(e["actions"], dtype=np.float64)
            lo = bisect.bisect_right(self._act_t, e["t"])
            hi = bisect.bisect_right(self._act_t, t)
            return chunk, min(hi - lo, len(chunk))

        buf, popped = self._replay_buffer(t)
        if self.mode == "temporal_ensembling":
            return None, 0
        if self.mode == "async":
            if buf._chunk is None:
                return None, 0
            chunk = buf._chunk.copy()
            index = int(np.clip(buf._global_t - buf._chunk_start_t,
                                0, len(chunk)))
            return chunk, index
        # temporal_smoothing / rtc: consumed + remaining == the combined chunk
        remaining = [np.asarray(a) for a in buf._chunk]
        rows = popped + remaining
        if not rows:
            return None, 0
        return np.stack(rows), len(popped)

    def frame_at(self, t: float, cam: str):
        """(frame_id, capture_t) for the frame whose capture time is the last <= t."""
        ts = self._frame_t.get(cam)
        if not ts:
            return None, None
        i = bisect.bisect_right(ts, t) - 1
        if i < 0:
            return None, None
        return self._frame_id[cam][i], ts[i]

    def read_frame(self, t: float, cam: str):
        """Decode the actual (H,W,3) RGB frame at t via the mp4, or None. Needs cv2."""
        fid, _ = self.frame_at(t, cam)
        if fid is None:
            return None
        import cv2

        cap = cv2.VideoCapture(str(self.dir / f"{cam}.mp4"))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, bgr = cap.read()
        finally:
            cap.release()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def at(self, t: float) -> dict:
        """Everything true at monotonic time t: chunk state + the last
        commanded row + inference phase + frames."""
        chunk, index = self.chunk_at(t)
        i = bisect.bisect_right(self._act_t, t) - 1
        cmd = self._acts[i] if i >= 0 else None

        snap = {
            "t": t,
            "mode": self.mode,
            "ready": chunk is not None and index < len(chunk),
            "index": index,
            "horizon": None if chunk is None else len(chunk),
            "action_row": None if cmd is None else list(cmd["row"]),
            "step": None if cmd is None else cmd.get("step"),
            "phase": self._phase_at(t),
        }
        if self.mode == "temporal_ensembling":
            buf, popped = self._replay_buffer(t)
            snap["votes"] = buf.telemetry()["votes"]
            snap["ready"] = bool(snap["votes"]) or cmd is not None
        for cam in self._frame_t:
            fid, ft = self.frame_at(t, cam)
            snap[f"{cam}_frame_id"] = fid
            snap[f"{cam}_frame_t"] = ft
        return snap

    def _phase_at(self, t: float) -> str:
        """idle / inferring / executing at t. The new schema has no
        infer_request event; each infer_result carries its own latency, so the
        in-flight window is [t_result - latency, t_result)."""
        for e in self._infer:
            t1 = e["t"]
            if t1 - float(e.get("latency", 0.0)) <= t < t1:
                return "inferring"
            if t1 > t:
                break
        if bisect.bisect_right(self._act_t, t) == 0 \
                and bisect.bisect_right(self._infer_t, t) == 0:
            return "idle"
        return "executing"

    # --- timeline for reports ----------------------------------------------

    def timeline(self) -> list:
        """The absolute event timeline, one compact row per meaningful event,
        sorted by time — the spine of a post-hoc behaviour report. Per-tick
        obs/action events are summarized away."""
        rows = []
        for e in self.events:
            k = e["kind"]
            if k == "latency_check":
                rows.append((e["t"], "latency_check",
                             f"{e.get('verdict', '?')}: {e.get('detail', '')}"))
            elif k == "infer_result":
                sp = e.get("splice") or {}
                tag = "LATE " if sp.get("late") else ""
                extra = (f" drop={sp.get('drop_count')}" if sp else "")
                rows.append((e["t"], "infer_result",
                             f"{tag}latency={1000 * e.get('latency', 0):.0f}ms "
                             f"H={e.get('horizon')}"
                             f"{' rtc' if e.get('rtc') else ''}{extra}"))
            elif k == "clamp":
                rows.append((e["t"], "clamp",
                             f"step {e.get('step')}: {e.get('max_step', 0):.3f} rad capped"))
            elif k == "tracking":
                rows.append((e["t"], "tracking",
                             f"worst {1000 * e.get('worst_m', 0):.0f} mm"))
            elif k == "worker_error":
                rows.append((e["t"], "worker_error", str(e.get("error"))))
            elif k == "estop":
                rows.append((e["t"], "estop", str(e.get("reason"))))
            elif k == "latency_check_refused":
                rows.append((e["t"], "latency_check_refused", ""))
        return rows

    def span(self):
        """(t_first, t_last) monotonic bounds of the recording."""
        ts = [e["t"] for e in self.events] + [f["t"] for f in self.frames]
        return (min(ts), max(ts)) if ts else (None, None)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="inspect a recorded deploy session")
    p.add_argument("session", help="path to a recordings/<...> directory")
    p.add_argument("--at", type=float, default=None,
                   help="seconds SINCE SESSION START to reconstruct (matches the "
                        "timeline column; default: print the timeline)")
    args = p.parse_args()

    s = Session(args.session)
    t0, t1 = s.span()
    if args.at is None:
        print(f"session {s.dir.name}: {t1 - t0:.1f}s, {len(s.events)} events, "
              f"mode={s.mode}")
        for t, kind, desc in s.timeline():
            print(f"  {t - t0:8.3f}  {kind:16s} {desc}")
    else:
        # --at is relative to session start, so add the clock's origin.
        snap = s.at(t0 + args.at)
        snap["t_rel"] = args.at
        print(json.dumps({k: v for k, v in snap.items() if k != "action_row"},
                         indent=2))
