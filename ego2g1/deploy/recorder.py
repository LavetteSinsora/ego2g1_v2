"""Record a deploy session so it can be reconstructed post-hoc at any tick.

Ported from the old deploy's recorder.py (third_party/openpi/ego2g1/deploy) —
the reason the jitter was diagnosable at all. jitter_root_cause.md's latency
and splice numbers were read straight out of one of these event streams. It is
not optional instrumentation: a rollout that was not recorded cannot be
debugged, only re-run.

The trick that makes it cheap: nothing is snapshotted. The runner's state (the
active chunk, the strategy's splice indices, what was sent to the executor) is
a pure function of a small, timestamped EVENT STREAM, so we log ~8 event kinds
at the existing seams and replay them offline.

Event kinds the new runner emits (each carries `t` monotonic + `kind`):

    meta            once: mode, action_mode, horizon, fps, hosts, strategy params
    latency_check   the startup self-check report
    obs             per tick: state age, camera frame id
    infer_result    per inference: latency, start_timestep, splice info, and
                    `actions` — the converted (H, 26) joint chunk, so
                    replay_record.py can rebuild the buffers exactly
    action          per tick: the popped joint row, clamped or not
    clamp           when the clamp actually limited a step
    tracking        per chunk (relative_eef): worst IK tracking error
    worker_error    the async inference worker died (with the exception)
    estop           the damp() call, with the watchdog's reason

ISOLATION contract: nothing here runs on, or blocks, a hot thread. `log()`
builds a small dict and enqueues; one daemon writer drains to JSONL; a second
daemon pumps cameras to MP4. Session layout::

    <root>/<task>_<ISO8601>/
        events.jsonl    one JSON object per line
        frames.jsonl    {"cam", "frame_id", "t"} — the video<->clock map
        <cam>.mp4       per camera
        meta.json       horizon, fps, layout, clock epochs, ...
"""

import datetime
import json
import logging
import pathlib
import queue
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

_SENTINEL = object()


def _san(v):
    """Make a value JSON-safe, eagerly, on the calling (possibly hot) thread so
    the writer never races an array the loop reuses after enqueue."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _san(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_san(x) for x in v]
    return v


class _VideoSink:
    """Append-only MP4, indexed by frame number. Offline we seek by frame_id
    and map time->frame_id through frames.jsonl; the container's own fps is
    nominal and its timestamps are never trusted."""

    def __init__(self, path: pathlib.Path, *, nominal_fps: float = 30.0):
        self.path = path
        self._fps = float(nominal_fps)
        self._writer = None
        self._n = 0

    def append(self, frame_bgr) -> int:
        import cv2

        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self._fps, (w, h))
            if not self._writer.isOpened():
                raise RuntimeError(f"could not open VideoWriter at {self.path}")
        fid = self._n
        self._writer.write(frame_bgr)
        self._n += 1
        return fid

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class Recorder:
    """One recording session. Construct, `start()`, feed `log()` from the
    loop's seams, `stop()` when done. `cameras` is {name: obj-with-.read()};
    None values (no wrist camera yet) are skipped."""

    def __init__(self, session_dir, *, meta: dict, cameras: dict | None = None,
                 pump_hz: float = 30.0):
        self.dir = pathlib.Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta = dict(meta)
        self._cams = {k: v for k, v in (cameras or {}).items() if v is not None}
        self._pump_period = 1.0 / float(pump_hz)

        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._writer_thread = None
        self._pump_thread = None

        self._sinks = {name: _VideoSink(self.dir / f"{name}.mp4") for name in self._cams}
        # Latest frame_id per camera, so `obs` events can name the frame the
        # model most likely saw. Plain int reads; one-frame-stale is harmless.
        self._latest = {name: -1 for name in self._cams}

        self._events = None
        self._frames = None

    # --- lifecycle ----------------------------------------------------------

    def start(self):
        self._meta.update({
            "t0_monotonic": time.monotonic(),
            "t0_wall": time.time(),
            "started_iso": datetime.datetime.now().isoformat(timespec="seconds"),
            "cameras": list(self._cams),
        })
        (self.dir / "meta.json").write_text(json.dumps(_san(self._meta), indent=2))
        self._events = open(self.dir / "events.jsonl", "w", buffering=1)
        self._frames = open(self.dir / "frames.jsonl", "w", buffering=1)

        self._writer_thread = threading.Thread(target=self._drain, name="rec-writer",
                                               daemon=True)
        self._writer_thread.start()
        if self._cams:
            self._pump_thread = threading.Thread(target=self._pump, name="rec-pump",
                                                 daemon=True)
            self._pump_thread.start()
        logger.info("recording -> %s", self.dir)

    def stop(self):
        self._stop.set()
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=2.0)
        self._q.put(_SENTINEL)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        for s in self._sinks.values():
            s.close()
        for f in (self._events, self._frames):
            if f is not None:
                f.close()
        logger.info("recording closed: %s", self.dir)

    # --- producer side (called from loop threads) ---------------------------

    def log(self, kind: str, **fields) -> None:
        """Enqueue one event. Hot-path safe: sanitise (small copies) + put."""
        rec = {"t": time.monotonic(), "kind": kind}
        for k, v in fields.items():
            rec[k] = _san(v)
        self._q.put(rec)

    def latest_frame_id(self, cam: str) -> int:
        return self._latest.get(cam, -1)

    # --- consumer side (own daemon threads) ---------------------------------

    def _drain(self):
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                break
            try:
                if item.get("kind") == "_frame":
                    json.dump({"cam": item["cam"], "frame_id": item["frame_id"],
                               "t": item["t"]}, self._frames)
                    self._frames.write("\n")
                else:
                    json.dump(item, self._events)
                    self._events.write("\n")
            except Exception:
                logger.exception("recorder writer dropped an event")

    def _pump(self):
        import cv2

        while not self._stop.is_set():
            t0 = time.perf_counter()
            for name, cam in self._cams.items():
                try:
                    frame = cam.read()
                except Exception:
                    frame = None
                if frame is None:
                    continue
                bgr = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR)
                try:
                    fid = self._sinks[name].append(bgr)
                except Exception:
                    logger.exception("video sink %s failed; disabling", name)
                    continue
                self._latest[name] = fid
                self._q.put({"kind": "_frame", "cam": name, "frame_id": fid,
                             "t": time.monotonic()})
            time.sleep(max(0.0, self._pump_period - (time.perf_counter() - t0)))


class NullRecorder:
    """Same producer API, writes nothing — for tests and --no-record."""

    def start(self):
        pass

    def stop(self):
        pass

    def log(self, kind: str, **fields) -> None:
        pass

    def latest_frame_id(self, cam: str) -> int:
        return -1


def new_session(root, task: str) -> pathlib.Path:
    """`<root>/<task-slug>_<ISO8601>` — a fresh directory for one recording."""
    slug = "".join(c if c.isalnum() else "_" for c in task)[:40].strip("_") or "session"
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    return pathlib.Path(root) / f"{slug}_{stamp}"
