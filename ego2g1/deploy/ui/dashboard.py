"""A live web dashboard for the deploy runner — a pure, pull-based add-on.

    python -m ego2g1.deploy.runner --host ... --prompt "..." --dashboard
    python -m ego2g1.deploy.dashboard --demo            # no hardware; synthetic data

Ported from the old deploy's dashboard.py (third_party/openpi/ego2g1/deploy).
It visualizes the core data structures of the new control loop:

    * the strategy's chunk state    telemetry(): horizon/index    -> the bar's cells
    * the execution pointer         (pop-and-send is one tick here, so robot-now
                                     and the pointer coincide)
    * the last commanded (26,) row  executor telemetry             -> the strip
    * the inference lifecycle, DelayBudget stats, clamp/watchdog counters, and
      the egocentric camera frame.

ISOLATION — this touches NOTHING on the hot path. The loop's threads (the
30 Hz control loop, the async inference worker, the vendored 500 Hz arm thread)
get no new code and never call in here. This server runs on its own daemon
thread and PULLS `runner.telemetry()` when an HTTP request arrives (~10 Hz);
telemetry() only reads existing state under existing locks / copies small
arrays. JPEG encoding (cv2, which releases the GIL) and all socket I/O happen
here, off every hot thread. Off by default.
"""

import http.server
import json
import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# --- relation_eef perception overlay --------------------------------------
# The renderer lives in ui/overlay.py, fed the RECORDED `percept` shape so
# the live page and replay_dashboard draw through one code path. Re-exported
# here under the old name for callers that knew it by this module.
from .overlay import project_to_pixel as _project_to_pixel  # noqa: E402,F401


class _Handler(http.server.BaseHTTPRequestHandler):
    # The owning Dashboard is attached to the server as `.dash`.

    def log_message(self, *args):  # silence the default per-request stderr spam
        pass

    def do_GET(self):
        dash = self.server.dash
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
        elif path == "/state":
            try:
                body = json.dumps(dash.loop.telemetry()).encode("utf-8")
            except Exception as e:  # a failed read must not kill the server thread
                self._send(500, "text/plain", str(e).encode("utf-8"))
                return
            self._send(200, "application/json", body)
        elif path == "/frame.jpg":
            jpg = dash.encode_frame()
            if jpg is None:
                self.send_response(204)
                self.end_headers()
                return
            self._send(200, "image/jpeg", jpg)
        elif path == "/perception.jpg":
            jpg = dash.encode_perception_frame()
            if jpg is None:
                self.send_response(204)
                self.end_headers()
                return
            self._send(200, "image/jpeg", jpg)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        """State-changing controls. GET stays pure telemetry; anything that
        affects the robot is a POST. Each route maps to a DeployRunner method
        (begin/pause/estop/record_toggle/reset_to_episode); anything the loop
        object doesn't provide answers 409 with the reason (the --demo loop
        stubs them all so the page can be exercised)."""
        dash = self.server.dash
        path = self.path.split("?", 1)[0]
        loop = dash.loop

        def need(name):
            fn = getattr(loop, name, None)
            if not callable(fn):
                raise RuntimeError(f"{name}: not supported by this runner")
            return fn

        try:
            if path == "/start":
                need("begin")()
                result = {"active": True}
            elif path == "/pause":
                need("pause")()
                result = {"active": False}
            elif path == "/estop":
                need("estop")("dashboard")
                result = {"tripped": True}
            elif path == "/record":
                result = need("record_toggle")()
            elif path == "/reset":
                body = self._read_json()
                result = need("reset_to_episode")(int(body.get("episode", 0)))
            elif path == "/seek":
                # replay_dashboard.py's ReplayLoop only — manual scrub. Same
                # need()-dispatch/409 contract as everything else here; a
                # live DeployRunner simply has no `.seek`.
                body = self._read_json()
                result = need("seek")(float(body.get("t", 0.0)))
            else:
                self._send(404, "text/plain", b"not found")
                return
        except Exception as e:  # a bad request must not kill the server thread
            self._send(409, "application/json",
                       json.dumps({"error": str(e)}).encode("utf-8"))
            return
        self._send(200, "application/json", json.dumps(result).encode("utf-8"))

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-response; nothing to do


class Dashboard:
    """Serves the live page. `loop` needs only `.telemetry()` and, for the
    camera pane, a `.camera` (or `.cam`) with `.read()`."""

    def __init__(self, loop, *, port: int = 8080, frame_width: int = 360):
        self.loop = loop
        self.port = int(port)
        self.frame_width = int(frame_width)
        self._server = None
        self._thread = None

    def encode_frame(self):
        """Latest camera frame as JPEG bytes, or None. Uses the same public
        `read()` the loop already calls (returns a copy), then cv2 to encode.

        A camera exposing `read_pair()` (umi_eef's `CameraPair`) is rendered as
        both views SIDE BY SIDE, captioned, rather than just the acting one.
        The captions are not decoration: this mode's two cameras go to
        different model input slots and swapping them produces a rollout that
        looks plausible and is wrong, so which-is-which has to be readable at a
        glance on the page.
        """
        cam = getattr(self.loop, "camera", None) or getattr(self.loop, "cam", None)
        if cam is None:
            return None
        try:
            import cv2
        except Exception:
            return None

        pair = getattr(cam, "read_pair", None)
        frames = pair() if pair is not None else (cam.read(),)
        labels = getattr(cam, "labels", ()) if pair is not None else ()
        if any(f is None for f in frames):
            return None

        tiles = []
        for i, frame in enumerate(frames):
            img = np.ascontiguousarray(frame)
            h, w = img.shape[:2]
            if self.frame_width and w > self.frame_width:
                scale = self.frame_width / float(w)
                img = cv2.resize(img, (self.frame_width, max(1, int(round(h * scale)))),
                                 interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # camera hands out RGB
            if i < len(labels):
                bgr = bgr.copy()
                text = str(labels[i])
                cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 18), (0, 0, 0), -1)
                cv2.putText(bgr, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(bgr)

        if len(tiles) > 1:
            # pad to a common height before hstack — the two cameras need not
            # be the same model or resolution
            th = max(t.shape[0] for t in tiles)
            padded = []
            for t in tiles:
                if t.shape[0] != th:
                    pad = np.zeros((th - t.shape[0], t.shape[1], 3), t.dtype)
                    t = np.vstack([t, pad])
                padded.append(t)
                padded.append(np.zeros((th, 2, 3), t.dtype))   # divider
            out = np.hstack(padded[:-1])
        else:
            out = tiles[0]

        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def encode_perception_frame(self):
        """`relation_eef`-only: the last detector-cadence frame (rgb_left)
        annotated with DINO+SAM2 boxes/masks, the live Kalman/OneEuro fast-
        tracker position per object (NOT CoTracker -- this repo's live
        pipeline doesn't use it, see `perception/tracker.py`'s docstring),
        while a hand is CANDIDATE/LATCHED the rigid-predicted pose `latch.py`
        is comparing it against, AND each hand's FK wrist position projected
        through the same `T_pelvis_camera`/`K_left` calibration (refreshed
        EVERY tick, not gated to detector cadence like the boxes/masks) --
        a hand-eye-calibration sanity check independent of the detector: if
        this marker doesn't sit on the visibly-real wrist, the extrinsics
        (or K_left) are wrong, not the detection/tracking cascade. JPEG
        bytes, or None if relation_eef perception hasn't produced a frame
        yet. All drawing happens HERE, on the dashboard's own HTTP thread,
        on each request --
        same hot-path isolation guarantee as `encode_frame()`."""
        perception = getattr(getattr(self.loop, "adapter", None), "perception", None)
        if perception is None or perception.last_rgb_left is None:
            return None
        try:
            import cv2
        except Exception:
            return None
        from .overlay import draw_perception_overlay

        # The renderer consumes the RECORDED `percept` shape (debug_snapshot)
        # — the same call the recorder logs — so this view and the replay
        # dashboard's are one code path. Raw detector masks are passed
        # alongside to skip a pointless PNG encode/decode round-trip live.
        snapshot = perception.debug_snapshot(include_masks=False)
        masks = {oid: det.mask
                 for oid, det in perception.last_detections.items()
                 if det.mask is not None}
        img = draw_perception_overlay(
            perception.last_rgb_left, snapshot,
            perception.calib.K_left, perception.T_pelvis_camera,
            flange_poses=getattr(perception, "last_flange_poses", None),
            masks=masks)

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def start(self):
        self._server = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
        self.port = self._server.server_address[1]   # resolves port=0 for tests
        self._server.dash = self
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="dashboard", daemon=True)
        self._thread.start()
        logger.info("dashboard → http://localhost:%d", self.port)

    def stop(self):
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None


# --- demo mode: synthetic telemetry so the page can be verified with no robot -----

class _DemoCam:
    def read(self):
        t = time.time()
        h, w = 200, 300
        gx, gy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        r = (0.5 + 0.5 * np.sin(2 * np.pi * (gx + 0.10 * t))) * 255
        g = (0.5 + 0.5 * np.sin(2 * np.pi * (gy + 0.13 * t))) * 255
        b = (0.5 + 0.5 * np.sin(2 * np.pi * (gx + gy + 0.07 * t))) * 255
        return np.stack([r, g, b], -1).astype(np.uint8)

    def age(self):
        return 0.02


class _DemoLoop:
    """Mimics DeployRunner.telemetry() with a believable sync cycle: a chunk
    executes at fps, then a short inference gap, repeat."""

    H = 50
    FPS = 30
    INFER_S = 0.45

    def __init__(self):
        self.camera = _DemoCam()
        self._t0 = time.monotonic()
        self._active = True
        self._recording = False
        self._tripped = False

    # Stub controls so all the dashboard buttons work in --demo.
    def begin(self): self._active = True
    def pause(self): self._active = False
    def estop(self, reason="demo"): self._tripped = True
    def reset_to_episode(self, ep, **kw): return {"episode": ep}
    def record_toggle(self):
        self._recording = not self._recording
        return {"recording": self._recording, "dir": "demo/session"}

    def telemetry(self):
        from .. import actions as _actions
        from .telemetry import TelemetrySnapshot

        now = time.monotonic()
        H, fps = self.H, self.FPS
        exec_s = H / fps
        period = exec_s + self.INFER_S
        elapsed = now - self._t0
        cyc = elapsed % period
        inferring = cyc > exec_s
        index = int(min(H, cyc * fps))
        chunks = int(elapsed // period) + 1

        k = elapsed * 3.0
        row = np.zeros(_actions.ROBOT_DIM)
        row[:_actions.ARM_DOF] = 0.8 * np.sin(
            np.linspace(0.0, 6.0, _actions.ARM_DOF) + k)
        row[_actions.ARM_DOF:] = 0.5 + 0.5 * np.sin(
            np.linspace(0.0, 3.0, _actions.ROBOT_DIM - _actions.ARM_DOF) + 0.7 * k)

        return TelemetrySnapshot(
            now=now, mode="sync",
            active=self._active and not self._tripped,
            recording=self._recording,
            task="DEMO — synthetic data, no robot",
            horizon=H, fps=fps,
            ready=True, index=index, wall_slot=index,
            trigger=H, d=12,
            action_row=row.tolist(),
            row_slot=max(0, index - 1),
            inferring=inferring,
            stats={"ticks": int(elapsed * fps), "chunks": chunks, "votes": None},
            budget={"d": 12, "violations": 0, "saturated": 0, "n": chunks,
                    "mean_ms": float(300.0 + 30 * np.sin(elapsed * 0.2)),
                    "p95_ms": 380.0},
            runway_s=(H - index) / fps, camera_age=0.02,
            watchdog={"tripped": self._tripped,
                      "reason": "dashboard" if self._tripped else None},
            arm_q=row[:14].tolist(), state_age=0.003,
            estopped=self._tripped,
        ).to_json()


def _run_demo(port: int):
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")
    dash = Dashboard(_DemoLoop(), port=port)
    dash.start()
    logger.info("demo running. ctrl-C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        dash.stop()


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ego2g1 deploy monitor</title>
<style>
:root{
  --bg:#0d0d0d; --surface:#1a1a19; --surface2:#232320;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --hair:#2c2c2a;
  --executed:#6b6a64; --committed:#199e70; --queued:#3987e5;
  --robot:#ffffff; --cursor:#c98500; --trigger:#d95926;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --eef-l:#3987e5; --hand-l:#199e70; --eef-r:#9085e9; --hand-r:#d95926;
}
@media (prefers-color-scheme: light){
  :root{ --bg:#f9f9f7; --surface:#fcfcfb; --surface2:#f0efec;
    --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --hair:#e1e0d9;
    --executed:#b7b6ae; --committed:#1baf7a; --queued:#2a78d6; --robot:#0b0b0b;
    --cursor:#eda100; --trigger:#eb6834; --eef-r:#4a3aa7; --hand-r:#eb6834; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:18px}
h1{font-size:15px;font-weight:600;margin:0;letter-spacing:.02em}
.sub{color:var(--muted);font-size:12px;margin-top:2px}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:10px;
  padding:14px 16px;margin-top:14px}
.row{display:flex;gap:14px;flex-wrap:wrap}
.row>.card{margin-top:0}
.head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
.badge{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
  padding:3px 9px;border-radius:999px;border:1px solid var(--hair);color:var(--ink2)}
/* 728px = two 360px tiles + the 2px divider, so umi_eef's side-by-side pair
   renders at full size; a single-camera mode is still capped at its own
   360px width by the source image and never stretches. */
#cam{display:block;width:auto;max-width:min(728px,100%);border-radius:8px;background:var(--surface2);
  aspect-ratio:3/2;object-fit:cover}
.camwrap{flex:0 0 auto}
.status{flex:1 1 240px;min-width:240px}
.light{display:flex;align-items:center;gap:10px;font-weight:600;font-size:15px}
.dot{width:14px;height:14px;border-radius:50%;background:var(--muted);flex:0 0 auto}
.dot.on{background:var(--good);box-shadow:0 0 0 0 rgba(12,163,12,.5);animation:pulse 1s infinite}
.dot.pend{background:var(--warn);animation:none}
.dot.trip{background:var(--crit);animation:none}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(12,163,12,.5)}70%{box-shadow:0 0 0 9px rgba(12,163,12,0)}100%{box-shadow:0 0 0 0 rgba(12,163,12,0)}}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;margin-top:12px;
  font-variant-numeric:tabular-nums}
.kv .k{color:var(--muted)}
.kv .v{text-align:right;color:var(--ink2)}
canvas{display:block;width:100%}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin-top:10px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.line{width:14px;height:0;border-top-width:2px;border-top-style:solid;display:inline-block}
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;
  font-variant-numeric:tabular-nums}
.stat{background:var(--surface2);border-radius:8px;padding:9px 11px}
.stat .n{font-size:19px;font-weight:600}
.stat .l{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.title{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;
  margin:0 0 10px}
.warnbar{color:var(--crit);font-weight:600;margin-top:8px;display:none}
.ctrl{margin-top:14px}
.ctrlrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.statelight{display:flex;align-items:center;gap:8px;font-weight:600;margin-right:6px}
.btn{font:inherit;font-weight:600;padding:7px 14px;border-radius:8px;cursor:pointer;
  border:1px solid var(--hair);background:var(--surface2);color:var(--ink);}
.btn:hover{border-color:var(--muted)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn.primary{background:var(--good);border-color:var(--good);color:#fff}
.btn.danger{background:var(--crit);border-color:var(--crit);color:#fff;margin-left:auto}
.btn.rec.on{background:var(--crit);border-color:var(--crit);color:#fff;
  animation:pulse 1.2s infinite}
.epin{width:64px;font:inherit;padding:6px 8px;border-radius:8px;
  border:1px solid var(--hair);background:var(--surface2);color:var(--ink);
  font-variant-numeric:tabular-nums}
.scrub{display:flex;align-items:center;gap:10px;width:100%}
.scrub input[type=range]{flex:1 1 auto}
.scrubtime{font-variant-numeric:tabular-nums;color:var(--ink2);white-space:nowrap}
.objrow{display:flex;justify-content:space-between;gap:10px;padding:4px 0;
  border-bottom:1px solid var(--hair);font-variant-numeric:tabular-nums}
.objrow:last-child{border-bottom:none}
#percep{display:block;width:100%;max-width:360px;border-radius:8px;
  background:var(--surface2);aspect-ratio:3/2;object-fit:cover}
</style></head>
<body><div class="wrap">
  <div class="head">
    <div><h1>ego2g1 deploy monitor</h1><div class="sub" id="task">connecting…</div></div>
    <div><span class="badge" id="mode">—</span> <span class="badge" id="cfg">—</span></div>
  </div>

  <div class="card ctrl">
    <div class="ctrlrow">
      <span class="statelight"><span class="dot" id="runlight"></span><span id="runstate">—</span></span>
      <input id="ep" class="epin" type="number" min="0" value="0" title="episode index">
      <button class="btn" id="b-reset">Reset</button>
      <button class="btn primary" id="b-start">Start</button>
      <button class="btn" id="b-pause">Pause</button>
      <button class="btn rec" id="b-record">● Record</button>
      <button class="btn danger" id="b-estop">E-STOP</button>
    </div>
    <div class="scrub" id="scrubRow" style="display:none;margin-top:12px">
      <input type="range" id="scrub" min="0" max="1" step="0.01" value="0">
      <span class="scrubtime" id="scrubTime">0.0 / 0.0s</span>
    </div>
    <div class="sub" id="ctrlmsg" style="margin-top:8px"></div>
  </div>

  <div class="row" style="margin-top:14px">
    <div class="card camwrap"><div class="title">camera (model input)</div>
      <img id="cam" alt="camera"></div>
    <div class="card status"><div class="title">inference</div>
      <div class="light"><span class="dot" id="dot"></span><span id="lstate">—</span></div>
      <div class="kv">
        <div class="k">infer p95</div><div class="v" id="p95">—</div>
        <div class="k">delay budget d</div><div class="v" id="dbud">—</div>
        <div class="k">infer mean</div><div class="v" id="lat">—</div>
        <div class="k">samples</div><div class="v" id="samp">—</div>
      </div>
      <div class="warnbar" id="wd"></div>
    </div>
  </div>

  <div class="card"><div class="title">action chunk &nbsp;·&nbsp; <span id="barsub">—</span></div>
    <canvas id="bar" height="96"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:var(--executed)"></i>executed</span>
      <span><i class="sw" style="background:var(--queued)"></i>queued</span>
      <span><i class="line" style="border-color:var(--cursor)"></i>pointer (index)</span>
      <span><i class="line" style="border-color:var(--trigger);border-top-style:dashed"></i>replan trigger</span>
    </div>
  </div>

  <div class="card"><div class="title">current action &nbsp;·&nbsp; <span id="stripsub">—</span></div>
    <canvas id="strip" height="88"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:var(--eef-l)"></i>L arm</span>
      <span><i class="sw" style="background:var(--hand-l)"></i>L hand</span>
      <span><i class="sw" style="background:var(--eef-r)"></i>R arm</span>
      <span><i class="sw" style="background:var(--hand-r)"></i>R hand</span>
    </div>
  </div>

  <div id="relationCard" style="display:none">
    <div class="row" style="margin-top:14px">
      <div class="card camwrap"><div class="title">perception &nbsp;·&nbsp; DINO+SAM2 detection &amp; fast tracker</div>
        <img id="percep" alt="perception overlay">
        <div class="legend">
          <span><i class="sw" style="background:#fff;border:1px solid var(--hair)"></i>tracked object</span>
          <span><i class="sw" style="background:rgb(255,90,0)"></i>latch predicted</span>
          <span><i class="sw" style="background:rgb(0,230,230)"></i>FK wrist (calib check)</span>
        </div>
      </div>
      <div class="card status"><div class="title">objects</div>
        <div id="objList"></div>
      </div>
    </div>
    <div class="card"><div class="title">grasp / kinematic latch</div>
      <div class="grid4" id="handList"></div>
    </div>
    <div class="card"><div class="title">latch &amp; hand-closed timeline &nbsp;·&nbsp; last 30s</div>
      <canvas id="latchTimeline" height="70"></canvas>
      <div class="legend">
        <span><i class="sw" style="background:var(--cursor)"></i>hand closed</span>
        <span><i class="sw" style="background:var(--warn)"></i>candidate</span>
        <span><i class="sw" style="background:var(--committed)"></i>latched</span>
      </div>
    </div>
  </div>
  <div id="umiCard" style="display:none">
    <div class="row" style="margin-top:14px">
      <div class="card status"><div class="title">state history &nbsp;·&nbsp; injected prompt tokens</div>
        <div class="grid4">
          <div class="stat"><div class="n" id="u-hist">—</div><div class="l">lags sent / total</div></div>
          <div class="stat"><div class="n" id="u-lags">—</div><div class="l">lag ticks</div></div>
        </div>
        <div class="sub" style="margin-top:6px">short = the loop is dropping ticks past the
          buffer tolerance; expected only just after Start</div>
      </div>
      <div class="card status"><div class="title">gripper &nbsp;·&nbsp; commanded vs measured</div>
        <div class="grid4">
          <div class="stat"><div class="n" id="u-gcmd">—</div><div class="l">commanded (rad)</div></div>
          <div class="stat"><div class="n" id="u-gmeas">—</div><div class="l">measured (rad)</div></div>
          <div class="stat"><div class="n" id="u-gdev">—</div><div class="l">gap</div></div>
          <div class="stat"><div class="n" id="u-gdevw">—</div><div class="l">worst gap</div></div>
        </div>
        <div class="sub" style="margin-top:6px">a gap while closing means the jaws met something;
          a gap near zero through a grasp means they closed on nothing</div>
      </div>
      <div class="card status"><div class="title">idle arm</div>
        <div class="grid4">
          <div class="stat"><div class="n" id="u-idle">—</div><div class="l">hold latched</div></div>
          <div class="stat"><div class="n" id="u-acting">—</div><div class="l">acting arm</div></div>
        </div>
      </div>
    </div>
  </div>
  <div class="sub" id="relationNA" style="margin-top:14px">
    per-mode panel: n/a (this run is not in relation_eef or umi_eef mode)</div>

  <div class="card"><div class="title">loop health</div>
    <div class="grid4">
      <div class="stat"><div class="n" id="s-chunks">—</div><div class="l">chunks</div></div>
      <div class="stat"><div class="n" id="s-ticks">—</div><div class="l">ticks</div></div>
      <div class="stat"><div class="n" id="s-runway">—</div><div class="l">runway (s)</div></div>
      <div class="stat"><div class="n" id="s-cam">—</div><div class="l">camera age (s)</div></div>
      <div class="stat"><div class="n" id="s-state">—</div><div class="l">state age (s)</div></div>
      <div class="stat"><div class="n" id="s-clamp">—</div><div class="l">clamped ticks</div></div>
      <div class="stat"><div class="n" id="s-mean">—</div><div class="l">infer mean (ms)</div></div>
      <div class="stat"><div class="n" id="s-viol">—</div><div class="l">budget violations</div></div>
    </div>
  </div>
  <div class="sub" id="conn" style="margin-top:10px">—</div>
</div>
<script>
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function fmt(x,d=2){return (x==null||isNaN(x))?"—":Number(x).toFixed(d);}

function fitCanvas(c){
  const dpr=window.devicePixelRatio||1, w=c.clientWidth, h=c.height;
  if(c.width!==Math.round(w*dpr)||c._dpr!==dpr){c.width=Math.round(w*dpr);c._dpr=dpr;}
  const g=c.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);return [g,w,h];
}

function drawBar(t){
  const c=$("bar");const [g,W,H]=fitCanvas(c);g.clearRect(0,0,W,H);
  const H0=t.horizon; if(!H0){return;}
  const padX=8, padTop=26, padBot=22, bw=(W-2*padX)/H0, barH=H-padTop-padBot;
  const idx=t.ready?t.index:0;
  for(let i=0;i<H0;i++){
    let col=css("--queued");
    if(t.ready){ if(i<idx) col=css("--executed"); }
    else col=css("--surface2");
    g.fillStyle=col;
    const x=padX+i*bw;
    roundRect(g,x+0.5,padTop,Math.max(1,bw-1.5),barH,2);g.fill();
  }
  // markers
  const mark=(slot,color,label,dash,above)=>{
    if(slot==null)return; const x=padX+slot*bw;
    g.strokeStyle=color;g.lineWidth=2;g.setLineDash(dash?[4,3]:[]);
    g.beginPath();g.moveTo(x,padTop-4);g.lineTo(x,padTop+barH+4);g.stroke();g.setLineDash([]);
    g.fillStyle=color;
    if(above!==false){g.beginPath();g.moveTo(x,padTop-4);g.lineTo(x-4,padTop-11);g.lineTo(x+4,padTop-11);g.closePath();g.fill();}
    g.font="10px system-ui";g.textAlign="center";
    g.fillText(label,Math.min(W-16,Math.max(16,x)),above===false?padTop+barH+16:padTop-14);
  };
  if(t.ready){
    mark(t.trigger,css("--trigger"),"replan",true,false);
    mark(idx,css("--cursor"),"index "+idx,false,true);
  }
}
function roundRect(g,x,y,w,h,r){r=Math.min(r,w/2,h/2);g.beginPath();
  g.moveTo(x+r,y);g.arcTo(x+w,y,x+w,y+h,r);g.arcTo(x+w,y+h,x,y+h,r);
  g.arcTo(x,y+h,x,y,r);g.arcTo(x,y,x+w,y,r);g.closePath();}

function drawStrip(t){
  const c=$("strip");const [g,W,H]=fitCanvas(c);g.clearRect(0,0,W,H);
  const row=t.action_row, groups=t.groups||[]; if(!row){return;}
  const n=row.length, padX=8, mid=H/2, half=(H-20)/2;
  const gap=6, span=W-2*padX-gap*Math.max(0,groups.length-1), bw=span/n;
  const colFor=lbl=>lbl[0]==='L'?(lbl.includes('hand')?css('--hand-l'):css('--eef-l'))
                                :(lbl.includes('hand')?css('--hand-r'):css('--eef-r'));
  let x=padX;
  g.strokeStyle=css("--hair");g.lineWidth=1;g.beginPath();g.moveTo(padX,mid);g.lineTo(W-padX,mid);g.stroke();
  for(const grp of groups){
    const col=colFor(grp.label);
    for(let i=grp.start;i<grp.stop;i++){
      const v=row[i];
      // hands are [0,1] -> up from baseline; arm joints (rad) -> squash symmetric
      const isHand=grp.label.includes('hand');
      const y = isHand ? (v) : (0.5+0.5*Math.tanh(v));
      const h = (y-0.5)*2*half;
      g.fillStyle=col;
      const bx=x, by=h>=0?mid-h:mid, bh=Math.max(1,Math.abs(h));
      roundRect(g,bx+0.5,by,Math.max(1,bw-1.2),bh,1.5);g.fill();
      x+=bw;
    }
    // group label
    g.fillStyle=css("--muted");g.font="9px system-ui";g.textAlign="center";
    g.fillText(grp.label, x-(grp.stop-grp.start)*bw/2, H-3);
    x+=gap;
  }
}

let lastFrame=0;
function refreshCam(){
  const now=performance.now(); if(now-lastFrame<66)return; lastFrame=now;
  const img=$("cam"); const nx=new Image();
  nx.onload=()=>{img.src=nx.src;}; nx.src="/frame.jpg?t="+Date.now();
}

let lastPercep=0;
function refreshPercep(){
  // ~5Hz: plenty for a ~2Hz detector refresh + a moving tracker marker,
  // without paying mask-compositing cost (cv2.addWeighted) every /state poll.
  const now=performance.now(); if(now-lastPercep<200)return; lastPercep=now;
  const img=$("percep"); const nx=new Image();
  nx.onload=()=>{img.src=nx.src;}; nx.src="/perception.jpg?t="+Date.now();
}

function drawLatchTimeline(events, nowT){
  const c=$("latchTimeline");const [g,W,H]=fitCanvas(c);g.clearRect(0,0,W,H);
  events=events||[];
  const hands=[...new Set(events.map(e=>e.hand))];
  if(!hands.length){
    g.fillStyle=css("--muted");g.font="11px system-ui";g.textAlign="left";
    g.fillText("no latch/hand-closed events yet",8,H/2+4);
    return;
  }
  const windowS=30, padX=8, padTop=6, padBot=16;
  const laneH=(H-padTop-padBot)/hands.length;
  const xFor=t=>Math.max(padX, Math.min(W-padX,
              W-padX-((nowT-t)/windowS)*(W-2*padX)));
  const stateColor=s=>s==="latched"?css("--committed"):s==="candidate"?css("--warn"):null;
  hands.forEach((h,li)=>{
    const y=padTop+li*laneH;
    const evs=events.filter(e=>e.hand===h).sort((a,b)=>a.t-b.t);
    let closedSince=null, stateSince=null, curState=null;
    for(const e of evs){
      if(e.kind==="hand"){
        if(e.closed) closedSince=e.t;
        else if(closedSince!=null){
          g.fillStyle=css("--cursor");
          const x0=xFor(closedSince), x1=xFor(e.t);
          g.fillRect(x0,y+laneH*0.55,Math.max(1,x1-x0),laneH*0.35);
          closedSince=null;
        }
      } else {
        if(stateSince!=null){
          const col=stateColor(curState);
          if(col){g.fillStyle=col;
            const x0=xFor(stateSince), x1=xFor(e.t);
            g.fillRect(x0,y+laneH*0.05,Math.max(1,x1-x0),laneH*0.4);}
        }
        stateSince=e.t; curState=e.state;
      }
    }
    if(closedSince!=null){g.fillStyle=css("--cursor");
      const x0=xFor(closedSince);
      g.fillRect(x0,y+laneH*0.55,Math.max(1,W-padX-x0),laneH*0.35);}
    if(stateSince!=null){const col=stateColor(curState);
      if(col){g.fillStyle=col;const x0=xFor(stateSince);
        g.fillRect(x0,y+laneH*0.05,Math.max(1,W-padX-x0),laneH*0.4);}}
    g.fillStyle=css("--ink2");g.font="10px system-ui";g.textAlign="left";
    g.fillText(h,padX,y+laneH-3);
  });
}

function renderRelation(t){
  const rel=t.relation;
  if(!rel){
    $("relationCard").style.display="none";
    $("umiCard").style.display="none";
    $("relationNA").style.display="";
    return;
  }
  // Panels from recordings written before `kind` existed are all relation_eef.
  if(rel.kind==="umi"){ renderUmi(rel); return; }
  $("umiCard").style.display="none";
  $("relationCard").style.display="";
  $("relationNA").style.display="none";
  refreshPercep();

  const ol=$("objList"); ol.innerHTML="";
  for(const o of rel.objects){
    const mark=o.detected_this_tick?"●":(o.tracked?"○":"—");
    const row=document.createElement("div"); row.className="objrow";
    row.innerHTML=`<span>${mark} ${o.instance_id}</span>`+
      `<span>${o.depth_m!=null?fmt(o.depth_m,2)+" m":"—"}</span>`;
    ol.appendChild(row);
  }

  const hl=$("handList"); hl.innerHTML="";
  for(const h of rel.hands){
    const label = h.state==="latched" ? ("LATCHED → "+h.latched_object)
                : h.state==="candidate" ? ("candidate → "+h.candidate_object)
                : "unlatched";
    const div=document.createElement("div"); div.className="stat";
    div.innerHTML=`<div class="n" style="color:${h.hand_closed?css('--crit'):css('--good')}">`+
      `${h.hand_closed?"closed":"open"}</div>`+
      `<div class="l">${h.hand} · ${label}</div>`;
    hl.appendChild(div);
  }

  drawLatchTimeline(rel.events, t.now);
}

function renderUmi(u){
  $("relationCard").style.display="none";
  $("relationNA").style.display="none";
  $("umiCard").style.display="";
  const n=u.n_lags||0, k=u.history_len||0;
  const el=$("u-hist");
  el.textContent=k+" / "+n;
  // short history is normal for the first half-second after Start, and a
  // problem after that — colour it rather than making the operator watch a number
  el.style.color = (k>=n) ? css('--good') : (k>0 ? css('--warn') : css('--crit'));
  $("u-lags").textContent=(u.lag_ticks||[]).join(" ");
  $("u-gcmd").textContent = u.grip_cmd!=null ? fmt(u.grip_cmd,2) : "—";
  $("u-gmeas").textContent = u.grip_measured!=null ? fmt(u.grip_measured,2) : "—";
  $("u-gdev").textContent = u.grip_dev!=null ? fmt(u.grip_dev,2) : "—";
  $("u-gdevw").textContent = u.grip_dev_worst!=null ? fmt(u.grip_dev_worst,2) : "—";
  const idle=$("u-idle");
  idle.textContent = u.idle_latched ? ("held · "+u.idle_hand) : "NOT LATCHED";
  idle.style.color = u.idle_latched ? css('--good') : css('--crit');
  $("u-acting").textContent = u.acting_hand || "—";
}

let scrubDragging=false;
function renderReplay(t){
  const rep=t.replay;
  const row=$("scrubRow");
  if(!rep){ row.style.display="none"; return; }
  row.style.display="";
  $("b-start").textContent="Play"; $("b-pause").textContent="Pause";
  const sc=$("scrub");
  sc.max=Math.max(0.01, rep.duration);
  if(!scrubDragging) sc.value=rep.t;
  $("scrubTime").textContent=fmt(rep.t,1)+" / "+fmt(rep.duration,1)+"s"+(rep.playing?" ▶":" ⏸");
}

async function tick(){
  try{
    const t=await (await fetch("/state",{cache:"no-store"})).json();
    $("task").textContent=t.task||"";
    $("mode").textContent=t.mode+(t.server_rtc?" · rtc":"");
    $("cfg").textContent="H="+t.horizon+" · "+t.fps+"Hz · dim "+t.dim;
    $("stripsub").textContent=t.dim+" dims (joint space)";
    // inference light
    const dot=$("dot"); dot.className="dot";
    let st="idle";
    if(t.watchdog&&t.watchdog.tripped){dot.classList.add("trip");st="WATCHDOG TRIPPED";}
    else if(t.worker_dead){dot.classList.add("trip");st="INFERENCE WORKER DEAD";}
    else if(t.inferring){dot.classList.add("on");st="inferring — request in flight";}
    else if(t.pending){dot.classList.add("pend");st="pending — chunk awaiting splice";}
    $("lstate").textContent=st;
    const b=t.budget||{};
    $("p95").textContent=b.p95_ms!=null?fmt(b.p95_ms,0)+" ms":"—";
    $("lat").textContent=b.mean_ms!=null?fmt(b.mean_ms,0)+" ms":"—";
    $("dbud").textContent=t.d==null?"—":t.d+" ticks";
    $("samp").textContent=b.n??"—";
    const wd=$("wd");
    if(t.watchdog&&t.watchdog.tripped){wd.style.display="block";wd.textContent="⚠ "+(t.watchdog.reason||"tripped");}
    else wd.style.display="none";
    // bar
    $("barsub").textContent=t.ready?("index "+t.index+" / "+t.horizon+(t.d!=null?"  ·  d="+t.d:"")):"waiting for first chunk…";
    drawBar(t); drawStrip(t);
    renderRelation(t);
    renderReplay(t);
    // health
    const s=t.stats||{};
    $("s-chunks").textContent=s.chunks??"—"; $("s-ticks").textContent=s.ticks??"—";
    $("s-runway").textContent=fmt(t.runway_s,3); $("s-cam").textContent=fmt(t.camera_age,2);
    $("s-state").textContent=fmt(t.state_age,3);
    $("s-clamp").textContent=t.clamped_ticks??"—";
    $("s-mean").textContent=b.mean_ms!=null?fmt(b.mean_ms,0):"—";
    $("s-viol").textContent=b.violations??"—";
    // control bar state
    const rl=$("runlight"); rl.className="dot";
    let rs="idle — holding pose";
    if(t.watchdog&&t.watchdog.tripped){rl.classList.add("trip");rs="TRIPPED";}
    else if(t.active){rl.classList.add("on");rs="running";}
    $("runstate").textContent=rs;
    $("b-start").disabled=t.active||(t.watchdog&&t.watchdog.tripped);
    $("b-pause").disabled=!t.active;
    $("b-reset").disabled=t.active||!t.has_dataset;
    $("b-record").classList.toggle("on",!!t.recording);
    $("b-record").textContent=t.recording?"■ Recording":"● Record";
    $("conn").textContent="connected · "+new Date().toLocaleTimeString();
  }catch(e){ $("conn").textContent="disconnected — is the deploy running? ("+e+")"; }
}

async function post(path,body){
  const msg=$("ctrlmsg");
  try{
    const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},
      body:body?JSON.stringify(body):null});
    const j=await r.json().catch(()=>({}));
    if(!r.ok){msg.textContent="⚠ "+(j.error||r.status);return;}
    msg.textContent=path.slice(1)+": "+JSON.stringify(j);
    tick();
  }catch(e){msg.textContent="⚠ "+e;}
}
$("b-start").onclick=()=>post("/start");
$("b-pause").onclick=()=>post("/pause");
$("b-record").onclick=()=>post("/record");
$("b-estop").onclick=()=>{if(confirm("E-STOP: damp the robot?"))post("/estop");};
$("b-reset").onclick=()=>post("/reset",{episode:parseInt($("ep").value||"0",10)});
$("scrub").addEventListener("mousedown",()=>scrubDragging=true);
$("scrub").addEventListener("touchstart",()=>scrubDragging=true);
$("scrub").addEventListener("input",()=>{
  $("scrubTime").textContent=fmt(parseFloat($("scrub").value),1)+" / "+fmt(parseFloat($("scrub").max),1)+"s";
});
$("scrub").addEventListener("change",async()=>{
  await post("/seek",{t:parseFloat($("scrub").value)});
  scrubDragging=false;
});

setInterval(tick,100);
setInterval(refreshCam,66);
tick();
</script>
</body></html>
"""


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="ego2g1 deploy dashboard")
    p.add_argument("--demo", action="store_true",
                   help="serve synthetic data (no robot / no policy server)")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    if not args.demo:
        p.error("run the dashboard via `python -m ego2g1.deploy.runner --dashboard`; "
                "this entrypoint only supports --demo")
    _run_demo(args.port)
