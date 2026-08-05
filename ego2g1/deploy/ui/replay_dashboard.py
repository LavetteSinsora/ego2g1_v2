"""Scrub a recorded `relation_eef` session through the SAME live dashboard
page `runner.py --dashboard` serves — no changes to `dashboard.py`'s
rendering needed. `ReplayLoop` below duck-types what that page reads:
`.telemetry()` (built through the same `ui.telemetry.TelemetrySnapshot` +
`relation_panel` every live producer uses), `.camera` (a plain
`.read()`/`.age()`), and `.adapter.perception` (the recorded `percept`
event dressed in the few attributes `Dashboard.encode_perception_frame`
reads — the overlay renderer itself consumes the recorded shape directly,
`ui/overlay.py`). Works on any recorded session with `percept` events
(`runner.py`'s relation_eef drain step, or `replay_relation_openloop.py`'s
Tier 2/3 rollouts).

    python -m ego2g1.deploy.replay_dashboard recordings/<session>
    python -m ego2g1.deploy.replay_dashboard recordings/<session> \\
        --stereo-calib stereo_calib.npz --camera-extrinsic camera_calib.npz

Without `--stereo-calib`/`--camera-extrinsic`, a placeholder calibration
(`perception_preview.py`'s own `_dummy_calibration()`) is used so the page
still renders — 2D boxes/masks stay pixel-accurate regardless, but the
projected 3D markers (tracked position, latch prediction, FK wrist) will NOT
land in the right place without the real calibration this session was
actually recorded under.

The existing Start/Pause buttons are REPURPOSED to mean "play/pause through
recorded time" (there is no robot to start/pause here) — a background thread
advances the scrub position at real time (`--speed` to go faster/slower). A
scrub slider (shown only when telemetry carries a `"replay"` key) POSTs to
`/seek` for manual scrubbing. Record/Reset/E-STOP are simply not implemented
on `ReplayLoop` — same "the button 409s, that's correct, this tool cannot
drive anything" contract `perception_preview.py`'s `_NoOpAdapter` already
established.
"""

from __future__ import annotations

import bisect
import threading
import time

import numpy as np

from .telemetry import TelemetrySnapshot, relation_panel


def _nearest_at_or_before(events: list[dict], t: float) -> dict | None:
    """The last event (by its own `t`) at or before `t`, or None."""
    ts = [e["t"] for e in events]
    i = bisect.bisect_right(ts, t) - 1
    return events[i] if i >= 0 else None


def _reconstruct_events(session) -> list[dict]:
    """Undo `DeployMode.record_tick`'s recorder re-labeling (`ev["kind"]` ->
    "hand_state"/"latch", original `t` stashed as `event_t`) back to
    `RelationPerception._events`'s own shape (`kind`: "hand"/"latch", `t`:
    the original tick time) — the exact shape `dashboard.py`'s
    `drawLatchTimeline` JS expects, since a live run feeds it
    `perception.recent_events()` directly."""
    out = []
    for e in session._by_kind.get("hand_state", []):
        out.append({k: v for k, v in e.items() if k not in ("t", "kind")}
                   | {"t": e["event_t"], "kind": "hand"})
    for e in session._by_kind.get("latch", []):
        out.append({k: v for k, v in e.items() if k not in ("t", "kind")}
                   | {"t": e["event_t"], "kind": "latch"})
    out.sort(key=lambda e: e["t"])
    return out


class _ReplayPerception:
    """One recorded `percept` payload, dressed in the few attributes
    `Dashboard.encode_perception_frame` reads off a live
    `RelationPerception`. The payload IS `debug_snapshot()`'s own shape, so
    there is nothing to reconstruct — the overlay renderer consumes it
    directly (masks decode from the recording's own PNG b64)."""

    def __init__(self, snapshot: dict | None, rgb_left: np.ndarray | None,
                 calib, T_pelvis_camera: np.ndarray,
                 flange_poses: dict[str, np.ndarray] | None,
                 events: list[dict]):
        self._snapshot = snapshot or {"objects": {}, "hands": {}}
        self.last_rgb_left = rgb_left
        self.calib = calib
        self.T_pelvis_camera = T_pelvis_camera
        self.last_flange_poses = flange_poses
        self.last_detections: dict = {}   # masks come from the snapshot b64
        self._events = events

    def debug_snapshot(self, *, include_masks: bool | None = None) -> dict:
        return self._snapshot

    def recent_events(self, since_t: float | None = None) -> list[dict]:
        if since_t is None:
            return list(self._events)
        return [e for e in self._events if e["t"] > since_t]


class _ReplayCamera:
    def __init__(self, loop: "ReplayLoop", cam: str):
        self._loop = loop
        self._cam = cam

    def read(self):
        return self._loop._frame_at_current(self._cam)

    def age(self) -> float:
        return 0.0


class _ReplayAdapter:
    """`.perception`/`.prompt` — the only attributes `dashboard.py` reads
    off `DeployRunner.adapter` for the relation_eef panels."""

    def __init__(self, loop: "ReplayLoop", prompt: str):
        self._loop = loop
        self.prompt = prompt

    @property
    def perception(self) -> _ReplayPerception:
        return self._loop._perception_at_current()


class ReplayLoop:
    """`Dashboard`-compatible stand-in over a recorded `Session`. Construct
    with the session directory; `Dashboard(ReplayLoop(...))` serves it on
    the same page a live `DeployRunner` would use.
    """

    def __init__(self, session_dir, *, calib=None, T_pelvis_camera=None,
                cam: str = "head", speed: float = 1.0):
        from ..record.session_reader import Session

        self.session = Session(session_dir)
        self.t0, self.t1 = self.session.span()
        if self.t0 is None:
            raise ValueError(f"{session_dir}: no events/frames — nothing to replay")
        self._t = self.t0
        self._cam = cam
        self._speed = float(speed)
        self._playing = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

        if calib is None or T_pelvis_camera is None:
            from .perception_preview import _dummy_calibration
            calib = calib if calib is not None else _dummy_calibration()
            T_pelvis_camera = (T_pelvis_camera if T_pelvis_camera is not None
                              else np.eye(4))
        self._calib = calib
        self._T_pelvis_camera = T_pelvis_camera

        self._kin = None   # lazy Kinematics(), only if a session has arm_q to FK

        self.camera = _ReplayCamera(self, cam)
        prompt = self.session.meta.get("prompt") or self.session.meta.get("task", "")
        self.adapter = _ReplayAdapter(self, prompt)

    # --- scrub/playback (repurposes the page's Start/Pause/seek controls) ---

    def begin(self) -> None:
        with self._lock:
            if self._playing:
                return
            self._playing = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._play_loop, daemon=True,
                                            name="replay-dashboard-play")
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def seek(self, t: float) -> dict:
        """`t`: seconds SINCE SESSION START (matches `replay_record.py`'s/
        `replay_mujoco.py`'s own `--at`/`--start` convention)."""
        with self._lock:
            self._t = float(np.clip(self.t0 + t, self.t0, self.t1))
        return {"t": self._t - self.t0}

    def _play_loop(self) -> None:
        last = time.monotonic()
        while True:
            time.sleep(0.03)
            with self._lock:
                if not self._playing:
                    last = time.monotonic()
                    continue
                now = time.monotonic()
                self._t = min(self.t1, self._t + (now - last) * self._speed)
                last = now
                if self._t >= self.t1:
                    self._playing = False

    # --- perception-shaped reads, at the CURRENT scrub position -------------

    def _frame_at_current(self, cam: str):
        with self._lock:
            t = self._t
        return self.session.read_frame(t, cam)

    def _perception_at_current(self) -> _ReplayPerception:
        with self._lock:
            t = self._t
        percept = self.session.percept_at(t)
        rgb_left = self.session.read_frame(t, self._cam)
        flange_poses = self._flange_poses_at(t)
        events = _reconstruct_events(self.session)
        return _ReplayPerception(percept, rgb_left, self._calib,
                                 self._T_pelvis_camera, flange_poses, events)

    def _flange_poses_at(self, t: float) -> dict[str, np.ndarray] | None:
        """FK of the recorded `arm_q` at `t` — recomputed here rather than
        recorded verbatim (it's a pure, cheap function of `arm_q`, which the
        `obs` event already carries; no reason to double the schema)."""
        obs = _nearest_at_or_before(self.session._by_kind.get("obs", []), t)
        if obs is None or "arm_q" not in obs:
            return None
        if self._kin is None:
            from ..core.kinematics import Kinematics
            self._kin = Kinematics()
        return self._kin.flange_poses(np.asarray(obs["arm_q"]))

    # --- Dashboard's telemetry contract ---------------------------------------

    def telemetry(self) -> dict:
        with self._lock:
            t = self._t
            playing = self._playing
        snap = self.session.at(t)
        obs = _nearest_at_or_before(self.session._by_kind.get("obs", []), t)
        estops = [e for e in self.session._by_kind.get("estop", []) if e["t"] <= t]

        ready = bool(snap["ready"])
        horizon = snap["horizon"] or 0
        index = snap["index"]
        percept = self.session.percept_at(t)
        rel = None
        if percept is not None:
            rel = relation_panel(percept, _reconstruct_events(self.session))

        return TelemetrySnapshot(
            now=time.monotonic(),
            mode=snap["mode"],
            active=playing,
            task=self.adapter.prompt,
            horizon=horizon, fps=self.session.fps,
            ready=ready, index=index,
            wall_slot=index if ready else None,
            action_row=snap["action_row"],
            row_slot=max(0, index - 1) if ready else None,
            inferring=snap["phase"] == "inferring",
            stats={"ticks": obs.get("step") if obs else None,
                   "chunks": len(self.session._infer), "votes": None},
            runway_s=(horizon - index) / self.session.fps if ready else None,
            camera_age=0.0,
            clamped_ticks=len([e for e in self.session._by_kind.get("clamp", [])
                               if e["t"] <= t]),
            watchdog={"tripped": bool(estops),
                      "reason": estops[-1].get("reason") if estops else None},
            arm_q=obs.get("arm_q") if obs else None,
            state_age=obs.get("state_age") if obs else None,
            estopped=bool(estops),
            relation=rel,
            replay={"t": t - self.t0, "duration": self.t1 - self.t0,
                    "playing": playing},
        ).to_json()


if __name__ == "__main__":
    import dataclasses as _dc
    import logging

    import tyro

    from .dashboard import Dashboard

    @_dc.dataclass
    class Args:
        session: str
        port: int = 8080
        cam: str = "head"
        speed: float = 1.0
        stereo_calib: str | None = None
        camera_extrinsic: str | None = None

    def main(args: Args) -> None:
        calib = None
        T_pelvis_camera = None
        if args.stereo_calib:
            from ..perception.depth import StereoCalibration
            calib = StereoCalibration.load(args.stereo_calib)
        if args.camera_extrinsic:
            T_pelvis_camera = np.load(args.camera_extrinsic)["T_pelvis_camera"]

        loop = ReplayLoop(args.session, calib=calib, T_pelvis_camera=T_pelvis_camera,
                          cam=args.cam, speed=args.speed)
        dash = Dashboard(loop, port=args.port)
        dash.start()
        print(f"replay dashboard -> http://localhost:{dash.port}  "
              f"({loop.t1 - loop.t0:.1f}s recorded)")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            dash.stop()

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
