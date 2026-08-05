"""Replay a recorded deploy session in MuJoCo, on the SAME G1 model that
generated the training labels and ran the IK (kin/g1.py + assets/unitree_g1).

Ported from the old deploy's replay_mujoco.py (third_party/openpi/ego2g1/
deploy) to the NEW recorder schema (deploy/recorder.py) and the v2 kinematics
backend — the old vendored-sim machinery (`vendor_g1_sim.py`) is gone because
this repo owns its assets.

It renders the robot at its MEASURED joints (the proprioception the robot
reported, logged per tick in the `obs` events) and overlays two markers per
hand:

    * RED sphere   — the flange pose the MODEL asked for: the per-slot flange
                     target the relative_eef converter logged with each
                     `infer_result` (post-One-Euro — the pose the IK was
                     actually judged against). "Where the policy wanted the
                     hand."
    * GREEN sphere — the flange the IK actually COMMANDED: FK of the joint row
                     that was executing (the logged `action` events, post-
                     clamp). "Where the solver got to."

So you can see the two error terms a tracking-error trip is made of:

    RED  -> GREEN   = IK tracking error (target unreachable / frames wrong —
                      the gap safety.py:check_tracking watches)
    GREEN -> robot  = servo lag (commanded vs measured; the robot trailing
                      its command)

Sessions recorded before arm_q entered the `obs` events fall back to rendering
the COMMANDED joints as the body (no servo-lag term); joint-mode sessions have
no flange targets, so no RED markers.

Usage (needs a display). On macOS the interactive viewer MUST run under
`mjpython` (it owns the main-thread GUI loop); `mjpython` ships with the
mujoco wheel and is in your venv's bin/. On Linux plain `python` is fine.
`--check` needs no display and runs under either.

    mjpython -m ego2g1.deploy.replay_mujoco recordings/<session>              # macOS
    mjpython -m ego2g1.deploy.replay_mujoco recordings/<session> --speed 0.25 --at-worst
    python   -m ego2g1.deploy.replay_mujoco recordings/<session> --check 200  # headless

`--at-worst` jumps to the moment of largest recorded tracking error — usually
the frame that tripped the watchdog. Once the viewer is open it is fully
scrubbable from the keyboard:

    SPACE  pause / resume
    ← / →  scrub back / forward while paused; set play direction while running
    , / .  step one frame back / forward (pauses)
    ↑ / ↓  speed ×1.5 / ÷1.5
    R      restart from the beginning
    L      toggle looping
    HOME   jump to start        END  jump to the worst-error frame
"""

import argparse
import bisect
import time

import numpy as np

from ...core import layout
from .. import actions as _actions
from ..record.session_reader import Session

# Marker colours (RGBA).
_RED = np.array([0.90, 0.15, 0.15, 0.9])     # model target, right hand
_GREEN = np.array([0.15, 0.80, 0.30, 0.9])   # commanded IK flange, right hand
_ORANGE = np.array([0.95, 0.55, 0.10, 0.9])  # left-hand tint of the target
_CYAN = np.array([0.10, 0.70, 0.85, 0.9])    # left-hand tint of the commanded

# Opacity of the rendered robot body. Kept semi-transparent so the target /
# commanded marker spheres stay visible when they sit inside the links.
_BODY_ALPHA = 0.55


class _Replayer:
    """Reconstructs, per query time, everything MuJoCo needs to draw one frame."""

    def __init__(self, session_dir):
        import mujoco

        from ...kin import g1 as _g1

        self.mj = mujoco
        self.s = Session(session_dir)
        self.t0, self.t1 = self.s.span()

        self.model = mujoco.MjModel.from_xml_path(_g1.MODEL_XML)
        # Make the robot body semi-transparent. Links get their colour from
        # either a material or a direct geom rgba, so dim the alpha on both.
        self.model.geom_rgba[:, 3] = _BODY_ALPHA
        if self.model.nmat:
            self.model.mat_rgba[:, 3] = _BODY_ALPHA
        # Render backend holds the pose the viewer shows; a second backend does
        # commanded-FK without disturbing it — same split as Kinematics.
        self.render = _g1.G1Backend(model=self.model)
        self.fk = _g1.G1Backend(model=self.model)
        self.arm_adr = np.concatenate(
            [self.render.arm_qpos_adr["left"], self.render.arm_qpos_adr["right"]])
        self.base = self.render.base_pose()

        # Measured samples (obs events that carry arm_q), for nearest lookup.
        meas = [e for e in self.s._by_kind.get("obs", []) if "arm_q" in e]
        self._meas_t = np.array([e["t"] for e in meas])
        self._meas_q = (np.array([e["arm_q"] for e in meas])
                        if meas else np.empty((0, _actions.ARM_DOF)))

        # Per-chunk flange targets (relative_eef sessions; pelvis frame).
        infer = self.s._by_kind.get("infer_result", [])
        self._tgt_events = [e for e in infer if e.get("flange_targets")]
        self._tgt_t = [e["t"] for e in self._tgt_events]
        self._act_t = [e["t"] for e in self.s._by_kind.get("action", [])]

    # --- per-frame reconstruction ------------------------------------------

    def measured_q(self, t):
        if len(self._meas_t) == 0:
            return None
        i = int(np.searchsorted(self._meas_t, t))
        i = min(max(i - 1, 0), len(self._meas_t) - 1)
        return self._meas_q[i]

    def targets_at(self, t):
        """{hand: (3,) pelvis-frame target position} for the slot executing at
        t: the newest chunk's logged flange_targets, indexed by how many action
        rows were popped since that chunk landed (uniform across modes)."""
        i = bisect.bisect_right(self._tgt_t, t) - 1
        if i < 0:
            return {}
        e = self._tgt_events[i]
        popped = bisect.bisect_right(self._act_t, t) - bisect.bisect_right(
            self._act_t, e["t"])
        slot = max(0, popped - 1)   # the row EXECUTING (last popped), matching
        out = {}                    # the action_row the GREEN marker FKs
        for h, pos in e["flange_targets"].items():
            pos = np.asarray(pos, dtype=np.float64)
            if len(pos):
                out[h] = pos[int(np.clip(slot, 0, len(pos) - 1))]
        return out

    def _write(self, backend, q14):
        backend.data.qpos[:] = 0.0        # waist + legs pinned to 0, as in training
        backend.data.qpos[self.arm_adr] = q14
        self.mj.mj_forward(backend.model, backend.data)

    def frame(self, t):
        """A dict with the pose to render and the marker world-points."""
        snap = self.s.at(t)
        measured = self.measured_q(t)
        commanded = (np.asarray(snap["action_row"], dtype=np.float64)[_actions.ARM]
                     if snap["action_row"] is not None else None)

        out = {"t": t, "phase": snap["phase"], "ready": snap["ready"],
               "measured_q": measured, "commanded_q": commanded,
               "targets": {}, "commanded_flange": {}, "tracking": {}}

        # Model target flange (pelvis frame -> world), for the executing slot.
        for h, pos in self.targets_at(t).items():
            out["targets"][h] = (self.base @ np.append(pos, 1.0))[:3]

        # Commanded IK flange: FK of the commanded joints.
        if commanded is not None:
            self._write(self.fk, commanded)
            for h in layout.HANDS:
                out["commanded_flange"][h] = self.fk.flange_pose(h)[:3, 3].copy()

        # The IK tracking error we can recompute here (target vs commanded).
        for h in layout.HANDS:
            if h in out["targets"] and h in out["commanded_flange"]:
                out["tracking"][h] = float(np.linalg.norm(
                    out["targets"][h] - out["commanded_flange"][h]))
        return out

    def worst_tracking_time(self):
        """Monotonic time of the largest recorded IK tracking error (the
        per-chunk `tracking` events) — usually the frame that tripped."""
        worst_t, worst = self.t0, -1.0
        for e in self.s._by_kind.get("tracking", []):
            m = float(e.get("worst_m") or 0.0)
            if m > worst:
                worst, worst_t = m, e["t"]
        return worst_t, worst


def _add_sphere(scn, pos, rgba, size=0.02):
    import mujoco

    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, size, size], dtype=np.float64),
                        np.asarray(pos, dtype=np.float64),
                        np.eye(3).flatten(), rgba.astype(np.float32))
    scn.ngeom += 1


def _draw_markers(scn, fr):
    scn.ngeom = 0
    for h, warm, cool in (("left", _ORANGE, _CYAN), ("right", _RED, _GREEN)):
        if h in fr["targets"]:
            _add_sphere(scn, fr["targets"][h], warm, size=0.025)
        if h in fr["commanded_flange"]:
            _add_sphere(scn, fr["commanded_flange"][h], cool, size=0.018)


# GLFW key codes (mujoco's viewer hands these to key_callback).
_K_SPACE, _K_RIGHT, _K_LEFT, _K_UP, _K_DOWN = 32, 262, 263, 265, 264
_K_HOME, _K_END, _K_PERIOD, _K_COMMA = 268, 269, 46, 44
_K_R, _K_L = 82, 76


class _Playback:
    """Mutable playback head, driven by the keyboard. Simple float/bool fields
    shared with the viewer's key-callback thread — a stale read for one frame
    is harmless."""

    def __init__(self, lo, hi, rel0, speed, worst_rel, loop):
        # lo/hi are the scrub bounds (the whole recording); rel0 is just where
        # playback BEGINS. --at-worst sets rel0, not a floor, so R/HOME still
        # reach the start.
        self.rel = rel0
        self.start = lo
        self.end = hi
        self.speed = speed
        self.dir = 1            # +1 forward, -1 reverse
        self.paused = False
        self.loop = loop
        self.worst = worst_rel
        self.step = 1.0 / 30.0

    def key(self, keycode):
        if keycode == _K_SPACE:
            self.paused = not self.paused
        elif keycode == _K_RIGHT:
            if self.paused: self.rel = min(self.end, self.rel + self.step)
            else: self.dir = 1
        elif keycode == _K_LEFT:
            if self.paused: self.rel = max(self.start, self.rel - self.step)
            else: self.dir = -1
        elif keycode == _K_PERIOD:                 # frame step forward (pauses)
            self.paused = True; self.rel = min(self.end, self.rel + self.step)
        elif keycode == _K_COMMA:                  # frame step back (pauses)
            self.paused = True; self.rel = max(self.start, self.rel - self.step)
        elif keycode == _K_UP:
            self.speed = min(16.0, self.speed * 1.5)
        elif keycode == _K_DOWN:
            self.speed = max(0.05, self.speed / 1.5)
        elif keycode == _K_R:                      # restart
            self.rel = self.start; self.paused = False; self.dir = 1
        elif keycode == _K_L:
            self.loop = not self.loop
        elif keycode == _K_HOME:
            self.rel = self.start
        elif keycode == _K_END and self.worst is not None:
            self.rel = self.worst; self.paused = True


_CONTROLS = (
    "controls:  SPACE pause/resume   ←/→ scrub (or set direction)   ,/. frame step\n"
    "           ↑/↓ speed ×/÷1.5      R restart   L loop   HOME start   END worst-error"
)


def run(session_dir, *, speed=1.0, start=0.0, end=None, at_worst=False,
        loop=False):
    import mujoco
    import mujoco.viewer

    r = _Replayer(session_dir)
    dur = r.t1 - r.t0
    # Scrub bounds cover the WHOLE recording; --start/--end/--at-worst only
    # choose where playback begins, so R/HOME always reach the true start.
    lo = 0.0
    hi = dur if end is None else min(end, dur)
    wt, wv = r.worst_tracking_time()
    worst_rel = wt - r.t0
    rel0 = start
    if at_worst:
        rel0 = max(0.0, worst_rel - 1.0)
        print(f"worst IK tracking error {wv*1000:.0f} mm at {worst_rel:.2f}s; "
              f"opening 1s before (R or HOME jumps to the recording start).")

    body = "measured joints" if len(r._meas_t) else \
        "COMMANDED joints (this session predates arm_q in obs events)"
    print(f"session {r.s.dir.name}: {dur:.1f}s | body={body} | "
          f"RED/orange=model target, GREEN/cyan=IK-commanded flange")
    print(_CONTROLS)

    pb = _Playback(lo, hi, rel0, speed, worst_rel, loop)

    try:
        viewer_cm = mujoco.viewer.launch_passive(
            r.render.model, r.render.data, key_callback=pb.key)
    except RuntimeError as e:
        if "mjpython" in str(e):
            raise SystemExit(
                "\nmacOS needs the MuJoCo viewer to run under `mjpython`, not "
                "`python`. Re-run with:\n"
                f"  mjpython -m ego2g1.deploy.replay_mujoco {session_dir} "
                f"--speed {speed}"
                + (" --at-worst" if at_worst else "")
                + "\n(mjpython ships with the mujoco package — it's in your "
                "venv's bin/. Or use --check N for a headless reconstruction.)"
            ) from None
        raise

    with viewer_cm as v:
        last = time.monotonic()
        last_status = 0.0
        while v.is_running():
            now = time.monotonic()
            dt, last = now - last, now

            if not pb.paused:
                pb.rel += dt * pb.speed * pb.dir
            if pb.rel >= pb.end:
                pb.rel = pb.start if pb.loop else pb.end
                if not pb.loop: pb.paused = True
            elif pb.rel <= pb.start:
                pb.rel = pb.end if pb.loop else pb.start

            fr = r.frame(r.t0 + pb.rel)
            q_body = fr["measured_q"] if fr["measured_q"] is not None \
                else fr["commanded_q"]
            if q_body is not None:
                r._write(r.render, q_body)
            _draw_markers(v.user_scn, fr)
            v.sync()

            if now - last_status > 0.25:                    # throttled status
                worst = max(fr["tracking"].values()) if fr["tracking"] else 0.0
                flag = "  <-- OVER 100mm" if worst > 0.10 else ""
                print(f"\r  t={pb.rel:6.2f}/{pb.end:.1f}s  {pb.speed:4.2f}x  "
                      f"{'PAUSED' if pb.paused else ('<<' if pb.dir < 0 else '>>')}  "
                      f"{fr['phase']:9s}  IK {worst*1000:4.0f}mm{flag}   ",
                      end="", flush=True)
                last_status = now

            time.sleep(0.004)
    print("\ndone.")


def check(session_dir, n):
    """Headless self-test: reconstruct N evenly spaced frames, no viewer.
    Prints the tracking-error profile so the pipeline can be validated on a
    box with no display."""
    r = _Replayer(session_dir)
    grid = np.linspace(r.t0, r.t1, n)
    worst = 0.0
    worst_t = None
    n_ready = n_meas = 0
    for t in grid:
        fr = r.frame(t)
        if fr["ready"]:
            n_ready += 1
        if fr["measured_q"] is not None:
            n_meas += 1
        m = max(fr["tracking"].values()) if fr["tracking"] else 0.0
        if m > worst:
            worst, worst_t = m, t - r.t0
    wt, wv = r.worst_tracking_time()
    print(f"checked {n} frames over {r.t1 - r.t0:.1f}s: {n_ready} with an "
          f"active chunk, {n_meas} with a measured pose")
    if worst_t is not None:
        print(f"  reconstructed worst IK tracking (on-grid): {worst*1000:.0f} mm "
              f"at {worst_t:.2f}s")
    else:
        print("  no flange targets on grid (joint-mode or pre-flange_targets "
              "session) — GREEN markers only")
    print(f"  recorded worst IK tracking (tracking events): {wv*1000:.0f} mm "
          f"at {wt - r.t0:.2f}s" if wv >= 0 else
          "  no tracking events recorded")
    print("  reconstruction OK — run without --check (needs a display) to view.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="replay a recorded deploy session in MuJoCo")
    p.add_argument("session", help="path to a recordings/<...> directory")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier")
    p.add_argument("--start", type=float, default=0.0,
                   help="start time (s from session start)")
    p.add_argument("--end", type=float, default=None,
                   help="end time (s from session start)")
    p.add_argument("--at-worst", action="store_true",
                   help="jump to the largest tracking error (the likely trip point)")
    p.add_argument("--loop", action="store_true", help="loop the playback")
    p.add_argument("--check", type=int, default=None, metavar="N",
                   help="headless: reconstruct N frames and print stats, no viewer")
    args = p.parse_args()

    if args.check is not None:
        check(args.session, args.check)
    else:
        run(args.session, speed=args.speed, start=args.start, end=args.end,
            at_worst=args.at_worst, loop=args.loop)
