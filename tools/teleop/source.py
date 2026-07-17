"""Where the hands come from. Two transports, one sample type.

Every source emits the SAME array the recordings hold: a (26, 7) hand pose per side,
rows [x y z qx qy qz qw] (quaternion scalar-LAST), in the raw OpenXR world frame --
i.e. exactly what `data_extraction/common/episode.py::load_episode` reads out of the
HDF5. That is deliberate and load-bearing: it means `HandRetargeter` and the wrist
algebra downstream cannot tell a live headset from a replayed episode, so the
offline-equivalence check (check.py rung 3) tests the REAL teleop code path rather
than a parallel one written to look like it.

The two transports:

  VuerSource  -- live. The PICO's browser opens a vuer page and streams WebXR hand
                 tracking back. NOTHING is installed on the headset.
  Hdf5Source  -- replays a recorded episode at wall-clock rate. Not a toy: it is what
                 rung 3 runs on, and it lets the whole system be built and debugged
                 with no headset and no robot.

WebXR gives 25 joints; OpenXR gives 26. The difference is OpenXR's extra `palm` joint
at index 0, and the remaining 25 are in the SAME order -- so `openxr_idx = webxr_idx + 1`
and row 0 is left as zeros. Nothing reads it: `XR_PALM` is referenced by no line of the
retargeter (it wants wrist=1, tips=5/10/15/20/25, knuckles=7/12/22), and the retargeter's
own validity test sums |pose| over ALL joints, so a zero row cannot make a frame look
invalid on its own.
"""

import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from ._vendor.de.common import frames
from ._vendor.de.common.episode import R_XR_TO_MJ

SIDES = ("left", "right")

# WebXR hand joints -> OpenXR-26 rows. OpenXR index 0 is `palm`, which WebXR does not
# have and we never read.
N_WEBXR_JOINTS = 25
WEBXR_TO_OPENXR = slice(1, 26)


@dataclass
class Sample:
    """One instant of both hands, in the recordings' own convention."""

    t_ns: int
    # {side: (26,7) float64}  [x y z qx qy qz qw], raw OpenXR world frame
    hand: dict[str, np.ndarray]
    # {side: bool} -- the tracker is actually seeing this hand right now
    active: dict[str, bool]

    def wrist_se3(self, side: str) -> np.ndarray:
        """(4,4) wrist pose in the MuJoCo world frame.

        The axis remap is, strictly, a no-op for teleoperation: the action is the
        relative delta T_w(t0)^-1 T_w(t), and a world-frame change left-multiplies
        both factors and cancels. We apply it anyway so that a wrist pose printed by
        `check.py` means the same thing it means everywhere else in this repo, and so
        that anything that ever does want an absolute pose gets a correct one.
        """
        p = self.hand[side][1, :3]
        # Normalize before use, as the offline resampler does. The recordings store
        # float32 quaternions, so |q| is off unity by ~1e-7 -- and `se3_inv` inverts by
        # TRANSPOSING, which is only the inverse of an orthonormal matrix. Skip this and
        # T(t0)^-1 T(t) stops cancelling exactly: the residual shows up as ~1e-6 deg of
        # phantom rotation, small enough to look like round-off and large enough not to be.
        q = frames.quat_normalize(frames.quat_wxyz_from_xyzw(self.hand[side][1, 3:7]))
        return frames.se3(p @ R_XR_TO_MJ.T,
                          frames.quat_mul(frames.quat_from_mat(R_XR_TO_MJ), q))


class HandSource(Protocol):
    def start(self) -> None: ...
    def latest(self) -> Sample | None: ...
    def age(self) -> float: ...
    def close(self) -> None: ...


def _pose26_from_webxr(positions: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """(25,3) + (25,3,3) -> (26,7) [xyz, qx qy qz qw], OpenXR row indexing."""
    out = np.zeros((26, 7), dtype=np.float64)
    out[WEBXR_TO_OPENXR, :3] = positions
    out[WEBXR_TO_OPENXR, 3:7] = Rotation.from_matrix(orientations).as_quat()  # xyzw
    return out


def _webxr_hand_ok(positions: np.ndarray, orientations: np.ndarray) -> bool:
    """Is the tracker actually seeing this hand?

    When a hand is not tracked, WebXR reports zeros rather than dropping the field, so
    the poses arrive looking structurally valid and numerically dead. televuer's own
    `safe_mat_update`/`safe_rot_update` guard the same way: a singular rotation is the
    tell. Check the WRIST specifically -- it is the joint the arm rides on, and a hand
    whose wrist is garbage must not move the robot no matter how good the fingers look.
    """
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(orientations)):
        return False
    if np.abs(positions).sum() == 0.0:
        return False
    det = float(np.linalg.det(orientations[0]))   # webxr joint 0 = wrist
    return np.isfinite(det) and abs(det - 1.0) < 1e-3


class VuerSource:
    """Live WebXR hand tracking off a PICO, via televuer.

    The headset runs no app: its browser opens the vuer page over HTTPS, enters an
    immersive WebXR session, and the page websockets the hand skeletons back. That is
    why "open the webpage and press Virtual Reality" is the entire setup on the device.

    `display_mode`:
      pass-through  the headset shows the REAL WORLD and no robot video. This is the
                    neck/chest-mounted case -- the operator watches the actual robot
                    with their own eyes, and the PICO is a pure hand-tracking sensor.
      ego           a small robot-view window inset in the real world.
      immersive     full robot first-person view (needs a zmq/webrtc image source).
    """

    def __init__(self, *, display_mode: str = "pass-through", cert: str | None = None,
                 key: str | None = None, img_shape: tuple[int, int] = (480, 1280),
                 stale_window: float = 0.2):
        import multiprocessing as mp
        import os

        # televuer runs its Vuer server in a multiprocessing.Process. On Linux (its target
        # platform) that FORKS, which shares the object as-is. macOS defaults to SPAWN,
        # which pickles the target -- and the bound method drags in the whole TeleVuer,
        # whose Vuer holds an unpicklable aiohttp FrozenList, so start() dies with
        # "self._frozen cannot be converted ... for pickling". Force fork before the
        # server is constructed. Safe here: teleop's only child process is this server.
        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass

        # Resolve the WebXR TLS cert. The page is HTTPS, so vuer MUST load a cert/key or
        # its server process dies with a bare "[Errno 2] No such file or directory" that
        # looks like anything. Prefer explicit args, then the XR_TELEOP_* env vars, then
        # the standard generated location (~/xr_teleop_certs/). Fail LOUD and actionable
        # rather than letting the child process crash silently and the PICO never connect.
        from pathlib import Path
        if cert is None and os.environ.get("XR_TELEOP_CERT"):
            cert, key = os.environ["XR_TELEOP_CERT"], os.environ.get("XR_TELEOP_KEY", key)
        if cert is None:
            d = Path.home() / "xr_teleop_certs"
            if (d / "cert.pem").exists() and (d / "key.pem").exists():
                cert, key = str(d / "cert.pem"), str(d / "key.pem")
        if not cert or not Path(cert).exists() or not key or not Path(key).exists():
            raise SystemExit(
                "no TLS cert for the WebXR page (vuer needs HTTPS). Generate one once:\n"
                "  mkdir -p ~/xr_teleop_certs && cd ~/xr_teleop_certs && \\\n"
                "  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\\n"
                "    -keyout key.pem -out cert.pem -subj '/CN=<this-mac-ip>'\n"
                "or pass --cert/--key, or set XR_TELEOP_CERT / XR_TELEOP_KEY.")

        from televuer import TeleVuer

        # img_shape is required by TeleVuer even when nothing is displayed (pass-through
        # never reads it); it only sizes the display plane.
        self._tv = TeleVuer(use_hand_tracking=True, binocular=True, img_shape=img_shape,
                            display_mode=display_mode, cert_file=cert, key_file=key)
        self.stale_window = stale_window
        self._lock = threading.Lock()
        self._sample: Sample | None = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_seen = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._pump, name="vuer-pump", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        """Poll televuer's shared memory into a timestamped Sample, detecting FREEZE.

        televuer exposes the stream as multiprocessing Arrays that its vuer process
        overwrites in place; there is no callback and no timestamp. Crucially, when a
        hand leaves the headset's FOV (or the whole WebXR session stalls), televuer does
        NOT zero the pose or drop `active` -- it HOLDS the last valid pose. So a frozen
        hand looks structurally perfect: finite, non-zero, a valid rotation. The only
        tell is that the numbers stop changing, and live optical tracking never repeats a
        frame to the bit -- so byte-equality with the previous read IS the freeze signal.

        A hand is `active` only if it is valid AND has changed within `stale_window`.
        `_stamp` (hence `age()`) advances only while SOME hand is fresh, so if the whole
        stream freezes `age()` grows and the loop's staleness watchdog can finally fire.
        The Sample is published every tick regardless, so `latest()` always carries the
        current per-hand active flags for the retargeter to hold a dropped arm.
        """
        period = 1.0 / 200.0
        last_raw = {s: None for s in SIDES}
        changed_at = {s: 0.0 for s in SIDES}
        while not self._stop.is_set():
            t0 = time.perf_counter()
            now = time.monotonic()
            try:
                pos = {"left": self._tv.left_hand_positions,
                       "right": self._tv.right_hand_positions}
                ori = {"left": self._tv.left_hand_orientations,
                       "right": self._tv.right_hand_orientations}
            except Exception:
                time.sleep(period)
                continue

            hand, active = {}, {}
            any_fresh = False
            for s in SIDES:
                valid = _webxr_hand_ok(pos[s], ori[s])
                pose26 = _pose26_from_webxr(pos[s], ori[s]) if valid else np.zeros((26, 7))
                fresh = False
                if valid:
                    if last_raw[s] is None or not np.array_equal(pose26, last_raw[s]):
                        changed_at[s] = now
                        last_raw[s] = pose26
                    fresh = (now - changed_at[s]) < self.stale_window
                hand[s] = pose26
                active[s] = fresh
                any_fresh = any_fresh or fresh

            with self._lock:
                self._sample = Sample(t_ns=time.monotonic_ns(), hand=hand, active=active)
                if any_fresh:
                    self._stamp = now
                    self.frames_seen += 1

            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def latest(self) -> Sample | None:
        with self._lock:
            return self._sample

    def age(self) -> float:
        with self._lock:
            if self._stamp == 0.0:
                return float("inf")
            return time.monotonic() - self._stamp

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class Hdf5Source:
    """Replays a recorded episode at wall-clock rate, as if it were a live headset.

    Indexes by elapsed time rather than stepping a cursor, so a slow consumer sees
    dropped samples exactly as it would from a real tracker instead of silently
    time-warping the human.
    """

    def __init__(self, path: str, *, speed: float = 1.0, loop: bool = False):
        import h5py

        with h5py.File(path, "r") as f:
            self.pose = {s: f[f"{s}_hand_pose"][:].astype(np.float64) for s in SIDES}
            self.act = {s: f[f"{s}_hand_active"][:].astype(bool) for s in SIDES}
            self.t_ns = f["timestamps_ns"][:].astype(np.int64)

        self.path = path
        self.speed = speed
        self.loop = loop
        self.n = len(self.t_ns)
        self._t0_wall = 0.0
        self._span_ns = int(self.t_ns[-1] - self.t_ns[0])

    def start(self) -> None:
        self._t0_wall = time.monotonic()

    def index_now(self) -> int | None:
        """Which recorded sample is 'now'. None once the episode is spent."""
        if self._t0_wall == 0.0:
            return None
        elapsed_ns = int((time.monotonic() - self._t0_wall) * self.speed * 1e9)
        if elapsed_ns > self._span_ns:
            if not self.loop:
                return None
            elapsed_ns %= max(self._span_ns, 1)
        return int(np.searchsorted(self.t_ns - self.t_ns[0], elapsed_ns, side="right") - 1)

    def latest(self) -> Sample | None:
        i = self.index_now()
        if i is None or i < 0:
            return None
        return Sample(
            t_ns=int(self.t_ns[i]),
            hand={s: self.pose[s][i] for s in SIDES},
            active={s: bool(self.act[s][i]) for s in SIDES},
        )

    def at(self, i: int) -> Sample:
        """Sample by index -- for offline replay that must not depend on wall time."""
        return Sample(
            t_ns=int(self.t_ns[i]),
            hand={s: self.pose[s][i] for s in SIDES},
            active={s: bool(self.act[s][i]) for s in SIDES},
        )

    def age(self) -> float:
        return 0.0 if self.index_now() is not None else float("inf")

    def close(self) -> None:
        pass


def _hand_status(sample: "Sample | None", side: str) -> str:
    """One-word diagnosis of why a hand is (not) tracked, for the wait heartbeat.

    The gate is `active`, but `active` False lumps together two very different states,
    and the operator needs to know which one they are in:

      tracked       valid AND fresh -- this hand is good, it counts toward the gate.
      frozen/still  a valid, non-zero pose that is not CHANGING. Either the hand is
                    genuinely motionless or it has left the FOV and televuer is holding
                    the last pose (see _pump). The fix is to move it / bring it into view.
      no data       the pose is zero: the tracker is not reporting this hand at all,
                    which upstream almost always means the WebXR session is not running
                    (PICO not on the page, or not in immersive VR yet).
    """
    if sample is None:
        return "no data"
    if sample.active[side]:
        return "tracked"
    # Wrist (row 1) carries the pose the arm rides on; a zero wrist is "no data",
    # a non-zero-but-not-active wrist is a frame that failed the freshness test.
    if float(np.abs(sample.hand[side][1, :3]).sum()) > 0.0:
        return "frozen/still"
    return "no data"


def wait_for_hands(src: HandSource, hands=SIDES, *, timeout: float | None = None,
                   heartbeat_s: float = 2.0) -> None:
    """Block until EVERY required hand is actually tracked (active).

    'latest() is not None' is not the right gate: the pump publishes a sample every tick
    (with per-hand active flags) so latest() is non-None the instant the source starts,
    even before the PICO connects. Gating on that marches straight into calibration with
    no hands. Gate on active hands instead -- that is what "waiting for hands" means.

    With no headset connected this blocks forever, so print a heartbeat every
    `heartbeat_s` seconds with a PER-HAND status (tracked / frozen-still / no data). A
    silent wait here is indistinguishable from a hang, and the per-hand line tells the
    operator exactly which hand is missing and why -- see `_hand_status`.
    """
    t0 = time.monotonic()
    next_beat = t0 + heartbeat_s
    ever_any_data = False
    while True:
        s = src.latest()
        if s is not None and all(s.active[h] for h in hands):
            return

        now = time.monotonic()
        if now >= next_beat:
            next_beat = now + heartbeat_s
            status = {h: _hand_status(s, h) for h in hands}
            ever_any_data = ever_any_data or any(v != "no data" for v in status.values())
            bits = "   ".join(f"{h}={status[h]}" for h in hands)
            line = f"  waiting for hands ({now - t0:3.0f}s):   {bits}"
            # If nothing has ever arrived, the headset side is the suspect, not the hands.
            if not ever_any_data:
                line += "\n    (no tracker data yet — open the page on the PICO and press "
                line += "'Virtual Reality')"
            print(line, flush=True)

        if timeout is not None and now - t0 > timeout:
            raise SystemExit("timed out waiting for hands — is the PICO in VR mode with "
                             "both hands in the headset's view?")
        time.sleep(0.1)
