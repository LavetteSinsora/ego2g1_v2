"""The single egocentric camera the model consumes. Ported from the old
deploy's camera.py (third_party/openpi/ego2g1/deploy/camera.py).

The model takes exactly ONE image (the input transform zero-fills and
attention-masks both wrist slots — it trained having never seen a wrist view).

The G1's head streams a STEREO PAIR over ZMQ from the `image_server` on the
robot's onboard board, so we pick one eye. Which eye, and how well its
FOV/mount matches the Pico-headset view the training video came from, is the
biggest open risk in the whole deployment — a policy fed a systematically
different viewpoint fails quietly and looks like a bad policy. Log the exact
array (the recorder does) and eyeball it against a training frame before
believing any rollout.

No resize here: read() hands back the RAW head frame, which is what check and
the recorder must see. The 224x224 resize happens once, at the wire, in
PolicyClient._prepare_image.

`relation_eef` mode (docs/relation_deploy_plan.md §5.2) needs BOTH eyes at
once — `perception/depth.py`'s `StereoSGBMDepthSource` needs the stereo pair,
not the single configured `eye=` the model itself consumes. `read_stereo()`
below hands back both halves regardless of `eye=`; `read()`/`age()` keep
their exact original single-eye behavior unchanged, so every existing
joint/relative_eef caller (and the model's own single-image input in
relation_eef mode too — training only ever saw one egocentric view) is
unaffected.
"""

import threading
import time

import numpy as np

DEFAULT_HOST = "192.168.123.164"   # G1 head board; override with --camera-host


class HeadCamera:
    """G1 head camera via the unitree_deploy ImageClient (ZMQ from the head board)."""

    def __init__(self, *, host: str = DEFAULT_HOST, eye: str = "left",
                 flip_bgr: bool = True, auto_start_server: bool = False,
                 ssh_user: str | None = None, ssh_password: str | None = None):
        if eye not in ("left", "right"):
            raise ValueError(f"eye must be left|right, got {eye}")
        self.host = host
        self.eye = eye
        self.flip_bgr = flip_bgr
        # If True, connect() SSHes into `host` and starts image_server.py
        # itself when it isn't already reachable, instead of just raising --
        # see remote_image_server.py's module docstring for why this can
        # never fully remove the need for SOME process on the robot board
        # (the camera is wired to ITS usb bus, not the deploy PC's), only
        # remove having to start that process by hand every session.
        self.auto_start_server = bool(auto_start_server)
        self._ssh_user = ssh_user
        self._ssh_password = ssh_password
        self._client = None
        self._lock = threading.Lock()
        self._frame = None
        # both eyes, kept independently of `self._frame`/`self.eye` so
        # read_stereo() works regardless of which eye the model is fed
        self._frame_left = None
        self._frame_right = None
        self._t = 0.0
        self._stop = threading.Event()
        self._thread = None

    def connect(self, *, timeout: float = 10.0) -> None:
        from unitree_deploy.robot_devices.cameras.configs import ImageClientCameraConfig
        from unitree_deploy.robot_devices.cameras.imageclient import ImageClientCamera

        if self.auto_start_server:
            from . import remote_image_server as _remote

            kwargs = {}
            if self._ssh_user is not None:
                kwargs["ssh_user"] = self._ssh_user
            if self._ssh_password is not None:
                kwargs["ssh_password"] = self._ssh_password
            was_running = _remote.ensure_running(self.host, **kwargs)
            if not was_running:
                # image_server was JUST started -- give it a moment to open
                # its ZMQ socket and produce a first frame before the normal
                # wait-loop below (which times out on its own regardless).
                time.sleep(2.0)

        self._client = ImageClientCamera(ImageClientCameraConfig(host_ip=self.host))
        self._client.connect()

        self._thread = threading.Thread(target=self._pump, name="camera", daemon=True)
        self._thread.start()

        t0 = time.monotonic()
        while self.age() > timeout:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"no frames from image_server at {self.host}. Is it running? "
                    "(ssh unitree@%s; conda activate tv; cd ~/image_server; "
                    "python image_server.py) -- or construct HeadCamera(..., "
                    "auto_start_server=True) to have this done for you." % self.host
                )
            time.sleep(0.05)

    def _to_rgb(self, img):
        if self.flip_bgr:
            img = img[..., ::-1]   # ImageClient yields BGR; the model wants RGB
        return np.ascontiguousarray(img, dtype=np.uint8)

    def _pump(self) -> None:
        # ImageClient splits ONE wide stereo frame into these two keys.
        eye_keys = {"left": "cam_left_high", "right": "cam_right_high"}
        while not self._stop.is_set():
            try:
                out = self._client.async_read()
            except Exception:
                time.sleep(0.01)
                continue

            if not isinstance(out, dict):
                # a transport that hands back a single array directly (not
                # ImageClient's split-dict shape) — only the configured eye
                # is knowable here; the other half of read_stereo() stays
                # None until a real dict arrives, exactly as documented.
                if out is not None:
                    with self._lock:
                        self._frame = self._to_rgb(out)
                        self._t = time.monotonic()
                time.sleep(1 / 60)
                continue

            left = out.get(eye_keys["left"])
            right = out.get(eye_keys["right"])
            if left is None and right is None:
                time.sleep(1 / 60)
                continue
            with self._lock:
                if left is not None:
                    self._frame_left = self._to_rgb(left)
                if right is not None:
                    self._frame_right = self._to_rgb(right)
                configured = left if self.eye == "left" else right
                if configured is not None:
                    # keep .read()'s exact original semantics: self._frame /
                    # self._t only move when the CONFIGURED eye arrived
                    self._frame = self._frame_left if self.eye == "left" else self._frame_right
                    self._t = time.monotonic()
            time.sleep(1 / 60)

    def read(self):
        """Latest frame as (H, W, 3) uint8 RGB, or None. Only the configured
        `eye=` — unchanged behavior, exactly what every joint/relative_eef
        caller and the model's own single-image input already expect."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def read_stereo(self):
        """Both eyes' latest frames as `(left, right)`, each (H, W, 3) uint8
        RGB, or None if either half hasn't arrived yet. Regardless of
        `eye=` — this is for `relation_eef` mode's stereo depth
        (`perception/depth.py`), never for the model's own image input."""
        with self._lock:
            if self._frame_left is None or self._frame_right is None:
                return None
            return self._frame_left.copy(), self._frame_right.copy()

    def age(self) -> float:
        with self._lock:
            return float("inf") if self._frame is None else time.monotonic() - self._t

    def close(self) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass


class StaticCamera:
    """A fixed frame, forever — for dry-runs and latency checks without the
    robot (the wire cost of a real-sized frame is part of the latency).

    `frame_right`, if given, is a second static frame for `read_stereo()`'s
    right eye (e.g. a dry-run exercising relation_eef's stereo depth path
    with two distinct fixtures); defaults to the SAME frame as `read()`,
    which is the simple, common case — a dry run has no real stereo baseline
    to speak of anyway."""

    def __init__(self, frame=None, shape=(480, 640, 3), frame_right=None):
        self._frame = (np.zeros(shape, np.uint8) if frame is None
                       else np.ascontiguousarray(frame, dtype=np.uint8))
        self._frame_right = (self._frame if frame_right is None
                             else np.ascontiguousarray(frame_right, dtype=np.uint8))

    def connect(self, **kwargs) -> None:
        pass

    def read(self):
        return self._frame.copy()

    def read_stereo(self):
        """Same static frame (or two, if `frame_right=` was given) for both
        eyes — see class docstring."""
        return self._frame.copy(), self._frame_right.copy()

    def age(self) -> float:
        return 0.0

    def close(self) -> None:
        pass
