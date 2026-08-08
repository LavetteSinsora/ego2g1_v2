"""The two wrist cameras `umi_eef` consumes.

Same transport as the head camera, and the SAME CLIENT OBJECT. unitree's
`image_server` publishes three cameras from one board — its
`cam_config_client.yaml` gives head 480x1280 binocular on port 55555,
left_wrist and right_wrist 480x640 each on 55556/55557 — and
`ImageClientCamera.async_read()` returns all of them in one dict::

    cam_left_high, cam_right_high    the head pair, split in half
    cam_left_wrist                   left wrist, 480x640
    cam_right_wrist                  right wrist, 480x640

Those last two key names are EXACTLY the training dataset's feature names
(`observation.images.cam_left_wrist` / `cam_right_wrist`, both 480x640), so the
recording came through this same path. Reading them here is therefore not a new
transport at all — it is `camera.HeadCamera`'s pump with two different keys.

No resize: `read()` hands back the raw frame, which is what the recorder and
the dashboard must see. The 224x224 resize happens once, on the wire, in
`PolicyClient._prepare_image`.

`CameraPair` (composing two arbitrary camera objects) is kept as an escape
hatch for a rig whose wrist cameras are NOT on the robot's image_server — two
USB cameras on the deploy PC, say. It is not the default and needs explicit
`--acting-camera`/`--context-camera` URIs.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from ..core import umi_layout
from .camera import DEFAULT_HOST


class WristCameraPair:
    """Both wrist cameras off the robot's image_server, via one ImageClient.

    Exposes the single-camera protocol (`connect/read/age/close`) so the
    recorder, the staleness watchdog and the dashboard work unchanged, plus
    `read_pair()` for the two-image request `umi_eef` sends.
    """

    labels = ("acting wrist -> right_wrist_0_rgb", "context -> base_0_rgb")

    def __init__(self, *, host: str = DEFAULT_HOST,
                 acting: str = umi_layout.ACTING_HAND,
                 flip_bgr: bool = True, auto_start_server: bool = False,
                 ssh_user: str | None = None, ssh_password: str | None = None):
        if acting not in ("left", "right"):
            raise ValueError(f"acting must be left|right, got {acting}")
        self.host = host
        self.acting = acting
        self.context = "left" if acting == "right" else "right"
        self.flip_bgr = flip_bgr
        self.auto_start_server = bool(auto_start_server)
        self._ssh_user, self._ssh_password = ssh_user, ssh_password
        self._client = None
        self._lock = threading.Lock()
        self._acting_frame = None
        self._context_frame = None
        self._t = 0.0
        self._stop = threading.Event()
        self._thread = None
        self.labels = (
            f"acting = cam_{self.acting}_wrist -> right_wrist_0_rgb",
            f"context = cam_{self.context}_wrist -> base_0_rgb",
        )

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
            if not _remote.ensure_running(self.host, **kwargs):
                time.sleep(2.0)   # just started: let it open its sockets

        self._client = ImageClientCamera(ImageClientCameraConfig(host_ip=self.host))
        self._client.connect()
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, name="umi-wrist-cams",
                                        daemon=True)
        self._thread.start()

        t0 = time.monotonic()
        while self.age() > timeout:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError(
                    f"no WRIST frames from image_server at {self.host}. The head "
                    "camera can be fine while these are not: check that "
                    "left_wrist_camera/right_wrist_camera have enable_zmq: true in "
                    "the board's cam_config_client.yaml and that both USB cameras "
                    "enumerate (video_id 2 and 4 in the stock config)."
                )
            time.sleep(0.05)

    def _to_rgb(self, img):
        if self.flip_bgr:
            img = img[..., ::-1]     # ImageClient yields BGR; the model wants RGB
        return np.ascontiguousarray(img, dtype=np.uint8)

    def _pump(self) -> None:
        acting_key = f"cam_{self.acting}_wrist"
        context_key = f"cam_{self.context}_wrist"
        while not self._stop.is_set():
            try:
                out = self._client.async_read()
            except Exception:
                time.sleep(0.01)
                continue
            if not isinstance(out, dict):
                time.sleep(1 / 60)
                continue
            a, c = out.get(acting_key), out.get(context_key)
            # BOTH or neither: a request carrying one live view and one stale
            # one is a silently wrong observation, and `age()` could not
            # express it (there is one timestamp). Dropping the tick lets the
            # staleness watchdog see it.
            if a is None or c is None:
                time.sleep(1 / 60)
                continue
            with self._lock:
                self._acting_frame = self._to_rgb(a)
                self._context_frame = self._to_rgb(c)
                self._t = time.monotonic()
            time.sleep(1 / 120)

    def read(self):
        """The ACTING view — what a single-image consumer should see."""
        with self._lock:
            return None if self._acting_frame is None else self._acting_frame.copy()

    def read_pair(self):
        with self._lock:
            if self._acting_frame is None or self._context_frame is None:
                return None, None
            return self._acting_frame.copy(), self._context_frame.copy()

    def age(self) -> float:
        with self._lock:
            return float("inf") if self._acting_frame is None else time.monotonic() - self._t

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._client is not None:
            with __import__("contextlib").suppress(Exception):
                self._client.disconnect()


class CameraPair:
    """Escape hatch: two INDEPENDENT camera objects behind one protocol.

    For a rig whose wrist cameras are not on the robot's image_server. Not the
    default — see `build_camera_pair`.
    """

    labels = ("acting wrist -> right_wrist_0_rgb", "context -> base_0_rgb")

    def __init__(self, acting, context):
        self.acting = acting
        self.context = context

    def connect(self, **kwargs) -> None:
        self.acting.connect(**kwargs)
        self.context.connect(**kwargs)

    def read(self):
        return self.acting.read()

    def read_pair(self):
        return self.acting.read(), self.context.read()

    def age(self) -> float:
        """The STALER of the two: a fresh acting view says nothing about a
        frozen context view, and the watchdog must trip on either."""
        return max(self.acting.age(), self.context.age())

    def close(self) -> None:
        for cam in (self.acting, self.context):
            try:
                cam.close()
            except Exception:  # noqa: BLE001 — closing must not mask the first failure
                pass


class LocalCamera:
    """A USB/V4L2 camera on the deploy PC, via cv2. RGB, raw resolution."""

    def __init__(self, index: int, *, width: int | None = None,
                 height: int | None = None):
        self.index = int(index)
        self.width, self.height = width, height
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._t = 0.0
        self._stop = threading.Event()
        self._thread = None

    def connect(self, *, timeout: float = 10.0) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.index)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera index {self.index}")
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name=f"umi-cam-{self.index}")
        self._thread.start()
        deadline = time.monotonic() + timeout
        while self._frame is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"camera index {self.index} opened but produced no frame in "
                    f"{timeout:.0f}s")
            time.sleep(0.02)

    def _pump(self) -> None:
        import cv2

        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb
                self._t = time.monotonic()

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def age(self) -> float:
        with self._lock:
            return float("inf") if self._frame is None else time.monotonic() - self._t

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


def make_camera(uri: str):
    """URI -> a camera object, for the escape-hatch path only.

    'static' | 'v4l2:<index>' | 'zmq:<host>[:<eye>]' (a head-camera eye).
    """
    scheme, _, rest = uri.partition(":")
    if scheme == "static":
        from .camera import StaticCamera
        return StaticCamera()
    if scheme == "v4l2":
        if not rest:
            raise ValueError(f"{uri!r}: v4l2 needs a device index, e.g. v4l2:0")
        return LocalCamera(int(rest))
    if scheme == "zmq":
        from .camera import HeadCamera
        host, _, eye = rest.partition(":")
        if not host:
            raise ValueError(f"{uri!r}: zmq needs a host, e.g. zmq:192.168.123.164")
        return HeadCamera(host=host, eye=eye or "left")
    raise ValueError(
        f"unknown camera URI {uri!r}; expected one of "
        "'static', 'v4l2:<index>', 'zmq:<host>[:<eye>]'")


def build_camera_pair(args, *, acting: str = umi_layout.ACTING_HAND):
    """`UmiEEFMode.build_camera`.

    DEFAULT: both wrist cameras off the robot's own image_server, the same
    host and the same client the head camera uses — so a normal run needs no
    new flags at all. Only a rig whose wrist cameras live somewhere else needs
    the explicit `--acting-camera`/`--context-camera` URIs.
    """
    if getattr(args, "dry_run", False):
        from .camera import StaticCamera
        return CameraPair(StaticCamera(), StaticCamera())

    acting_uri = getattr(args, "acting_camera", None)
    context_uri = getattr(args, "context_camera", None)
    if acting_uri or context_uri:
        if not (acting_uri and context_uri):
            missing = "--context-camera" if acting_uri else "--acting-camera"
            raise ValueError(
                f"{missing} is also required: overriding one wrist camera without "
                "the other would silently pair a custom device with an "
                "image_server one, and which camera is which is not recoverable "
                "from the frames.")
        return CameraPair(make_camera(acting_uri), make_camera(context_uri))

    return WristCameraPair(
        host=getattr(args, "camera_host", DEFAULT_HOST), acting=acting,
        auto_start_server=bool(getattr(args, "auto_start_camera_server", False)))
