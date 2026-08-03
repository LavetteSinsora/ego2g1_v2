"""Live GroundingDINO+SAM2 overlay preview on the deploy dashboard --
real camera, real detector, CPU, and NO robot actuation whatsoever.

There is no executor, no policy client, and no Kinematics/IK anywhere in this
file -- `PreviewLoop` below is the whole "loop," and the only network traffic
it generates is the camera's READ-ONLY ZMQ stream from the robot's head
board. `Dashboard`'s Start/Pause/E-STOP buttons will 409 ("not supported by
this runner") if clicked; that is the correct, intended behavior here, not a
bug -- this tool cannot drive the robot even if you press them.

    python -m ego2g1.deploy.perception_preview \\
        --camera-host 192.168.123.164 --eye left \\
        --prompts "obj0:a red cup,obj1:a mug,obj2:a black pen holder"

    python -m ego2g1.deploy.perception_preview --fake-camera   # no hardware;
                                                                # a blank test
                                                                # frame, just
                                                                # to check the
                                                                # tool itself
                                                                # runs/serves

CPU only (`GroundingDinoSam2Detector(device="cpu")` explicitly, since this
machine has no CUDA -- see chat: torch reports cuda=False here). Expect each
detector tick (GroundingDINO once per prompt + SAM2 once per found box) to
take real wall-clock seconds, not milliseconds -- `--detect-hz` paces how
often a new detector call is KICKED OFF, but a call already running is never
interrupted, so the real cadence is "as fast as this CPU allows," not exactly
`--detect-hz`.

The depth/3D numbers this preview reports (object depth_m, the "tracked"/
"predicted" markers on /perception.jpg, position_pelvis) are NOT physically
meaningful -- there is no measured stereo calibration or camera extrinsic for
this session, only made-up placeholders (see `_ConstantDepth`/
`_dummy_calibration` below). The 2D box + mask overlay IS pixel-accurate
(straight from the real detector's own output); that is the actual point of
this tool.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS = "obj0:a red cup,obj1:a mug,obj2:a black pen holder"


def _parse_prompts(spec: str):
    from ego2g1.deploy.perception.task_config import ObjectSpec

    objects = []
    for i, entry in enumerate(spec.split(",")):
        entry = entry.strip()
        if not entry:
            continue
        instance_id, _, prompt = entry.partition(":")
        if not prompt:
            instance_id, prompt = f"obj{i}", instance_id
        instance_id, prompt = instance_id.strip(), prompt.strip()
        objects.append(ObjectSpec(instance_id=instance_id, category=prompt,
                                  detector_prompt=prompt, graspable=True))
    if not objects:
        raise ValueError(f"--prompts parsed to zero objects: {spec!r}")
    return tuple(objects)


def _dummy_calibration():
    """A plausible-looking, NOT measured, `StereoCalibration` -- only used to
    give `RelationPerception` something to back-project 2D detections
    through. See module docstring: the resulting 3D numbers are placeholders,
    not real geometry."""
    from ego2g1.deploy.perception.depth import StereoCalibration

    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    return StereoCalibration(
        K_left=K, K_right=K.copy(), dist_left=np.zeros(5), dist_right=np.zeros(5),
        R=np.eye(3), T=np.array([0.06, 0.0, 0.0]), image_size=(640, 480))


class _ConstantDepth:
    """Depth source stand-in: every pixel at a fixed, made-up distance. This
    preview has no measured stereo calibration, so there is nothing truer to
    report -- see module docstring. Only feeds `RelationPerception`'s 3D
    lift; the 2D box/mask overlay never touches this."""

    def __init__(self, depth_m: float = 0.5):
        self._depth_m = float(depth_m)

    def estimate(self, rgb_left, rgb_right):
        h, w = np.asarray(rgb_left).shape[:2]
        return np.full((h, w), self._depth_m, dtype=np.float32)


class _NoOpAdapter:
    """The only thing `Dashboard.encode_perception_frame`/
    `build_relation_telemetry` need from an "adapter": a `.perception`
    attribute (and `.last_percept`, refreshed by `PreviewLoop` itself). No
    `.infer`, no policy client, nothing that could ever drive a robot."""

    def __init__(self, perception):
        self.perception = perception
        self.last_percept = None


class PreviewLoop:
    """`Dashboard`-compatible stand-in for `DeployRunner`: a real camera
    (read-only) + a real `RelationPerception` (real detector), nothing else.
    No executor, no policy client, no Kinematics/IK, no watchdog -- there is
    no code path here that can send the robot anything."""

    def __init__(self, camera, perception, *, detect_hz: float = 0.5):
        self.camera = camera
        self._perception = perception
        self.adapter = _NoOpAdapter(perception)
        self._period_s = 1.0 / float(detect_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="perception-preview")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        hands = self._perception.task_config.hands
        flange = {h: np.eye(4) for h in hands}
        hand_cmds_last = {h: 0.0 for h in hands}
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                rgb_left, rgb_right = self.camera.read_stereo()
                self.adapter.last_percept = self._perception.observe(
                    rgb_left, rgb_right, flange, hand_cmds_last)
            except Exception as e:   # noqa: BLE001 -- a bad/empty tick must not kill the loop
                # Very common early on (or with --fake-camera): the detector
                # hasn't found one of the prompts in-frame yet. Log tersely,
                # not a full traceback every retry.
                logger.warning("perception tick failed (will retry): %s", e)
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.0, self._period_s - elapsed))

    # --- Dashboard's telemetry contract (see dashboard.py's Dashboard class) ---
    def telemetry(self) -> dict:
        from .runner import build_relation_telemetry

        rel = build_relation_telemetry(self._perception, self.adapter.last_percept)
        return {
            "now": time.monotonic(), "mode": "perception-preview",
            "server_rtc": False, "active": True, "recording": False,
            "has_dataset": False,
            "task": "perception preview -- real detector, NO robot/policy",
            "horizon": 0, "fps": 0, "dim": 0,
            "ready": False, "index": 0, "wall_slot": None, "trigger": None,
            "d": None, "action_row": None, "row_slot": None, "groups": [],
            "inferring": False, "pending": False, "worker_dead": False,
            "last_splice": {},
            "stats": {"ticks": 0, "chunks": None, "votes": None},
            "budget": None, "runway_s": None,
            "camera_age": float(self.camera.age()),
            "clamped_ticks": 0,
            "watchdog": {"tripped": False, "reason": None},
            "arm_q": None, "state_age": None, "estopped": False,
            "relation": rel,
        }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-host", default="192.168.123.164",
                  help="G1 head board (ignored with --fake-camera)")
    p.add_argument("--eye", default="left", choices=("left", "right"))
    p.add_argument("--fake-camera", action="store_true",
                  help="no hardware: a blank static frame, just to check the "
                       "tool itself runs/serves")
    p.add_argument("--prompts", default=_DEFAULT_PROMPTS,
                  help="comma-separated instance_id:prompt pairs, e.g. "
                       "'obj0:a red cup,obj1:a mug'")
    p.add_argument("--detect-hz", type=float, default=0.5,
                  help="how often to KICK OFF a new detector call (a call "
                       "already running is never interrupted -- see module "
                       "docstring)")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")

    from ego2g1.deploy.perception.task_config import DeployTaskConfig

    objects = _parse_prompts(args.prompts)
    task_config = DeployTaskConfig(objects=objects, hands=("left", "right"))
    logger.info("watching for: %s",
               ", ".join(f"{o.instance_id}={o.detector_prompt!r}" for o in objects))

    if args.fake_camera:
        from ego2g1.deploy.camera import StaticCamera
        cam = StaticCamera(shape=(480, 640, 3))
        logger.info("--fake-camera: a blank static frame (no hardware)")
    else:
        from ego2g1.deploy.camera import HeadCamera
        cam = HeadCamera(host=args.camera_host, eye=args.eye)
    cam.connect()

    logger.info("loading GroundingDINO+SAM2 on CPU -- first run downloads "
               "weights from HuggingFace, this can take a while...")
    from ego2g1.deploy.perception.detector import GroundingDinoSam2Detector
    detector = GroundingDinoSam2Detector(device="cpu")
    logger.info("detector loaded.")

    from ego2g1.deploy.perception.relation_perception import RelationPerception
    perception = RelationPerception(
        task_config, detector, _ConstantDepth(), _dummy_calibration(),
        T_pelvis_camera=np.eye(4), fps=max(0.1, args.detect_hz),
        detector_period_ticks=1)   # every observe() call IS a detector call

    loop = PreviewLoop(cam, perception, detect_hz=args.detect_hz)
    loop.start()

    from ego2g1.deploy.dashboard import Dashboard
    dash = Dashboard(loop, port=args.port)
    dash.start()
    logger.info("perception preview -> http://localhost:%d  "
               "(no robot, no policy server -- ctrl-C to stop)", dash.port)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        dash.stop()
        loop.stop()
        cam.close()


if __name__ == "__main__":
    main()
