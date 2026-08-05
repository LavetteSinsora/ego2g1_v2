"""Measure real, on-THIS-machine wall-clock latency of the relation_eef
perception cascade -- GroundingDINO+SAM2 detection, StereoSGBM depth, and a
full `RelationPerception.observe()` pass -- BEFORE deciding whether it's
affordable to run the whole thing fresh on every policy inference.

Why this exists: `RelationPerception`'s `detector_period_ticks` cadence gate
(relation_perception.py) assumes `observe()` is called at the 30 Hz control
rate ("~2 Hz" = fps/2). It isn't -- `observe()` only runs once per policy
inference (once per chunk in sync mode, ~inference_hz in async/rtc modes),
so that cadence number was never grounded in a real measurement of what one
full pass actually costs on real hardware. This script answers "how
expensive IS one from-scratch DINO+SAM2+SGBM pass, right here, right now" so
the cadence (or dropping the gate entirely, running fresh every inference)
can be set from a number instead of a guess.

    python -m ego2g1.deploy.perception_latency \\
        --camera-host 192.168.123.164 \\
        --task-config path/to/task.yaml --stereo-calib path/to/calib.npz

    python -m ego2g1.deploy.perception_latency --fake-camera
        # no hardware/calibration required: still times REAL DINO/SAM2/SGBM
        # compute on this machine's GPU/CPU against a blank frame (a
        # placeholder calibration is built matching the frame size -- see
        # `_resolve_calibration`; fine for a COMPUTE-cost number, the
        # resulting depth VALUES are not physically meaningful)

Reports three numbers, separately, because they answer different questions:

  detect()    GroundingDINO + SAM2, one prompt per configured object --
              scales ~linearly with object count (one DINO forward pass +
              one SAM2 predict PER OBJECT, see detector.py's `detect()`
              loop). The number you'd extrapolate from if the task config's
              object count changes.
  estimate()  StereoSGBM over the full frame -- CPU-bound (cv2), independent
              of object count or what's actually in view.
  observe()   The full `RelationPerception` pass end-to-end (detect +
              estimate + pixel->3D lift + tracker + latch), forced to run
              the detector on EVERY call (`detector_period_ticks=1`) -- the
              number that actually answers "what does calling this fresh
              every inference cost."

This script only MEASURES. It does not change `detector_period_ticks` or any
other deploy-time default -- that decision is a separate step once you have
real numbers from the actual deployment machine.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS = "obj0:a red cup,obj1:a mug,obj2:a black pen holder"


def _parse_prompts(spec: str):
    """Same shape as `perception_preview.py`'s own `_parse_prompts` --
    duplicated (not imported) on purpose: this is a standalone diagnostic
    tool, not meant to couple to another CLI script's private helper."""
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


def _resolve_task_config(task_config_path, prompts: str):
    from ego2g1.deploy.perception.task_config import DeployTaskConfig, load_task_config

    if task_config_path:
        return load_task_config(task_config_path)
    return DeployTaskConfig(objects=_parse_prompts(prompts), hands=("left", "right"))


def _resolve_calibration(stereo_calib_path, rgb_shape: tuple[int, int, int]):
    """A real, loaded `StereoCalibration` if `--stereo-calib` was given, else
    a placeholder sized to the ACTUAL frame the camera handed back. SGBM's
    compute cost depends on image size/matcher params, not on calibration
    accuracy -- a placeholder still gives an honest latency number, just not
    a trustworthy depth VALUE (same stance as `perception_preview.py`'s
    `_dummy_calibration`, generalized to whatever frame size is really in
    play here instead of a hardcoded 640x480)."""
    from ego2g1.deploy.perception.depth import StereoCalibration

    if stereo_calib_path:
        return StereoCalibration.load(stereo_calib_path)
    h, w = rgb_shape[:2]
    logger.warning(
        "--stereo-calib not given: using a PLACEHOLDER calibration sized to "
        "the real %dx%d frame this camera returned. Fine for a COMPUTE-cost "
        "measurement (StereoSGBM's cost depends on image size/params, not "
        "calibration accuracy) -- the resulting depth VALUES are not "
        "physically meaningful. Pass a real --stereo-calib for numbers "
        "you'd also trust for grasping.", w, h)
    K = np.array([[600.0, 0.0, w / 2.0], [0.0, 600.0, h / 2.0], [0.0, 0.0, 1.0]])
    return StereoCalibration(
        K_left=K, K_right=K.copy(), dist_left=np.zeros(5), dist_right=np.zeros(5),
        R=np.eye(3), T=np.array([0.06, 0.0, 0.0]), image_size=(w, h))


def _time_calls(fn, *, n: int, warmup: int, clock=time.perf_counter
               ) -> tuple[list[float], list[float], BaseException | None]:
    """(warmup_samples_s, steady_samples_s, last_error). `fn` is timed even
    when it raises (a --fake-camera run with objects never actually in
    frame still exercises -- and should still time -- every real compute
    stage up to `observe()`'s final "no pose to report" check; a benchmark
    must survive that and keep reporting, not abort)."""
    warmup_samples: list[float] = []
    last_error: BaseException | None = None
    for _ in range(max(0, warmup)):
        t0 = clock()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 -- see docstring
            last_error = e
        warmup_samples.append(clock() - t0)
    samples: list[float] = []
    for _ in range(n):
        t0 = clock()
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            last_error = e
        samples.append(clock() - t0)
    return warmup_samples, samples, last_error


def _report_stage(name: str, warmup_s: list[float], samples_s: list[float],
                  error: BaseException | None) -> float:
    """Print one stage's numbers, return its p95 in milliseconds."""
    w_ms = np.asarray(warmup_s) * 1000.0
    s_ms = np.asarray(samples_s) * 1000.0
    print(f"\n--- {name} ---")
    if error is not None:
        print(f"  every call raised {type(error).__name__}: {error}")
        print("  (timing above/below is still the real compute cost incurred "
              "before that error -- expected with --fake-camera or a frame "
              "that doesn't show the configured objects)")
    if len(w_ms):
        print(f"  warmup   ({len(w_ms)} call(s), includes first-call GPU/JIT "
              f"warmup): {', '.join(f'{x:.0f}' for x in w_ms)} ms")
    p50, p95, p99 = np.percentile(s_ms, [50, 95, 99])
    print(f"  steady   (n={len(s_ms)}): mean {s_ms.mean():.0f}  p50 {p50:.0f}  "
          f"p95 {p95:.0f}  p99 {p99:.0f}  max {s_ms.max():.0f}  ms")
    return float(p95)


def main(
    *,
    camera_host: str = "192.168.123.164",
    eye: str = "left",
    fake_camera: bool = False,
    task_config: str | None = None,
    prompts: str = _DEFAULT_PROMPTS,
    stereo_calib: str | None = None,
    device: str | None = None,
    box_threshold: float = 0.3,
    n: int = 20,
    warmup: int = 2,
    fps: int = 30,
    horizon: int = 50,
) -> None:
    """Measure detect()/estimate()/observe() latency on this machine.

    `fps`/`horizon` are only used to put the final `observe()` number in
    context against a sync-mode chunk duration (`horizon / fps`) -- they do
    not affect what's measured, only how the summary is phrased. Use the
    connected checkpoint's real values (client.fps / client.action_horizon)
    for a meaningful comparison; the defaults here are just placeholders.
    """
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s")

    if fake_camera:
        from ego2g1.deploy.camera import StaticCamera
        cam = StaticCamera(shape=(480, 640, 3))
        logger.info("--fake-camera: a blank static frame (no hardware)")
    else:
        from ego2g1.deploy.camera import HeadCamera
        cam = HeadCamera(host=camera_host, eye=eye)
    cam.connect()

    stereo = cam.read_stereo()
    if stereo is None:
        raise TimeoutError("camera never produced a stereo pair -- check the "
                            "camera link, or pass --fake-camera")
    rgb_left, rgb_right = stereo
    logger.info("timing against ONE held %dx%d frame pair for all stages "
               "below (freshness doesn't matter for a compute-cost "
               "measurement; holding it fixed keeps the three stages "
               "directly comparable)", rgb_left.shape[1], rgb_left.shape[0])

    cfg = _resolve_task_config(task_config, prompts)
    logger.info("task config: %d object(s): %s", len(cfg.objects),
               ", ".join(f"{o.instance_id}={o.detector_prompt!r}" for o in cfg.objects))

    calib = _resolve_calibration(stereo_calib, rgb_left.shape)

    logger.info("loading GroundingDINO+SAM2 (device=%s) -- first run "
               "downloads weights from HuggingFace, this can take a "
               "while...", device or "auto")
    from ego2g1.deploy.perception.detector import GroundingDinoSam2Detector
    detector = GroundingDinoSam2Detector(device=device, box_threshold=box_threshold)
    logger.info("detector loaded on %s.", detector.device)

    from ego2g1.deploy.perception.depth import StereoSGBMDepthSource
    depth_source = StereoSGBMDepthSource(calib)

    from ego2g1.deploy.perception.relation_perception import RelationPerception
    perception = RelationPerception(
        cfg, detector, depth_source, calib, T_pelvis_camera=np.eye(4),
        fps=fps, detector_period_ticks=1)   # force the full cascade on EVERY observe() call

    print("\n" + "=" * 70)
    print(f"perception latency probe -- device={detector.device}  "
         f"n={n} timed call(s) per stage, {warmup} warmup call(s) discarded "
         "from the steady-state stats")

    detect_w, detect_s, detect_err = _time_calls(
        lambda: detector.detect(rgb_left, cfg.objects), n=n, warmup=warmup)
    detect_p95 = _report_stage("detect()  [GroundingDINO + SAM2]", detect_w, detect_s, detect_err)

    depth_w, depth_s, depth_err = _time_calls(
        lambda: depth_source.estimate(rgb_left, rgb_right), n=n, warmup=warmup)
    _report_stage("estimate()  [StereoSGBM]", depth_w, depth_s, depth_err)

    hands = cfg.hands
    flange = {h: np.eye(4) for h in hands}
    hand_cmds_last = {h: 0.0 for h in hands}
    observe_w, observe_s, observe_err = _time_calls(
        lambda: perception.observe(rgb_left, rgb_right, flange, hand_cmds_last),
        n=n, warmup=warmup)
    observe_p95 = _report_stage(
        "observe()  [full RelationPerception pass, forced every call]",
        observe_w, observe_s, observe_err)

    chunk_ms = 1000.0 * horizon / fps
    tick_ms = 1000.0 / fps
    print("\n" + "=" * 70)
    print(f"context: fps={fps}  horizon={horizon}  ->  one sync-mode chunk "
         f"lasts {chunk_ms:.0f} ms, one control tick is {tick_ms:.1f} ms")
    print(f"one full observe() pass costs ~{observe_p95:.0f} ms p95 on THIS machine "
         f"({len(cfg.objects)} object(s)).")
    if observe_p95 > chunk_ms:
        print(f"  -> LONGER than a whole action chunk ({chunk_ms:.0f} ms). Calling the "
             "full cascade on every inference would make perception the dominant cost "
             "of every chunk boundary, on top of whatever the policy server itself takes.")
    else:
        print(f"  -> {100 * observe_p95 / chunk_ms:.0f}% of one chunk's duration "
             f"({chunk_ms:.0f} ms). Affordable to run every inference IF the policy "
             "server's own latency still leaves headroom under your actual timing "
             "budget -- combine this with a real startup_self_check (latency.py) "
             "run against the connected checkpoint for the true combined total.")
    print(f"(detect() alone: ~{detect_p95:.0f} ms p95 for {len(cfg.objects)} object(s) -- "
         "scales ~linearly with object count, one DINO+SAM2 pass per object; "
         "re-run with a larger --task-config/--prompts to see that scaling directly.)")
    print("\nThis script only measures -- it does not change "
         "detector_period_ticks or any other deploy default.")

    cam.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
