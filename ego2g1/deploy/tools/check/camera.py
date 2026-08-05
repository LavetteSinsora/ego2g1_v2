"""Rungs 4/4b (`check camera` / `check stereo-capture`): one frame to disk
(LOOK AT IT next to a training frame — a wrong viewpoint fails
silently), and auto-numbered stereo pairs for the intrinsics
calibration (perception/stereo_calib.py consumes the naming)."""

from __future__ import annotations

import time

from ego2g1.deploy.camera import DEFAULT_HOST as _CAMERA_HOST

# --- 4. camera ---------------------------------------------------------------

def camera(host: str = _CAMERA_HOST, eye: str = "left",
           out: str = "check_camera.png", auto_start_server: bool = True) -> None:
    """Grab one frame and write it out. Then LOOK AT IT next to a training frame.

    This is the highest-risk open item in the deployment: the model trained on
    Pico-headset egocentric video, and a systematically different viewpoint
    fails quietly and looks like a bad policy.

    `auto_start_server=True` (default): SSH into `host` and start
    `image_server.py` there if it isn't already reachable, instead of just
    failing with "is it running?" -- see `remote_image_server.py`'s module
    docstring. Set False to get the old fail-fast behavior (e.g. if you're
    intentionally testing that the server ISN'T up)."""
    import cv2

    from ego2g1.deploy.camera import HeadCamera

    cam = HeadCamera(host=host, eye=eye, auto_start_server=auto_start_server)
    cam.connect()
    img = cam.read()
    cam.close()
    print(f"frame: {img.shape} {img.dtype}  range [{img.min()}, {img.max()}]")
    cv2.imwrite(out, img[..., ::-1])
    print(f"wrote {out} — compare against a training video frame before "
          "trusting a rollout.")


# --- 4b. stereo-capture -------------------------------------------------------

def _next_pair_index(out_path) -> int:
    """The lowest non-negative integer N such that neither left_{N:03d}.png
    nor right_{N:03d}.png exists yet in `out_path` — so repeated invocations
    never collide or need the operator to track a counter by hand, and a
    manually deleted/re-shot pair doesn't leave a permanent gap unfilled."""
    existing = set()
    for p in out_path.glob("*_*.png"):
        stem_suffix = p.stem.rsplit("_", 1)[-1]
        if stem_suffix.isdigit():
            existing.add(int(stem_suffix))
    idx = 0
    while idx in existing:
        idx += 1
    return idx


def stereo_capture(host: str = _CAMERA_HOST,
                   out_dir: str = "calibration/camera_intrinsics_calibration",
                   timeout: float = 10.0, auto_start_server: bool = True) -> None:
    """Grab ONE stereo pair (both eyes, from the same wire frame) and save it
    as an auto-numbered `left_NNN.png`/`right_NNN.png` pair -- the capture
    half of stereo calibration (docs/relation_deploy_plan.md §9 task 6b;
    `perception/stereo_calib.py`'s CLI consumes exactly this naming).

    Run this once per checkerboard position/tilt: hold the board somewhere
    new, run this command, repeat 15-20 times, covering different positions
    across the frame (not just the center — that's what teaches the solver
    about lens distortion, worst at the edges), different distances, and
    some tilt (not always dead-on). The pair index is picked automatically
    from whatever `left_*.png` files already exist in `out_dir` (the lowest
    unused number), so you never have to track `_000`, `_001`, ... by hand,
    and re-running after deleting a bad pair reuses that slot rather than
    leaving a permanent gap.

    Both eyes come from `HeadCamera.read_stereo()` (one wire frame, split in
    two) rather than two separate single-eye captures — the board is static,
    so a few milliseconds apart is fine, but the point is the two files are
    GUARANTEED to be the same physical moment, never at risk of drifting out
    of sync the way two independent `check camera --eye ...` calls could
    (e.g. if the operator moves the board between them without noticing).

    `auto_start_server=True` (default): SSH into `host` and start
    `image_server.py` there if it isn't already reachable, rather than just
    failing -- see `remote_image_server.py`'s module docstring. This is the
    point of this flag existing: run this command repeatedly across a whole
    capture session without ever manually SSHing in yourself.
    """
    import pathlib

    import cv2

    from ego2g1.deploy.camera import HeadCamera

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    idx = _next_pair_index(out_path)

    cam = HeadCamera(host=host, eye="left", auto_start_server=auto_start_server)
    cam.connect(timeout=timeout)
    t0 = time.monotonic()
    stereo = cam.read_stereo()
    while stereo is None:
        if time.monotonic() - t0 > timeout:
            cam.close()
            raise TimeoutError(
                f"connected to {host} but never got a full stereo pair (both "
                f"eyes) within {timeout}s — the image_server may be sending "
                "only one eye; check its cam_config.")
        time.sleep(0.05)
        stereo = cam.read_stereo()
    cam.close()

    left, right = stereo
    left_path = out_path / f"left_{idx:03d}.png"
    right_path = out_path / f"right_{idx:03d}.png"
    cv2.imwrite(str(left_path), left[..., ::-1])
    cv2.imwrite(str(right_path), right[..., ::-1])
    print(f"pair #{idx}: wrote {left_path} and {right_path}  ({left.shape} {left.dtype})")
