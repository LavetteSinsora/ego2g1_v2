"""`camera.py`'s stereo-read extension (docs/relation_deploy_plan.md §5.2/§9):
`HeadCamera`/`StaticCamera` must hand back BOTH eyes for relation_eef's
depth path, while `.read()`'s existing single-configured-eye behavior stays
byte-for-byte unchanged for every joint/relative_eef caller (regression-
tested explicitly below).

`HeadCamera._pump()` is exercised directly against a fake
`ImageClientCamera` double (no unitree_deploy/ZMQ needed) — same "swap the
transport, keep the pump logic" seam `connect()` itself uses.
"""

import threading
import time

import numpy as np
import pytest

from ego2g1.deploy.camera import HeadCamera, StaticCamera


class _FakeImageClient:
    """Minimal ImageClientCamera stand-in: `async_read()` returns the
    split-dict shape `HeadCamera._pump` expects (`cam_{eye}_high` keys),
    one entry (or omission, to simulate a miss) per call."""

    def __init__(self, left_frames=(), right_frames=()):
        self._left = list(left_frames)
        self._right = list(right_frames)
        self._i = 0

    def async_read(self):
        out = {}
        if self._i < len(self._left):
            out["cam_left_high"] = self._left[self._i]
        if self._i < len(self._right):
            out["cam_right_high"] = self._right[self._i]
        self._i += 1
        return out

    def disconnect(self):
        pass


def _frame(fill, shape=(4, 4, 3)):
    return np.full(shape, fill, dtype=np.uint8)


def _pump(cam: HeadCamera, client: _FakeImageClient, timeout: float = 2.0) -> HeadCamera:
    """Wire the fake client straight in and start the pump thread — exactly
    what `connect()` does, minus the real `ImageClientCamera` construction."""
    cam._client = client
    cam._thread = threading.Thread(target=cam._pump, daemon=True)
    cam._thread.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cam.read() is not None:
            return cam
        time.sleep(0.005)
    raise TimeoutError("fake camera never produced a frame")


# --------------------------------------------------------------------------
# HeadCamera: read_stereo() new behavior
# --------------------------------------------------------------------------


def test_read_stereo_returns_both_eyes_regardless_of_configured_eye():
    cam = HeadCamera(host="fake", eye="left", flip_bgr=False)
    left, right = _frame(10), _frame(20)
    try:
        _pump(cam, _FakeImageClient([left], [right]))
        stereo = cam.read_stereo()
        assert stereo is not None
        got_left, got_right = stereo
        np.testing.assert_array_equal(got_left, left)
        np.testing.assert_array_equal(got_right, right)
    finally:
        cam.close()


def test_read_stereo_none_until_both_eyes_have_arrived():
    cam = HeadCamera(host="fake", eye="left", flip_bgr=False)
    try:
        # only the left eye ever arrives (right never does)
        _pump(cam, _FakeImageClient([_frame(5)], []))
        assert cam.read_stereo() is None
        # the configured (left) eye is available on its own via read()
        assert cam.read() is not None
    finally:
        cam.close()


def test_read_stereo_reflects_flip_bgr_on_both_eyes():
    cam = HeadCamera(host="fake", eye="left", flip_bgr=True)
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 1   # B
    bgr[..., 2] = 3   # R
    try:
        _pump(cam, _FakeImageClient([bgr], [bgr]))
        left, right = cam.read_stereo()
        # flip_bgr reverses the channel axis: old R (index 2) becomes index 0
        assert left[0, 0, 0] == 3 and left[0, 0, 2] == 1
        assert right[0, 0, 0] == 3 and right[0, 0, 2] == 1
    finally:
        cam.close()


def test_read_stereo_copies_are_independent_of_internal_state():
    cam = HeadCamera(host="fake", eye="left", flip_bgr=False)
    try:
        _pump(cam, _FakeImageClient([_frame(1)], [_frame(2)]))
        left, right = cam.read_stereo()
        left[0, 0, 0] = 255
        left2, _right2 = cam.read_stereo()
        assert left2[0, 0, 0] == 1   # mutating the returned copy didn't leak in
    finally:
        cam.close()


# --------------------------------------------------------------------------
# HeadCamera: read()/age() regression — UNCHANGED single-eye behavior
# --------------------------------------------------------------------------


def test_read_unchanged_returns_only_the_configured_eye():
    cam = HeadCamera(host="fake", eye="right", flip_bgr=False)
    left, right = _frame(1), _frame(2)
    try:
        _pump(cam, _FakeImageClient([left], [right]))
        got = cam.read()
        np.testing.assert_array_equal(got, right)   # configured eye == right
        assert cam.read().shape == right.shape
    finally:
        cam.close()


def test_read_returns_none_before_any_frame_and_a_copy_after():
    cam = HeadCamera(host="fake", eye="left", flip_bgr=False)
    assert cam.read() is None            # nothing pumped yet
    assert cam.age() == float("inf")
    try:
        _pump(cam, _FakeImageClient([_frame(9)], [_frame(9)]))
        a = cam.read()
        a[0, 0, 0] = 0
        b = cam.read()
        assert b[0, 0, 0] == 9            # independent copy, same as before
        assert cam.age() < 1.0
    finally:
        cam.close()


def test_read_only_updates_from_the_configured_eyes_own_arrival():
    """Regression for the exact original semantics: `.read()`/`.age()` must
    only reflect the CONFIGURED eye arriving, not just "some" frame — a
    right-eye-only tick must not make a left-configured camera think it has
    a fresh frame."""
    cam = HeadCamera(host="fake", eye="left", flip_bgr=False)
    try:
        cam._client = _FakeImageClient([], [_frame(7)])   # right only, ever
        cam._thread = threading.Thread(target=cam._pump, daemon=True)
        cam._thread.start()
        time.sleep(0.05)
        assert cam.read() is None
    finally:
        cam.close()


# --------------------------------------------------------------------------
# StaticCamera: read_stereo() + read() regression
# --------------------------------------------------------------------------


def test_static_camera_read_stereo_defaults_to_the_same_frame_both_eyes():
    frame = np.full((3, 3, 3), 7, dtype=np.uint8)
    cam = StaticCamera(frame=frame)
    left, right = cam.read_stereo()
    np.testing.assert_array_equal(left, frame)
    np.testing.assert_array_equal(right, frame)
    assert left is not right   # independent copies, not the same array object


def test_static_camera_read_stereo_with_distinct_right_frame():
    frame_l = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_r = np.ones((2, 2, 3), dtype=np.uint8)
    cam = StaticCamera(frame=frame_l, frame_right=frame_r)
    left, right = cam.read_stereo()
    np.testing.assert_array_equal(left, frame_l)
    np.testing.assert_array_equal(right, frame_r)


def test_static_camera_read_unchanged():
    cam = StaticCamera()
    frame = cam.read()
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8
    assert cam.age() == 0.0
