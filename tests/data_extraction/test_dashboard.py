"""The dashboard's payload: RLE, and the centroid it is built around.

The RLE is decoded by hand-written JS in the page, so its contract has to be
exact — an off-by-one in the leading background run shifts every mask in the
episode by one pixel and looks like a calibration error, not a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_extraction.dashboard import _fit, _rle
from ego2g1.deploy.perception.v2.sam3_source import SlotObservation


def decode(runs, n):
    """The page's decoder, in Python. Runs alternate bg, fg, bg, ..."""
    out = np.zeros(n, dtype=np.uint8)
    p, v = 0, 0
    for r in runs:
        if v:
            out[p:p + r] = 1
        p += r
        v ^= 1
    return out


def roundtrip(mask: np.ndarray) -> np.ndarray:
    flat = decode(_rle(mask), mask.size)
    return flat.reshape(mask.shape).astype(bool)


def test_a_mask_survives_the_round_trip():
    rng = np.random.default_rng(0)
    m = rng.random((17, 23)) > 0.6
    np.testing.assert_array_equal(roundtrip(m), m)


def test_a_mask_starting_at_pixel_zero_gets_a_zero_length_background_run():
    """The alternation must always START with background.

    A mask whose very first pixel is set has no leading background, so the
    encoder has to emit an explicit 0 — otherwise the decoder reads the first
    run as background and every pixel in the frame inverts.
    """
    m = np.ones((3, 4), dtype=bool)
    runs = _rle(m)
    assert runs[0] == 0
    np.testing.assert_array_equal(roundtrip(m), m)


def test_an_all_background_mask_is_one_run():
    m = np.zeros((4, 5), dtype=bool)
    assert _rle(m) == [20]
    np.testing.assert_array_equal(roundtrip(m), m)


def test_runs_sum_to_the_pixel_count():
    rng = np.random.default_rng(3)
    for shape in [(1, 1), (2, 9), (31, 17)]:
        m = rng.random(shape) > 0.5
        assert sum(_rle(m)) == m.size


def test_a_solid_blob_compresses_to_a_handful_of_runs():
    """The reason this encoding exists: 77 KB of mask becomes ~2 KB of runs,
    which is the difference between a 14 MB page and a 150 MB one."""
    m = np.zeros((240, 320), dtype=bool)
    m[60:180, 90:230] = True
    assert len(_rle(m)) < 300


def test_the_centroid_is_the_plain_mean_of_mask_pixels():
    """Pinning what the dot on screen means.

    Not the bbox centre, not a median — `xs.mean(), ys.mean()`. This test
    exists because a reader looking at the dashboard has to be able to trust
    that the marker is the pixel the deploy loop back-projects.
    """
    m = np.zeros((10, 10), dtype=bool)
    m[2, 2] = m[2, 8] = m[8, 2] = True          # deliberately L-shaped
    obs = SlotObservation(instance_id="o", mask=m, box_xyxy=None,
                          det_score=None, tracker_score=0.0,
                          mask_area_px=3, occluded=False)
    u, v = obs.centroid_uv()
    assert u == pytest.approx((2 + 8 + 2) / 3)
    assert v == pytest.approx((2 + 2 + 8) / 3)
    # ...and that is NOT the bbox centre, which is what makes it informative
    assert u != pytest.approx(5.0)


def test_colour_slots_are_assigned_by_entity_never_cycled():
    """A 4th object must not wrap around to slot 1's hue — colour follows the
    entity, so a repeat would claim two objects are the same one."""
    assert _fit(["a", "b", "c"], 2) == ["a", "b"]
    assert _fit(["a", "b", "c"], 5) == ["a", "b", "c", "c", "c"]
