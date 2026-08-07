"""Stereo derivation, presence mapping, symmetry summary.

The stereo transform is derived from recorded extrinsics rather than measured,
so the guards that check it against physics are the only thing standing
between a convention mistake and a whole dataset of confident wrong depths.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_extraction.stereo import _quat_xyzw_to_R, rig_from_episode
from data_extraction.symmetry import ALPHA_MEANING, SymmetryFitter, summarise


# -- quaternion -------------------------------------------------------------

def test_identity_quaternion_is_the_identity_rotation():
    np.testing.assert_allclose(_quat_xyzw_to_R([0, 0, 0, 1]), np.eye(3),
                               atol=1e-12)


def test_the_scalar_is_the_LAST_element():
    """Scalar-last, matching what the recording documents for every other pose
    field it writes. Read as scalar-first, [1,0,0,0] would be the identity;
    read correctly it is a 180 deg turn about x."""
    R = _quat_xyzw_to_R([1, 0, 0, 0])
    np.testing.assert_allclose(R, np.diag([1.0, -1.0, -1.0]), atol=1e-12)


def test_a_quarter_turn_about_z():
    R = _quat_xyzw_to_R([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9)


def test_an_unnormalised_quaternion_is_normalised():
    R = _quat_xyzw_to_R([0, 0, 0, 5])
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_a_degenerate_quaternion_falls_back_to_identity():
    np.testing.assert_allclose(_quat_xyzw_to_R([0, 0, 0, 0]), np.eye(3))


# -- rig derivation ---------------------------------------------------------

class FakeEpisode:
    """Two parallel eyes 64 mm apart, like the real recordings."""

    name = "fake/episode_0"
    eye = "left"
    width, height = 640, 480
    K = np.array([[407.0, 0, 319.5], [0, 407.0, 239.5], [0, 0, 1.0]])
    K_other = K.copy()
    has_stereo = True

    def __init__(self, *, baseline=0.064, right_quat=(0, 0, 0, 1)):
        self.extrinsics = {
            "left": np.array([-baseline / 2, 0, 0, 0, 0, 0, 1.0]),
            "right": np.array([baseline / 2, 0, 0, *right_quat]),
        }


def test_the_left_to_right_transform_matches_opencvs_convention():
    """OpenCV's T takes LEFT-camera coordinates to RIGHT-camera coordinates,
    so a rig whose right eye sits at +x has T pointing at -x."""
    rig = rig_from_episode(FakeEpisode())
    assert rig is not None
    assert rig.baseline_m == pytest.approx(0.064, abs=1e-6)
    np.testing.assert_allclose(rig.calib.T, [-0.064, 0, 0], atol=1e-9)
    np.testing.assert_allclose(rig.calib.R, np.eye(3), atol=1e-9)


def test_parallel_eyes_need_essentially_no_rectification():
    rig = rig_from_episode(FakeEpisode())
    assert rig.rectify_shift_px < 0.1
    assert rig.rel_rot_deg < 1e-6


def test_an_implausible_baseline_is_refused_rather_than_used(caplog):
    """A silent wrong baseline scales EVERY depth in the dataset by a constant
    and still looks entirely reasonable. Refusing beats producing."""
    assert rig_from_episode(FakeEpisode(baseline=2.0)) is None
    assert "baseline" in caplog.text


def test_eyes_that_disagree_in_rotation_are_refused(caplog):
    """Two eyes of one rigid headset are near-parallel. A large relative
    rotation means the quaternion layout is not what is assumed, and the
    right response is to stop, not to rectify nonsense."""
    q = (0, np.sin(np.pi / 8), 0, np.cos(np.pi / 8))       # 45 deg about y
    assert rig_from_episode(FakeEpisode(right_quat=q)) is None
    assert "scalar-last" in caplog.text


def test_no_stereo_is_survivable_not_fatal(caplog):
    ep = FakeEpisode()
    ep.has_stereo = False
    assert rig_from_episode(ep) is None
    assert "depth unavailable" in caplog.text


# -- symmetry ---------------------------------------------------------------

def test_alpha_zero_is_a_real_answer_not_a_missing_one():
    """0 means the fit ran and declined to claim symmetry; -1 means it never
    ran. Collapsing them would turn "the model is unsure" into "we didn't
    look", which are different problems."""
    assert 0 in ALPHA_MEANING
    assert "no confident" in ALPHA_MEANING[0]
    s = summarise(np.array([0, 0, 0, -1], dtype=np.int8))
    assert s["mode"] == 0
    assert s["measured_frames"] == 3          # the -1 is excluded


def test_summarise_reports_the_mode_and_how_much_the_episode_agreed():
    s = summarise(np.array([2, 2, 2, 4], dtype=np.int8))
    assert s["mode"] == 2
    assert s["agreement"] == pytest.approx(0.75)
    assert s["histogram"] == {2: 3, 4: 1}


def test_an_episode_with_nothing_measured_reports_nothing():
    s = summarise(np.full(5, -1, dtype=np.int8))
    assert s["mode"] is None
    assert s["measured_frames"] == 0


def test_a_disabled_fitter_returns_not_measured_for_every_row():
    f = SymmetryFitter(enabled=False)
    assert not f.available
    out = f(np.zeros((4, 360)))
    assert out.tolist() == [-1, -1, -1, -1]


def test_a_one_dimensional_distribution_is_accepted_as_a_batch_of_one():
    assert SymmetryFitter(enabled=False)(np.zeros(360)).shape == (1,)


# -- presence probe mapping -------------------------------------------------

class FakeProbe:
    """Exercises PresenceProbe's ordering contract without loading SAM 3."""


def test_presence_maps_detector_calls_to_slots_in_prompt_order():
    from data_extraction.sam3_offline import PresenceProbe

    probe = PresenceProbe.__new__(PresenceProbe)
    probe.slots_in_order = ["objA", "objB", "objC"]
    probe._calls = [0.9, 0.1, 0.5]
    assert probe.take() == {"objA": 0.9, "objB": 0.1, "objC": 0.5}


def test_a_miscounted_call_sequence_is_dropped_not_misaligned(caplog):
    """If upstream ever stops calling the detector once per prompt, a
    positional mapping would attribute one object's presence to another. That
    is worse than having no presence score, so the frame is dropped."""
    from data_extraction.sam3_offline import PresenceProbe

    probe = PresenceProbe.__new__(PresenceProbe)
    probe.slots_in_order = ["objA", "objB", "objC"]
    probe._calls = [0.9, 0.1]
    assert probe.take() == {}
    assert "dropping" in caplog.text


def test_taking_twice_does_not_repeat_the_previous_frame():
    from data_extraction.sam3_offline import PresenceProbe

    probe = PresenceProbe.__new__(PresenceProbe)
    probe.slots_in_order = ["objA"]
    probe._calls = [0.7]
    assert probe.take() == {"objA": 0.7}
    assert probe.take() == {}


def test_presence_survives_a_frame_with_no_mask():
    """The case the signal exists for: the concept is visibly in frame and
    nothing was tracked. That row must not be blank."""
    from data_extraction.sam3_offline import OfflineSam3, SlotFrame

    a = SlotFrame(presence=0.93)              # no mask
    picked = OfflineSam3._pick(a, None)
    assert not picked.has_mask
    assert picked.presence == pytest.approx(0.93)


def test_presence_survives_hotstart_retraction():
    from data_extraction.sam3_offline import OfflineSam3, SlotFrame

    m = np.zeros((4, 4), dtype=bool)
    m[1:3, 1:3] = True
    per_frame = [{"o": SlotFrame(mask_bits=np.packbits(m), mask_area_px=4,
                                 obj_id=7, presence=0.88)}]
    OfflineSam3._retract(per_frame, {7})
    assert not per_frame[0]["o"].has_mask
    assert per_frame[0]["o"].presence == pytest.approx(0.88)
