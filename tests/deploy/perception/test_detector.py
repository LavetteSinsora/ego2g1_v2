"""detector.py: FakeDetector's own sanity check, plus GroundingDinoSam2Detector's
constructor/interface shape -- none of this needs real model weights or a GPU.
"""

import numpy as np
import pytest

from ego2g1.deploy.perception.detector import (
    Detection,
    FakeDetector,
    GroundingDinoSam2Detector,
    ObjectQuery,
)


def _image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)


class TestDetection:
    def test_requires_mask_or_box(self):
        with pytest.raises(ValueError, match="needs a mask, a box"):
            Detection(instance_id="obj1", confidence=0.9)

    def test_centroid_from_box(self):
        det = Detection(
            instance_id="obj1", confidence=0.9,
            box_xyxy=np.array([10.0, 20.0, 30.0, 60.0]),
        )
        np.testing.assert_allclose(det.centroid_uv(), [20.0, 40.0])

    def test_centroid_from_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:6, 2:4] = True  # rows 4-5, cols 2-3 -> centroid (2.5, 4.5)
        det = Detection(instance_id="obj1", confidence=0.9, mask=mask)
        np.testing.assert_allclose(det.centroid_uv(), [2.5, 4.5])

    def test_rejects_non_2d_mask(self):
        with pytest.raises(ValueError, match="must be"):
            Detection(
                instance_id="obj1", confidence=0.9,
                mask=np.zeros((3, 4, 4), dtype=bool),
            )


class TestFakeDetector:
    def test_returns_programmed_detections(self):
        det_a = Detection(
            instance_id="a", confidence=0.95,
            box_xyxy=np.array([0.0, 0.0, 10.0, 10.0]),
        )
        det_b = Detection(
            instance_id="b", confidence=0.5,
            box_xyxy=np.array([5.0, 5.0, 15.0, 15.0]),
        )
        fake = FakeDetector({"a": det_a, "b": det_b})
        queries = [ObjectQuery("a", "prompt a ."), ObjectQuery("b", "prompt b .")]
        out = fake.detect(_image(), queries)
        assert out == {"a": det_a, "b": det_b}

    def test_missing_instance_is_simply_absent(self):
        fake = FakeDetector()  # nothing programmed
        out = fake.detect(_image(), [ObjectQuery("ghost", "a ghost .")])
        assert out == {}

    def test_set_and_clear_detection(self):
        fake = FakeDetector()
        det = Detection(
            instance_id="a", confidence=0.9,
            box_xyxy=np.array([0.0, 0.0, 1.0, 1.0]),
        )
        fake.set_detection("a", det)
        assert fake.detect(_image(), [ObjectQuery("a", "x")]) == {"a": det}
        fake.clear_detection("a")
        assert fake.detect(_image(), [ObjectQuery("a", "x")]) == {}

    def test_set_detection_rejects_id_mismatch(self):
        fake = FakeDetector()
        det = Detection(
            instance_id="a", confidence=0.9,
            box_xyxy=np.array([0.0, 0.0, 1.0, 1.0]),
        )
        with pytest.raises(ValueError, match="instance_id"):
            fake.set_detection("b", det)

    def test_records_calls(self):
        fake = FakeDetector()
        image = _image()
        queries = [ObjectQuery("a", "x"), ObjectQuery("b", "y")]
        fake.detect(image, queries)
        fake.detect(image, queries[:1])
        assert len(fake.calls) == 2
        assert fake.calls[0][1] == ("a", "b")
        assert fake.calls[1][1] == ("a",)
        np.testing.assert_array_equal(fake.calls[0][0], image)

    def test_deterministic_regardless_of_image_content(self):
        det = Detection(
            instance_id="a", confidence=0.9,
            box_xyxy=np.array([0.0, 0.0, 1.0, 1.0]),
        )
        fake = FakeDetector({"a": det})
        out1 = fake.detect(_image(seed=1), [ObjectQuery("a", "x")])
        out2 = fake.detect(_image(seed=2), [ObjectQuery("a", "x")])
        assert out1 == out2 == {"a": det}


class TestGroundingDinoSam2DetectorShape:
    def test_missing_optional_dependency_raises_clear_error(self):
        """Without the `perception` dependency group installed, constructing
        this class must fail with an actionable message (pointing at the
        pyproject dependency group), not a bare ImportError/ModuleNotFoundError
        surfacing from deep inside transformers/sam2 -- mirrors
        `deploy/executor.py`'s lazy-import-with-clear-message pattern."""
        try:
            import transformers  # noqa: F401
            import sam2  # noqa: F401
            pytest.skip("perception deps are installed in this environment")
        except ImportError:
            pass

        with pytest.raises(RuntimeError, match="perception"):
            GroundingDinoSam2Detector()

    def test_constructor_accepts_expected_keyword_args(self):
        """The constructor's signature must accept every documented knob
        without raising a TypeError before it even gets to the (possibly
        missing) heavy imports -- verified by checking the failure is always
        the dependency RuntimeError, never a TypeError from a bad signature."""
        try:
            import transformers  # noqa: F401
            import sam2  # noqa: F401
            pytest.skip("perception deps are installed in this environment")
        except ImportError:
            pass

        with pytest.raises(RuntimeError):
            GroundingDinoSam2Detector(
                dino_model_id="IDEA-Research/grounding-dino-tiny",
                sam2_config="sam2_hiera_l.yaml",
                sam2_repo_id="facebook/sam2-hiera-large",
                sam2_checkpoint_name="sam2_hiera_large.pt",
                box_threshold=0.25,
                device="cpu",
            )

    def test_is_an_object_detector(self):
        from ego2g1.deploy.perception.detector import ObjectDetector
        assert issubclass(GroundingDinoSam2Detector, ObjectDetector)
        assert issubclass(FakeDetector, ObjectDetector)
