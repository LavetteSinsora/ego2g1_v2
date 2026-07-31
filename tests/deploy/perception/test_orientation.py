"""orientation.py: cube-symmetry snapping recovers the correct branch under
small noise, and the identity/no-symmetry group is an exact pass-through.
Pure numpy math -- no model weights, no GPU, no image needed.
"""

import itertools

import numpy as np
import pytest

from ego2g1.core import rotvec
from ego2g1.deploy.perception.orientation import (
    OrientationRefiner,
    cube_symmetry_group,
    identity_symmetry_group,
    nearest_symmetric_rotation,
)


def _angle(R: np.ndarray) -> float:
    # theta=pi (present among the cube's 180-degree symmetries) triggers a
    # harmless, correctly-overridden RuntimeWarning inside mat_to_rotvec.
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.linalg.norm(rotvec.mat_to_rotvec(R)))


@pytest.fixture(scope="module")
def cube_group():
    group = cube_symmetry_group()
    assert len(group) == 24
    for R in group:
        # every element must actually be a proper rotation
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) > 0.5
    return group


@pytest.fixture(scope="module")
def min_group_gap_rad(cube_group):
    """Smallest geodesic angle between any two DISTINCT elements of the cube
    group -- used to size the test's noise so it can never straddle the
    boundary between two symmetry branches."""
    gaps = [
        _angle(cube_group[i].T @ cube_group[j])
        for i, j in itertools.permutations(range(len(cube_group)), 2)
    ]
    return min(gaps)


class TestCubeSymmetryGroup:
    def test_has_24_elements_including_identity(self, cube_group):
        assert any(np.allclose(R, np.eye(3)) for R in cube_group)

    def test_closed_under_inverse(self, cube_group):
        for R in cube_group:
            assert any(np.allclose(R.T, S) for S in cube_group)

    def test_min_gap_is_meaningfully_large(self, min_group_gap_rad):
        # sanity: the cube's rotational symmetries are well separated (the
        # smallest step is a 90 degree face rotation), so "small noise" has
        # a lot of safe margin to live in.
        assert min_group_gap_rad > np.deg2rad(45)


class TestNearestSymmetricRotationRecoversTrueBranch:
    def test_exact_recovery_with_zero_noise(self, cube_group):
        reference = rotvec.rotvec_to_mat(np.array([0.3, -0.2, 0.1]))
        s_true = next(S for S in cube_group if not np.allclose(S, np.eye(3)))
        measured = reference @ s_true  # same physical pose, "wrong" branch

        candidate = nearest_symmetric_rotation(reference, measured, cube_group)

        np.testing.assert_allclose(candidate, reference, atol=1e-9)

    @pytest.mark.parametrize("noise_scale", [0.01, 0.05, 0.15])
    def test_small_noise_does_not_change_which_branch_is_picked(
        self, cube_group, min_group_gap_rad, noise_scale
    ):
        rng = np.random.default_rng(0)
        reference = rotvec.rotvec_to_mat(np.array([0.3, -0.2, 0.1]))
        s_true = next(S for S in cube_group if not np.allclose(S, np.eye(3)))

        # Safety margin: noise well under a quarter of the smallest gap
        # between distinct symmetry branches, so it can never look closer to
        # the wrong branch than to the true one.
        safe_bound = min_group_gap_rad / 4.0
        assert noise_scale < safe_bound

        noise_vec = rng.normal(scale=noise_scale / np.sqrt(3), size=3)
        noise = rotvec.rotvec_to_mat(noise_vec)
        measured = noise @ (reference @ s_true)

        candidate = nearest_symmetric_rotation(reference, measured, cube_group)

        # candidate must equal noise @ reference exactly (deterministic
        # algebra, not an approximation) -- i.e. the algorithm picked
        # S_true^{-1} and only S_true^{-1}, undoing exactly the injected
        # branch ambiguity and leaving only the (small) noise behind.
        np.testing.assert_allclose(candidate, noise @ reference, atol=1e-9)
        # and that residual really is small -- confirms "same symmetry
        # class as ground truth" in the geodesic sense too.
        assert _angle(reference.T @ candidate) < np.deg2rad(30)

    def test_result_is_always_a_literal_group_coset_of_measured(self, cube_group):
        """candidate must be measured @ S for SOME S actually in the group --
        never an interpolated/averaged rotation outside the coset."""
        reference = rotvec.rotvec_to_mat(np.array([-0.4, 0.6, 0.2]))
        measured = rotvec.rotvec_to_mat(np.array([1.0, -0.3, 0.5]))
        candidate = nearest_symmetric_rotation(reference, measured, cube_group)
        assert any(
            np.allclose(candidate, measured @ S, atol=1e-9) for S in cube_group
        )


class TestIdentitySymmetryGroupIsPassThrough:
    def test_identity_group_has_one_element(self):
        group = identity_symmetry_group()
        assert len(group) == 1
        np.testing.assert_allclose(group[0], np.eye(3))

    def test_measured_returned_unchanged_regardless_of_reference(self):
        group = identity_symmetry_group()
        measured = rotvec.rotvec_to_mat(np.array([0.9, 0.3, -0.4]))
        for reference_vec in ([0.0, 0.0, 0.0], [0.1, -0.2, 0.05], [2.0, -1.0, 0.5]):
            reference = rotvec.rotvec_to_mat(np.array(reference_vec))
            candidate = nearest_symmetric_rotation(reference, measured, group)
            np.testing.assert_allclose(candidate, measured, atol=1e-12)


class TestOrientationRefiner:
    def test_requires_nonempty_symmetry_group(self):
        with pytest.raises(ValueError):
            OrientationRefiner(())

    def test_first_refresh_seeds_reference_unchanged(self, cube_group):
        refiner = OrientationRefiner(cube_group)
        measured = rotvec.rotvec_to_mat(np.array([0.2, 0.1, -0.3]))
        out = refiner.refresh(measured)
        np.testing.assert_allclose(out, measured)
        np.testing.assert_allclose(refiner.rotation, measured)

    def test_rotation_property_before_any_refresh_raises(self, cube_group):
        refiner = OrientationRefiner(cube_group)
        with pytest.raises(RuntimeError):
            _ = refiner.rotation

    def test_subsequent_refresh_snaps_to_the_stored_reference(self, cube_group):
        refiner = OrientationRefiner(cube_group)
        canonical = rotvec.rotvec_to_mat(np.array([0.3, -0.2, 0.1]))
        refiner.refresh(canonical)

        s_true = next(S for S in cube_group if not np.allclose(S, np.eye(3)))
        wrong_branch = canonical @ s_true
        out = refiner.refresh(wrong_branch)

        np.testing.assert_allclose(out, canonical, atol=1e-9)

    def test_identity_group_refiner_is_pass_through_every_call(self):
        refiner = OrientationRefiner(identity_symmetry_group())
        rng = np.random.default_rng(1)
        for _ in range(5):
            measured = rotvec.rotvec_to_mat(rng.normal(scale=0.5, size=3))
            out = refiner.refresh(measured)
            np.testing.assert_allclose(out, measured, atol=1e-12)
