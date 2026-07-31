"""The ~0.2 Hz orientation-refresh stage (docs/relation_deploy_plan.md §5.3).

Cheap enough to call often, but this module has no clock of its own -- the
plan is explicit that "a caller controls the cadence, not this module", so
`OrientationRefiner.refresh(...)` just does the (symmetry-group-aware
rotation-snapping) work whenever it is called, at whatever rate the
perception loop decides.

Algorithm reference: `data_extraction_zh/src/ego_relation/
s2_object_relations/stereo_fusion.py::_nearest_symmetric_rotation` (read-only
reference, reimplemented cleanly here, not imported -- that repo is its own
uv project). That function hardcodes the cube's 24-element rotational
symmetry group; this version takes the symmetry group as a parameter so a
non-cube-symmetric object (this checkpoint's pen holder, per
docs/relation_deploy_plan.md §5.3) can use `identity_symmetry_group()` --
a single-element group containing only the identity -- to get an exact
pass-through instead of a hardcoded cube special-case.
"""

from __future__ import annotations

import itertools

import numpy as np

from ...core import rotvec as _rotvec

SymmetryGroup = tuple[np.ndarray, ...]


def cube_symmetry_group() -> SymmetryGroup:
    """The 24 proper (det = +1) rotational symmetries of a cube: every
    signed permutation matrix with determinant +1. Reimplementation of
    `stereo_fusion.py::_proper_cube_symmetries` (same construction: for each
    axis permutation and each sign pattern, keep the ones that are proper
    rotations, i.e. determinant +1 not -1)."""
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            matrix[np.arange(3), permutation] = signs
            if np.linalg.det(matrix) > 0.5:
                rotations.append(matrix)
    return tuple(rotations)


def identity_symmetry_group() -> SymmetryGroup:
    """No rotational symmetry: the only "equivalent" representation of a
    measured rotation is itself. Use this for objects like a pen holder that
    have no meaningful rotational symmetry -- `nearest_symmetric_rotation`
    with this group is an exact pass-through (see its docstring)."""
    return (np.eye(3, dtype=np.float64),)


def _rotation_angle(R: np.ndarray) -> float:
    """Geodesic angle (rad) of a rotation matrix, via the existing SO(3) log
    map (`ego2g1.core.rotvec.mat_to_rotvec`) rather than reimplementing an
    angle formula -- reuse, don't re-derive (docs/relation_deploy_plan.md's
    own stated principle for `core/rotvec.py`).

    The cube symmetry group contains 180-degree rotations, and
    `mat_to_rotvec`'s generic-case branch divides by `sin(theta)` before its
    own near-pi override kicks in -- a real (harmless, correctly-overridden)
    RuntimeWarning at exactly theta=pi. Suppressed here at the call site
    rather than touching `rotvec.py`, which this module reuses verbatim.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.linalg.norm(_rotvec.mat_to_rotvec(R)))


def nearest_symmetric_rotation(
    reference: np.ndarray,
    measured: np.ndarray,
    symmetry_group: SymmetryGroup,
) -> np.ndarray:
    """Snap `measured` onto whichever symmetry-equivalent representation is
    geodesically closest to `reference`.

    For a symmetric object, `measured @ S` for any `S` in its rotational
    symmetry group describes the exact same physical orientation as
    `measured` alone (the object is invariant under `S`). A raw per-frame
    orientation estimate can therefore land on any of those equivalent
    branches independently frame to frame, which -- fed downstream as a
    literal rotation matrix -- looks like the object spontaneously snapping
    through a large-angle jump even though nothing physically moved. Picking
    the branch closest to `reference` (typically: whatever this tracker's
    own last accepted/smoothed rotation was) keeps the reported rotation
    temporally consistent.

    This does NOT denoise `measured` -- it only resolves the discrete
    symmetry ambiguity. With `symmetry_group = identity_symmetry_group()`
    (a single identity element) this is an exact, unconditional pass-through:
    `measured @ eye(3) == measured`, for any `reference`.
    """
    reference = np.asarray(reference, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    best_candidate = None
    best_angle = np.inf
    for symmetry in symmetry_group:
        candidate = measured @ symmetry
        angle = _rotation_angle(reference.T @ candidate)
        if angle < best_angle:
            best_angle = angle
            best_candidate = candidate
    return best_candidate


class OrientationRefiner:
    """Stateful wrapper: remembers the last snapped rotation as the
    `reference` for the next call, so consecutive `refresh()` calls stay on
    a temporally-consistent symmetry branch. Holds no notion of time/cadence
    -- call `refresh()` whenever the perception loop's own ~0.2 Hz timer (or
    any other policy) decides a fresh orientation measurement is ready.
    """

    def __init__(
        self,
        symmetry_group: SymmetryGroup,
        *,
        initial_rotation: np.ndarray | None = None,
    ):
        self._group = tuple(
            np.asarray(s, dtype=np.float64) for s in symmetry_group
        )
        if not self._group:
            raise ValueError("symmetry_group must have at least one element")
        self._current = (
            None
            if initial_rotation is None
            else np.asarray(initial_rotation, dtype=np.float64).copy()
        )

    def refresh(self, measured_rotation: np.ndarray) -> np.ndarray:
        """Feed a new raw rotation measurement, get back the symmetry-snapped
        rotation (also stored as the reference for the next call)."""
        measured = np.asarray(measured_rotation, dtype=np.float64)
        if self._current is None:
            self._current = measured.copy()
        else:
            self._current = nearest_symmetric_rotation(
                self._current, measured, self._group
            )
        return self._current.copy()

    @property
    def rotation(self) -> np.ndarray:
        if self._current is None:
            raise RuntimeError(
                "OrientationRefiner.rotation read before any refresh() call"
            )
        return self._current.copy()
