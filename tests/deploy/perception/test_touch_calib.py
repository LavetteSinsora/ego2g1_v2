"""solve_camera_extrinsic: the single most important test in this set, per
the task brief -- the entire touch-calibration approach (§6.2) depends on
this math being exactly right. Fully deterministic, no camera/robot needed.
"""

import numpy as np
import pytest

from ego2g1.deploy.perception.touch_calib import solve_camera_extrinsic


def _rotation_from_euler_xyz(rx, ry, rz):
    """Plain numpy R_z @ R_y @ R_x, degrees in -- avoids a scipy dependency
    in the test itself."""
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def test_recovers_known_rigid_transform_exactly():
    rng = np.random.default_rng(0)
    R_true = _rotation_from_euler_xyz(23.0, -41.0, 77.0)
    t_true = np.array([0.12, -0.34, 0.56])

    points_camera = rng.normal(scale=0.5, size=(20, 3))
    points_pelvis = (R_true @ points_camera.T).T + t_true

    T, rms = solve_camera_extrinsic(points_camera, points_pelvis)

    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, :3], R_true, atol=1e-9)
    np.testing.assert_allclose(T[:3, 3], t_true, atol=1e-9)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])
    assert rms < 1e-9

    # rotation must actually be a proper rotation (orthonormal, det=+1) --
    # a reflection would silently pass a looser "close to R_true" check
    # only because R_true itself is proper; assert the property directly.
    R_est = T[:3, :3]
    np.testing.assert_allclose(R_est @ R_est.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R_est) == pytest.approx(1.0, abs=1e-9)


def test_minimum_three_noncollinear_points():
    rng = np.random.default_rng(1)
    R_true = _rotation_from_euler_xyz(10, 20, 30)
    t_true = np.array([0.01, 0.02, -0.03])
    points_camera = rng.normal(size=(3, 3))
    points_pelvis = (R_true @ points_camera.T).T + t_true

    T, rms = solve_camera_extrinsic(points_camera, points_pelvis)
    np.testing.assert_allclose(T[:3, :3], R_true, atol=1e-8)
    np.testing.assert_allclose(T[:3, 3], t_true, atol=1e-8)
    assert rms < 1e-8


def test_rejects_fewer_than_three_points():
    points_camera = np.zeros((2, 3))
    points_pelvis = np.zeros((2, 3))
    with pytest.raises(ValueError, match=">= 3"):
        solve_camera_extrinsic(points_camera, points_pelvis)


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        solve_camera_extrinsic(np.zeros((5, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        solve_camera_extrinsic(np.zeros((5, 2)), np.zeros((5, 2)))


def test_nonzero_residual_reported_under_noise():
    rng = np.random.default_rng(2)
    R_true = _rotation_from_euler_xyz(5, -5, 15)
    t_true = np.array([0.2, 0.1, -0.1])
    points_camera = rng.normal(scale=0.3, size=(30, 3))
    points_pelvis = (R_true @ points_camera.T).T + t_true
    noise = rng.normal(scale=0.005, size=points_pelvis.shape)  # 5 mm noise
    points_pelvis_noisy = points_pelvis + noise

    T, rms = solve_camera_extrinsic(points_camera, points_pelvis_noisy)

    # recovered transform still close to ground truth, and the RMS residual
    # is on the same order as the injected noise (not zero, not huge)
    np.testing.assert_allclose(T[:3, :3], R_true, atol=0.02)
    np.testing.assert_allclose(T[:3, 3], t_true, atol=0.01)
    assert 0.001 < rms < 0.02
