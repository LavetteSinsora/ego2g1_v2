"""6D rotation encoding (Zhou et al.) + 9-dim pose vectors.

Conventions (must match the loader transform and the dashboard exactly):
- 6d(R) = concat(R[:, 0], R[:, 1])  - the first two COLUMNS of the rotation.
- vec9(T) = [t (3), 6d(R) (6)] for a 4x4 SE(3) pose T.
Decoding Gram-Schmidts the two columns, so any regressed 6-vector maps to a
valid rotation.
"""

import numpy as np


def mat_to_6d(R):
    R = np.asarray(R)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_mat(d6):
    d6 = np.asarray(d6, dtype=np.float64)
    a, b = d6[..., :3], d6[..., 3:6]
    x = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b - (x * b).sum(axis=-1, keepdims=True) * x
    y = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)   # columns


def se3_to_vec9(T):
    T = np.asarray(T)
    return np.concatenate([T[..., :3, 3], mat_to_6d(T[..., :3, :3])], axis=-1)


def vec9_to_se3(v):
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-1] + (4, 4))
    out[..., :3, :3] = rot6d_to_mat(v[..., 3:9])
    out[..., :3, 3] = v[..., :3]
    out[..., 3, 3] = 1.0
    return out
