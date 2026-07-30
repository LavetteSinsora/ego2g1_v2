"""Rotation-vector (axis-angle) encoding: the SO(3) log/exp maps.

Why this exists alongside rot6d: 6D is the right encoding for ABSOLUTE poses
(no singularities, no wraparound, and what the datasets store), but it is the
wrong one for the small RELATIVE rotations of an anchor-relative action chunk.
Measured on red_block_in_pen_holder_ego, chunk slot 0: the 6D diagonal entries
have mean 0.99994 and std 1.4e-4, so any normalizer that gives them unit scale
amplifies by ~7000x, and the informative signal lives entirely in that noise
floor. The same rotations in rotation-vector form have std 7e-3..2e-2 -- 50x
better by std, 204x by quantile span -- because the encoding is centred on zero
instead of on the identity matrix.

Validity domain: the log map is unique only for |theta| < pi, and the axis is
undefined at theta = pi exactly. That is a real constraint on the horizon, not
a formality: it bounds how much relative rotation a chunk may contain. On
red_block_in_pen_holder_ego at H=50 the maximum observed relative rotation is
1.89 rad, so there is comfortable margin -- but re-measure before raising H.

Measured accuracy (20k random rotations, vs scipy): log(exp(v)) == v to 4e-16
for |theta| < 1.9, which is the whole operating range here. Agreement degrades
to ~8e-12 only for theta within 1e-5 of pi, where the log map is intrinsically
ill-conditioned (perturbing R by eps moves the rotvec by eps/sin theta) -- not
worth a quaternion detour given the horizon bound above.

numpy only, like the rest of core (scipy is available in the project but core's
numpy-only invariant is what lets deploy and data share this code).
"""

import numpy as np

# Below this angle, sin(theta)/theta and its inverse are evaluated by their
# Taylor series instead of by division: the direct form loses all precision as
# theta -> 0, and theta -> 0 is the COMMON case here (slot 0 of every chunk).
_SMALL_ANGLE = 1e-8


def mat_to_rotvec(R):
    """SO(3) log map: rotation matrix -> axis*angle. R (..., 3, 3) -> (..., 3).

    Uses the antisymmetric part for the generic case and falls back to an
    eigenvector-style construction near theta = pi, where the antisymmetric
    part vanishes and carries no axis information.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {R.shape}")

    # vee(R - R^T) = 2 sin(theta) * axis
    vee = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )

    # theta via atan2(sin, cos), NOT arccos(cos): arccos loses half the
    # available precision wherever |cos| -> 1, and random rotations concentrate
    # near theta = pi (the Haar density goes as 1 - cos theta). Both arguments
    # here are read straight off the matrix with absolute accuracy ~eps, which
    # is exactly the regime atan2 handles well.
    sin_theta = np.linalg.norm(vee, axis=-1) / 2.0
    cos_theta = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) / 2.0
    theta = np.arctan2(sin_theta, cos_theta)

    small = theta < _SMALL_ANGLE
    # Switch to the eigenvector branch while sin(theta) is still large enough to
    # normalize safely. sin(pi - d) ~ d, so d < 1e-6 is where vee/|vee| starts
    # losing precision; below that the antisymmetric part is numerical noise.
    near_pi = theta > np.pi - 1e-6

    # out = vee * theta / (2 sin theta). Small-angle limit of that factor is
    # 1/2 * (1 + theta^2/6); the guarded denominator keeps the unused branch
    # from producing a warning or a nan under `where`.
    safe_sin = np.where(small, 1.0, sin_theta)
    factor = np.where(small, 0.5 + theta**2 / 12.0, theta / (2.0 * safe_sin))
    out = vee * factor[..., None]

    if np.any(near_pi):
        # At theta = pi, R = I + 2*outer(a, a) - 2*I_perp, i.e. (R + I)/2 =
        # outer(a, a). The axis is the column of that matrix with the largest
        # norm (any column is a multiple of a; the largest is best conditioned).
        M = (R + np.eye(3)) / 2.0
        col_norms = np.linalg.norm(M, axis=-2)             # (..., 3)
        best = np.argmax(col_norms, axis=-1)               # (...,)
        axis = np.take_along_axis(M, best[..., None, None], axis=-1)[..., 0]
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = axis / np.maximum(norm, 1e-12)
        # Sign is genuinely ambiguous at exactly pi (+a and -a are the same
        # rotation). Resolve it consistently from the antisymmetric part where
        # that still carries signal, else fix a deterministic convention so the
        # output is at least reproducible.
        sign = np.sign(np.sum(vee * axis, axis=-1))
        sign = np.where(sign == 0.0, 1.0, sign)
        cand = axis * (theta * sign)[..., None]
        out = np.where(near_pi[..., None], cand, out)

    return out


def rotvec_to_mat(v):
    """SO(3) exp map (Rodrigues): axis*angle -> rotation matrix.
    v (..., 3) -> (..., 3, 3). Exact inverse of `mat_to_rotvec` for |theta| < pi.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"expected (..., 3), got {v.shape}")

    theta = np.linalg.norm(v, axis=-1)                      # (...,)
    small = theta < _SMALL_ANGLE

    # Rodrigues: R = I + sin(t) K + (1 - cos(t)) K^2, with K = skew(axis).
    # Written in terms of the UNNORMALIZED v to avoid dividing by theta:
    #   R = I + a1 * skew(v) + a2 * skew(v)^2
    # where a1 = sin(t)/t and a2 = (1 - cos(t))/t^2, both analytic at t = 0.
    t2 = theta**2
    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = np.where(small, 1.0 - t2 / 6.0, np.sin(theta) / np.where(small, 1.0, theta))
        a2 = np.where(small, 0.5 - t2 / 24.0, (1.0 - np.cos(theta)) / np.where(small, 1.0, t2))

    zero = np.zeros(v.shape[:-1])
    K = np.stack(
        [
            np.stack([zero, -v[..., 2], v[..., 1]], axis=-1),
            np.stack([v[..., 2], zero, -v[..., 0]], axis=-1),
            np.stack([-v[..., 1], v[..., 0], zero], axis=-1),
        ],
        axis=-2,
    )
    eye = np.broadcast_to(np.eye(3), v.shape[:-1] + (3, 3))
    return eye + a1[..., None, None] * K + a2[..., None, None] * (K @ K)


def se3_to_vec6(T):
    """SE(3) pose -> [t (3), rotvec (3)]. The relative-action encoding."""
    T = np.asarray(T)
    return np.concatenate([T[..., :3, 3], mat_to_rotvec(T[..., :3, :3])], axis=-1)


def vec6_to_se3(v):
    """Inverse of `se3_to_vec6`."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-1] + (4, 4))
    out[..., :3, :3] = rotvec_to_mat(v[..., 3:6])
    out[..., :3, 3] = v[..., :3]
    out[..., 3, 3] = 1.0
    return out
