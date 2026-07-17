"""Causal smoothing for the target path — the actual jitter fix.

docs/jitter_root_cause.md, measured: the servo is quiet on a constant command,
but Pico tracking noise (1197 deg/s² angular accel RMS raw vs 156 smoothed)
passes through a *converged* IK into 26+ rad/s² joint zig-zag, and more IK
iterations change nothing. So smoothing lives here, in the target path, in two
places with different jobs:

  OneEuro / OneEuroSE3   BEFORE IK, on the EEF target stream. Adaptive: a still
                         hand is smoothed hard, a fast deliberate move tracks
                         with almost no lag (Casiez et al. 2012) — a fixed EMA
                         costs up to 8° orientation lag on fast segments for
                         the same smoothing (measured, episode_1).
  JointFilter            AFTER IK, 4-tap weighted MA on the joint solution.
                         Safety net for residual null-space wander (the
                         redundant elbow). Ported from zh_deploy_inference's
                         g1_kinematics._MovingFilter — the proven-smooth stack.

Offline (non-causal) label smoothing is a different animal and lives in the
extraction pipeline (SavGol, ego2g1.data s08_smooth).
"""

import numpy as np

from ..core import frames


class OneEuro:
    """One-Euro filter, per-component over an arbitrary vector.

    The cutoff rises with estimated speed (`min_cutoff + beta*|dx|`), which is
    the whole point: jitter suppression scales inversely with intent. Ported
    from human_hand_teleoperate.retarget._OneEuro (proven on Revo2 finger
    commands); dtype relaxed to float64 for joint/pose use.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, x=None) -> None:
        """Seed so the next value starts FROM `x` (no snap on re-engage)."""
        if x is None:
            self.x_prev = self.dx_prev = None
            return
        self.x_prev = np.asarray(x, dtype=np.float64).copy()
        self.dx_prev = np.zeros_like(self.x_prev)

    def filter(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None or dt <= 0.0:
            self.reset(x)
            return self.x_prev.copy()
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)   # per-component
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat.copy()


class OneEuroSE3:
    """One-Euro over an SE3 pose: position per-axis, rotation by adaptive slerp.

    Rotation cannot go through the vector filter (it would leave SO(3)), so the
    same adaptive-alpha idea is applied as a slerp toward the new sample, with
    the speed estimate taken from the geodesic angle rate. Rotational noise is
    what the Jacobian amplifies worst (wrist joints), so beta_rot defaults
    lower — smooth harder — than translation.
    """

    def __init__(self, min_cutoff: float = 1.0, beta_pos: float = 0.5,
                 min_cutoff_rot: float = 0.6, beta_rot: float = 0.15,
                 d_cutoff: float = 1.0):
        self._pos = OneEuro(min_cutoff, beta_pos, d_cutoff)
        self.min_cutoff_rot = float(min_cutoff_rot)
        self.beta_rot = float(beta_rot)
        self.d_cutoff = float(d_cutoff)
        self.q_prev: np.ndarray | None = None      # wxyz
        self.w_prev: float = 0.0                   # filtered |angular rate|, rad/s

    def reset(self) -> None:
        self._pos.reset()
        self.q_prev = None
        self.w_prev = 0.0

    def filter(self, T: np.ndarray, dt: float) -> np.ndarray:
        """T: (4,4) SE3. Returns the smoothed (4,4) SE3."""
        pos, quat = frames.pose_of(T)
        p_hat = self._pos.filter(pos, dt)
        if self.q_prev is None or dt <= 0.0:
            self.q_prev = np.asarray(quat, dtype=np.float64).copy()
            self.w_prev = 0.0
            return frames.se3(p_hat, self.q_prev)
        # keep the quaternion on the same hemisphere as the running state
        q = np.asarray(quat, dtype=np.float64)
        if np.dot(q, self.q_prev) < 0.0:
            q = -q
        ang = np.deg2rad(frames.rot_geodesic_deg(
            frames.mat_from_quat(self.q_prev), frames.mat_from_quat(q)))
        a_d = OneEuro._alpha(self.d_cutoff, dt)
        self.w_prev = a_d * (ang / dt) + (1.0 - a_d) * self.w_prev
        cutoff = self.min_cutoff_rot + self.beta_rot * self.w_prev
        a = OneEuro._alpha(cutoff, dt)
        q_hat = frames.quat_normalize(frames.quat_slerp(self.q_prev, q, a))
        self.q_prev = q_hat
        return frames.se3(p_hat, q_hat)


class JointFilter:
    """Weighted moving average over the last few IK solutions; newest first.

    Verbatim port of the deploy `_JointFilter` (itself from zh's
    g1_kinematics._MovingFilter). Warm-up passes through so the first ticks
    after a reseed are not dragged toward zero.
    """

    def __init__(self, weights=(0.4, 0.3, 0.2, 0.1)):
        self._w = np.asarray(weights, dtype=np.float64)
        if not np.isclose(self._w.sum(), 1.0):
            raise ValueError("filter weights must sum to 1")
        self._q: list[np.ndarray] = []

    def add(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        self._q.append(value.copy())
        self._q = self._q[-len(self._w):]
        if len(self._q) < len(self._w):
            return value                       # warm-up: pass through
        stack = np.asarray(self._q[::-1])      # newest gets the largest weight
        return np.sum(stack * self._w[:, None], axis=0)

    def clear(self) -> None:
        self._q.clear()
