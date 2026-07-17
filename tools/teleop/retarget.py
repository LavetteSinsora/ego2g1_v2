"""Human hands -> robot flange targets + Revo2 motor commands, one frame at a time.

This is the causal half of `data_extraction`'s s002 stages, reusing their code rather
than re-deriving it: `HandRetargeter.step` IS the offline finger solve (proven
bit-identical), and `B` is loaded from the very `b_calib` npz the training labels used.
Only what cannot run causally is replaced -- the placement fit `S` is dropped (it
cancels; see below), and calibration comes from a session-start open-hand hold instead
of scanning an episode.

--------------------------------------------------------------------------------------
How a wrist pose becomes a flange target
--------------------------------------------------------------------------------------

The pipeline's label decomposes the flange pose as   G(t) = pelvis^-1 . S . T_w(t) . B,
i.e. left factor C := pelvis^-1 . S (a heading yaw + the base), the wrist pose T_w(t),
and right factor B (the wrist->flange convention). Orientation and position are:

    R_flange(t) = C . R_w(t) . B_R
    p_flange(t) = C . p_w(t) + (translation of pelvis^-1 . S)

DEFAULT MODE = "absolute" orientation, position relative to engage. This is what a human
actually wants to teleoperate with:

    R_target(t) = C . R_w(t) . B_R                       # ABSOLUTE: hand pose <-> flange
                                                          #   pose is a FIXED correspondence
    p_target(t) = p_engage + C . ( p_w(t) - p_w_engage )  # RELATIVE to the engage anchor

  * Orientation is absolute because a rotation has no origin, no scale, and no workspace
    limit -- so it maps through the single fixed `C` with nothing to calibrate per
    engage. Rotate your hand 90 deg and the flange rotates 90 deg the same way, every
    time, no matter where or when you engaged. That fixed correspondence is the thing
    that makes teleop feel natural.
  * Position is relative because it has all three problems: the Pico world origin is
    wherever the headset booted, a human's reach exceeds the G1's, and mapped points must
    be reachable. The engage anchor is the live stand-in for S's fitted translation --
    it picks the origin correspondence at engage, and re-engaging (the clutch) re-picks
    it when you run out of workspace. Note the displacement is rotated by the SAME fixed
    `C`, so "world right" maps to "base right" regardless of how your hand is turned.

  `C` is a single session-fixed heading yaw (shared by both hands): the only unknown that
  absolute orientation introduces is how the operator's Pico-world "forward" relates to
  the robot's "forward". `set_heading` estimates it once from a matched engage pose; it
  cannot be positioned away (it is the operator's heading), but it has no workspace cost.

  At engage, position is identity (no jump) but the absolute orientation may differ from
  where the flange is -- so the orientation is RAMPED from the anchor to the absolute
  target over `engage_ramp_s`. At the instant of engage the target still equals the
  anchor exactly.

MODE = "relative" keeps the fully-relative scheme (both orientation and position
differenced against engage) for A/B comparison:

    G_target(t) = G_engage . B^-1 . ( T_w_engage^-1 . T_w(t) ) . B

Here S and the world frame cancel without ever being known -- the property the offline
equivalence test leans on. The trade is that the zero-point is whatever pose you engage
in, so engaging twisted feels twisted.

Both modes record the identical body-frame action delta downstream (the schema
differences the flange poses either way), so the choice is pure operator feel.
"""

import json
import pathlib

import numpy as np

from ._vendor.de.common import frames
from ._vendor.de.hand.constants import CMD_RATE_LIMIT, MOTOR_ORDER
from ._vendor.de.hand.retarget import HandRetargeter, wrist_frame_tips

SIDES = ("left", "right")
RATES = np.array([CMD_RATE_LIMIT[m] for m in MOTOR_ORDER], dtype=np.float32)


class _OneEuro:
    """One-Euro filter (Casiez et al. 2012), per-component over the 6-vector command.

    The velocity clamp (`TeleopRetargeter._limit`) caps how fast a finger MAY move but does
    nothing to jitter that lives under the cap -- so raw tracker noise passes straight to the
    Revo2. This adds the missing low-pass, and it is adaptive: the cutoff rises with |speed|
    (`min_cutoff + beta*|dx|`), so a still hand is smoothed hard while a deliberate fast close
    is tracked with almost no lag. That speed/lag trade is exactly why VR hand tracking uses
    it over a fixed EMA.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, x) -> None:
        """Seed the state so the next filtered value starts FROM `x` (used to hold a dropped
        hand's command without a snap on re-acquire)."""
        self.x_prev = np.asarray(x, dtype=np.float32).copy()
        self.dx_prev = np.zeros_like(self.x_prev)

    def filter(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None or dt <= 0.0:
            self.reset(x)
            return self.x_prev.copy()
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)   # per-component
        a = self._alpha(cutoff, dt)
        x_hat = (a * x + (1.0 - a) * self.x_prev).astype(np.float32)
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat.copy()


def load_B(source: str | pathlib.Path) -> dict[str, np.ndarray]:
    """The wrist->flange alignment, from wherever the training labels recorded it.

    Accepts either the pipeline's `b_calib` npz or a LeRobot dataset root (whose
    `extraction_meta.json` carries B per episode). It MUST be the same B the labels
    used: B sits inside every flange target, so a teleop session running a different one
    is steering a different robot than the policy was trained on.
    """
    p = pathlib.Path(source)
    if p.is_dir():
        meta = json.loads((p / "extraction_meta.json").read_text())
        ep = next(iter(meta["episodes"].values()))
        return {s: np.asarray(ep[f"B_{s}"], dtype=np.float64) for s in SIDES}
    with np.load(p) as z:
        return {s: np.asarray(z[f"B_{s}"], dtype=np.float64) for s in SIDES}


def _yaw_angle(R: np.ndarray) -> float:
    """The heading (rotation about vertical z) of a near-upright rotation: the angle of
    its x-axis in the ground plane. Same idea as televuer's head-yaw extraction."""
    x = R[:, 0]
    return float(np.arctan2(x[1], x[0]))


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _mean_angle(angles) -> float:
    a = np.asarray(angles)
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def _se3(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _slerp_R(Ra: np.ndarray, Rb: np.ndarray, u: float) -> np.ndarray:
    q = frames.quat_slerp(frames.quat_from_mat(Ra), frames.quat_from_mat(Rb), u)
    return frames.mat_from_quat(q)


class TeleopRetargeter:
    """Stateful, per-frame. Owns the calibration, the heading, the anchor, the limiter."""

    def __init__(self, B: dict[str, np.ndarray], *, hands=SIDES, align: str = "geometric",
                 rate_limit: bool = True, orientation: str = "absolute",
                 engage_ramp_s: float = 0.5, reacquire_ramp_s: float = 0.3,
                 reacquire_gap_s: float = 0.12, finger_smooth: bool = True,
                 smooth_min_cutoff: float = 1.5, smooth_beta: float = 0.05,
                 smooth_d_cutoff: float = 1.0, arm_follow: bool = True):
        if orientation not in ("absolute", "relative"):
            raise ValueError(f"orientation must be 'absolute' or 'relative', got {orientation!r}")
        self.hands = tuple(hands)
        self.B = {s: np.asarray(B[s], dtype=np.float64) for s in self.hands}
        self.B_R = {s: self.B[s][:3, :3] for s in self.hands}
        self.B_inv = {s: frames.se3_inv(self.B[s]) for s in self.hands}
        self.hand_rt = {s: HandRetargeter(s, align=align) for s in self.hands}
        self.rate_limit = rate_limit
        # One-Euro finger smoother, applied per hand BEFORE the velocity clamp. Off here
        # means the raw solve reaches _limit unchanged -- which is what the offline-
        # equivalence checks (check.py replay, tests) need to see, since the offline path
        # has no causal smoother of its own.
        self.finger_smooth = finger_smooth
        self._euro = {s: _OneEuro(smooth_min_cutoff, smooth_beta, smooth_d_cutoff)
                      for s in self.hands}
        # arm_follow=False -- "fingers only": the flange target is PINNED at the engage
        # anchor and your wrist motion is ignored; only the finger commands stay live.
        # The arm is the half that can leave the workspace (a normal reach from a bad
        # anchor trips the IK watchdog), so pinning it lets the hand retarget be exercised
        # on its own, with no reach, no heading and no clutch in the way.
        self.arm_follow = arm_follow
        self.orientation = orientation
        self.engage_ramp_s = engage_ramp_s
        # A hand back in view after longer than this counts as re-acquired: it re-anchors
        # (position picks up from where the arm HELD, no teleport) and ramps orientation
        # over reacquire_ramp_s. Shorter than a blink so a micro-flicker is not a re-anchor.
        self.reacquire_ramp_s = reacquire_ramp_s
        self.reacquire_gap_s = reacquire_gap_s

        self._C: np.ndarray | None = None        # session heading yaw (shared, 3x3)
        self.heading_spread_deg: float = 0.0      # left/right disagreement, a diagnostic

        # per-hand anchor state (engage AND re-acquire both write it, via _anchor_hand)
        self._engaged = False
        self._p_engage: dict[str, np.ndarray] = {}   # anchor flange position (pelvis frame)
        self._R_engage: dict[str, np.ndarray] = {}   # orientation the ramp starts FROM
        self._pw_engage: dict[str, np.ndarray] = {}  # wrist position at anchor (world)
        self._G_engage: dict[str, np.ndarray] = {}   # full anchor pose (relative mode)
        self._Tw_engage: dict[str, np.ndarray] = {}  # full wrist pose at anchor (relative mode)
        self._held: dict[str, np.ndarray] = {}       # last good target, held on dropout
        self._ramp_t0: dict[str, float] = {}         # per-hand orientation-ramp start time
        self._ramp_s: dict[str, float] = {}          # per-hand ramp duration (engage vs reacquire)
        self._last_active_t: dict[str, float] = {}   # last time this hand was stepped fresh

        self._last_cmd: dict[str, np.ndarray] = {}
        self._last_t: float | None = None

    # ---------------- calibration ----------------

    def calibrate(self, open_hand_poses: dict[str, np.ndarray]) -> None:
        """open_hand_poses: {side: (T,26,7)} of the operator holding a flat open hand."""
        for s in self.hands:
            self.hand_rt[s].calibrate(open_hand_poses[s])

    @property
    def calibrated(self) -> bool:
        return all(self.hand_rt[s].R_align is not None for s in self.hands)

    # ---------------- heading (absolute mode only) ----------------

    def set_heading(self, sample, anchor_flange: dict[str, np.ndarray]) -> float:
        """Estimate `C`, the operator's heading, from a matched pose.

        We want C . R_w . B_R ~= R_flange, so C ~= R_flange . B_R^T . R_w^T. That full
        rotation is projected to its heading yaw (C should be a pure yaw, exactly as the
        offline pelvis^-1 . S is): the convention offset is already carried by B, so the
        operator's calibration pose only has to get the FORWARD DIRECTION right, not a
        perfect full-orientation match. Both hands see the same heading, so we average
        them and keep the disagreement as a health check -- a large spread means B is off
        (the same thing `check measure-c` looks for).
        """
        yaws = {}
        for s in self.hands:
            R_w = sample.wrist_se3(s)[:3, :3]
            R_F = np.asarray(anchor_flange[s], dtype=np.float64)[:3, :3]
            yaws[s] = _yaw_angle(R_F @ self.B_R[s].T @ R_w.T)
        yaw = _mean_angle(list(yaws.values()))
        self._C = _rot_z(yaw)
        if len(yaws) == 2:
            d = (yaws["left"] - yaws["right"] + np.pi) % (2 * np.pi) - np.pi  # wrapped
            self.heading_spread_deg = float(abs(np.degrees(d)))
        return yaw

    def set_heading_matrix(self, C: np.ndarray) -> None:
        """Set `C` directly. For the offline equivalence test (which knows the exact
        pelvis^-1 . S) and for a saved session heading."""
        self._C = np.asarray(C, dtype=np.float64)[:3, :3].copy()

    @property
    def heading_set(self) -> bool:
        return self._C is not None

    # ---------------- clutch ----------------

    def engage(self, sample, anchor_flange: dict[str, np.ndarray], now: float = 0.0) -> None:
        """Latch the robot's measured flange pose and the human's wrist pose.

        `anchor_flange` must be the MEASURED FK of the real joints, so the delta rides on
        where the arm actually is and IK error is absorbed at the anchor. In absolute
        mode, the first engage also fixes the heading `C` if it is not set yet.
        """
        if not self.calibrated:
            raise RuntimeError("calibrate() before engage()")
        for s in self.hands:
            if not sample.active[s]:
                raise RuntimeError(f"cannot engage: {s} hand is not being tracked")

        if self.orientation == "absolute" and not self.heading_set:
            self.set_heading(sample, anchor_flange)

        for s in self.hands:
            self._anchor_hand(s, anchor_flange[s], sample.wrist_se3(s), now,
                              self.engage_ramp_s)
        self._last_t = None
        self._engaged = True

    def _anchor_hand(self, side: str, flange, wrist: np.ndarray, now: float,
                     ramp_s: float) -> None:
        """(Re)establish one hand's reference at the given flange pose and current wrist.

        The single operation behind BOTH engage and re-acquisition. Position telescopes
        from `flange`, so the arm continues from there rather than teleporting; orientation
        ramps from `flange`'s orientation to the live absolute target over `ramp_s`. At
        engage `flange` is the measured FK; on re-acquire it is where the arm was HOLDING.
        """
        G = np.asarray(flange, dtype=np.float64).copy()
        self._G_engage[side] = G
        self._Tw_engage[side] = wrist
        self._p_engage[side] = G[:3, 3].copy()
        self._R_engage[side] = G[:3, :3].copy()
        self._pw_engage[side] = wrist[:3, 3].copy()
        self._held[side] = G
        self._ramp_t0[side] = now
        self._ramp_s[side] = ramp_s
        self._last_active_t[side] = now

    def disengage(self) -> None:
        self._engaged = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    # ---------------- per-frame ----------------

    def step(self, sample, now: float) -> tuple[dict, dict, dict]:
        """-> (targets {side: (4,4)}, hand_cmds {side: (6,)}, info)

        A hand whose tracking has dropped holds BOTH its arm target and its finger
        command. It must not extrapolate, and it must not snap open: releasing a grasped
        object because the headset briefly lost sight of the hand is the worst failure.
        """
        if not self._engaged:
            raise RuntimeError("engage() before step()")

        dt = 0.0 if self._last_t is None else max(now - self._last_t, 0.0)
        self._last_t = now

        targets, cmds, info = {}, {}, {"dropped": [], "reacquired": []}
        for s in self.hands:
            if not sample.active[s]:
                info["dropped"].append(s)
                targets[s] = self._held[s]
                held = self._last_cmd.get(s, np.zeros(6, dtype=np.float32))
                cmds[s] = held
                # keep the smoother pinned to the held command so re-acquisition eases from
                # it instead of snapping through a stale filter state
                self._euro[s].reset(held)
                continue

            Tw = sample.wrist_se3(s)
            # A gap since this hand was last fresh means it dropped out and is now back:
            # re-anchor at the HELD pose (no teleport) and ramp orientation in. The gap
            # check -- not a bool "was active last tick" -- also catches the case where the
            # loop skipped ticks entirely while BOTH hands were out (nothing stepped).
            if now - self._last_active_t.get(s, now) > self.reacquire_gap_s:
                self._anchor_hand(s, self._held[s], Tw, now, self.reacquire_ramp_s)
                info["reacquired"].append(s)

            if not self.arm_follow:
                targets[s] = self._G_engage[s]      # pinned: wrist motion drives nothing
            else:
                targets[s] = (self._target_absolute(s, Tw, now)
                              if self.orientation == "absolute"
                              else self._target_relative(s, Tw))
            self._held[s] = targets[s]
            self._last_active_t[s] = now

            # fingers: the offline solve, verbatim
            tips = wrist_frame_tips(sample.hand[s][None])[0]     # (5,3), wrist frame
            cmd, residual, snaps = self.hand_rt[s].step(tips)
            if self.finger_smooth:
                cmd = self._euro[s].filter(cmd, dt)   # smooth jitter, THEN clamp velocity
            cmds[s] = self._limit(s, cmd, dt)
            info[f"residual_{s}"] = residual
            info[f"snap_{s}"] = snaps

        return targets, cmds, info

    def _target_absolute(self, side: str, Tw: np.ndarray, now: float) -> np.ndarray:
        """Absolute orientation, engage-relative position (the default)."""
        R_abs = self._C @ Tw[:3, :3] @ self.B_R[side]
        ramp_s = self._ramp_s[side]
        u = 1.0 if ramp_s <= 0.0 else \
            min(max((now - self._ramp_t0[side]) / ramp_s, 0.0), 1.0)
        R = R_abs if u >= 1.0 else _slerp_R(self._R_engage[side], R_abs, u)
        p = self._p_engage[side] + self._C @ (Tw[:3, 3] - self._pw_engage[side])
        return _se3(R, p)

    def _target_relative(self, side: str, Tw: np.ndarray) -> np.ndarray:
        """Fully-relative: delta against the engage pose, conjugated into the flange
        frame. S and the world frame cancel without being known."""
        delta = frames.se3_inv(self._Tw_engage[side]) @ Tw
        return self._G_engage[side] @ self.B_inv[side] @ delta @ self.B[side]

    def _limit(self, side: str, cmd: np.ndarray, dt: float) -> np.ndarray:
        """The URDF finger-velocity limit, applied online. The offline path rate-limits
        the whole track in one pass; the same ceiling has to hold here or the fingers can
        be commanded to slam between frames."""
        prev = self._last_cmd.get(side)
        if prev is None or not self.rate_limit or dt <= 0.0:
            self._last_cmd[side] = cmd.copy()
            return cmd
        step = RATES * dt
        out = np.clip(cmd, prev - step, prev + step).astype(np.float32)
        self._last_cmd[side] = out
        return out
