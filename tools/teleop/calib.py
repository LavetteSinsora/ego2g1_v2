"""Two calibrations the live path needs and the offline pipeline does not.

1. OPEN-HAND. `HandRetargeter.calibrate` fits the alignment rotation and the per-finger
   scales from an episode's most-open frame -- it scans the whole recording to find it,
   which a live stream obviously cannot do. So the operator is asked to hold both hands
   flat and open for a second at session start, and we calibrate from that.

2. THE WRIST CONVENTION, `W`. This is the one genuine unknown in moving from the
   recordings to the live WebXR stream, and it is worth being precise about why. (It is
   NOT the retargeter's heading `C`, which is the operator's per-session forward-yaw; `W`
   is a per-device axis convention that is either identity or a one-time constant.)

   The world frame does NOT matter: the action is a relative delta T(t0)^-1 T(t), and a
   change of world frame left-multiplies both factors and cancels (see the package
   docstring). The FINGERS do not matter either: in the default "geometric" align mode
   the wrist rotation drops out of the finger solve algebraically --

       R_align . (R_wrist^T . rel) = (robot_palm . G_h^T) . (R_wrist^T . rel)
                                   = robot_palm . (R_wrist^T P)^T . R_wrist^T . rel
                                   = robot_palm . P^T . rel

   -- leaving only landmark POSITIONS, which both streams agree on by construction.

   What does not cancel is the wrist's own LOCAL axis convention. If WebXR's wrist frame
   differs from the Pico OpenXR API's by a fixed rotation W (R_webxr = R_openxr . W),
   then in absolute mode the flange orientation C . R_w . B_R comes out wrong by exactly
   W (R_w carries the W), and the fix is to fold it in as B' = W^T . B.

   televuer's own docs say its raw data is already in the OpenXR convention (the wrist
   +Z pointing away from the fingertips, +Y out the back of the hand), which would make
   W the identity. That is a claim about someone else's code, and B is not a thing to
   take on trust -- so measure it.

   The measurement needs no simultaneous capture, which is what makes it practical: the
   PALM FRAME built from landmark positions is a convention-free common reference. For
   any frame define

       G = R_wrist^T . palm_frame_from_points(landmarks)

   Positions are identical in both streams, so G differs only by the wrist convention:
   G_webxr = W^T . G_openxr. Take the mean G over the HDF5 recordings, the mean G over a
   short live capture, and read W off directly:  W = G_openxr . G_webxr^T.
"""

import glob
import time

import numpy as np

from ._vendor.de.common import frames
from ._vendor.de.hand.fk_tables import palm_frame_from_points
from ._vendor.de.hand.retarget import XR_INDEX_K, XR_MIDDLE_K, XR_PINKY_K
from ._vendor.de.hand.constants import XR_WRIST

SIDES = ("left", "right")


def _quat_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    return frames.mat_from_quat(
        frames.quat_normalize(frames.quat_wxyz_from_xyzw(q_xyzw)))


def wrist_palm_frame(pose26: np.ndarray, side: str) -> np.ndarray:
    """G = R_wrist^T . P: the palm frame expressed in the wrist's own axes.

    Invariant to the world frame (both factors are world-frame rotations, so the world
    cancels) and built from POSITIONS on the palm side, so the only thing it can report
    is the wrist's local axis convention. That is exactly what we are trying to measure.
    """
    R_w = _quat_to_R(pose26[XR_WRIST, 3:7])
    P = palm_frame_from_points(pose26[XR_WRIST, :3], pose26[XR_INDEX_K, :3],
                               pose26[XR_MIDDLE_K, :3], pose26[XR_PINKY_K, :3], side)
    return R_w.T @ P


def _chordal_mean(rots: np.ndarray) -> np.ndarray:
    """The rotation closest (Frobenius) to the arithmetic mean. Same construction as
    b_alignment.chordal_mean -- averaging rotations elementwise does not give one."""
    U, _, Vt = np.linalg.svd(np.mean(rots, axis=0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _spread_deg(rots: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.array([frames.rot_geodesic_deg(R, mean) for R in rots])


def mean_G_from_hdf5(pattern: str, side: str, *, stride: int = 5) -> tuple:
    """Mean G over the recordings -- the OpenXR reference."""
    import h5py

    Gs = []
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no recordings matched {pattern!r}")
    for p in paths:
        with h5py.File(p, "r") as f:
            pose = f[f"{side}_hand_pose"][::stride].astype(np.float64)
            act = f[f"{side}_hand_active"][::stride].astype(bool)
        for i in np.flatnonzero(act):
            Gs.append(wrist_palm_frame(pose[i], side))
    Gs = np.stack(Gs)
    mean = _chordal_mean(Gs)
    return mean, _spread_deg(Gs, mean), len(paths)


def mean_G_from_source(source, side: str, *, seconds: float = 5.0) -> tuple:
    """Mean G over a short live capture -- the WebXR measurement."""
    Gs, seen = [], set()
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        s = source.latest()
        if s is not None and s.active[side] and s.t_ns not in seen:
            seen.add(s.t_ns)
            Gs.append(wrist_palm_frame(s.hand[side], side))
        time.sleep(0.005)
    if len(Gs) < 10:
        raise SystemExit(f"only {len(Gs)} tracked {side}-hand frames in {seconds:.0f}s "
                         "— keep the hand still and in view of the headset")
    Gs = np.stack(Gs)
    mean = _chordal_mean(Gs)
    return mean, _spread_deg(Gs, mean), len(Gs)


def measure_wrist_convention(G_openxr: np.ndarray, G_webxr: np.ndarray) -> np.ndarray:
    """The fixed rotation between the WebXR wrist frame and the recordings' wrist frame,
    call it W: R_wrist_webxr = R_wrist_openxr . W, i.e. G_webxr = W^T . G_openxr, so
    W = G_openxr . G_webxr^T.

    NB this is a different thing from the retargeter's heading `C` (retarget.py): that is
    the operator's per-session forward-yaw; this is a per-DEVICE convention offset that
    is either identity (televuer's claim) or a constant to fold into B once and forget.
    """
    return G_openxr @ G_webxr.T


def corrected_B(B: np.ndarray, W: np.ndarray) -> np.ndarray:
    """B' = W^T . B -- the alignment to use when the wrists come from WebXR, given the
    device convention W from `measure_wrist_convention`."""
    out = np.eye(4)
    out[:3, :3] = W.T @ B[:3, :3]
    return out


def collect_open_hand(source, *, seconds: float = 1.5, hands=SIDES,
                      settle_s: float = 3.0) -> dict[str, np.ndarray]:
    """Ask the operator to hold both hands flat and open; return the frames.

    This replaces the offline "scan the episode for its most-open frame" step, and it is
    the ONLY thing standing between a live stream and the training-identical finger
    solve. A sloppy calibration here rescales every finger command for the whole session.
    """
    print(f"\n  CALIBRATION — hold BOTH hands FLAT and OPEN, palms down, in view of "
          f"the headset.")
    for k in range(int(settle_s), 0, -1):
        print(f"    capturing in {k}...", flush=True)
        time.sleep(1.0)

    out = {s: [] for s in hands}
    seen = set()
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        s = source.latest()
        if s is not None and s.t_ns not in seen:
            seen.add(s.t_ns)
            for h in hands:
                if s.active[h]:
                    out[h].append(s.hand[h])
        time.sleep(0.005)

    for h in hands:
        if len(out[h]) < 5:
            raise SystemExit(f"only {len(out[h])} tracked frames for the {h} hand — "
                             "calibration needs both hands visible")
        print(f"    {h:5s}: {len(out[h])} frames")
    return {h: np.stack(out[h]) for h in hands}
