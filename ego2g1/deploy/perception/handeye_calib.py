"""Eye-to-hand calibration: solves `T_pelvis_camera` by having the robot grip
an AprilTag/ArUco marker (position on the flange UNKNOWN and un-measured --
just whatever the fingers happened to grab) and moving the arm through
several different orientations.

Why this is a different method from `touch_calib.py`, not a rewrite of it:
touch calibration correlates a detector-estimated OBJECT centroid (noisy --
segmentation + stereo depth, multi-mm to cm-level per docs/relation_deploy_plan
.md's own measured floor) against FK. This module instead correlates a rigid
MARKER's PnP-estimated pose (sub-mm/sub-degree) against FK, and -- because the
tag is just gripped rather than precisely mounted -- treats the flange-to-tag
offset as a second unknown that must be eliminated algebraically rather than
measured. Both are valid; this one trades touch_calib's simplicity (plain
Kabsch point fit) for better accuracy on the vision side.

The derivation (worked through with the user turn-by-turn; recorded here so
the code matches the math, not the other way around):

Frames -- `T_A_B` means "converts a point expressed in frame B into frame A's
coordinates" (so `T_A_B @ T_B_C = T_A_C`, `T_A_B^-1 = T_B_A`):
  Ba = robot base/pelvis        F_i = flange at sample i (moves, from FK)
  Ca = camera (fixed)           M_i = marker as observed at sample i (moves
                                       with the flange -- rigidly gripped)

At every sample, the marker's real pose can be written two independent ways
(through the robot: FK then the unknown grip offset; through vision: the
unknown camera extrinsic then what was observed) -- both equal "marker pose
in base frame", so:

    T_Ba_F_i @ T_F_M  =  T_Ba_Ca @ T_Ca_M_i          (*)

`T_F_M` (how the tag sits in the gripper -- unknown, uncontrolled) and
`T_Ba_Ca` (== T_pelvis_camera, what we want) are BOTH constant across i but
unknown from a single sample. Taking two samples i, j and eliminating T_F_M
(solve (*) for T_F_M at each, set equal, left/right-multiply to cancel) gives
the classical hand-eye equation in exactly ONE unknown:

    A_ij @ X = X @ B_ij
    A_ij = T_Ba_F_j^-1 @ T_Ba_F_i         (flange's own relative motion, FK only)
    B_ij = T_Ca_M_j    @ T_Ca_M_i^-1      (marker's relative motion AS SEEN BY
                                            the fixed camera, vision only)
    X    = T_Ba_Ca == T_pelvis_camera

`cv2.calibrateHandEye` solves exactly this (forming every pairwise A_ij/B_ij
internally from a list of poses) -- but it is written assuming EYE-IN-HAND
(camera moves with the gripper, target fixed in the world), where its inputs
are the gripper's pose IN base frame directly. Our situation is the mirror
image (camera fixed, marker moves with the flange) -- the standard, documented
trick is to feed it the flange poses INVERTED (T_F_Ba_i = T_Ba_F_i^-1) in the
"gripper2base" argument slot, with the marker's camera-frame pose fed AS-IS in
the "target2cam" slot. Substituting those into the function's own (eye-in-hand
shaped) internal derivation reproduces exactly the A_ij/B_ij pair above, and
what it returns as "cam2gripper" comes out numerically equal to `T_Ba_Ca` --
i.e. `T_pelvis_camera`, directly, no extra recovery step. See `solve_eye_to_hand`.

`T_F_M` (the grip offset) is never solved for or needed -- eliminating it is
the entire point of using the pairwise relative-motion form instead of a
per-sample Kabsch fit (which is exactly what `touch_calib.py` does, and
exactly why that approach NEEDS to know the touch-point offset up front,
while this one does not).
"""

import dataclasses

import numpy as np

from ego2g1.core.se3 import se3_inv

# Below this many rotationally-diverse samples, AX=XB is a known
# ill-conditioned solve (a real effect, not a made-up floor -- e.g. OpenCV's
# own calibrateHandEye docs and most robotics hand-eye tutorials recommend
# 10+ poses with genuinely varied ROTATION axes, not just varied positions;
# translation alone cannot disambiguate the rotation part of X at all, since
# a pure-translation motion pair has A_ij and B_ij both rotation-free).
MIN_RECOMMENDED_SAMPLES = 10
MIN_ROTATION_SPREAD_DEG = 15.0

# Consistency-check go/no-go bounds (see `mount_consistency_report`) -- same
# spirit as touch_calib.py's RESIDUAL_WARN_M, a different number because this
# measures a different thing (spread of an ESTIMATED constant across samples,
# not distance from a fitted point).
CONSISTENCY_WARN_TRANSLATION_M = 0.01
CONSISTENCY_WARN_ROTATION_DEG = 3.0


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Angle (deg) of the axis-angle representation of a (3,3) rotation."""
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def detect_tag_pose(
    image: np.ndarray,
    tag_size_m: float,
    K: np.ndarray,
    dist: np.ndarray,
    *,
    dictionary_name: str | None = None,
) -> tuple[np.ndarray, int] | None:
    """Detect ONE ArUco/AprilTag marker in `image`, return `(T_camera_marker,
    marker_id)` -- (4,4) rigid pose, camera optical frame -- or `None` if no
    marker was found. If several are visible, returns the first one
    `ArucoDetector.detectMarkers` reports (grip one tag at a time; multiple
    simultaneous tags are out of scope for this procedure).

    `dictionary_name`: e.g. `"DICT_APRILTAG_36h11"`. `None` auto-detects via
    `stereo_calib.detect_aruco_dictionary` against this one image -- that
    function's own candidate list already includes the AprilTag families (see
    its module comment), so this deliberately reuses it rather than
    maintaining a second dictionary list.

    Pose via `cv2.solvePnP` against the marker's own 4 corners (NOT the
    deprecated/removed `estimatePoseSingleMarkers` convenience wrapper, which
    newer OpenCV builds drop) -- object points in the marker's local frame,
    origin at its center, matching `ArucoDetector`'s own corner order
    (top-left, top-right, bottom-right, bottom-left):
        (-s/2,  s/2, 0), (s/2,  s/2, 0), (s/2, -s/2, 0), (-s/2, -s/2, 0)

    Do not eyeball the returned rotation expecting "identity == marker
    facing the camera straight-on": this object-frame convention's +Y is
    tied to the marker's own printed bit-pattern orientation, not to image
    "up" -- OpenCV's pixel frame is Y-DOWN, so a marker held flat, facing the
    camera, right-side-up, actually solves to a rotation that includes a
    ~180 deg flip (confirmed by hand while testing this function: rendering
    with a literal identity rotation produces an UNDECODABLE marker image --
    see `tests/deploy/perception/test_handeye_calib.py`'s `_CANONICAL_BASE_R`
    for the empirical confirmation). This is a real, standard ArUco/IPPE
    convention fact, not a bug here -- it is irrelevant to
    `solve_eye_to_hand`'s correctness (every quantity it uses is a
    RELATIVE motion between two samples of this SAME convention, so any
    fixed labeling offset cancels out identically), it only matters if you
    print a single sample's pose and try to sanity-check it by eye.
    """
    import cv2  # lazy, see package __init__ docstring

    from .stereo_calib import detect_aruco_dictionary

    image = np.asarray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    if dictionary_name is None:
        dictionary_name = detect_aruco_dictionary([gray], min_markers=1)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    half = tag_size_m / 2.0
    obj_points = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float64,
    )
    img_points = corners[0].reshape(4, 2).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, np.asarray(K, dtype=np.float64),
        np.asarray(dist, dtype=np.float64), flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T, int(ids[0][0])


@dataclasses.dataclass
class HandEyeSample:
    """One eye-to-hand correspondence: the SAME instant's flange pose (FK)
    and gripped-marker pose (vision), nothing more -- the grip offset
    connecting them is deliberately never computed per-sample (see module
    docstring)."""

    T_base_flange: np.ndarray  # (4,4) FK, pelvis frame -- Kinematics.flange_poses()[hand]
    T_camera_marker: np.ndarray  # (4,4) detect_tag_pose()'s output
    arm_q: np.ndarray | None = None  # (14,) joint config at capture, for the record
    note: str = ""


def rotation_spread_deg(samples: list[HandEyeSample]) -> float:
    """Max pairwise flange-ROTATION angular separation across samples
    (degrees) -- a direct check of the "genuine rotational diversity" AX=XB
    needs, independent of how many samples there are (many samples all at
    nearly the same orientation are just as ill-conditioned as too few)."""
    Rs = [s.T_base_flange[:3, :3] for s in samples]
    spread = 0.0
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            angle = _rotation_angle_deg(Rs[j] @ Rs[i].T)
            spread = max(spread, angle)
    return spread


def solve_eye_to_hand(
    samples: list[HandEyeSample], *, method: int | None = None
) -> tuple[np.ndarray, dict]:
    """Solve `T_pelvis_camera` (4,4) from `samples` via the classical AX=XB
    hand-eye equation, eye-to-hand configuration (module docstring has the
    full derivation). Does NOT require knowing the flange-to-marker grip
    offset -- that unknown is eliminated by construction, which is the whole
    reason to use this over a per-sample Kabsch fit when the tag is just
    gripped, not precisely mounted.

    Returns `(T_pelvis_camera, report)`; `report` is a dict with
    `n_samples`, `rotation_spread_deg`, and the `mount_consistency_report`
    (see below) -- print/inspect both before trusting the result, same
    go/no-go spirit as `stereo_calib.py`'s `CalibrationReport`.
    """
    import cv2  # lazy

    if len(samples) < 3:
        raise ValueError(
            f"need >= 3 samples to solve a unique rigid transform, got {len(samples)} "
            "(and >= 3 is the bare mathematical minimum -- "
            f"{MIN_RECOMMENDED_SAMPLES}+ with real rotational diversity is what "
            "actually gives a trustworthy solve, see rotation_spread_deg)"
        )

    spread = rotation_spread_deg(samples)
    if spread < MIN_ROTATION_SPREAD_DEG:
        print(
            f"WARNING: max pairwise flange-rotation spread is only {spread:.1f} deg "
            f"(< {MIN_ROTATION_SPREAD_DEG} deg) -- AX=XB cannot disambiguate the "
            "rotation part of the extrinsic from poses this close together. Capture "
            "more samples with genuinely different arm ORIENTATIONS, not just "
            "different positions, before trusting this solve."
        )
    if len(samples) < MIN_RECOMMENDED_SAMPLES:
        print(
            f"WARNING: only {len(samples)} sample(s) "
            f"(< {MIN_RECOMMENDED_SAMPLES} recommended) -- treat this as provisional."
        )

    R_flange2base, t_flange2base = [], []
    R_marker2cam, t_marker2cam = [], []
    for s in samples:
        T_flange_base = se3_inv(np.asarray(s.T_base_flange, dtype=np.float64))
        R_flange2base.append(T_flange_base[:3, :3])
        t_flange2base.append(T_flange_base[:3, 3])
        T_cam_marker = np.asarray(s.T_camera_marker, dtype=np.float64)
        R_marker2cam.append(T_cam_marker[:3, :3])
        t_marker2cam.append(T_cam_marker[:3, 3])

    kwargs = {} if method is None else {"method": method}
    # NOTE: arg names below are cv2's own (eye-in-hand) naming; we feed
    # INVERTED flange poses into the "gripper2base" slot -- see module
    # docstring for why the returned "cam2gripper" is then T_pelvis_camera.
    R_out, t_out = cv2.calibrateHandEye(
        R_flange2base, t_flange2base, R_marker2cam, t_marker2cam, **kwargs
    )

    T_pelvis_camera = np.eye(4, dtype=np.float64)
    T_pelvis_camera[:3, :3] = R_out
    T_pelvis_camera[:3, 3] = t_out.flatten()

    report = {
        "n_samples": len(samples),
        "rotation_spread_deg": spread,
        **mount_consistency_report(samples, T_pelvis_camera),
    }
    return T_pelvis_camera, report


def mount_consistency_report(
    samples: list[HandEyeSample], T_pelvis_camera: np.ndarray
) -> dict:
    """Go/no-go check that does NOT require ever knowing the true grip
    offset: back out an ESTIMATED `T_flange_marker` at every sample --

        T_flange_marker_i = T_base_flange_i^-1 @ T_pelvis_camera @ T_camera_marker_i

    -- from equation (*) in the module docstring. Since the tag is rigidly
    gripped, this estimate should be near-IDENTICAL across every sample if
    `T_pelvis_camera` is correct; its SPREAD across samples (not its value,
    which is thrown away) is the residual. A wrong extrinsic makes each
    sample "explain" a different apparent grip offset, which is exactly
    what a large spread here catches.

    Returns `translation_std_m` (per-axis std of the estimated translation
    across samples) and `rotation_spread_deg` (max pairwise angular
    separation of the estimated rotations) -- compare against
    `CONSISTENCY_WARN_TRANSLATION_M`/`CONSISTENCY_WARN_ROTATION_DEG`.
    """
    T_pelvis_camera = np.asarray(T_pelvis_camera, dtype=np.float64)
    estimates = []
    for s in samples:
        T_flange_base = se3_inv(np.asarray(s.T_base_flange, dtype=np.float64))
        estimates.append(
            T_flange_base @ T_pelvis_camera @ np.asarray(s.T_camera_marker, dtype=np.float64)
        )

    translations = np.array([T[:3, 3] for T in estimates])
    translation_std = np.std(translations, axis=0)
    translation_std_m = float(np.linalg.norm(translation_std))

    rotations = [T[:3, :3] for T in estimates]
    rotation_spread = 0.0
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            angle = _rotation_angle_deg(rotations[j] @ rotations[i].T)
            rotation_spread = max(rotation_spread, angle)

    return {
        "translation_std_m": translation_std_m,
        "rotation_spread_deg_estimate": rotation_spread,
    }


def _cli_solve(samples_npz: str, out_npz: str = "camera_calib.npz") -> None:
    """CLI: solve from a saved `.npz` of stacked samples (however they were
    logged during a real capture session -- see `check.py`'s `handeye-capture`
    rung), print the transform + go/no-go report, save.

    Expected keys in `samples_npz`: `T_base_flange` (N,4,4), `T_camera_marker`
    (N,4,4) -- exactly what `handeye-capture` writes.
    """
    data = np.load(samples_npz)
    T_base_flange = data["T_base_flange"]
    T_camera_marker = data["T_camera_marker"]
    samples = [
        HandEyeSample(T_base_flange=T_base_flange[i], T_camera_marker=T_camera_marker[i])
        for i in range(len(T_base_flange))
    ]

    T_pelvis_camera, report = solve_eye_to_hand(samples)

    print(f"T_pelvis_camera ({report['n_samples']} samples) =\n{T_pelvis_camera}")
    print(f"max pairwise flange-rotation spread: {report['rotation_spread_deg']:.1f} deg")
    print(
        f"consistency check -- estimated grip-offset translation std: "
        f"{report['translation_std_m'] * 1000:.2f} mm, rotation spread: "
        f"{report['rotation_spread_deg_estimate']:.2f} deg"
    )
    suspect = (
        report["translation_std_m"] > CONSISTENCY_WARN_TRANSLATION_M
        or report["rotation_spread_deg_estimate"] > CONSISTENCY_WARN_ROTATION_DEG
    )
    if suspect:
        print(
            "\nSUSPECT CALIBRATION -- the estimated grip offset disagrees too much "
            "across samples (thresholds: "
            f"{CONSISTENCY_WARN_TRANSLATION_M * 1000:.0f} mm / "
            f"{CONSISTENCY_WARN_ROTATION_DEG:.1f} deg) -- do not trust this before "
            "capturing more/better-varied samples."
        )
    np.savez(
        out_npz,
        T_pelvis_camera=T_pelvis_camera,
        n_samples=np.int64(report["n_samples"]),
        rotation_spread_deg=np.float64(report["rotation_spread_deg"]),
        translation_std_m=np.float64(report["translation_std_m"]),
        rotation_spread_deg_estimate=np.float64(report["rotation_spread_deg_estimate"]),
    )
    print(f"saved -> {out_npz}" + ("  (SUSPECT, see above)" if suspect else ""))


if __name__ == "__main__":
    import tyro

    tyro.cli(_cli_solve)
