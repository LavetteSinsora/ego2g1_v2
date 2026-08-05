"""Rung 4c (`check handeye-capture`): interactively grip an ArUco marker
and capture eye-to-hand calibration samples (AX=XB solve of
T_pelvis_camera — perception/handeye_calib.py). Folds hand-jog's
controls in so there is no tool switch mid-session."""

from __future__ import annotations

import sys
import time

import numpy as np

from ego2g1.core import layout
from ego2g1.deploy import actions as _actions
from ego2g1.deploy._util import dds_init
from ego2g1.deploy.camera import DEFAULT_HOST as _CAMERA_HOST

from .hands import _read_key

# --- 4c. eye-to-hand calibration capture ---------------------------------------

# motor_state slots 12-14 -- G1_29_JointArmIndex order, same numbering
# `listen()`'s own comment documents ("legs 0-11, waist 12-14, arm 15-28").
_WAIST_IDX = (12, 13, 14)

# `Kinematics.flange_poses()` (kinematics.py) hardcodes waist qpos to EXACTLY
# 0 -- it does not read the real robot's waist encoders at all, because
# training itself only ever saw waist==0 (kinematics.py's own docstring).
# `handeye_capture` computes T_base_flange through that same FK, so if the
# REAL waist has drifted off 0 (bumped while handling the arm/tag -- a real
# risk during a manual capture session, not during autonomous inference
# where nothing should be touching the torso), the computed T_base_flange is
# silently wrong for that sample, by however much the waist actually moved --
# same order of magnitude as docs/relation_deploy_plan.md §6.1's own
# "~1 deg orientation error ~= 5-10mm at 0.3-0.6m reach" estimate, since the
# waist sits upstream of the ENTIRE arm chain (a lever-arm effect, not a
# local one). This is a REJECT threshold, not a warning-only one -- a sample
# built on a wrong T_base_flange is worse than no sample at all.
_WAIST_WARN_RAD = np.radians(0.5)


def _read_waist_rad(state_msg) -> np.ndarray:
    return np.array([state_msg.motor_state[i].q for i in _WAIST_IDX], dtype=np.float64)


def _save_handeye_samples(out_path, samples) -> None:
    """Persist the raw (T_base_flange, T_camera_marker, arm_q) arrays for
    every sample captured SO FAR -- no `T_pelvis_camera` solve required (that
    needs >= 3 samples with real rotational diversity; this needs none of
    that, it's just "don't lose what's already been measured"). Called after
    EVERY successful capture, not only at the end: a session that crashes or
    is quit early (Ctrl-C, a bad `q` mid-sentence, the process dying) must
    never lose samples already on disk -- exactly what happened once before
    this existed (see `handeye_capture`'s docstring), when only the debug
    PNGs survived and the arm-pose half of each correspondence, which only
    ever lived in memory, was gone. Overwrites `out_path` each call (cheap at
    these sample counts); `perception.handeye_calib._cli_solve` can solve
    directly from whatever is here at any point, session still running or not.
    """
    import numpy as np

    np.savez(
        out_path,
        T_base_flange=np.stack([s.T_base_flange for s in samples]),
        T_camera_marker=np.stack([s.T_camera_marker for s in samples]),
        arm_q=np.stack([s.arm_q for s in samples]),
    )


def _save_annotated_detection(img, aruco_dict_name: str | None, out_path) -> None:
    """Draw whatever `ArucoDetector` actually found (corners + ID, via
    `cv2.aruco.drawDetectedMarkers`) on top of `img` and save it -- the
    fastest way to SEE why a capture locked onto an unexpected marker id
    (background clutter decoded as a different real tag) instead of
    guessing from the printed id alone."""
    import cv2

    from ego2g1.deploy.perception.stereo_calib import detect_aruco_dictionary

    img = np.asarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    gray = np.ascontiguousarray(gray, dtype=np.uint8)

    if aruco_dict_name is None:
        aruco_dict_name = detect_aruco_dictionary([gray], min_markers=1)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _rejected = detector.detectMarkers(gray)

    annotated = np.ascontiguousarray(img[..., ::-1].copy())  # RGB -> BGR for cv2 drawing/imwrite
    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    cv2.imwrite(str(out_path), annotated)


def _next_index(dir_path, pattern: str) -> int:
    """Lowest non-negative integer N such that no existing file in
    `dir_path` matching `pattern` ends in `_{N}` -- same non-colliding,
    resume-safe convention as `_next_pair_index`, generalized to a caller-
    supplied glob (this one names files `capture_NNN[...].png`, not
    `left_NNN.png`/`right_NNN.png`)."""
    existing = set()
    for p in dir_path.glob(pattern):
        suffix = p.stem.split("_")[1] if "_" in p.stem else ""
        if suffix.isdigit():
            existing.add(int(suffix))
    idx = 0
    while idx in existing:
        idx += 1
    return idx


def handeye_capture(
    iface: str | None = None,
    domain: int = 0,
    camera_host: str = _CAMERA_HOST,
    stereo_calib_npz: str = "stereo_calib.npz",
    tag_size_m: float = 0.05,
    aruco_dict: str | None = None,
    tag_id: int | None = None,
    hand: str = "right",
    out: str = "handeye_samples.npz",
    image_dir: str = "calibration/camera_extrinsic_calibration",
    auto_start_server: bool = True,
    grip_step: float = 0.02,
    hold_arm: bool = True,
    fps: int = 30,
    yes: bool = False,
) -> None:
    """Interactively GRIP an AprilTag/ArUco marker with `hand` AND capture
    eye-to-hand calibration samples, in the same session -- folds `hand-jog`'s
    motor-jogging controls directly into this rung so there's no separate
    "grip the tag first, then switch tools" step: the exact grip doesn't need
    to be measured or repeatable (position on the flange UNKNOWN/uncontrolled
    is fine -- see `perception/handeye_calib.py`'s module docstring for the
    derivation of why), it only needs to stay FIXED once you start capturing
    (module docstring: `T_F_M` is treated as one constant across the whole
    session -- a re-grip partway through invalidates every sample before it).

    Solves `T_pelvis_camera` from FK + the tag's camera-observed pose via the
    classical AX=XB hand-eye equation -- an alternative to `touch_calib.py`'s
    object-centroid Kabsch fit, trading its simplicity for not depending on
    detector/depth accuracy at all.

    `out` is written after EVERY successful capture, not only at quit --
    confirmed the hard way: a session that crashes or exits before `q` used
    to lose every sample (only in memory), leaving nothing but the debug
    PNGs behind, which do NOT carry the arm pose half of each correspondence
    and so cannot reconstruct it after the fact. If this session ends before
    you solve (crash, Ctrl-C, too few/insufficiently-varied samples for
    `solve_eye_to_hand` to be worth trusting yet), whatever was captured up
    to that point is already safely in `out` -- solve it directly later with
    `python -m ego2g1.deploy.perception.handeye_calib --samples-npz <out>`.

    Every CAPTURE ATTEMPT (successful or not) saves the raw frame to
    `image_dir` as `capture_NNN.png` -- inspect these to see exactly what the
    detector saw. When a marker WAS found (regardless of whether its ID
    matched `tag_id`), an ADDITIONALLY saved `capture_NNN_annotated.png` draws
    the detected corners + ID directly on the image (`cv2.aruco
    .drawDetectedMarkers`) -- the fastest way to tell "wrong marker" (real
    detection, wrong ID -- e.g. background clutter decoded as a DIFFERENT
    valid-looking tag) from "no detection at all" (blur, occlusion, out of
    frame, bad lighting) without guessing from the printed message alone.

    `aruco_dict`/`tag_id` are LOCKED after the first successful detection,
    not re-detected every capture, on purpose: auto-detecting the dictionary
    fresh each call (this rung's first version did this) tries all ~27 known
    dictionaries against whatever is in frame, and with a real cluttered
    workspace behind the tag, some OTHER dictionary occasionally decodes a
    false-positive marker out of background texture -- confirmed on real
    hardware: consecutive captures returned marker ids 0, 101, 93, 0 from
    what was supposedly the SAME physical tag (ids >=50 are not even valid in
    DICT_4X4_50, proving those particular hits came from a DIFFERENT
    auto-detected dictionary entirely, i.e. a false positive, not the real
    tag), and eventually raised an uncaught `ValueError` when NO dictionary
    found anything in one particular frame -- which crashed the whole session
    and lost every sample already captured, since nothing had been persisted
    to disk. If you know which dictionary/id you printed, pass them
    explicitly (`--aruco-dict`/`--tag-id`) and they're used from the start;
    left as `None` (the default), the FIRST successful capture auto-detects
    once, prints what it locked onto, and every later capture in the session
    reuses that SAME dictionary/id -- it only ever looks for that one real
    tag from then on, and a later detection with a DIFFERENT id is flagged
    as a likely false positive rather than silently accepted. Every
    per-capture detection is also wrapped so a bad frame prints an error and
    lets you retry, instead of ending the session.

    Keys:
      1-6    select which of `hand`'s motors to adjust (HAND_MOTOR_ORDER)
      j/k    decrease / increase the selected motor by `grip_step`
      o      open `hand` fully (all motors -> 0.0)
      g      grip: close `hand` fully (all motors -> 1.0) -- a coarse
             starting point, fine-tune with 1-6/j/k afterward
      c      CAPTURE a calibration sample: reads current FK + detects the
             tag in the latest camera frame. Prints a clear message and
             captures NOTHING if the tag isn't visible (or the wrong ID was
             seen), rather than silently recording a bad detection.
      p      print sample count + current hand motor vector
      q      quit -- solves + saves if there are enough samples (prints the
             go/no-go report either way)

    Once the grip looks solid (tag firmly held, won't shift), move the ARM
    through several distinctly different ORIENTATIONS by hand, then press 'c'
    to capture whatever pose it is CURRENTLY in. Real rotational diversity
    across samples matters far more than sample count or how far apart the
    positions are (see `handeye_calib.MIN_ROTATION_SPREAD_DEG`) -- roll the
    wrist and rotate the whole arm through several distinct orientations
    between captures, don't just slide it sideways. Do NOT re-grip the tag
    between captures.

    `hold_arm=True` (default) connects the REAL joint-level executor
    (`UnitreeExecutor`, the same one every other hardware rung uses) so the
    waist is ACTIVELY held at 0 rad (unitree_deploy's own G1 controller does
    this internally, at the documented kp=300/kd=3, the instant it's
    connected -- not something this function builds, just something
    connecting the real executor turns on) -- mechanical prevention of the
    exact waist-drift the `_WAIST_WARN_RAD` check below only ever detects
    after the fact. Consequences, all new relative to `hold_arm=False` (the
    old, read-only behavior) and worth knowing before you start:
      - `executor.connect()` immediately soft-ramps the ARM to its vendor
        init pose -- expect real motion the moment you start this rung, same
        as every other hardware rung that uses `UnitreeExecutor`. Don't grip
        the tag until AFTER that ramp settles.
      - Between captures, this rung continuously re-sends the arm's own
        CURRENT measured position as its target (`executor.hold()`, a safe
        no-op waypoint) so you can still move it by hand -- but real PD gains
        (shoulder/elbow 80/3, wrist 40/1.5) are active throughout, not zero
        stiffness. It will feel a bit springy, not freely floating -- move it
        slowly and gently, particularly right after each hold() refresh.
      - `q` (clean exit) drives the arm back to its vendor init pose on
        close(), same as every other rung's normal exit. Ctrl-C or an
        unhandled exception instead calls `executor.damp()` (the real
        e-stop: arm goes limp, latched) -- same convention as `replay-actions`.
      - Prompts for a `[y/N]` confirmation before connecting, unless `--yes`
        -- same gate `replay-actions` uses before touching the real arm.
    `hold_arm=False` restores the original read-only behavior (only the
    `_WAIST_WARN_RAD` detect-and-reject check, no arm commands at all, no
    confirmation prompt, no motion) if you'd rather not have this rung touch
    the arm at all.

    The waist check itself is against a SESSION BASELINE, not literal 0 --
    confirmed on real hardware that at least one waist axis can sit at a
    persistent few-degree reading that never approaches 0, whether or not
    `hold_arm` is actively commanding it (most likely a per-unit encoder-zero
    offset, or simply this unit's genuine settled resting value -- see the
    printed "waist baseline" line and the code comment right above it for the
    full reasoning). The `_WAIST_WARN_RAD` reject-check is against DEVIATION
    from that printed baseline, captured once after the init-pose ramp
    settles -- this does not change what `Kinematics.flange_poses()` assumes
    for the actual FK (still literal waist==0); if the baseline itself turns
    out to be a real physical deviation rather than a reporting quirk, every
    sample in the session carries a corresponding small systematic bias.

    Needs `stereo_calib_npz` (`check stereo-capture` + `perception.stereo_
    calib`'s solve, already done) for the camera intrinsics `detect_tag_pose`
    needs -- this rung only ever reads ONE eye (left), stereo depth is not
    involved in this calibration method at all.
    """
    import pathlib
    import termios
    import tty

    import cv2
    import numpy as np

    from ego2g1.core import layout
    from ego2g1.deploy.camera import HeadCamera
    from ego2g1.deploy.core.kinematics import Kinematics
    from ego2g1.deploy.perception.depth import StereoCalibration
    from ego2g1.deploy.perception.handeye_calib import HandEyeSample, detect_tag_pose, solve_eye_to_hand

    if hand not in layout.HANDS:
        raise ValueError(f"hand must be one of {layout.HANDS}, got {hand!r}")

    calib = StereoCalibration.load(stereo_calib_npz)

    executor = None
    if hold_arm:
        from ego2g1.deploy.core.executor import UnitreeExecutor
        if not yes and input(
                "connect the REAL arm executor to actively hold the waist at "
                "0 rad (arm will soft-ramp to its init pose now, then follow "
                "your hand with real PD gains active)? [y/N] "
                ).strip().lower() != "y":
            return
        executor = UnitreeExecutor(fps=fps, network_interface=iface)
        executor.connect()
        print("executor connected — arm ramping to its init pose (expect "
              "motion), waist now actively pinned at 0 rad.")

    image_dir_path = pathlib.Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)
    capture_idx = _next_index(image_dir_path, "capture_*.png")
    print(f"debug images -> {image_dir_path.resolve()} (capture_NNN.png / "
          f"capture_NNN_annotated.png)")

    # None until the first successful detection resolves/locks them (see
    # docstring) -- from then on every capture in this session reuses the
    # SAME dictionary/id rather than re-detecting or silently accepting
    # whatever a later frame happens to decode.
    locked_dict = aruco_dict
    locked_id = tag_id

    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    dds_init(iface, domain)

    state = {"msg": None}

    def on_state(msg):
        state["msg"] = msg

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    t0 = time.monotonic()
    while state["msg"] is None:
        if time.monotonic() - t0 > 5.0:
            sys.exit("no rt/lowstate in 5 s — check the link / DDS domain / iface.")
        time.sleep(0.05)

    if hold_arm:
        time.sleep(1.5)  # let the init-pose soft-ramp settle before baselining
    # Session BASELINE, not literal zero: confirmed on real hardware that at
    # least one waist axis can sit at a persistent, non-drifting few-degree
    # reading (e.g. 4.7 deg) that never approaches 0 no matter how long you
    # wait or whether the arm/waist is actively held -- most likely either a
    # per-unit encoder-zero offset (the joint is truly at whatever position
    # the model considers "zero", the RAW value we read is just shifted by a
    # fixed constant) or simply this unit's actual settled resting value.
    # Either way, comparing against literal 0 made every single capture
    # reject on this hardware -- comparing against THIS session's own
    # observed baseline still catches real NEW drift (someone bumping the
    # torso mid-session) without permanently blocking on a fixed, harmless
    # characteristic. This does NOT change `Kinematics.flange_poses()`'s own
    # FK, which still assumes literal waist==0 for the actual T_base_flange
    # math -- if this baseline reading turns out to reflect a REAL physical
    # deviation rather than a sensor-reporting quirk, every sample in this
    # session carries a corresponding small systematic bias; treat a large
    # baseline (much beyond a couple degrees) as a reason to double check,
    # not just accept, before trusting the resulting calibration.
    waist_baseline_rad = _read_waist_rad(state["msg"])
    print(f"waist baseline (session reference, NOT literal 0): "
          f"{np.round(np.degrees(waist_baseline_rad), 2)} deg — captures "
          "are rejected on DEVIATION from this, not from absolute zero.")

    # `hand_msgs` is the in-memory desired grip state (what 1-6/j/k/o/g edit,
    # and the status line reads) regardless of `hold_arm` -- only WHO
    # actually publishes it to `rt/brainco/{h}/cmd` differs:
    #   hold_arm=True:  `executor.send()` does, as part of one combined
    #                   arm+hand row every tick (see the publish loop below)
    #                   -- unitree_deploy's own "unitree_g1_brainco" robot
    #                   bundles Brainco hand control into the SAME send_action
    #                   the arm-hold uses, so a SEPARATE raw publisher here
    #                   would be a second, independent writer to the exact
    #                   same DDS topic. Confirmed as a real bug, not a
    #                   theoretical one: `executor.hold()`'s row zeroes
    #                   everything outside the arm slice, so a second
    #                   publisher racing it looked like "grip commands do
    #                   nothing" -- they were being overwritten every tick.
    #   hold_arm=False: a plain `ChannelPublisher` here, same as `hand_jog`
    #                   (nothing else is publishing to this topic in that mode).
    hand_pubs, hand_msgs = {}, {}
    for h in layout.HANDS:
        if executor is None:
            hand_pubs[h] = ChannelPublisher(f"rt/brainco/{h}/cmd", MotorCmds_)
            hand_pubs[h].Init()
        m = MotorCmds_()
        m.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(layout.HAND_DIM)]
        for i in range(layout.HAND_DIM):
            m.cmds[i].q = 0.0
            m.cmds[i].dq = 1.0
        hand_msgs[h] = m
    selected_motor = 0

    kin = Kinematics()
    cam = HeadCamera(host=camera_host, eye="left", auto_start_server=auto_start_server)
    cam.connect()

    arm_idx = list(range(15, 29))  # G1_29_JointArmIndex order, same as listen()
    samples: list[HandEyeSample] = []

    def _hand_status_line() -> str:
        vals = ", ".join(
            f"{'*' if i == selected_motor else ' '}{name}={hand_msgs[hand].cmds[i].q:.3f}"
            for i, name in enumerate(layout.HAND_MOTOR_ORDER)
        )
        waist_dev_rad = _read_waist_rad(state["msg"]) - waist_baseline_rad
        waist_flag = "!" if np.abs(waist_dev_rad).max() > _WAIST_WARN_RAD else " "
        return (f"  [{hand}] {vals}  ({len(samples)} sample(s))  "
                f"waist{waist_flag}Δ{np.round(np.degrees(waist_dev_rad), 2)}deg")

    def _attempt_capture() -> None:
        """One 'c' press: reject on a drifted waist, else record one
        (T_base_flange, T_camera_marker) sample -- see the module docstring
        for why waist is checked at all and why the dictionary/id lock
        happens here, on the first hit, rather than being pre-supplied."""
        nonlocal capture_idx, locked_dict, locked_id

        waist_rad = _read_waist_rad(state["msg"])
        waist_dev_rad = waist_rad - waist_baseline_rad
        if np.abs(waist_dev_rad).max() > _WAIST_WARN_RAD:
            print(
                f"\nwaist at {np.round(np.degrees(waist_rad), 2)} deg, "
                f"{np.round(np.degrees(waist_dev_rad), 2)} deg from this session's "
                f"baseline (> {np.degrees(_WAIST_WARN_RAD):.1f} deg) — FK assumes "
                "waist==0 exactly, so T_base_flange would be wrong by however much "
                "this REALLY moved. NOT captured — stop leaning/bumping the torso "
                "and retry."
            )
            return

        arm_q = np.array([state["msg"].motor_state[i].q for i in arm_idx])
        T_base_flange = kin.flange_poses(arm_q)[hand]
        img = cam.read()
        if img is None:
            print("\nno camera frame yet — try again")
            return

        this_idx = capture_idx
        capture_idx += 1
        raw_path = image_dir_path / f"capture_{this_idx:03d}.png"
        cv2.imwrite(str(raw_path), img[..., ::-1])

        result = detect_tag_pose(
            img, tag_size_m, calib.K_left, calib.dist_left, dictionary_name=locked_dict,
        )
        if result is None:
            print(f"\nno tag detected — saved {raw_path} for inspection, reposition and retry")
            return

        T_camera_marker, marker_id = result
        if locked_dict is None:
            # First hit: resolve which dictionary THIS detection actually
            # used, once, and reuse it for every later capture instead of
            # re-auto-detecting per call (see docstring for why that's
            # unstable).
            from ego2g1.deploy.perception.stereo_calib import detect_aruco_dictionary
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
            gray = np.ascontiguousarray(gray, dtype=np.uint8)
            locked_dict = detect_aruco_dictionary([gray], min_markers=1)
            print(f"\nlocked ArUco dictionary: {locked_dict} (reused for the rest of this session)")
        if locked_id is None:
            locked_id = marker_id
            print(f"\nlocked expected tag id: {locked_id} (reused for the rest of this session)")

        annotated_path = image_dir_path / f"capture_{this_idx:03d}_annotated.png"
        _save_annotated_detection(img, locked_dict, annotated_path)
        if marker_id != locked_id:
            print(
                f"\nsaw marker id {marker_id}, not the locked {locked_id} — likely a "
                f"false-positive detection on background clutter, NOT captured. See "
                f"{annotated_path} to confirm what it actually locked onto."
            )
            return

        samples.append(HandEyeSample(
            T_base_flange=T_base_flange, T_camera_marker=T_camera_marker,
            arm_q=arm_q, note=f"marker_id={marker_id}",
        ))
        _save_handeye_samples(out, samples)
        print(f"\ncaptured sample {len(samples)} (marker id {marker_id}, see {annotated_path})")

    print(f"Eye-to-hand capture, {hand} hand. Grip: 1-6 select motor, j/k -/+ step, "
          "o open, g grip closed. Capture: c=capture, p=print, q=quit+solve.")
    print(f"Motor order: {layout.HAND_MOTOR_ORDER}")
    print("Grip the tag first (watch the hand). Once it's solid, move the arm "
          "through DIFFERENT ORIENTATIONS between captures -- see the docstring.\n")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key(timeout=0.05)
            if key:
                key = key.lower()
                if key == "q":
                    break
                elif key in "123456":
                    selected_motor = int(key) - 1
                elif key == "j":
                    hand_msgs[hand].cmds[selected_motor].q = float(
                        np.clip(hand_msgs[hand].cmds[selected_motor].q - grip_step, 0.0, 1.0))
                elif key == "k":
                    hand_msgs[hand].cmds[selected_motor].q = float(
                        np.clip(hand_msgs[hand].cmds[selected_motor].q + grip_step, 0.0, 1.0))
                elif key == "o":
                    for i in range(layout.HAND_DIM):
                        hand_msgs[hand].cmds[i].q = 0.0
                elif key == "g":
                    for i in range(layout.HAND_DIM):
                        hand_msgs[hand].cmds[i].q = 1.0
                elif key == "p":
                    print(f"\n{_hand_status_line()}")
                elif key == "c":
                    # Wrapped broadly on purpose: a bad frame or a detection
                    # edge case must never crash the session and lose every
                    # sample already captured (in memory only) -- see the
                    # docstring's account of the real crash this guards.
                    try:
                        _attempt_capture()
                    except Exception as exc:  # noqa: BLE001 -- see docstring, must never crash the session
                        print(f"\ncapture attempt failed ({exc!r}) — samples so far are safe, try again")

            if executor is not None:
                # ONE combined row, not `executor.hold()` -- that helper
                # zeroes everything outside the arm slice, which would
                # publish "both hands open" every tick via unitree_deploy's
                # own Brainco control (bundled into the same send_action;
                # see `hand_pubs`' comment above for the bug this caused).
                # Arm slice = current measured pose (a safe no-op waypoint,
                # same intent as hold()); hand slices = whatever 1-6/j/k/o/g
                # currently have `hand_msgs` set to, for BOTH hands.
                row = np.zeros(_actions.ROBOT_DIM)
                row[_actions.ARM] = executor.arm_q()
                for h in layout.HANDS:
                    row[_actions.HAND[h]] = [hand_msgs[h].cmds[i].q for i in range(layout.HAND_DIM)]
                executor.send(row)
            else:
                for h in layout.HANDS:
                    hand_pubs[h].Write(hand_msgs[h])
            print(_hand_status_line(), end="\r")
    except KeyboardInterrupt:
        print("\ninterrupted", end="")
        if executor is not None:
            print(" — DAMPING.")
            executor.damp()
        else:
            print(".")
    except Exception:
        if executor is not None:
            print("\nunexpected error — DAMPING.")
            executor.damp()
        raise
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        cam.close()
        if executor is not None:
            executor.close()

    print(f"\n{len(samples)} sample(s) total.")
    if len(samples) < 3:
        print("fewer than 3 samples — nothing to solve. Re-run and capture more.")
        return

    T_pelvis_camera, report = solve_eye_to_hand(samples)
    print(f"T_pelvis_camera =\n{T_pelvis_camera}")
    print(f"max pairwise flange-rotation spread: {report['rotation_spread_deg']:.1f} deg")
    print(f"consistency check — estimated grip-offset translation std: "
          f"{report['translation_std_m'] * 1000:.2f} mm, rotation spread: "
          f"{report['rotation_spread_deg_estimate']:.2f} deg")

    np.savez(
        out,
        T_base_flange=np.stack([s.T_base_flange for s in samples]),
        T_camera_marker=np.stack([s.T_camera_marker for s in samples]),
        arm_q=np.stack([s.arm_q for s in samples]),
        T_pelvis_camera=T_pelvis_camera,
    )
    print(f"saved -> {out}")
