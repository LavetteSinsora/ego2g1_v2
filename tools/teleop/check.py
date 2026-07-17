"""The bring-up ladder. Climb it in order; each rung assumes the one below passed.

    python -m tools.teleop.check stream      # 2  tracker only, no robot
    python -m tools.teleop.check measure-c   # 2b wrist convention vs recordings
    python -m tools.teleop.check replay      # 3  offline equivalence, no hardware

Rung 1 is not here: it is `python -m ego2g1.deploy.check hand-sweep`, which settles the
Brainco motor order against our MOTOR_ORDER. That mapping is flagged UNVERIFIED in
deploy/dds.py, and it must be settled BEFORE a human hand ever commands the fingers --
a permuted motor order means the thumb closes when you curl your ring finger, which is
not something you want to discover with an object in the hand.

Rung 3 is the one that earns its keep. It drives the REAL teleop code path -- the same
source, retargeter and IK the robot runs -- from a recorded episode, and checks it
against the offline pipeline's own output on the same data. It needs no headset and no
robot, and it catches every frame, index and convention bug there is.
"""

import argparse
import sys
import time

import numpy as np

from ._vendor.de.hand.retarget import HandRetargeter
from tools.teleop import calib as _calib
from tools.teleop.retarget import SIDES, TeleopRetargeter, load_B
from tools.teleop.source import Hdf5Source, VuerSource, wait_for_hands

DEFAULT_EPISODES = "data/put_bottle_in_box_ego/*.hdf5"


# --------------------------------------------------------------------------- rung 2

def cmd_stream(args) -> int:
    """Is the tracker giving us usable hands, and how often does it drop them?

    This is also the rung that answers the mounting question. Run it with the headset on
    your NECK: if the WebXR session survives off-head and the dropout stays low, the
    operator gets to watch the real robot with their own eyes and the PICO is a pure
    sensor. If the session dies the moment the headset leaves your head, that is the
    proximity sensor, and the answer is the head-worn fallback (display_mode="ego").
    """
    src = VuerSource(display_mode=args.display_mode, cert=args.cert, key=args.key)
    src.start()
    print(f"vuer up. On the PICO, open  https://<this-host>:8012/?ws=wss://<this-host>:8012")
    print("then press 'Virtual Reality'.\n")

    hands = tuple(args.hands)
    rt = TeleopRetargeter(load_B(args.B), hands=hands) if args.B else None
    if rt is not None:
        # calibration needs real hands; block for them first
        print("waiting for both hands in view...")
        wait_for_hands(src, hands)
        rt.calibrate(_calib.collect_open_hand(src, hands=hands))
        print("\n  calibrated.\n")
    else:
        # pure monitor: proceed and just show data (out of view until hands appear)
        while src.latest() is None:
            time.sleep(0.2)

    # `active` now means valid AND changing (see source._pump): a hand that leaves the
    # FOV freezes -- televuer holds its last pose rather than dropping it -- so `active`
    # False here is the true out-of-view signal, and the fresh-frame rate comes from
    # `frames_seen` (a re-read of frozen data is not a frame).
    t0 = time.monotonic()
    polls, stale = 0, {h: 0 for h in hands}
    last_print, last_frames, last_t = 0.0, src.frames_seen, t0
    try:
        while time.monotonic() - t0 < args.seconds:
            s = src.latest()
            now = time.monotonic()
            if s is not None:
                polls += 1
                for h in hands:
                    if not s.active[h]:
                        stale[h] += 1

            if s is not None and now - last_print > 0.5:
                fresh_hz = (src.frames_seen - last_frames) / max(now - last_t, 1e-6)
                last_print, last_frames, last_t = now, src.frames_seen, now
                bits = [f"{now - t0:5.1f}s  {fresh_hz:5.1f} Hz fresh"]
                for h in hands:
                    if s.active[h]:
                        p = s.wrist_se3(h)[:3, 3]
                        bits.append(f"{h[0]}wrist [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]")
                    else:
                        bits.append(f"{h[0]}wrist --out of view--")
                if rt is not None and rt.calibrated:
                    for h in hands:
                        if s.active[h]:
                            from ._vendor.de.hand.retarget import wrist_frame_tips
                            cmd, _, _ = rt.hand_rt[h].step(wrist_frame_tips(s.hand[h][None])[0])
                            bits.append(f"{h[0]}hand " + " ".join(f"{v:.2f}" for v in cmd))
                print("  " + "  ".join(bits), flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()

    el = time.monotonic() - t0
    print(f"\n  fresh-frame rate ~{src.frames_seen / max(el, 1e-6):.0f} Hz over {el:.1f}s")
    print("  'out of view %' = fraction of the run a hand's tracking was frozen (hands "
          "below the\n  neck mount's FOV, or the session stalled). Hold your hands where "
          "you would WORK\n  and it should be near zero; it does NOT need to be zero while "
          "your arms are resting.")
    for h in hands:
        pct = 100.0 * stale[h] / max(polls, 1)
        print(f"  {h:5s} out of view {pct:5.1f}%")
    return 0


# -------------------------------------------------------------------------- rung 2b

def cmd_measure_c(args) -> int:
    """Does the live wrist frame agree with the one the labels were fitted in?

    See calib.py for why this is the only convention that does not cancel on its own,
    and why the palm frame lets us measure it without a simultaneous capture.
    """
    src = VuerSource(display_mode=args.display_mode, cert=args.cert, key=args.key)
    src.start()
    print("vuer up — open the page on the PICO and press 'Virtual Reality'.")
    print("waiting for both hands in view...")
    wait_for_hands(src, SIDES)

    print(f"\n  Hold both hands STILL, flat and open, in view. Capturing "
          f"{args.seconds:.0f}s per hand.\n")
    time.sleep(2.0)

    ok = True
    for side in SIDES:
        G_rec, spread_rec, n_eps = _calib.mean_G_from_hdf5(args.episodes, side)
        G_live, spread_live, n_live = _calib.mean_G_from_source(
            src, side, seconds=args.seconds)

        W = _calib.measure_wrist_convention(G_rec, G_live)
        angle = _calib.frames.rot_geodesic_deg(W, np.eye(3))

        print(f"  {side}:")
        print(f"    recordings : {n_eps} episodes, wrist-vs-palm spread "
              f"{spread_rec.mean():5.1f} +/- {spread_rec.std():.1f} deg")
        print(f"    live WebXR : {n_live} frames,   spread "
              f"{spread_live.mean():5.1f} +/- {spread_live.std():.1f} deg")
        print(f"    W = {angle:6.2f} deg from identity")

        # The spreads are the honest error bar on W: G varies frame to frame with how the
        # hand is posed, so a W smaller than that variation is not a measurement of
        # anything. Only claim a non-identity W when it clearly exceeds the noise.
        noise = float(spread_rec.std() + spread_live.std())
        if angle < max(5.0, noise):
            print(f"    -> IDENTITY (within the {max(5.0, noise):.1f} deg noise floor). "
                  f"Use B as fitted; televuer's OpenXR convention claim holds.\n")
        else:
            ok = False
            print(f"    -> NOT identity, and above the {noise:.1f} deg noise floor.")
            print(f"       The live wrist frame differs from the recordings'. Apply")
            print(f"       B' = W^T . B  (calib.corrected_B) or every EEF rotation this")
            print(f"       session commands is wrong by {angle:.1f} deg.")
            print(f"       W =\n{np.array2string(W, precision=4, prefix='         ')}\n")

    src.close()
    return 0 if ok else 1


# --------------------------------------------------------------------------- rung 3

def cmd_replay(args) -> int:
    """The killer test: the live code path, over offline ground truth.

    Drives the real `Hdf5Source -> TeleopRetargeter -> mink IK` chain on a recorded
    episode and checks two independent things:

      1. the FINGERS reproduce the offline batch retarget on the same episode, to the
         bit. Any error in the (26,7) plumbing -- a transposed quaternion, an off-by-one
         landmark index, a swapped hand -- shows up here and nowhere else.
      2. the IK can actually TRACK the flange targets the retarget produces. This is a
         property of the retargeting, not of the robot: if the human's wrist path maps
         to poses the G1 cannot reach, that is the retarget telling you so, and it is far
         better to learn it here than from a watchdog trip with the arm moving.
    """
    from ._vendor.eg.deploy.kinematics import Kinematics

    src = Hdf5Source(args.episode)
    B = load_B(args.B) if args.B else {s: np.eye(4) for s in SIDES}

    # Calibrate BOTH paths identically, so the only thing under test is the plumbing.
    # (Letting each pick its own most-open frame would compare two different retargets
    # and call the difference a bug.)
    offline = {s: HandRetargeter(s) for s in SIDES}
    ref = {}
    for s in SIDES:
        ref[s] = offline[s].retarget(src.pose[s], timestamps_ns=None, active=src.act[s])

    # no ramp AND no finger smoothing: this rung tests the (26,7) plumbing against the
    # offline solve to the bit, and the One-Euro smoother is a live-only causal filter the
    # offline path has no counterpart for.
    rt = TeleopRetargeter(B, rate_limit=False, engage_ramp_s=0.0, finger_smooth=False)
    for s in SIDES:
        rt.hand_rt[s].R_align = offline[s].R_align
        rt.hand_rt[s].scales = offline[s].scales

    kin = Kinematics()

    # WHERE TO ANCHOR -- the subtlest thing in this file.
    #
    # Offline, the placement fit S is what puts the human's wrists inside the robot's
    # reachable set (reach violation is the dominant term in refine_placement's cost).
    # Teleop has no S: it cancels out of the relative action. The ANCHOR takes its place.
    # G_engage decides where the human's engage pose lands on the robot, so it must be a
    # pose from which the human's subsequent motion is actually reachable.
    #
    # Anchoring at the nominal ready pose (arms hanging: NOMINAL_ARM_QPOS sets only the
    # shoulder rolls) does NOT satisfy that -- it puts this episode ~190 mm outside the
    # arm's reach, because a human starts a reach with their hands up in front of them,
    # not by their thighs. That is a real, physical result and it looks exactly like a
    # broken transform, so it is worth being explicit: it is not one.
    #
    # The anchor that IS right is the pose the training data starts from, G(t0) =
    # pelvis^-1 . S . T_w(t0) . B, which the placement fit guarantees is reachable. On the
    # real robot that is exactly what `deploy --start-from-episode` ramps to. So this is
    # also an operational requirement, not just a test detail: THE ROBOT MUST BE RAMPED TO
    # A START POSE MATCHING THE OPERATOR'S ENGAGE POSTURE.
    #
    # Anchoring there makes this a genuine cross-check against the pipeline: the teleop
    # targets must now equal the pipeline's own labels G(t), and the IK must track them
    # exactly as well as the offline stage does.
    #
    # This runs the DEFAULT (absolute-orientation) path. Absolute mode reproduces G(t)
    # exactly -- not by cancelling S, but by USING it: with the heading C set to the exact
    # pelvis^-1 . S, orientation is C . R_w . B_R = G(t)'s orientation, and the
    # engage-relative position telescopes to G(t)'s position (both shown in
    # tests/test_cancellation.py). Live, C is a yaw estimate from a matched pose; here it
    # is exact, so the composition itself is what is under test.
    from ._vendor.de.common import frames
    from ._vendor.eg.deploy._g1_sim.sim.g1 import G1Backend

    S = _load_S(args.episode)
    pelvis_inv = frames.se3_inv(G1Backend().base_pose())
    C_exact = (pelvis_inv @ S)[:3, :3]

    def label(side, i):
        """The pipeline's own G(t), for the same episode."""
        return pelvis_inv @ S @ src.at(i).wrist_se3(side) @ B[side]

    first = next(i for i in range(src.n) if all(src.at(i).active[s] for s in SIDES))
    anchor = {s: label(s, first) for s in SIDES}
    kin.ground(np.zeros(14))          # cold IK seed; it converges on the first tick
    rt.set_heading_matrix(C_exact)
    rt.engage(src.at(first), anchor)

    worst_cmd, worst_pose = 0.0, 0.0
    err_pos, t_ik, n = [], 0.0, 0
    for i in range(first, src.n):
        sample = src.at(i)
        targets, cmds, _ = rt.step(sample, now=i / 40.0)

        for s in SIDES:
            if sample.active[s]:
                worst_cmd = max(worst_cmd,
                                float(np.abs(cmds[s] - ref[s]["cmds_raw"][i]).max()))
                worst_pose = max(worst_pose,
                                 float(np.abs(targets[s] - label(s, i)).max()))

        t0 = time.perf_counter()
        kin.solve(targets)
        t_ik += time.perf_counter() - t0
        err_pos.append(max(kin.tracking_error(targets).values()))
        n += 1

    err_pos = np.array(err_pos)
    print(f"\n  episode      : {args.episode}  ({n} frames)")
    print(f"  EEF targets  : max |teleop - pipeline label| = {worst_pose:.3e}  "
          f"(absolute orientation + engage-relative position == G(t))")
    print(f"  FINGERS      : max |live - offline| = {worst_cmd:.3e}  "
          f"({'BIT-IDENTICAL' if worst_cmd == 0.0 else 'MISMATCH'})")
    print(f"  IK tracking  : mean {err_pos.mean()*1e3:6.1f} mm   p95 "
          f"{np.percentile(err_pos,95)*1e3:6.1f} mm   max {err_pos.max()*1e3:6.1f} mm")
    print(f"  IK cost      : {t_ik/max(n,1)*1e3:.2f} ms/tick  "
          f"(budget at 60 Hz is 16.7 ms)\n")

    ok = True
    if worst_pose > 1e-9:
        print("  FAIL: the teleop targets are not the pipeline's labels. The anchor/B\n"
              "        composition is wrong.")
        ok = False
    if worst_cmd != 0.0:
        print("  FAIL: the live hand path does not reproduce the offline one. The (26,7)\n"
              "        plumbing is wrong — check the landmark index shift and the quat order.")
        ok = False
    if err_pos.max() > 0.10:
        print(f"  FAIL: IK cannot track the retargeted poses (max {err_pos.max()*1e3:.0f} mm\n"
              f"        > the 100 mm watchdog limit). On the robot this trips the e-stop.")
        ok = False
    if ok:
        print("  PASS")
    return 0 if ok else 1


def _load_S(episode: str) -> np.ndarray:
    """The episode's placement, from the pipeline's own cache."""
    import pathlib

    stem = pathlib.Path(episode).stem
    p = pathlib.Path("data_extraction/work") / stem / "s003_placement.npz"
    if not p.exists():
        raise SystemExit(
            f"{p} not found — run the pipeline first:\n"
            f"  .venv/bin/python -m data_extraction.run_pipeline --through b_calib "
            f"--set episodes_dir=$PWD/data/put_bottle_in_box_ego")
    with np.load(p) as z:
        return np.asarray(z["S"], dtype=np.float64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--cert", default=None, help="TLS cert for the vuer page")
        q.add_argument("--key", default=None)
        q.add_argument("--display-mode", default="pass-through",
                       choices=["pass-through", "ego", "immersive"],
                       help="pass-through = operator sees the real world (neck-mounted); "
                            "ego = robot view inset in the real world (head-worn)")

    s = sub.add_parser("stream", help="rung 2: tracker only, no robot")
    common(s)
    s.add_argument("--seconds", type=float, default=60.0)
    s.add_argument("--hands", nargs="+", default=list(SIDES))
    s.add_argument("--B", default=None,
                   help="b_calib npz or LeRobot dataset root; enables live hand commands")
    s.set_defaults(fn=cmd_stream)

    s = sub.add_parser("measure-c", help="rung 2b: wrist convention vs the recordings")
    common(s)
    s.add_argument("--seconds", type=float, default=5.0)
    s.add_argument("--episodes", default=DEFAULT_EPISODES)
    s.set_defaults(fn=cmd_measure_c)

    s = sub.add_parser("replay", help="rung 3: live code path over offline ground truth")
    s.add_argument("--episode", default="data/put_bottle_in_box_ego/episode_1.hdf5")
    s.add_argument("--B", default=None)
    s.set_defaults(fn=cmd_replay)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
