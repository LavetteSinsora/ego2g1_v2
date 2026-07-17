"""Convert a Unitree teleop recording into the s005 LeRobot schema.

    uv run python -m ego2g1.data.teleop_to_lerobot \
        --source data/put_bottle_in_box_teleop \
        --repo-id ego2g1/put_bottle_in_box_teleop

The teleop recorder writes `data.json` (info/text/data[T]) + `colors/` jpegs.
Every frame carries `states` (achieved) and `actions` (commanded), each with
7 arm qpos, a 6-DOF hand vector, and a 6-vector `*_arm_pose`. Output is a
LeRobot dataset with the EXACT features s005 writes, so `eval_replay.rollout`
and `deploy.eval_real` consume it with no code change.

Three decisions, none of them free:

  * The recorded `*_arm_pose` is IGNORED. It is Unitree's own EEF definition —
    a different frame origin and a different tip point than ours (~4.6 cm in x,
    a few degrees). EEF is a definition, not a measurement, and the policy's
    definition is `sim/g1.py`. So we FK the joint angles ourselves, with the
    same model that produced the training labels. `--audit-pose` prints the
    residual against the recorded pose rather than using it.

  * The hand block of `state` is the COMMAND (`actions.*_ee`), not the encoder
    (`states.*_ee`) — matching training (where no robot ever executed the
    retargeted commands, so no encoders exist) and deploy (kinematics.state()
    takes the last command sent). Encoders stall against the object during a
    grasp, which is exactly when it matters.

  * `state`'s EEF block is FK of the ACHIEVED joints; `pose.*` is FK of the
    COMMANDED joints. Same achieved/target split as s003_state vs s002_01.

Teleop episodes have no tracker dropouts, no camera gaps, and no IK (the joints
are ground truth), so there is nothing for s004 to filter: one recording is one
LeRobot episode, `anchor_bad` is empty, and `S`/`B` are identity — those describe
the human->robot retarget, and there is no human here.
"""

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from ..core.rot6d import se3_to_vec9
from .config import PipelineConfig
from .s005_write_lerobot import INSTALL_CMD, _features, _ffmpeg_available
from ..kin.g1 import ARM_JOINTS, G1Backend

# data.json group -> our side name. The teleop recorder is left/right, same as us.
SIDES = ("left", "right")


class _FK:
    """The training FK: 14 arm joints -> flange pose (4x4) in the PELVIS frame.

    Waist and legs are pinned at 0, exactly as the training labels assume — so
    zero the whole qpos and write only the 14 arm joints. This is the same
    computation deploy/kinematics.py does on measured encoders.
    """

    def __init__(self):
        self._sim = G1Backend()
        self._adr = {s: [self._sim.model.joint(n).qposadr[0] for n in names]
                     for s, names in ARM_JOINTS.items()}

    def poses(self, qpos: dict) -> dict:
        """qpos: {side: (7,)} -> {side: (4,4)} in the pelvis frame."""
        self._sim.data.qpos[:] = 0.0
        for side in SIDES:
            self._sim.data.qpos[self._adr[side]] = qpos[side]
        mujoco.mj_forward(self._sim.model, self._sim.data)
        return {s: self._sim.world_to_base(self._sim.flange_pose(s)) for s in SIDES}

    def vec9(self, qpos: dict) -> dict:
        return {s: se3_to_vec9(T).astype(np.float32)
                for s, T in self.poses(qpos).items()}


def _arm(rec: dict, group: str) -> dict:
    return {s: np.asarray(rec[group][f"{s}_arm"]["qpos"], dtype=np.float64)
            for s in SIDES}


def _hand(rec: dict, group: str) -> dict:
    return {s: np.asarray(rec[group][f"{s}_ee"]["qpos"], dtype=np.float32)
            for s in SIDES}


def _load(source: Path) -> list[dict]:
    doc = json.loads((source / "data.json").read_text())
    data = doc["data"]
    if not data:
        raise SystemExit(f"{source}/data.json has no frames")
    for t, rec in enumerate(data):
        if int(rec["idx"]) != t:
            raise SystemExit(
                f"data.json frame {t} has idx={rec['idx']}: the recording is not a "
                "contiguous 0..T-1 timeline, which every tick index here assumes")
    return data


def _audit_pose(data, fk) -> None:
    """The recorded `*_arm_pose` vs our FK, for the same joints.

    Ran on put_bottle_in_box_teleop (627 frames, both arms), the residual in our
    flange frame is:

        translation  +50.0000 mm along flange x, std 0.0000, range 0.2 um
        rotation      0.00 deg, std 0.00

    i.e. Unitree's kinematics and ours are THE SAME KINEMATICS to numerical
    precision. Same pelvis origin, same link geometry. The one difference is the
    tip point: they report a point 50 mm beyond our `*_ee_site` (a palm/hand-mount
    frame; ours is the flange at the wrist_yaw_link origin).

    That settles the question this audit exists to answer: our G1 model is not the
    "wrong robot". A wrong URDF cannot produce a constant-to-the-micron offset
    across a moving trajectory — link-length error shows up as a residual that
    VARIES with arm configuration. No regenerated dataset, no retrain.

    Note the rotation is extrinsic-XYZ euler (rpy), NOT axis-angle. Read it as a
    rotvec and you get a spurious residual that grows with rotation magnitude (up
    to 25 deg on the busy right arm, ~3 deg on the near-static left) — a convention
    error that masquerades convincingly as a kinematic one.
    """
    from scipy.spatial.transform import Rotation as R

    rows = {s: [] for s in SIDES}
    for rec in data:
        poses = fk.poses(_arm(rec, "states"))
        for s in SIDES:
            rec_pose = np.asarray(rec["states"][f"{s}_arm_pose"]["qpos"], dtype=np.float64)
            T_rec = np.eye(4)
            T_rec[:3, 3] = rec_pose[:3]
            T_rec[:3, :3] = R.from_euler("xyz", rec_pose[3:6]).as_matrix()
            # residual expressed in OUR flange frame: T_ours^-1 @ T_recorded. A pure
            # convention offset is a fixed rigid transform here, whatever the arm does.
            D = np.linalg.inv(poses[s]) @ T_rec
            rows[s].append(np.concatenate(
                [D[:3, 3], R.from_matrix(D[:3, :3]).as_rotvec()]))

    print("\n  recorded *_arm_pose vs our FK  (residual in our flange frame)")
    for s in SIDES:
        D = np.stack(rows[s])
        pos_mm, rot_deg = D[:, :3] * 1e3, np.degrees(D[:, 3:])
        print(f"    {s:5s} translation mm  mean {np.round(pos_mm.mean(0), 3)}  "
              f"std {np.round(pos_mm.std(0), 3)}")
        print(f"    {s:5s} rotation    deg  mean {np.round(rot_deg.mean(0), 3)}  "
              f"std {np.round(rot_deg.std(0), 3)}")
    print("    constant residual => pure tip/frame convention (our geometry is right);")
    print("    residual varying with arm configuration => wrong link geometry.\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, type=Path,
                   help="teleop recording dir (contains data.json + colors/)")
    p.add_argument("--repo-id", default="ego2g1/put_bottle_in_box_teleop")
    p.add_argument("--output-root", default=None,
                   help="default: the pipeline's cfg.output_root")
    p.add_argument("--camera", default="color_0",
                   help="which of the recorder's cameras is the model's image. "
                        "color_0/color_1 are the head stereo pair (left/right eye), "
                        "color_2/color_3 are the wrist cams. color_0 is the eye "
                        "deploy/camera.py streams, so it is the default.")
    p.add_argument("--task", default=None,
                   help="language prompt; default cfg.task_prompt. The recorder's "
                        "own text.goal is boilerplate and is never used.")
    p.add_argument("--source-episode", default=None,
                   help="name recorded in extraction_meta (default: the source dir "
                        "stem + /episode_0)")
    p.add_argument("--audit-pose", action="store_true",
                   help="print the recorded-pose vs our-FK residual and exit")
    args = p.parse_args()

    source = args.source.expanduser().resolve()
    cfg = PipelineConfig(**({"task_prompt": args.task} if args.task else {}),
                         repo_id=args.repo_id,
                         **({"output_root": args.output_root} if args.output_root else {}))

    data = _load(source)
    T = len(data)
    fk = _FK()
    print(f"{source.name}: {T} frames, cameras={sorted(data[0]['colors'])}")

    if args.audit_pose:
        _audit_pose(data, fk)
        return

    if args.camera not in data[0]["colors"]:
        raise SystemExit(f"--camera {args.camera} not in {sorted(data[0]['colors'])}")

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise SystemExit("needs the openpi-pinned lerobot; install it with:\n  "
                         + INSTALL_CMD)

    # ---- build every array up front: a decode or FK failure must not leave a
    # ---- half-written dataset behind.
    state = np.zeros((T, cfg.state_dim), dtype=np.float32)
    arm_qpos = np.zeros((T, 14), dtype=np.float32)
    pose = {s: np.zeros((T, cfg.eef_dim), dtype=np.float32) for s in SIDES}
    hand = {s: np.zeros((T, cfg.hand_dim), dtype=np.float32) for s in SIDES}

    for t, rec in enumerate(data):
        q_achieved, q_cmd = _arm(rec, "states"), _arm(rec, "actions")
        hand_cmd = _hand(rec, "actions")            # command, not encoder — see module docstring

        eef_achieved = fk.vec9(q_achieved)
        eef_target = fk.vec9(q_cmd)

        state[t] = np.concatenate(
            [np.concatenate([eef_achieved[h], hand_cmd[h]]) for h in cfg.hands])
        arm_qpos[t] = np.concatenate([q_achieved["left"], q_achieved["right"]])
        for h in cfg.hands:
            pose[h][t] = eef_target[h]
            hand[h][t] = hand_cmd[h]

    probe = np.asarray(Image.open(source / data[0]["colors"][args.camera]).convert("RGB"))
    img_shape = probe.shape
    print(f"image: {args.camera} {img_shape[1]}x{img_shape[0]}")

    use_videos = _ffmpeg_available()
    root = Path(cfg.output_root) / cfg.repo_id
    if root.exists():
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=int(cfg.control_hz),
        root=root,
        robot_type=cfg.robot_type,
        features=_features(cfg, img_shape, use_videos),
        use_videos=use_videos,
        image_writer_threads=4,
    )

    for t, rec in enumerate(data):
        frame = {
            "image": np.asarray(
                Image.open(source / rec["colors"][args.camera]).convert("RGB"),
                dtype=np.uint8),
            "state": state[t],
            "arm_qpos": arm_qpos[t],
            "task": cfg.task_prompt,
        }
        for h in cfg.hands:
            frame[f"pose.{h}"] = pose[h][t]
            frame[f"hand.{h}"] = hand[h][t]
        dataset.add_frame(frame)
    dataset.save_episode()

    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()

    source_episode = args.source_episode or f"{source.name}/episode_0"
    (root / "extraction_meta.json").write_text(json.dumps({
        "config_hash": cfg.config_hash,
        "config": cfg.to_dict(),
        "teleop_source": {"dir": str(source), "camera": args.camera,
                          "hand_block": "actions.*_ee (command, not encoder)",
                          "eef": "FK(sim/g1.py) of the recorded joints; the "
                                 "recorder's *_arm_pose is unused"},
        "episodes": {"0": {
            "source_file": str(source / "data.json"),
            "source_episode": source_episode,
            "tick_start": 0,
            "tick_end": T,
            "episode_real_end": True,
            "anchor_bad": [],
            # identity: S and B describe the human->robot retarget, and a teleop
            # recording has no human in it.
            "S": np.eye(4).tolist(),
            "B_left": np.eye(4).tolist(),
            "B_right": np.eye(4).tolist(),
            "filter_stats": {},
        }},
    }, indent=2, default=str))

    print(f"\nwrote {root}")
    print(f"  1 episode, {T} frames @ {int(cfg.control_hz)} Hz, "
          f"storage={'video' if use_videos else 'image'}")
    print(f"  task: {cfg.task_prompt!r}")
    print(f"  source_episode: {source_episode}")


if __name__ == "__main__":
    main()
