"""s005 [global]: write the LeRobot training dataset + extraction sidecar.

One LeRobot episode per s004 sub-episode. Per-frame features follow
SPEC.md's table (image, state, pose.left/right, hand.left/right, arm_qpos);
task string = cfg.task_prompt. The dataset is written to
cfg.output_root/cfg.repo_id (root= arg, never the default HF cache) and is a
pure build artifact: an existing root is deleted and rebuilt.

extraction_meta.json at the dataset root maps every LeRobot episode back to
its source recording + tick range and records the full config (hash asserted
by the training config).
"""

import io as _pyio
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .. import io

# datasets is pinned to openpi's uv.lock version: newer datasets (>=5) both
# breaks the pinned lerobot's API use AND writes parquet schema metadata the
# training env cannot read back.
INSTALL_CMD = ('uv pip install --python .venv/bin/python "datasets==3.6.0" "lerobot @ '
               'git+https://github.com/huggingface/lerobot@'
               '0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"')


def _ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _features(cfg, img_shape, use_videos):
    feats = {
        "image": {"dtype": "video" if use_videos else "image",
                  "shape": tuple(int(x) for x in img_shape),
                  "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (cfg.state_dim,), "names": None},
        "arm_qpos": {"dtype": "float32", "shape": (14,), "names": None},
    }
    for hand in cfg.hands:
        feats[f"pose.{hand}"] = {"dtype": "float32", "shape": (cfg.eef_dim,), "names": None}
        feats[f"hand.{hand}"] = {"dtype": "float32", "shape": (cfg.hand_dim,), "names": None}
    return feats


class _JpegSource:
    """Decode camera jpegs from the source hdf5, caching per camera index
    (consecutive ticks often share the nearest camera frame)."""

    def __init__(self, source_path):
        self._f = h5py.File(source_path, "r")
        self._ds = self._f["camera/images_left_jpeg"]
        self._cache_idx = -1
        self._cache_img = None

    def image(self, cam_idx, size=None):
        cam_idx = int(cam_idx)
        if cam_idx != self._cache_idx:
            img = Image.open(_pyio.BytesIO(np.asarray(self._ds[cam_idx]).tobytes()))
            img = img.convert("RGB")
            if size is not None:
                img = img.resize((size[1], size[0]), Image.BILINEAR)  # size=(H,W)
            self._cache_idx = cam_idx
            self._cache_img = np.asarray(img, dtype=np.uint8)
        return self._cache_img

    def close(self):
        self._f.close()


def run_global(cfg, ep_paths):
    if cfg.state_content != "eef+hand":
        raise SystemExit(
            f"s005 writes state as eef+hand (30-dim) only; "
            f"state_content={cfg.state_content!r} is not implemented here")
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise SystemExit(
            "s005 needs the openpi-pinned lerobot; install it with:\n  " + INSTALL_CMD)

    assert float(cfg.control_hz).is_integer(), \
        f"LeRobot fps must be an integer, got control_hz={cfg.control_hz}"
    fps = int(cfg.control_hz)

    use_videos = _ffmpeg_available()
    video_codec_actual = None
    root = Path(cfg.output_root) / cfg.repo_id
    if root.exists():
        shutil.rmtree(root)

    # native image size from the first episode's first matched camera frame
    first_s001, first_meta = io.load_stage(cfg, ep_paths[0].stem, "s001")
    src0 = _JpegSource(first_meta["source"])
    probe = src0.image(first_s001["cam_match"][0],
                       size=cfg.image_size)
    src0.close()
    img_shape = probe.shape  # (H, W, 3)

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=fps,
        root=root,
        robot_type=cfg.robot_type,
        features=_features(cfg, img_shape, use_videos),
        use_videos=use_videos,
        image_writer_threads=4,
    )
    if use_videos:
        # At the pinned rev, LeRobotDataset.encode_episode_videos calls
        # encode_video_frames(img_dir, video_path, fps, overwrite=True) with
        # vcodec left at its default - the codec is hardwired from s005's
        # point of view, so keep it and record what was actually used.
        import inspect
        from lerobot.common.datasets.video_utils import encode_video_frames
        video_codec_actual = inspect.signature(
            encode_video_frames).parameters["vcodec"].default
        if video_codec_actual != cfg.video_codec:
            print(f"  [s005] note: cfg.video_codec={cfg.video_codec} but the "
                  f"pinned lerobot hardwires vcodec={video_codec_actual}; "
                  f"recording video_codec_actual in meta")

    sidecar_episodes = {}
    n_frames_per_ep = []
    ep_index = 0
    B, _ = io.load_stage(cfg, None, "b_calib")
    pre = {"left": "l", "right": "r"}

    for ep_path in ep_paths:
        stem = ep_path.stem
        s001, s001_meta = io.load_stage(cfg, stem, "s001")
        # action labels come from s004b_smooth (fingers + EEF pose smoothed within
        # each good run); proprioception (state_eef, arm_qpos) comes from
        # s004c_resolve — FK of the same IK re-run on those SMOOTHED targets, so
        # the written joints are as smooth as what deploy will track
        # (docs/jitter_root_cause.md). s003's raw-target FK remains upstream as
        # the s004 filter-signal source only.
        resolved, _ = io.load_stage(cfg, stem, "s004c_resolve")
        smooth, _ = io.load_stage(cfg, stem, "s004b_smooth")
        s004, s004_meta = io.load_stage(cfg, stem, "s004")
        placement, _ = io.load_stage(cfg, stem, "s003_placement")

        state = np.concatenate(
            [np.concatenate([resolved[f"state_eef_{pre[h]}"],
                             smooth[f"hand_cmds_{pre[h]}"]], axis=1)
             for h in cfg.hands], axis=1).astype(np.float32)

        jpegs = _JpegSource(s001_meta["source"])
        anchor_bad_all = s004.get("anchor_bad",
                                  np.zeros(len(s001["ticks_ns"]), dtype=bool))
        for start, end, real_end in zip(s004["subep_start"], s004["subep_end"],
                                        s004["subep_real_end"]):
            start, end = int(start), int(end)
            for t in range(start, end):
                frame = {
                    "image": jpegs.image(s001["cam_match"][t], size=cfg.image_size),
                    "state": state[t],
                    "arm_qpos": resolved["arm_qpos"][t].astype(np.float32),
                    "task": cfg.task_prompt,
                }
                for h in cfg.hands:
                    frame[f"pose.{h}"] = smooth[f"pose_{pre[h]}"][t]
                    frame[f"hand.{h}"] = smooth[f"hand_cmds_{pre[h]}"][t]
                dataset.add_frame(frame)
            dataset.save_episode()

            sidecar_episodes[str(ep_index)] = {
                "source_file": str(s001_meta["source"]),
                "source_episode": s001_meta["episode"],
                "tick_start": start,
                "tick_end": end,
                "episode_real_end": bool(real_end),
                # frame offsets WITHIN this LeRobot episode whose tick was
                # bridged (bad but kept): excluded as datapoint anchors by the
                # loader's boundary indexing
                "anchor_bad": np.flatnonzero(
                    anchor_bad_all[start:end]).tolist(),
                "S": placement["S"].tolist(),
                "B_left": B["B_left"].tolist(),
                "B_right": B["B_right"].tolist(),
                "filter_stats": s004_meta["bad_counts"],
            }
            n_frames_per_ep.append(end - start)
            ep_index += 1
        jpegs.close()
        print(f"  [s005] {stem}: wrote {int(s004['subep_start'].shape[0])} "
              f"episode(s), running total {ep_index}")

    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()

    (root / "extraction_meta.json").write_text(json.dumps({
        "config_hash": cfg.config_hash,
        "config": cfg.to_dict(),
        "episodes": sidecar_episodes,
    }, indent=2, default=str))

    meta = {
        "n_source_episodes": len(ep_paths),
        "n_lerobot_episodes": ep_index,
        "n_frames": int(sum(n_frames_per_ep)),
        "fps": fps,
        "image_shape": [int(x) for x in img_shape],
        "use_videos": use_videos,
        "video_codec_requested": cfg.video_codec,
        "video_codec_actual": video_codec_actual if use_videos
                              else "none (ffmpeg unavailable - image storage)",
        "root": str(root),
    }
    arrays = {"n_frames": np.asarray(n_frames_per_ep, dtype=np.int64)}
    print(f"  [s005] dataset at {root}: {ep_index} episodes, "
          f"{meta['n_frames']} frames, storage="
          f"{'video' if use_videos else 'image'}")
    return arrays, meta
