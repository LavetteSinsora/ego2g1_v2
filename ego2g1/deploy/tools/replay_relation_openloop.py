"""Open-loop rollout of a REAL served `relation_eef` checkpoint against one
validation-set episode's own recorded (image, relation_state) pairs — drives
either MuJoCo or the real robot, same code path for both (the `--dry-run`
switch, exactly like `check.py`'s `replay_actions` / `replay_dataset.py`).

This is the tool for a specific question: "the live robot consistently
reaches for a spot out of reach — does the SAME checkpoint do that against
genuine validation-distribution images too, or only against the live
camera?" At each chunk-start anchor: read the MEASURED/simulated arm
(`executor.arm_q()`, exactly how a real deployment always anchors) but feed
the POLICY the validation episode's OWN recorded image + 56-dim relation
state at the corresponding dataset timestep — `RelationPolicyAdapter`'s
`perception=None` pass-through contract (`policy_adapter.py`), the same one
every existing relation test already uses. The episode's own ground-truth
action chunk (via `ego2g1.train.relation_transforms.RelativeEEFRotvecActions`,
the SAME transform training used) is computed and printed alongside the
prediction at every chunk, so a systematic overreach shows up as a number,
not just an impression from watching the arm.

If the predicted reach is sane here (fed real validation images) but the live
robot still overreaches, that points at a live-camera/perception visual
distribution gap, not the base policy. If it overreaches here too, the
policy itself has the bias, on data it should already handle well.

MuJoCo (Tier 2): `--dry-run` runs a `MockExecutor` and writes a real
recording; view it afterward with the EXISTING, UNCHANGED `replay_mujoco.py`
— it only reads the shared 26-dim executor row + `flange_targets` recorder
fields (mode-blind by the project's own design, docs/deploy.md), so nothing
there needed to change for this to work:

    python -m ego2g1.deploy.replay_relation_openloop \\
        --dataset data/lerobot_datasets/ego2g1/red_block_in_pen_holder_ego \\
        --episode 3 --host 127.0.0.1 --port 8000 --dry-run
    mjpython -m ego2g1.deploy.replay_mujoco recordings/<printed session> --at-worst

Real robot (Tier 3), once the MuJoCo pass looks sane:

    python -m ego2g1.deploy.replay_relation_openloop \\
        --dataset data/lerobot_datasets/ego2g1/red_block_in_pen_holder_ego \\
        --episode 3 --host <serve-box> --network-interface en11

No dataset-provided starting joint pose exists for this schema (relation
datasets store only relative object/TCP geometry, never absolute robot
joints — unlike the joint-space schema `replay_dataset.py` reads) — the
first anchor is whatever `executor.arm_q()` reports (a fixed IK-seed zero
pose for `--dry-run`'s MockExecutor, the robot's actual current position for
the real one). Since every prediction is anchor-RELATIVE, the reach-length
being investigated doesn't depend on that starting pose either way.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import numpy as np

from ...core import se3
from .. import actions as _actions
from ..core import safety as _safety
from . import cli as _cli

logger = logging.getLogger(__name__)


def _load_relation_episode(root: str, episode: int) -> dict:
    """One relation-mode episode's raw streams, straight from parquet (no
    lerobot/openpi import for the I/O itself — mirrors `replay_dataset.py`'s
    own `load_episode` philosophy). Video frames are read lazily by the
    caller (imageio reader kept open, not decoded here) since only a handful
    of frames — one per chunk anchor — are ever actually needed.
    """
    import json

    import pandas as pd

    root = pathlib.Path(root)
    files = sorted(root.glob("data/*/*.parquet")) or sorted(root.glob("data/*.parquet"))
    if not files:
        sibs = [d.name for d in root.parent.glob("*") if d.is_dir()]
        raise FileNotFoundError(
            f"no parquet under {root}/data/ — check the folder name; siblings here: {sibs}")
    ep_path = files[min(episode, len(files) - 1)]
    ep_idx = int(ep_path.stem.rsplit("_", 1)[-1])

    df = pd.read_parquet(ep_path, columns=[
        "observation.state", "action", "observation.action_reference_tcp"])
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)     # (T, 56)
    action = np.stack(df["action"].to_numpy()).astype(np.float64)               # (T, 20)
    ref = np.stack(df["observation.action_reference_tcp"].to_numpy()).astype(np.float64)  # (T, 18)

    video_path = (root / "videos" / f"chunk-{ep_idx // 1000:03d}"
                 / "observation.images.camera0" / ep_path.name.replace(".parquet", ".mp4"))
    if not video_path.exists():
        raise FileNotFoundError(f"no video at {video_path} for {ep_path.name}")

    prompt = ""
    episodes_meta = root / "meta" / "episodes.jsonl"
    if episodes_meta.exists():
        with episodes_meta.open() as f:
            for line in f:
                rec = json.loads(line)
                if int(rec["episode_index"]) == ep_idx:
                    tasks = rec.get("tasks") or []
                    prompt = tasks[0] if tasks else ""
                    break

    return {
        "name": ep_path.name, "episode_index": ep_idx, "length": len(state),
        "state": state, "action_raw": action, "ref": ref,
        "video_path": video_path, "prompt": prompt,
    }


def _ground_truth_chunk(tf, action_raw: np.ndarray, ref: np.ndarray, t0: int,
                        horizon: int) -> np.ndarray:
    """(H, 14) ground-truth action chunk anchored at tick `t0`, via the SAME
    transform training used — repeat-pads the tail exactly like
    `ego2g1.train.dataset.relation_raw_action_chunks` (terminal padding =
    "hold at the final pose"), reproduced here for a single anchor rather
    than the whole split at once."""
    length = len(action_raw)
    pad = np.repeat(action_raw[-1:], horizon, axis=0)
    padded = np.concatenate([action_raw, pad], axis=0)
    window = padded[t0: t0 + horizon]
    return tf({"action": window, "observation/action_reference_tcp": ref[t0]})["actions"]


def _reach_summary(chunk14: np.ndarray, hands: tuple[str, ...]) -> str:
    """Max per-hand translation norm over a (H, 14) rotvec chunk — the
    plain-English "how far did it try to reach" number, hand-major layout
    (`relation_layout.EEF6`: 6 dims/hand, translation first 3)."""
    parts = []
    for i, h in enumerate(hands):
        d = chunk14[:, i * 6: i * 6 + 3]
        parts.append(f"{h}={np.linalg.norm(d, axis=-1).max() * 100:.1f}cm")
    return " ".join(parts)


@dataclasses.dataclass(kw_only=True)
class Args(_cli.RobotArgs, _cli.RunArgs, _cli.IKArgs):
    dataset: str
    episode: int = 0
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str | None = None          # override the episode's own recorded task string
    max_step: float = 0.2094           # safety clamp, rad/tick — same default as the runner
    record_dir: str = "recordings"
    max_chunks: int | None = None      # stop after N chunks regardless of episode length


def main(args: Args) -> None:
    from ..core import client as _client
    from .. import policy_adapter as _policy_adapter
    from ..record import recorder as _recorder

    ep = _load_relation_episode(args.dataset, args.episode)
    print(f"{ep['name']}: {ep['length']} ticks, prompt={ep['prompt']!r}")

    client = _client.PolicyClient(args.host, args.port)
    if client.control_mode != "relation_eef":
        raise SystemExit(
            f"connected checkpoint advertises control_mode={client.control_mode!r}, "
            "not 'relation_eef' — this tool is for relation checkpoints only "
            "(the connected server is likely a relative_eef/joint checkpoint instead)."
        )
    fps = client.fps
    horizon = client.action_horizon
    prompt = args.prompt if args.prompt is not None else ep["prompt"]

    adapter = _policy_adapter.make_adapter(
        "relation_eef", client, prompt, ik_iters=args.ik_iters,
        posture_cost=args.posture_cost, collision_min_dist=args.collision_min_dist)

    from ego2g1.train import relation_transforms as _rt
    tf = _rt.RelativeEEFRotvecActions(hands=tuple(client.hands))

    if args.dry_run:
        from ..core.executor import MockExecutor
        executor = MockExecutor(fps=fps, initial_q=np.zeros(_actions.ARM_DOF))
        print("--dry-run: MockExecutor + a recording — view it after with "
              "`mjpython -m ego2g1.deploy.replay_mujoco <session> --at-worst`")
    else:
        from ..core.executor import UnitreeExecutor
        executor = UnitreeExecutor(fps=fps, network_interface=args.network_interface,
                                   max_pos_speed=args.max_pos_speed)
    executor.connect()

    if not args.dry_run and not args.yes:
        q_now = np.round(executor.arm_q(), 3)
        if input(
            f"\nno dataset start pose exists for relation_eef (see module docstring) — "
            f"the robot's CURRENT position {q_now} is the open-loop anchor.\n"
            "run the served policy's OWN predictions on the real arm from here? [y/N] "
        ).strip().lower() != "y":
            return

    from ..core.session import ExecutorSession

    from ..record import schema as _schema

    # new_session's slug/stamp convention + build_meta's required strategy
    # params (inert for this tool's chunk-by-chunk sync semantics, but a
    # Session reader must never have to guess them) — this tool used to
    # hand-roll both and its recordings silently defaulted the params.
    stamp = _recorder.new_session(args.record_dir,
                                  f"relation_openloop_{ep['name']}")
    rec = _recorder.Recorder(stamp, meta=_schema.build_meta(
        mode="sync", action_mode="relation_eef", fps=fps, horizon=horizon,
        source="replay_relation_openloop",
        strategy_params={"inference_hz": 0.0, "exp_weight_m": 0.0,
                         "max_latency_steps": 0, "min_smooth_steps": 0},
        dataset=str(args.dataset), episode=ep["episode_index"], prompt=prompt,
        dry_run=args.dry_run))
    rec.start()
    print(f"recording -> {stamp}")

    import imageio.v2 as imageio
    reader = imageio.get_reader(str(ep["video_path"]))

    sess = ExecutorSession(
        executor, fps=fps, recorder=rec,
        limits=_safety.SafetyLimits(max_joint_step=args.max_step))
    sess.ground()

    hand_cmds = {"left": 0.0, "right": 0.0}   # relation_eef's converter ignores this (see actions.py)
    t0 = 0
    n_chunks = 0
    interrupted = False
    try:
        while t0 < ep["length"]:
            if args.max_chunks is not None and n_chunks >= args.max_chunks:
                break
            arm_q = executor.arm_q()
            image = np.asarray(reader.get_data(min(t0, ep["length"] - 1)))
            state = ep["state"][t0].astype(np.float32)

            out = adapter.infer({
                "arm_q": arm_q, "hand_cmds": hand_cmds,
                "relation_state": state, "image": image, "prompt": prompt,
            })
            joints = out["actions"]                                        # (horizon, 26)
            gt_chunk = _ground_truth_chunk(tf, ep["action_raw"], ep["ref"], t0, horizon)

            print(f"  chunk @ t0={t0:4d}: IK worst {adapter.last_tracking_error * 1000:5.1f} mm  "
                  f"| predicted reach {_reach_summary(out['raw_chunk'], client.hands)}  "
                  f"| ground-truth reach {_reach_summary(gt_chunk, client.hands)}")

            rec.log("obs", step=n_chunks, state_age=executor.state_age(), arm_q=arm_q)
            rec.log("infer_result", latency=out.get("client_latency_s", 0.0),
                    horizon=horizon, mode="sync", actions=joints,
                    raw_chunk=out["raw_chunk"], request_state=out["request_state"],
                    slot_errors_m=out["slot_errors_m"], flange_targets=out["flange_targets"])
            rec.log("tracking", step=n_chunks, worst_m=adapter.last_tracking_error)

            # session.py owns clamp/sanity/stamp/pace + the `action` events
            if not sess.stream(joints, start_step=t0):
                interrupted = True
                break

            t0 += horizon
            n_chunks += 1
        if not interrupted:
            print("rollout complete.")
    except KeyboardInterrupt:
        # between-chunk interrupt (inside adapter.infer); in-stream Ctrl-C is
        # already damped by sess.stream
        print("\ninterrupted — DAMPING.")
        executor.damp()
    finally:
        reader.close()
        rec.stop()
        executor.close()
        adapter.reset()


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
