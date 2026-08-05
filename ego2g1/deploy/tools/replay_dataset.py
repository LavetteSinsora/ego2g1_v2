"""Play a LeRobot dataset's joint stream through the executor — the hardware A/B.

This is the experiment jitter_root_cause.md sets up: the OLD dataset's stored
joints were IK'd from unsmoothed Pico targets (26 rad/s² worst-joint accel RMS
in the parquet itself), so replaying them judders through ANY executor,
including the proven-smooth vendored one. The re-extracted dataset (IK re-solved
after label smoothing, ego2g1.data) should not. Same robot, same executor, same
command path — the only variable is the data. If the new extraction replays
smooth and the old one still judders, the target-path fix is confirmed on
hardware; if BOTH judder, the executor setup is wrong and no policy will help.

Two joint sources, same streaming path:

  default      stream the episode's STORED joints (`arm_qpos`). No IK. Catches
               plumbing (joint order, sign, units, rates, hand mapping) with a
               trajectory we KNOW — anything wrong here is the wiring.
  --from-eef   ignore stored joints; re-solve IK offline from the stored flange
               poses (`pose.{hand}`) through the SAME deploy solver
               (posture-tracks-last @ 0.05 + JointFilter). Exercises the
               eef->joint conversion in isolation, and doubles as an on-robot
               preview of what a re-extraction would store.

Hands stream unchanged in both modes (hand commands are already actuator
space). The first motion is the vendored executor's own drive_to_waypoint soft
ramp — the first send() ramps from wherever the arm is.

    python -m ego2g1.deploy.replay_dataset --dataset data/lerobot_datasets/ego2g1/put_bottle_in_box
    python -m ego2g1.deploy.replay_dataset --dataset ... --from-eef
    python -m ego2g1.deploy.replay_dataset --dataset ... --dry-run   # no robot: stats only
"""

from __future__ import annotations

import dataclasses
import logging
import json
import pathlib
import time

import numpy as np

from ...core import layout, se3
from .. import actions as _actions

logger = logging.getLogger(__name__)


def load_episode(root: str, episode: int = 0) -> dict:
    """Read one episode's streams straight from the parquet (no lerobot import)."""
    import pandas as pd

    files = sorted(pathlib.Path(root).glob("data/*/*.parquet")) \
        or sorted(pathlib.Path(root).glob("data/*.parquet"))
    if not files:
        sibs = [d.name for d in pathlib.Path(root).parent.glob("*") if d.is_dir()]
        raise FileNotFoundError(
            f"no parquet under {root}/data/ — check the folder name; "
            f"siblings here: {sibs}")
    f = files[min(episode, len(files) - 1)]
    df = pd.read_parquet(f)
    if "arm_qpos" in df.columns:
        return {
            "name": f.name,
            "arm": np.stack(df["arm_qpos"].to_numpy()).astype(np.float64),
            "pose": {h: np.stack(df[f"pose.{h}"].to_numpy()).astype(np.float64)
                     for h in layout.HANDS},
            "hand": {h: np.stack(df[f"hand.{h}"].to_numpy()).astype(np.float64)
                     for h in layout.HANDS},
        }
    if "observation.state" in df.columns:
        # Foreign schema: NEVER guess the layout from the shape — read the
        # feature names the dataset itself declares. (A ZH "EEF30" dataset is
        # 30-D of [wrist pose 9|9 | hand 12]; slicing that as joints would
        # drive pose numbers into the motors.)
        names = []
        meta = pathlib.Path(root) / "meta" / "info.json"
        if meta.exists():
            feat = json.loads(meta.read_text())["features"].get("observation.state", {})
            n = feat.get("names") or []
            names = list(n[0]) if n and isinstance(n[0], (list, tuple)) else list(n)
        if any("Wrist" in str(x) for x in names):
            raise SystemExit(
                f"{f.name} is an EEF-pose dataset ({names[:3]}...) in ZH's frame "
                "conventions — not directly replayable as joints, and our IK "
                "does not share their wrist frame. Use a joint-space dataset "
                "(names like kLeftShoulderPitch) or this repo's own datasets.")
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        joint_like = names and all(("Wrist" not in str(x)) for x in names) and \
            any(("Shoulder" in str(x) or "Elbow" in str(x)) for x in names)
        if not joint_like:
            raise SystemExit(
                f"cannot establish that {f.name} is joint-space "
                f"(names: {names[:6] or 'MISSING'}); refusing to guess.")
        if state.shape[1] < 26:
            raise ValueError(f"observation.state is {state.shape[1]}-D, expected >=26")
        logger.warning("joint-space dataset (%s): using observation.state, "
                       "assuming [arm14|handL6|handR6] order — verify with a "
                       "single-joint sanity wiggle before a full replay", f.name)
        return {
            "name": f.name,
            "arm": state[:, :14],
            "pose": None,                      # no EEF labels; --from-eef unavailable
            "hand": {"left": state[:, 14:20], "right": state[:, 20:26]},
        }
    raise ValueError(
        f"unrecognized dataset schema in {f.name}; columns: {list(df.columns)}")


def accel_rms(q: np.ndarray, fps: float) -> np.ndarray:
    """Per-joint acceleration RMS (rad/s²) — the jitter_root_cause metric."""
    a = np.diff(q, n=2, axis=0) * fps * fps
    return np.sqrt((a ** 2).mean(axis=0))


def solve_from_eef(ep: dict, fps: int, *, ik_iters: int = 40,
                   posture_cost: float = 0.05, tol_m: float = 0.02) -> np.ndarray:
    """Re-solve the episode's joints from its stored flange poses, offline,
    through the deploy solver. Aborts if any tick cannot be tracked to `tol_m`
    — the QP silently approximates an unreachable pose, and we refuse to send
    an approximation we did not vet to the real arm."""
    from ..core.kinematics import Kinematics

    kin = Kinematics(ik_iters=ik_iters, fps=fps, posture_cost=posture_cost)
    kin.ground(ep["arm"][0])

    n = len(ep["arm"])
    q = np.empty((n, layout.ARM_DOF))
    err = np.empty(n)
    logger.info("re-solving IK for %d ticks (iters=%d, posture_cost=%g) ...",
                n, ik_iters, posture_cost)
    t0 = time.perf_counter()
    for t in range(n):
        targets = {h: se3.vec9_to_se3(ep["pose"][h][t]) for h in layout.HANDS}
        q[t] = kin.solve(targets)
        err[t] = max(kin.tracking_error(targets).values())
    logger.info("  flange track mean %.2f mm  max %.2f mm  (%.1f ms/tick)",
                err.mean() * 1000, err.max() * 1000,
                (time.perf_counter() - t0) / n * 1000)
    if err.max() > tol_m:
        raise SystemExit(
            f"FAIL — IK cannot track the stored poses to {tol_m*1000:.0f} mm "
            f"(worst {err.max()*1000:.1f} mm). Refusing to drive the arm.")
    return q


@dataclasses.dataclass
class Args:
    dataset: str
    episode: int = 0
    fps: int = 30
    from_eef: bool = False
    ik_iters: int = 40
    ik_tol: float = 0.02
    network_interface: str | None = None
    max_pos_speed: float | None = None     # soften the interpolator for bring-up
    hands: bool = True
    dry_run: bool = False                  # stats + mock executor, no robot
    yes: bool = False                      # skip the confirmation prompt
    max_joint_step: float = 0.15           # session.py's per-tick clamp; a
                                           # legitimate replay never hits it


def main(args: Args) -> None:
    ep = load_episode(args.dataset, args.episode)
    if args.from_eef and ep["pose"] is None:
        raise SystemExit("--from-eef needs pose columns; this dataset is joint-space only")
    if args.from_eef:
        arm = solve_from_eef(ep, args.fps, ik_iters=args.ik_iters,
                             tol_m=args.ik_tol)
        src = "IK re-solved from stored eef poses"
    else:
        arm = ep["arm"]
        src = "stored joints"

    rms = accel_rms(arm, args.fps)
    worst = int(np.argmax(rms))
    joints = [j for h in layout.HANDS for j in layout.ARM_JOINTS[h]]
    print(f"{ep['name']}: {len(arm)} frames @ {args.fps} Hz = "
          f"{len(arm)/args.fps:.1f} s  [{src}]")
    print(f"accel RMS worst joint: {rms[worst]:.1f} rad/s² ({joints[worst]})  "
          f"mean {rms.mean():.1f}")
    print("  (old unsmoothed extractions measure ~26 rad/s² worst; the "
          "re-extracted target is <8 — jitter_root_cause.md)")

    if args.dry_run:
        from ..core.executor import MockExecutor
        executor = MockExecutor(fps=args.fps, initial_q=arm[0])
    else:
        from ..core.executor import UnitreeExecutor
        executor = UnitreeExecutor(fps=args.fps,
                                   network_interface=args.network_interface,
                                   max_pos_speed=args.max_pos_speed)

    if not args.yes and not args.dry_run:
        q_preview = np.round(arm[0], 3)
        if input(f"\nfirst waypoint {q_preview}\nsoft-ramp there and replay? "
                 "[y/N] ").strip().lower() != "y":
            return

    executor.connect()
    from ..core import safety as _safety
    from ..core.session import ExecutorSession
    sess = ExecutorSession(executor, fps=args.fps,
                           limits=_safety.SafetyLimits(
                               max_joint_step=args.max_joint_step))

    def row_at(k: int) -> np.ndarray:
        row = np.zeros(_actions.ROBOT_DIM)
        row[_actions.ARM] = arm[k]
        for h in layout.HANDS:
            row[_actions.HAND[h]] = ep["hand"][h][k] if args.hands else 0.0
        return row

    try:
        # First send: unitree_deploy's own drive_to_waypoint soft ramp takes
        # the arm from wherever it is to the episode start. Give it time.
        sess.soft_start(row_at(0), settle_s=0.0 if args.dry_run else 2.0)
        if sess.stream(row_at(k) for k in range(len(arm))):
            print("replay complete.")
    finally:
        executor.close()
        if sess.clamp.clamped_ticks:
            print(f"NOTE: clamp limited {sess.clamp.clamped_ticks} tick(s) "
                  f"(max step seen {sess.clamp.max_seen:.3f} rad) — a stored-"
                  "joint replay should never hit the clamp; check the data.")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
