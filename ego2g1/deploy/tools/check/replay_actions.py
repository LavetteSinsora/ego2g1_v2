"""Rung 7 (`check replay-actions`): the episode's ACTION-shaped deltas
through the REAL conversion path (measured anchor, OneEuroSE3, IK,
JointFilter) + ExecutorSession. Rung 6 (replay_dataset) proves the
plumbing; this proves the transforms — run 6 first so 7 is
interpretable."""

from __future__ import annotations

import numpy as np

from ego2g1.core import layout, se3
from ego2g1.deploy import actions as _actions
from ego2g1.deploy.tools.replay_dataset import load_episode

# --- 7. replay the ACTION labels through the real conversion path ---------------

def replay_actions(dataset: str, episode: int = 0, fps: int = 30,
                   horizon: int = 50, ik_iters: int = 25,
                   posture_cost: float = 0.05, max_step: float = 0.2094,
                   network_interface: str | None = None,
                   max_pos_speed: float | None = None,
                   dry_run: bool = False, yes: bool = False,
                   out: str = "replay_actions.npz") -> None:
    """Drive the arm from ACTION-shaped chunks with the policy replaced by the
    recording: at each chunk start, read the MEASURED arm, anchor there, build
    the chunk's deltas from the stored poses (delta_k = T(t0)⁻¹ T(t0+k) — what
    a perfect policy would output), and run the real conversion (OneEuroSE3 ->
    IK posture-tracks-last -> JointFilter) + clamp + executor. Rung 6 proves
    the plumbing; this proves the transforms."""
    from ego2g1.deploy.core import safety as _safety
    from ego2g1.deploy.actions import RelativeEEFChunks

    ep = load_episode(dataset, episode)
    n = len(ep["arm"])
    print(f"{ep['name']}: {n} frames @ {fps} Hz, chunks of {horizon}")

    if dry_run:
        from ego2g1.deploy.core.executor import MockExecutor
        executor = MockExecutor(fps=fps, initial_q=ep["arm"][0])
    else:
        from ego2g1.deploy.core.executor import UnitreeExecutor
        executor = UnitreeExecutor(fps=fps, network_interface=network_interface,
                                   max_pos_speed=max_pos_speed)
        if not yes and input(
                "replay the action labels on the REAL arm? [y/N] "
                ).strip().lower() != "y":
            return
    executor.connect()

    from ego2g1.deploy.core.session import ExecutorSession

    converter = RelativeEEFChunks(fps=fps, ik_iters=ik_iters,
                                  posture_cost=posture_cost)
    sess = ExecutorSession(executor, fps=fps,
                           limits=_safety.SafetyLimits(max_joint_step=max_step))

    log_cmd, log_meas = [], []

    def capture(_k: int, sent: np.ndarray) -> None:
        log_cmd.append(sent[_actions.ARM].copy())
        log_meas.append(executor.arm_q())

    try:
        # soft-ramp to the start via the vendor's first-send drive_to_waypoint
        row = np.zeros(_actions.ROBOT_DIM)
        row[_actions.ARM] = ep["arm"][0]
        for h in layout.HANDS:
            row[_actions.HAND[h]] = ep["hand"][h][0]
        sess.soft_start(row, settle_s=0.0 if dry_run else 2.0)

        for t0_idx in range(0, n - 1, horizon):
            k_max = min(horizon, n - 1 - t0_idx)
            arm_q = executor.arm_q()
            hand_cmds = {h: ep["hand"][h][t0_idx] for h in layout.HANDS}
            # what a perfect policy would output against this anchor
            chunk = np.zeros((k_max, layout.DIM))
            for h in layout.HANDS:
                T0 = se3.vec9_to_se3(ep["pose"][h][t0_idx])
                for k in range(k_max):
                    Tk = se3.vec9_to_se3(ep["pose"][h][t0_idx + 1 + k])
                    chunk[k, layout.EEF[h]] = se3.se3_to_vec9(se3.se3_inv(T0) @ Tk)
                    chunk[k, layout.HAND[h]] = ep["hand"][h][t0_idx + 1 + k]
            joints = converter.convert(chunk, arm_q, hand_cmds)
            print(f"  chunk @ {t0_idx}: IK worst "
                  f"{converter.last_tracking_error*1000:.1f} mm")
            if not sess.stream(joints, on_tick=capture, start_step=t0_idx):
                break
        else:
            print("replay complete.")
    finally:
        executor.close()

    if log_cmd:
        cmd, meas = np.stack(log_cmd), np.stack(log_meas)
        err = np.abs(cmd - meas)
        print(f"\ntracking: mean {err.mean():.4f} rad   max {err.max():.4f} rad")
        print(f"clamped ticks: {sess.clamp.clamped_ticks}  "
              f"(max step seen {sess.clamp.max_seen:.3f} rad)")
        np.savez(out, q_cmd=cmd, q_meas=meas, episode=episode, fps=fps)
        print(f"wrote {out}")
