"""Instrumented rung 6: replay stored JOINTS through the executor and measure
WHERE any jitter is.

Adapted from the old deploy's replay_diag.py (third_party/openpi/ego2g1/
deploy/replay_diag.py). The old tool drove a raw 500 Hz lowcmd emitter; the
new architecture delegates the 500 Hz path to the vendored executor (already
proven smooth), so this version instruments what remains OURS: the 30 Hz step
loop and the measured response. Same two suspects, same verdict shape:

  1. LOOP TIMING — our fps step loop does not actually run at fps (sleep
     granularity, GIL contention with DDS RX, per-tick work). Symptom: send
     intervals scatter far from the loop's own median. The executor's
     interpolator absorbs SMALL scatter; a multi-period stall still parks the
     trajectory.
  2. SERVO / EXECUTOR — the command knots are smooth (they are: stored joints
     + the interpolator) but the MEASURED joints step or lag. With gravity
     comp in the vendored path this should be quiet; if it is not, check
     sniff_lowcmd (foreign publisher) and measure_rate (starved link) before
     blaming gains.

    python -m ego2g1.deploy.replay_diag --dataset data/lerobot_datasets/ego2g1/put_bottle_in_box
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from ...core import layout
from .. import actions as _actions
from . import cli as _cli
from .replay_dataset import accel_rms, load_episode

logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True)
class Args(_cli.RobotArgs, _cli.RunArgs):
    dataset: str
    episode: int = 0
    fps: int = 30
    hands: bool = True
    out: str | None = None


def main(args: Args) -> None:
    ep = load_episode(args.dataset, args.episode)
    arm = ep["arm"]
    period = 1.0 / args.fps
    print(f"{ep['name']}: {len(arm)} frames @ {args.fps} Hz = "
          f"{len(arm)/args.fps:.1f} s")
    rms = accel_rms(arm, args.fps)
    print(f"source accel RMS: worst {rms.max():.1f} rad/s² — "
          "if this is ~26, expect judder from the DATA regardless of the executor\n")

    if args.dry_run:
        from ..core.executor import MockExecutor
        executor = MockExecutor(fps=args.fps, initial_q=arm[0])
    else:
        from ..core.executor import UnitreeExecutor
        executor = UnitreeExecutor(fps=args.fps,
                                   network_interface=args.network_interface,
                                   max_pos_speed=args.max_pos_speed)
        if not args.yes and input("replay + instrument on the REAL arm? [y/N] "
                                  ).strip().lower() != "y":
            return
    executor.connect()

    n = len(arm)
    t_log = np.empty(n)
    qc_log = np.empty((n, layout.ARM_DOF), dtype=np.float32)
    qm_log = np.empty((n, layout.ARM_DOF), dtype=np.float32)

    from ..core.session import ExecutorSession
    sess = ExecutorSession(executor, fps=args.fps)

    def row_at(k: int) -> np.ndarray:
        row = np.zeros(_actions.ROBOT_DIM)
        row[_actions.ARM] = arm[k]
        for h in layout.HANDS:
            row[_actions.HAND[h]] = ep["hand"][h][k] if args.hands else 0.0
        return row

    def capture(k: int, sent: np.ndarray) -> None:
        # per-tick instrumentation: the commanded (post-clamp) arm row and
        # the measured arm, stamped — the raw material of _report's verdict
        t_log[k] = time.monotonic()
        qc_log[k] = sent[_actions.ARM]
        qm_log[k] = executor.arm_q()

    try:
        # vendor drive_to_waypoint soft ramp to the start
        sess.soft_start(row_at(0), settle_s=0.0 if args.dry_run else 2.0)
        if sess.stream((row_at(k) for k in range(n)), on_tick=capture):
            print("replay complete.")
    finally:
        executor.close()

    _report(t_log, qc_log, qm_log, period)
    out = args.out or f"replay_diag_{args.fps}hz.npz"
    np.savez(out, t=t_log - t_log[0], q_cmd=qc_log, q_meas=qm_log,
             period=period, fps=args.fps, episode=args.episode)
    print(f"\nwrote {out} — plot q_cmd/q_meas and their velocities per joint.")


def _report(t: np.ndarray, qc: np.ndarray, qm: np.ndarray, period: float) -> None:
    dt = np.diff(t)
    dt_ms = dt * 1e3
    period_ms = period * 1e3
    p50, p99 = np.percentile(dt_ms, 50), np.percentile(dt_ms, 99)

    print("\n" + "=" * 64)
    print("SUSPECT 1 — LOOP TIMING  (does the step loop actually run at fps?)")
    print("=" * 64)
    eff_hz = (len(t) - 1) / (t[-1] - t[0])
    print(f"  ticks {len(t)}   wall {t[-1]-t[0]:.1f}s   effective {eff_hz:.1f} Hz "
          f"(target {1/period:.0f})")
    print(f"  interval ms: mean {dt_ms.mean():.2f}  p50 {p50:.2f}  "
          f"p95 {np.percentile(dt_ms, 95):.2f}  p99 {p99:.2f}  max {dt_ms.max():.2f}")
    spikes = int((dt_ms > 1.8 * p50).sum())
    print(f"  steadiness: p99/p50 {p99/max(p50, 1e-9):.2f}x   "
          f"spikes >1.8x p50: {spikes} ({100*spikes/len(dt):.2f}%)")
    timing_ragged = p99 > 1.8 * p50 and spikes > 0.02 * len(dt)

    print("\n" + "=" * 64)
    print("SUSPECT 2 — SERVO/EXECUTOR  (does the arm track a smooth command?)")
    print("=" * 64)
    vc = np.diff(qc, axis=0) / dt[:, None]
    vm = np.diff(qm, axis=0) / dt[:, None]
    jc = float(np.abs(np.diff(vc, axis=0)).mean())
    jm = float(np.abs(np.diff(vm, axis=0)).mean())
    lag = np.abs(qc - qm)
    print(f"  |q_cmd - q_meas|  mean {lag.mean():.4f}  max {lag.max():.4f} rad "
          f"(joint {np.unravel_index(lag.argmax(), lag.shape)[1]})")
    print(f"  velocity roughness (mean |Δv|): command {jc:.4f}  "
          f"measured {jm:.4f} rad/s/step  ratio {jm/max(jc, 1e-9):.1f}x")
    servo_bad = jm > 3 * max(jc, 1e-9)

    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    if timing_ragged:
        print("  -> LOOP TIMING is ragged (suspect #1). Intervals scatter far from")
        print("     the loop's own cadence. The executor absorbs small scatter; a")
        print("     multi-period stall parks the trajectory. Find what steals the")
        print("     tick (camera read? recorder? GIL).")
    elif servo_bad:
        print("  -> COMMAND is smooth but the ARM is not (suspect #2). With gravity")
        print("     comp in the vendored path this should not happen. In order:")
        print("     sniff_lowcmd (foreign publisher), measure_rate (starved link),")
        print("     then gains.")
    else:
        print("  -> Neither signal is strong. If the arm still looked jerky, the")
        print("     roughness is most likely IN THE DATA (see source accel RMS at")
        print("     the top) — the extraction-side fix, not a deploy knob.")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
