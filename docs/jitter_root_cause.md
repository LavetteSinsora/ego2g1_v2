# Why the arm juddered (and why gravity comp didn't fix it)

Measured 2026-07-17 from the live-deploy recording
`recordings/put_the_bottle_in_the_box_20260716T153733` (old repo: 30 Hz commanded
IK knots + 37 Hz measured joints + splice/watchdog events) and offline IK
experiments on `episode_1`. The old diagnosis — tau=0 / no gravity compensation —
was fixed (`run_unitree` routes through unitree_deploy's gravity-comp executor)
and the judder survived. These measurements explain why.

## The servo is innocent

During a 7.7 s stretch where the command was held constant, the measured arm is
dead quiet:

| metric (all 14 joints) | value |
|---|---|
| position std | < 1 mrad |
| peak-to-peak | < 5 mrad |
| velocity RMS | ~0.005 rad/s |

No hunting, no stick-slip at rest. Gains, damping, gravity comp, DDS transport —
none of it oscillates on its own. Whatever judders comes in with the **targets**.

## Cause 1 — the joint targets are rough at the source

The commanded knots themselves carry violent accelerations *before* the robot
ever sees them:

| joint stream | accel RMS (worst joint) |
|---|---|
| live deploy knots, 30 Hz | 38–75 rad/s² (R_elbow) |
| dataset `arm_qpos` (s003, what replay replays) | 26 rad/s² (R_wrist_pitch) |

And the IK is **not** under-converged: at `iters=5` it already tracks the EEF
targets to 0.00 cm; 25 iterations is bit-identical roughness. The IK is
*faithfully* converting noise that lives in the EEF targets — Pico hand-tracking
noise, mostly rotational — into joint zig-zag:

| episode_1 right hand EEF targets | angular accel RMS |
|---|---|
| raw (what s003 IK consumed) | 1197 deg/s² |
| after s004b SavGol smoothing | 156 deg/s² |

Small orientation wobble maps through the Jacobian into large wrist-joint motion;
the zig-zag is ~10 mrad in position (invisible in a MuJoCo animation — "the model
moves smoothly") but reverses at 10–15 Hz, which the real arm renders as buzz and
judder. This is also why **pure dataset replay judders through any executor,
including unitree_deploy's**: the stored joints were IK'd from *unsmoothed*
targets (s004b smooths only the action labels, never `arm_qpos`).

Knob sensitivity (episode_1, offline):

| IK variant | joint accel RMS (worst) | EEF cost |
|---|---|---|
| iters=5 (production) | 26.4 rad/s² | 0.00 cm |
| iters=25 | 26.4 | 0.00 cm |
| posture_cost 1e-3 → 5e-2 | 20.4 | 1.3 cm mean |
| + EMA(0.35) on q | **6.2** | 1.4 cm mean, ≤8° ori peaks |

Filtering works; iterating doesn't. The fix belongs in the *target path*, not the
solver: smooth EEF targets first (causal One-Euro at deploy; SavGol offline),
re-solve IK after smoothing during extraction, and keep a joint-space filter as
the safety net.

## Cause 2 — live inference latency blows the timing design

Same recording: chunk inference latencies **1.31 s, 4.31 s, 5.09 s** against a
budget of `initial_d=12` slots = 0.4 s. The rollout was: play ~1.7 s of chunk →
**freeze 8.7 s** holding slot 49 → lurch onto a chunk computed from a 5 s-stale
observation (`late=True`, splice at start_index=49) → watchdog. Tracking error
peaked at 1.4 rad on the right elbow; the measured arm trails commands by ~400 ms
(cross-correlation, r=0.98) and overshoots rough commands 2–4× (underdamped
~1 Hz). No RTC/ensembling scheme survives 10× its latency budget — serve latency
on the PPU box has to be profiled and fixed before any timing parameter matters.

## Why ZH's stack never showed this

Their policy emits **joint-space chunks learned from real robot joints** — smooth
by construction, no noisy-EEF→IK step exists anywhere in their path. Their casadi
IK variant (when used) carries an explicit smoothness cost plus a 4-tap weighted
moving filter. Their 500 Hz interpolator would track our zig-zag just as
faithfully as ours did — the executor was never the differentiator.

## What this repo does about it

1. Extraction re-solves IK **after** label smoothing, so stored `arm_qpos` /
   replay joints are smooth (`ego2g1/data`).
2. Deploy smooths EEF targets causally (One-Euro) before IK and filters joints
   after it, with measured EEF-error budgets (`ego2g1/deploy`).
3. Joint-space policies are a first-class action mode — when the model outputs
   joint chunks, deploy does no IK at all (ZH-style).
4. Serving has a latency self-check: report chunk latency vs budget at startup
   and refuse RTC timing that the measured latency can't honor (`ego2g1/serve`).
