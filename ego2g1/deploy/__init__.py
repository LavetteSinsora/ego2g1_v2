"""deploy: run a policy (or a recording) on the real G1-D, smoothly, on purpose.

The design answers docs/jitter_root_cause.md point by point:

  * the EXECUTOR is vendored, not rewritten: unitree_deploy's 500 Hz
    interpolating arm controller with gravity-comp tau feedforward
    (third_party/unitree_deploy) is measured-smooth — the servo holds <1 mrad
    on a constant command. We wrap it (`executor`), we never reimplement it.
  * the TARGET PATH is where jitter is made, so that is where the fix lives:
    EEF chunks are One-Euro-smoothed, IK'd with the posture task retargeted to
    the previous solution each tick (cost 0.05 — measured 25 -> 4-7 rad/s²
    worst-joint accel RMS), then joint-filtered (`actions`, `kinematics`).
  * LATENCY is checked before the robot moves, not discovered mid-rollout:
    measured 1.3-5.1 s inferences against a 0.4 s budget made the freeze-lurch;
    `latency.startup_self_check` refuses to start a timing strategy the
    measured latency cannot honor.
  * everything is RECORDED (`recorder`, events.jsonl) — the jitter was only
    diagnosable because the old deploy logged every seam.

Package map (docs/deploy_refactor_plan.md §1 — the directory tree matches the
semantic planes; imports are kept lazy so joint-mode never touches mujoco and
no module needs the robot to import):

  core/          the mode-blind execution spine: runner (the 30 Hz step loop,
                 precise_wait-paced, future-stamped), strategies (the five
                 chunk consumers), session (ExecutorSession — the ONE road
                 rows take to the executor: sanity + clamp + stamp + pace +
                 damp-on-interrupt), safety, latency, executor (unitree_deploy
                 wrapper + damp() e-stop + MockExecutor), kinematics, client,
                 fast_crc
  modes/         one DeployMode object per policy family (joint /
                 relative_eef / relation_eef) — the observation shape, adapter
                 wiring, hand bookkeeping, and per-mode telemetry/recorder
                 extras. Adding a policy family = one file here.
  actions        the action-mode boundary: policy chunks -> (H, 26) JOINT
                 chunks (_EEFChunksBase + per-mode decode), model-space guards
  policy_adapter wraps the policy client so the runner only ever sees joint
                 chunks (ZH's adapter pattern; carries our RTC prefix contract)
  perception/    the relation_eef cascade: detector / depth / tracker / latch
                 / relation_perception + the calibration solvers
  record/        the recording contract: schema (event kinds + build_meta,
                 the single source of truth), recorder (events.jsonl + mp4),
                 session_reader (reconstruct any session at any t)
  ui/            telemetry (TelemetrySnapshot — the page's ONE declared data
                 shape) + overlay (the perception overlay renderer, fed the
                 recorded `percept` shape), dashboard, replay_dashboard,
                 perception_preview
  tools/         the bring-up rung ladder (check — walk it in order), the
                 replay tools (replay_dataset: the hardware A/B; replay_diag;
                 replay_mujoco; replay_relation_openloop), and the wire
                 diagnostics (measure_rate, sniff_lowcmd)
  camera / remote_image_server / gripper_calib / _util   shared leaves

Every documented `python -m ego2g1.deploy.<name>` entrypoint keeps working:
the old flat module paths are shims onto the new locations.
"""
