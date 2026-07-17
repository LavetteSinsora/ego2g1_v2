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

Module map (imports are kept lazy so joint-mode never touches mujoco and no
module needs the robot to import):

  actions        the action-mode boundary: policy chunks (joint OR
                 relative_eef) -> timestamped JOINT chunks. The one place the
                 two modes differ.
  policy_adapter wraps the policy client so the runner only ever sees joint
                 chunks (ZH's adapter pattern; carries our RTC prefix contract)
  strategies     the five chunk consumers (sync / naive-async / temporal
                 ensembling / temporal smoothing / RTC), ported from
                 zh_deploy_inference
  runner         the 30 Hz step loop: obs -> strategy -> clamp -> executor,
                 precise_wait-paced, future-stamped targets
  executor       unitree_deploy wrapper (arm+Brainco) + damp() e-stop +
                 MockExecutor for hardware-free tests
  kinematics     FK anchors/state + the deploy IK (ego2g1.kin) with
                 posture-tracks-last-solution
  latency        DelayBudget + the startup latency self-check
  safety         Clamp, Watchdog, action sanity checks
  recorder       events.jsonl + mp4 session recording
  client         websocket PolicyClient to `python -m ego2g1.serve`
  camera         the single egocentric head camera (image_server ZMQ client)
  check          the bring-up rung ladder — walk it in order
  replay_dataset the hardware A/B: play a LeRobot dataset through the executor
  replay_diag / measure_rate / sniff_lowcmd   transport + servo diagnostics
"""
