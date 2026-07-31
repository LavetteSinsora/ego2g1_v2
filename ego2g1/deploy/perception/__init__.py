"""perception: live object detection/tracking feeding `relation_eef` mode.

Phase 2 of docs/relation_deploy_plan.md (§5). A `RelationPolicyAdapter`
(`ego2g1.deploy.policy_adapter`) needs, every tick, a 56-dim hand-major
relation vector built from *live* object poses instead of proprioception --
this subpackage is where that comes from.

Imports here are kept lazy/minimal on purpose (mirrors `kinematics.py`'s
"mujoco/mink imported only where needed" discipline): a `joint`- or
`relative_eef`-mode deploy must never pay for perception's DINO/SAM2/
torch-vision/cv2 dependencies just by importing `ego2g1.deploy`. Submodules
stay independently importable (numpy-only where possible) rather than
routing everything through this `__init__`.

Module map (docs/relation_deploy_plan.md §9):

  task_config    ObjectSpec/DeployTaskConfig: what objects the operator
                 wants detected, in the order the checkpoint expects, plus
                 a fail-loud cross-check against the server's own metadata
                 (§5.1, task 6). Implemented.
  depth          `DepthSource` interface + `StereoSGBMDepthSource`, and the
                 `StereoCalibration` data container depth.py and
                 stereo_calib.py share (§5.2, task 7). Implemented.
  stereo_calib   checkerboard/ChArUco stereo calibration (`cv2.stereoCalibrate`)
                 that PRODUCES a `StereoCalibration` for depth.py to consume
                 (§9 task 6b -- a prerequisite for metric depth; run before
                 touch_calib). Implemented.
  touch_calib    solves `T_pelvis_camera` (camera -> pelvis rigid transform,
                 §6) from FK/detector correspondence pairs, via this repo's
                 existing `core.hand.retarget._kabsch` fit (§6.2, task 11).
                 Implemented.
  detector       GroundingDINO + SAM2 cascade tier (§5.3, task 8).
                 Implemented, heavy deps behind a lazy-import guard.
  tracker        fast between-detector tracking tier: Kalman + causal
                 outlier rejection + OneEuroSE3 smoothing (§5.3, task 8).
                 Implemented.
  orientation    cube-symmetry snapping tier (§5.3, task 8). Implemented.
  latch          per-hand grasp-confirmation state machine (§5.4, task 9) --
                 pure numpy, no camera/detector/robot dependency, testable
                 in full isolation with synthetic trajectories. Implemented.
  relation_perception   `RelationPerception.observe(...)`: the integration
                 glue wiring all of the above into one 56-dim relation
                 vector per tick (§5.5, task 10). Implemented, and wired
                 into `ego2g1.deploy.policy_adapter.RelationPolicyAdapter`
                 via its `perception=` constructor arg; `ego2g1.deploy.runner
                 .main()` constructs the real detector/depth stack and
                 passes it through for the `relation_eef` CLI entrypoint.
  gripper_calib  see `ego2g1.deploy.gripper_calib` (top-level, not here).
"""
