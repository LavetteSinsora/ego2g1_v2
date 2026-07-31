"""perception: live object detection/tracking feeding `relation_eef` mode.

Phase 2 of docs/relation_deploy_plan.md (§5). A `RelationPolicyAdapter`
(`ego2g1.deploy.policy_adapter`) needs, every tick, a 56-dim hand-major
relation vector built from *live* object poses instead of proprioception —
this subpackage is where that comes from.

Imports here are kept lazy/minimal on purpose (mirrors `kinematics.py`'s
"mujoco/mink imported only where needed" discipline): a `joint`- or
`relative_eef`-mode deploy must never pay for perception's eventual
DINO/SAM2/torch-vision dependencies just by importing `ego2g1.deploy`.
`task_config` itself only needs `pyyaml` (already a transitive dependency
of this project, see `tests/deploy/test_task_config.py`), so it is safe to
import unconditionally.

Module map (as Phase 2 tasks land, per docs/relation_deploy_plan.md §9):

  task_config    ObjectSpec/DeployTaskConfig: what objects the operator
                 wants detected, in the order the checkpoint expects, plus
                 a fail-loud cross-check against the server's own metadata
                 (§5.1, task 6). Implemented.
  detector       GroundingDINO + SAM2 cascade tier (§5.3, task 8). Not yet.
  tracker        fast between-detector tracking tier (§5.3, task 8). Not yet.
  orientation    cube-symmetry snapping tier (§5.3, task 8). Not yet.
  latch          per-hand grasp-confirmation state machine (§5.4, task 9).
                 Not yet.
  gripper_calib  see `ego2g1.deploy.gripper_calib` (top-level, not here).
"""
