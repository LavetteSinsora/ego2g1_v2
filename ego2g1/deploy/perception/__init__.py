"""Live perception for `relation_eef` deploy (docs/relation_deploy_plan.md, §5).

New subpackage, imported lazily by whatever wires it up (`policy_adapter.py`'s
`RelationPolicyAdapter`) -- a `joint`- or `relative_eef`-mode deploy must never
pay for camera/detector/torch-vision imports, mirroring `kinematics.py`'s
"mujoco/mink imported only where needed" discipline. Submodules here should
stay independently importable (numpy-only where possible) rather than routing
everything through this `__init__`.

Module map (filled in as Phase 2 tasks land, docs/relation_deploy_plan.md §9):

  latch          per-hand grasp-confirmation state machine (§5.4) -- pure
                 numpy, no camera/detector/robot dependency, testable in full
                 isolation with synthetic trajectories.
  task_config    object-prompt configuration cross-checked against the
                 server's checkpoint metadata (§5.1).
  depth          DepthSource interface + StereoSGBM implementation (§5.2).
"""
