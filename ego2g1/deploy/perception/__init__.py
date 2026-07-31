"""Live object-detection cascade for `relation_eef` deploy (docs/relation_deploy_plan.md §5).

Subpackages/modules in here follow the same "heavy deps imported lazily,
behind a mockable interface" discipline as `deploy/kinematics.py`
(mujoco/mink only inside `__init__`) and `deploy/camera.py` (`HeadCamera` vs
`StaticCamera` behind one interface): a `joint`- or `relative_eef`-mode
deploy must never pay for GroundingDINO/SAM2/torch imports just because this
package exists on the path.

Modules (this pass):
  detector.py    ObjectDetector interface, GroundingDinoSam2Detector (real,
                 lazy-imported), FakeDetector (deterministic test double).
  tracker.py     Fast (~20-30 Hz) per-object position tracker: constant-
                 velocity Kalman filter + causal MAD-based outlier rejection
                 + ego2g1.kin.filters.OneEuroSE3 smoothing.
  orientation.py Lightweight (~0.2 Hz, caller-paced) orientation refresh:
                 symmetry-group snapping (cube for the training checkpoint's
                 cubes, identity pass-through for non-symmetric objects like
                 the pen holder).

Not in this pass (parallel/other work, see docs/relation_deploy_plan.md §9
tasks 6/6b/7/9/10/11/12): `task_config.py` (DeployTaskConfig/ObjectSpec),
`depth.py` (DepthSource/StereoSGBM), `latch.py` (grasp-confirmation state
machine), `RelationPerception.observe(...)` (the module wiring all of the
above together). This package's `__init__.py` intentionally does not import
any of those or re-export anything, so partially-merged sibling work never
breaks `import ego2g1.deploy.perception`.
"""
