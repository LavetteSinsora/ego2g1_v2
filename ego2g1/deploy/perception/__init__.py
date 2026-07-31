"""Live perception for `relation_eef` mode: stereo depth + camera calibration.

docs/relation_deploy_plan.md §5 is the design doc; this package is Phase 2
of §9's task breakdown. Imported LAZILY everywhere else in `deploy` (mirrors
`kinematics.py`'s "mujoco/mink imported only where needed" discipline) so a
`joint`- or `relative_eef`-mode deploy never pays for `cv2`/detector imports.

Module map (as of this pass -- `detector.py`/`tracker.py`/`orientation.py`/
`latch.py`/`task_config.py` are separate tasks in §9's breakdown and may not
exist yet; do not assume this list is the whole of §5):

  depth         `DepthSource` interface + `StereoSGBMDepthSource` (§5.2) and
                the `StereoCalibration` data container both depth.py and
                stereo_calib.py share.
  stereo_calib  checkerboard/ChArUco stereo calibration (`cv2.stereoCalibrate`)
                that PRODUCES a `StereoCalibration` for `depth.py` to consume
                (§9 task 6b -- a prerequisite for metric depth, run before
                touch_calib).
  touch_calib   solves `T_pelvis_camera` (camera -> pelvis rigid transform,
                §6) from FK/detector correspondence pairs, via this repo's
                existing `core.hand.retarget._kabsch` fit.
"""
