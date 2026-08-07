"""Perception v2: SAM 3 + Orient Anything V2, free-running async.

Implements docs/perception_v2_pipeline.md. It REPLACES the v1 cascade one
level up (`../detector.py` GroundingDINO+SAM2, `../tracker.py`, `../latch.py`,
`../relation_perception.py`) but does not delete it yet, on purpose:

  * v1 is what `modes/relation_eef.py` builds today and it runs. Deleting a
    working path before its replacement has been validated on the 4090 is how
    you end up with neither.
  * The plan's §8 gates integration behind steps 1-5, all of which need
    hardware this package cannot reach (a GPU, the SAM 3 weights, the stereo
    rig). Until those pass, v2 is code that must be *importable and tested*,
    not code that is *wired in*.

At §8 step 7 (open-loop eval passes), the cutover is: delete the v1 modules,
move these up one level, drop the `v2` package. Nothing outside this package
imports v1 and v2 at the same time.

Module map (plan §5):

  snapshot          `PerceptionSnapshot` (§4.2) — the one immutable object
                    that crosses the two clocks — plus `ControlTickLog`, the
                    30 Hz ring buffer a camera frame binds against (T4).
                    Pure numpy.
  sam3_source       `Sam3Source`: one session, all prompts, detect+track every
                    frame, prune-not-reset (M1/M3/R1). Also `VisibilityGate`
                    (S1), which is pure numpy and where the crop_usable
                    decision actually lives.
  orientation_v2    `OrientAnythingV2` (R2) + the crop/angle geometry around
                    it. The torch part is one class; everything else is pure.
  object_tracker    `ObjectTracker`: causal MAD gate + OneEuroSE3, NO
                    constant-velocity extrapolation, every window in seconds
                    (S2). Pure numpy.
  latch             `GraspLatch`: two-clock state machine, displacement
                    confirmation, position-only relative-motion divergence
                    (§6). Pure numpy.
  async_perception  `AsyncPerception`: the free-running thread and
                    `wait_for_current_round`, which is both the replan
                    primitive (T3) and the GPU arbitration (R3).
  relation          `RelationStateBuilder`: snapshot -> 56-dim hand-major
                    vector, everything from one instant (T2, §5.4).
  timing            the `d = P + L` arithmetic and the chunk-slot accounting
                    (T3, T4). Pure; nothing calls it yet — `DelayBudget` is
                    still fed policy latency alone (§8 step 9).
  config            `PerceptionV2Config` (§7): one YAML owner for every knob,
                    embedded verbatim into each recording's meta.json. The v1
                    cadence keys fail loudly rather than defaulting away.

Import discipline, same as the parent package: torch/transformers stay inside
the two modules that need them, imported at construction, never at module
scope. `snapshot`, `object_tracker`, `latch` and `relation` are numpy-only and
import on any machine — that is what makes them testable without a GPU, and
they are where the logic that can silently corrupt a state vector lives.
"""
