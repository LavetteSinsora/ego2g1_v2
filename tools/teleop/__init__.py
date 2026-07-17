"""Bare-hand XR teleoperation of G1-D, through the ego2g1 retargeting stack.

The operator wears no controllers. Their bare hands are tracked by a PICO headset
(nothing is installed on it -- the headset's BROWSER opens a vuer page and streams
WebXR hand data back), retargeted by the SAME code that produced the training labels,
and executed by the SAME control path the policy uses.

Why this exists: it is the only end-to-end test of the retarget. If a human can pick up
the bottle by moving their own hand, then S, B, the flange convention, the Revo2 mount
rotation and the Brainco motor order are all right. Today a policy failure and a
retargeting bug look identical.

SELF-CONTAINED. Everything this package needs is vendored into `_vendor/` (the retarget
from data_extraction, the deploy control path from ego2g1) -- so it imports nothing
outside its own directory and runs on the robot PC without the rest of the repo present.
`_vendor/_build.py` rebuilds those copies from source; `tests/test_vendor_drift.py` fails
if the source drifted, so the copies cannot silently diverge from the code that made the
training labels.

The one piece of algebra the whole thing rests on. The training label is

    G(t) = pelvis^-1 . S . T_wrist(t) . B          (s002_action_label/eef_label.py)

which factors as orientation `C . R_w . B_R` and position `C . p_w + const`, with
C = pelvis^-1 . S a heading yaw. Default mode is ABSOLUTE orientation (fixed hand<->flange
correspondence) with position RELATIVE to the engage anchor. See retarget.py.

Run from the repo root:

    .venv/bin/python -m tools.teleop.check stream
    .venv/bin/python -m tools.teleop --dataset lerobot_datasets/ego2g1/put_bottle_in_box
"""
