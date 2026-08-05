"""The bring-up ladder. Walk it in order; each rung gates the next.

Adapted from the old deploy's check.py (third_party/openpi/ego2g1/deploy) to
the vendored-executor architecture, then split one-module-per-rung-family
(docs/deploy_refactor_plan.md §7). Rungs 1/4/4c/5/6/7 touch the robot;
2/3/8 do not.

    python -m ego2g1.deploy.check listen      # 1. DDS only, no commands   [robot]
    python -m ego2g1.deploy.check fk          # 2. FK vs dataset state     [offline]
    python -m ego2g1.deploy.check ik          # 3. IK vs dataset joints    [offline]
    python -m ego2g1.deploy.check tcp-orientation  # 3b. TCP/flange convention [offline]
    python -m ego2g1.deploy.check camera      # 4. one frame, to disk      [robot]
    python -m ego2g1.deploy.check stereo-capture
                                               # (writes into calibration/
                                               #  camera_intrinsics_calibration/)
                                               # 4b. one auto-numbered stereo
                                               #     pair, for calibration    [robot]
    python -m ego2g1.deploy.check handeye-capture --hand right
                                               # 4c. interactively solve camera
                                               #     extrinsics from a gripped
                                               #     AprilTag/ArUco marker      [robot]
    python -m ego2g1.deploy.check hand-sweep  # 5. one finger at a time    [robot]
    python -m ego2g1.deploy.check hand-jog --hand right
                                               # 5b. interactively close a hand
                                               #     around a real object, for
                                               #     BRAINCO_CLOSED_POSE       [robot]
    python -m ego2g1.deploy.replay_dataset --dataset ...        # 6. stored JOINTS  [robot]
    python -m ego2g1.deploy.replay_dataset --dataset ... --from-eef  # 6b. eef->IK  [robot]
    python -m ego2g1.deploy.check replay-actions --dataset ...  # 7. ACTION labels  [robot]
    python -m ego2g1.deploy.check latency     # 8. round trip to the server [no robot]

Rungs 2 and 3 need no hardware and no checkpoint, and between them validate
joint order, the waist==0 assumption, the flange frame, the pelvis frame, the
vec9 encoding, and the IK — most of what can silently be wrong.

Rung 3b is `EgoRelationTrainConfig`-specific (docs/relation_deploy_plan.md
§4.4): it does not touch a dataset or a checkpoint at all, only the MuJoCo
model, and exists to let a human eyeball whether the fixed
`TCP_TO_INWARD_PALM` rotation convention the relational training pipeline
assumes for the human-tracked palm (in the sibling `data_extraction_zh`
repo — never imported here) actually matches this robot's own flange
orientation once a BrainCo hand is mounted on it. That correspondence is an
assumption (plan §1 item 1), not something FK/IK can confirm on their own —
a wrong 90°-family axis label here would make a relational policy's
predicted rotations move the real arm the wrong way.

Rungs 6 and 7 both drive the real arm from a recording with the policy out of
the loop, and they are NOT the same test. 6 streams stored joints straight to
the executor: it never touches an action label and proves the plumbing (order,
sign, units, rates, hands, e-stop). 7 feeds the episode's ACTION-shaped deltas
through the real conversion path — measured-FK anchor, delta composition,
OneEuroSE3, mink IK, JointFilter, clamp — and proves the TRANSFORMS. A frame
or anchor bug leaves 6 perfect and shows up only in 7; run 6 first so 7 is
interpretable.

Rung modules: dds (1), kin (2/3/3b), camera (4/4b), calib_capture (4c),
hands (5/5b), replay_actions (7), latency (8). Everything a rung needs is
importable from this package directly (the old flat `ego2g1.deploy.check`
import path keeps working via the top-level shim).
"""

from .calib_capture import handeye_capture  # noqa: F401
from .camera import _next_pair_index, camera, stereo_capture  # noqa: F401
from .dds import listen  # noqa: F401
from .hands import hand_jog, hand_sweep  # noqa: F401
from .kin import fk, ik, tcp_orientation  # noqa: F401
from .latency import latency  # noqa: F401
from .replay_actions import replay_actions  # noqa: F401

# rung name -> callable, exactly the old monolith's dispatcher table
RUNGS = {
    "listen": listen,
    "fk": fk,
    "ik": ik,
    "tcp-orientation": tcp_orientation,
    "camera": camera,
    "stereo-capture": stereo_capture,
    "handeye-capture": handeye_capture,
    "hand-sweep": hand_sweep,
    "hand-jog": hand_jog,
    "replay-actions": replay_actions,
    "latency": latency,
}
