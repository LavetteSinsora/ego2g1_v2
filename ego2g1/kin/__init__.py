"""kin: the G1+Revo2 model — FK, IK, self-collision, target smoothing.

g1          G1Backend (MuJoCo kinematic backend), DualArmIK (mink QP),
            ARM_JOINTS order, collision helpers. Loads assets/ via core.paths.
g1_hands    G1 + mounted Revo2 composite (render/sim; mount from b_calib)
placement   Pico-world -> robot-world rigid placement fit (extraction)
filters     causal smoothing: OneEuro / OneEuroSE3 (before IK),
            JointFilter (after IK) — see docs/jitter_root_cause.md

One upward exception, by design: g1_hands.default_mount_rotations lazily reads
the b_calib cache through ego2g1.data (function-local import, no cycle) —
callers who don't have a pipeline cache pass `mount=` explicitly.
"""

from .g1 import ARM_JOINTS, EE_SITES, DualArmIK, G1Backend  # noqa: F401
from .filters import JointFilter, OneEuro, OneEuroSE3       # noqa: F401
