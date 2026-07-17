"""G1 with BrainCo Revo2 hands rigidly mounted on both flanges.

Built at load time with mujoco.MjSpec: each patched Revo2 hand model is
attached to a frame at the flange site, and the G1's stock rubber-hand
visual meshes are hidden (alpha 0 - MjSpec has no geom delete). The mount
rotation between flange and Revo2 base is per side:

- explicit (4x4/3x3 or rpy degrees) if passed;
- else the b_calib stage's `mount_R_{side}` - the rotation that makes the
  mounted hand's palm coincide with the (dataset-mean) human palm whenever
  the flange tracks the stored pose T_wrist·B. Under b_alignment=geometric
  this equals cfg.revo2_mount_rpy_deg; under dataset_mean it is the
  data-consistent estimate of the same physical mount;
- else identity.

The composite model stays a kinematic backend: arm IK writes qpos, hand
commands write the actuated finger joints directly with the distal coupling
applied (same COUPLE ratios as hand.screen). Dynamic feasibility screening
stays in hand.screen.HandSim - this class is for rendering/inspection.
"""

import warnings

import mujoco
import numpy as np

from ..core.hand.constants import ACTUATOR_NAME, MJCF_PATH, MOTOR_ORDER
from ..core.hand.screen import COUPLE
from .g1 import MODEL_XML, G1Backend

HAND_PREFIX = {"left": "lh_", "right": "rh_"}
RUBBER_MESHES = {"left": "left_rubber_hand", "right": "right_rubber_hand"}
# Extra twist about the hand-base axis applied on top of the calibrated
# palm-alignment mount. The palm-frame construction pins the palm NORMAL but
# leaves a visual twist ambiguity; +90 deg confirmed by the user against the
# episode_1 dashboard (2026-07-10: -90 read as 180 off). Render-only:
# never enters labels.
MOUNT_EXTRA_Z_DEG = 90.0


def _as_rot(m):
    m = np.asarray(m, dtype=float)
    if m.shape == (3,):                      # rpy degrees
        from scipy.spatial.transform import Rotation
        return Rotation.from_euler("xyz", m, degrees=True).as_matrix()
    return m[:3, :3]


def default_mount_rotations(cfg=None):
    """Per-side flange->Revo2-base rotation from the b_calib stage output, with
    the MOUNT_EXTRA_Z_DEG visual twist applied. Raises if the b_calib cache is
    missing: silently falling back to identity drops BOTH the calibrated palm
    alignment and the twist, rendering the hands at the bare flange orientation
    - a wrong-but-plausible picture that reads as a broken transform. Fail loud
    and point at the fix instead."""
    from scipy.spatial.transform import Rotation

    from ..data import io
    from ..data.config import PipelineConfig
    cfg = cfg or PipelineConfig()
    Rz = Rotation.from_euler("z", MOUNT_EXTRA_Z_DEG, degrees=True).as_matrix()
    try:
        arrays, _ = io.load_stage(cfg, None, "b_calib")
    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(
            "b_calib cache missing - the mounted-hand render needs mount_R_* "
            "from the b_calib stage. Run it first:\n"
            "  uv run python -m ego2g1.data.run_pipeline --through b_calib\n"
            "(or pass an explicit `mount=` to build_g1_hands_model)."
        ) from e
    return {s: np.asarray(arrays[f"mount_R_{s}"], dtype=float) @ Rz
            for s in ("left", "right")}


def build_g1_hands_spec(mount=None):
    """The composite as an UNCOMPILED MjSpec, so a caller can extend the scene
    before compiling - `human_hand_teleoperate.sim` adds a table and a graspable
    object this way. `build_g1_hands_model` is just this plus compile()."""
    from ..core import frames

    if mount is None:
        mount = default_mount_rotations()
    spec = mujoco.MjSpec.from_file(MODEL_XML)
    for g in spec.geoms:
        if g.meshname in RUBBER_MESHES.values():
            g.rgba[3] = 0.0                 # hide the stock rubber hand
    for side in ("left", "right"):
        hand = mujoco.MjSpec.from_file(str(MJCF_PATH[side]))
        site = next(s for s in spec.sites if s.name == f"{side}_ee_site")
        q_mount = frames.quat_from_mat(_as_rot(mount[side]))
        quat = _quat_mul(site.quat, q_mount)
        frame = site.parent.add_frame(pos=site.pos.copy(), quat=quat)
        frame.attach_body(hand.worldbody.first_body(), HAND_PREFIX[side], "")
    return spec


def build_g1_hands_model(mount=None):
    """-> compiled MjModel of the fixed-base G1 with both Revo2 hands on the
    flange sites. `mount`: side -> rotation (3x3/4x4 or rpy deg), or None
    for default_mount_rotations()."""
    with warnings.catch_warnings():
        # Attach reports an option conflict: the Revo2 asks for timestep 5 ms /
        # RK4, the G1 scene for 2 ms / implicitfast, and the parent wins. Benign
        # for this class (kinematic: qpos + mj_forward, no integration), and the
        # parent's finer step + 100 solver iterations is the better choice for
        # the one caller that DOES integrate (human_hand_teleoperate.sim).
        warnings.simplefilter("ignore", UserWarning)
        return build_g1_hands_spec(mount).compile()


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class G1HandsBackend(G1Backend):
    """G1Backend on the composite model + kinematic hand-command writing."""

    def __init__(self, mount=None):
        super().__init__(model=build_g1_hands_model(mount))
        # per side: qpos addresses of the 6 actuated joints (MOTOR_ORDER) and
        # of each finger's distal joint, plus ctrl ranges for cmd -> rad
        self._hand = {}
        for side, pre in HAND_PREFIX.items():
            joint_to_motor = {v: k for k, v in ACTUATOR_NAME[side].items()}
            adr = np.empty(6, dtype=int)
            rng = np.empty(6)
            for jname, motor in joint_to_motor.items():
                j = self.model.joint(pre + jname)
                m = MOTOR_ORDER.index(motor)
                adr[m] = j.qposadr[0]
                rng[m] = j.range[1]
            dist = {f: int(self.model.joint(
                        f"{pre}{side}_{f}_distal_joint").qposadr[0])
                    for f in COUPLE}
            self._hand[side] = (adr, rng, dist)

    def set_hand_cmds(self, side, cmd6):
        """Write normalized [0,1] MOTOR_ORDER commands as joint angles
        (kinematic; distal joints follow their coupling ratio). Call
        mj_forward via set_qpos/solve_tick afterwards, or use apply()."""
        adr, rng, dist = self._hand[side]
        q = np.clip(np.asarray(cmd6, dtype=float), 0.0, 1.0) * rng
        self.data.qpos[adr] = q
        prox_of = dict(zip(MOTOR_ORDER, adr))
        for f, ratio in COUPLE.items():
            key = "thumb_flex" if f == "thumb" else f
            self.data.qpos[dist[f]] = ratio * self.data.qpos[prox_of[key]]

    def apply(self):
        mujoco.mj_forward(self.model, self.data)
