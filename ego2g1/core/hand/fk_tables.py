"""Precompute Revo2 fingertip FK tables from the patched vendor MJCF.

For each 1-DOF finger (index/middle/ring/pinky) the fingertip pad traces a
fixed 1-D arc in the hand base frame as the motor command sweeps 0..1; the
thumb's two motors (flex, rot) span a 2-D surface. This module samples those
into dense tables so retargeting per frame is pure lookup, no iterative IK.

Kinematics only: qpos is set directly (distal joints follow their mechanical
coupling ratio), mj_kinematics is run, no dynamics. Fingertip keypoint = the
centroid of the "touch" pad geom (present on both hands; on the left model the
pads sit on the distal links, on the right on the tip bodies).

Tables are cached to assets/revo2/fk_tables_{hand}.npz; delete to rebuild.
"""

import numpy as np

from .constants import ACTUATOR_NAME, ASSETS_DIR, FINGERS, MJCF_PATH, MJCF_PATH_FREE

N_ARC = 241  # samples per finger arc
N_GRID = 81  # samples per thumb axis (flex x rot grid)

# distal = ratio * proximal (from the URDF mimic tags / patched MJCF equality)
COUPLING = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}


def palm_frame_from_points(wrist, index_k, middle_k, pinky_k, hand):
    """Right-handed palm frame (3,3, columns = axes) from 4 landmarks:
    x = wrist->middle-knuckle (forward), z = back-of-hand normal, y = z cross x.

    The same construction applied to the human joints and to the robot model's
    joint anchors yields corresponding frames by pure geometry -- no fitted
    alignment, hence no calibration bias. `hand` flips the normal so z points
    out of the back of the hand for either handedness.
    """
    x = middle_k - wrist
    x = x / max(np.linalg.norm(x), 1e-9)
    d = index_k - pinky_k if hand == "right" else pinky_k - index_k
    z = np.cross(x, d)
    z = z / max(np.linalg.norm(z), 1e-9)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _touch_pad_centroids(model, data, hand):
    """World position of each finger's touch-pad centroid, keyed by finger."""
    import mujoco

    touch_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "touch")
    out = {}
    for gid in range(model.ngeom):
        if model.geom_matid[gid] != touch_mat:
            continue
        body = model.body(model.geom_bodyid[gid]).name
        finger = next((f for f in COUPLING if f in body), None)
        if finger is None:
            continue
        center = model.geom_aabb[gid, :3]
        pos = data.geom_xpos[gid] + data.geom_xmat[gid].reshape(3, 3) @ center
        # visual + collision copies of the pad share a centroid; keep either
        out[finger] = pos.copy()
    missing = set(COUPLING) - set(out)
    if missing:
        raise RuntimeError(f"{hand}: no touch pad found for {sorted(missing)}")
    return out


class Revo2FK:
    """Kinematic fingertip FK for one Revo2 hand."""

    def __init__(self, hand, free=False):
        import mujoco

        self.hand = hand
        path = (MJCF_PATH_FREE if free else MJCF_PATH)[hand]
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self.free_qadr = (
            int(self.model.joint("floating_base_joint").qposadr[0]) if free else None
        )

        self.base_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"{hand}_base_link"
        )
        # motor -> (proximal joint qpos addr, distal joint qpos addr, range, ratio)
        self.chain = {}
        for finger in COUPLING:
            prox = self.model.joint(f"{hand}_{finger}_proximal_joint")
            dist = self.model.joint(f"{hand}_{finger}_distal_joint")
            self.chain[finger] = (
                int(prox.qposadr[0]),
                int(dist.qposadr[0]),
                float(prox.range[1]),
                COUPLING[finger],
            )
        meta = self.model.joint(f"{hand}_thumb_metacarpal_joint")
        self.thumb_rot_adr = int(meta.qposadr[0])
        self.thumb_rot_range = float(meta.range[1])

    def palm_frame(self):
        """Geometric palm frame (3,3) in the base_link frame (open pose),
        built by the same construction as the human side: wrist = base
        origin, knuckles = index/middle/pinky proximal joint anchors."""
        zero = {m: 0.0 for m in ["thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky"]}
        self.tips_at(zero)
        base_pos = self.data.xpos[self.base_bid]
        base_rot = self.data.xmat[self.base_bid].reshape(3, 3)

        def anchor(joint_name):
            jid = self.model.joint(joint_name).id
            return base_rot.T @ (self.data.xanchor[jid] - base_pos)

        k = {f: anchor(f"{self.hand}_{f}_proximal_joint") for f in ("index", "middle", "pinky")}
        return palm_frame_from_points(np.zeros(3), k["index"], k["middle"], k["pinky"], self.hand)

    def tips_at(self, cmd):
        """Fingertip pad centroids in the base_link frame for a command dict.

        cmd: {"thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky"},
        each normalized [0,1]. Returns {finger: (3,) array}.
        """
        qpos = self.data.qpos
        qpos[:] = 0.0
        if self.free_qadr is not None:
            qpos[self.free_qadr + 3] = 1.0  # identity base quat (w,x,y,z)
        for finger, (prox_adr, dist_adr, rng, ratio) in self.chain.items():
            c = cmd["thumb_flex"] if finger == "thumb" else cmd[finger]
            q = float(np.clip(c, 0.0, 1.0)) * rng
            qpos[prox_adr] = q
            qpos[dist_adr] = ratio * q
        qpos[self.thumb_rot_adr] = float(np.clip(cmd["thumb_rot"], 0.0, 1.0)) * self.thumb_rot_range
        self._mujoco.mj_kinematics(self.model, self.data)

        base_pos = self.data.xpos[self.base_bid]
        base_rot = self.data.xmat[self.base_bid].reshape(3, 3)
        world = _touch_pad_centroids(self.model, self.data, self.hand)
        return {f: base_rot.T @ (p - base_pos) for f, p in world.items()}


def build_tables(hand):
    """Sample finger arcs + thumb grid. Returns dict of arrays."""
    fk = Revo2FK(hand)
    zero = {m: 0.0 for m in ["thumb_flex", "thumb_rot", "index", "middle", "ring", "pinky"]}

    arc_cmds = np.linspace(0.0, 1.0, N_ARC)
    arcs = {f: np.empty((N_ARC, 3)) for f in FINGERS}
    for i, c in enumerate(arc_cmds):
        tips = fk.tips_at({**zero, "index": c, "middle": c, "ring": c, "pinky": c})
        for f in FINGERS:
            arcs[f][i] = tips[f]

    grid_cmds = np.linspace(0.0, 1.0, N_GRID)
    thumb_grid = np.empty((N_GRID, N_GRID, 3))  # [flex, rot]
    for i, cf in enumerate(grid_cmds):
        for j, cr in enumerate(grid_cmds):
            thumb_grid[i, j] = fk.tips_at({**zero, "thumb_flex": cf, "thumb_rot": cr})["thumb"]

    return {
        "arc_cmds": arc_cmds,
        **{f"arc_{f}": arcs[f] for f in FINGERS},
        "grid_cmds": grid_cmds,
        "thumb_grid": thumb_grid,
        "robot_palm": fk.palm_frame(),
    }


def load_tables(hand, rebuild=False):
    cache = ASSETS_DIR / f"fk_tables_{hand}.npz"
    if cache.exists() and not rebuild:
        with np.load(cache) as z:
            return dict(z)
    tables = build_tables(hand)
    np.savez_compressed(cache, **tables)
    return tables
