"""Deployment-shaped closed-loop replay of an EpisodeRecord.

Mirrors wrist_replay.replay.run_mode, but consumes DATASET quantities: the
stored poses are already flange-frame vec9 in the PELVIS frame, so the
wrist->flange alignment B never appears here. World targets come from
`G1Backend.base_pose() @ vec9_to_se3(pose)`.

Chunked loop (chunk length H = cfg.action_horizon): at each chunk boundary
t0 the anchor is

- "measured":      the robot's IK-achieved FK flange pose (all a deployed
                   runtime can know — IK error feeds forward as drift);
- "ground-truth":  the stored pose at t0 (diagnostic — isolates per-chunk
                   IK quality).

Targets for k = 1..H:  T_anchor_world @ (inv(se3(pose[t0])) @ se3(pose[t0+k]))
— exactly the relative-action composition the trained policy will use.

Hands are replayed dynamically through hand.screen.HandSim (position servos
+ physics + self-collision), independently of the anchor mode.
"""

import numpy as np

from ...core import frames
from ...core.rot6d import vec9_to_se3
from ..config import PipelineConfig
from ...core.hand.screen import ERR_HARD, HandSim, blocked_mask
from ...kin.g1 import DualArmIK
from ...kin.g1_hands import G1HandsBackend

ANCHOR_MODES = ("measured", "ground-truth")
G1_IMG_W, G1_IMG_H = 360, 270
HAND_IMG = 220


def pose_err(T_a, T_b):
    """(position err cm, orientation geodesic err deg)."""
    return (float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 100.0),
            frames.rot_geodesic_deg(T_a[:3, :3], T_b[:3, :3]))


# ------------------------------------------------------------------ arms

def _replay_arms_subep(backend, ik, base, pose_seq, hand_seq, mode, H,
                       ik_iters, render, frames_out):
    """One sub-episode, one anchor mode. pose_seq: side -> (Ts,9),
    hand_seq: side -> (Ts,6) commands for the mounted-hand render.
    Returns dict side -> {pos_cm, ori_deg} arrays (Ts,), plus chunk (Ts,)."""
    sides = ("left", "right")
    P = {s: vec9_to_se3(pose_seq[s]) for s in sides}       # (Ts,4,4) pelvis
    W = {s: np.einsum("ij,tjk->tik", base, P[s]) for s in sides}  # world
    Ts = len(P["left"])

    # start from the nominal pose, converge onto the stored t0 pose (the
    # robot is placed at the sub-episode start before the loop begins)
    backend.reset_nominal()
    ik.config.update(backend.data.qpos.copy())
    ik.solve_static(W["left"][0], W["right"][0])

    tgt = {s: {0: W[s][0]} for s in sides}

    def plan(t0):
        for s in sides:
            anchor = (backend.flange_pose(s) if mode == "measured"
                      else W[s][t0])
            inv0 = frames.se3_inv(P[s][t0])
            K = min(H, Ts - 1 - t0)
            for k in range(1, K + 1):
                tgt[s][t0 + k] = anchor @ (inv0 @ P[s][t0 + k])

    plan(0)
    out = {s: {"pos_cm": np.zeros(Ts), "ori_deg": np.zeros(Ts)} for s in sides}
    chunk = np.zeros(Ts, dtype=np.int32)
    for k in range(Ts):
        chunk[k] = max(0, (k - 1) // H)
        ik.solve_tick(tgt["left"][k], tgt["right"][k], iters=ik_iters)
        for s in sides:
            p, o = pose_err(backend.flange_pose(s), W[s][k])
            out[s]["pos_cm"][k] = p
            out[s]["ori_deg"][k] = o
        if render:
            # pose the mounted BrainCo hands at this tick's commands so the
            # full robot render shows arm + fingers together
            for s in sides:
                backend.set_hand_cmds(s, hand_seq[s][k])
            backend.apply()
            frames_out.append(backend.render(G1_IMG_W, G1_IMG_H))
        # deployment order: after executing the chunk's last action,
        # re-observe and plan the next chunk
        if k > 0 and k % H == 0 and k < Ts - 1:
            plan(k)
    return out, chunk


# ------------------------------------------------------------------ hands

class HandRenderer:
    """Kinematic renderer on the same Revo2 model: qpos is copied from the
    dynamic sim after each frame's stepping, then drawn from a fixed camera
    looking at the palm."""

    def __init__(self, sim, size=HAND_IMG):
        import mujoco
        self.mujoco = mujoco
        self.renderer = mujoco.Renderer(sim.model, size, size)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(sim.model, self.cam)
        self.cam.lookat[:] = sim.model.stat.center
        self.cam.distance = 2.2 * sim.model.stat.extent
        # look at the palm from in front of the fingers
        self.cam.azimuth = 90 if sim.hand == "left" else -90
        self.cam.elevation = -25

    def render(self, data):
        self.renderer.update_scene(data, self.cam)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def _hand_frames(sim, cmds, ticks_ns, size=HAND_IMG):
    """Re-run the HandSim stepping loop (same integration as
    HandSim.replay) purely to render the achieved state per tick."""
    import mujoco
    model, data = sim.model, sim.data
    mujoco.mj_resetData(model, data)
    init = sim.ctrl_of(cmds[0])
    data.qpos[sim.qpos_adr] = init
    from ...core.hand.screen import COUPLE, MOTOR_ORDER
    prox_of = dict(zip(MOTOR_ORDER, sim.qpos_adr))
    for f, ratio in COUPLE.items():
        key = "thumb_flex" if f == "thumb" else f
        data.qpos[sim.dist_adr[f]] = ratio * data.qpos[prox_of[key]]
    data.ctrl[sim.act_idx] = init
    mujoco.mj_forward(model, data)

    dt = np.diff(np.asarray(ticks_ns, dtype=np.float64)) * 1e-9
    frame_dt = float(np.median(dt)) if len(dt) else 1 / 30
    n_sub = max(1, round(frame_dt / model.opt.timestep))

    rend = HandRenderer(sim, size)
    out = []
    for t in range(len(cmds)):
        data.ctrl[sim.act_idx] = sim.ctrl_of(cmds[t])
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        out.append(rend.render(data))
    rend.close()
    return out


# ------------------------------------------------------------------ driver

def replay_record(rec, modes=("measured", "ground-truth"), render=True,
                  render_hands=True, ik_iters=None, verbose=True):
    """Run the closed loop over every sub-episode of an EpisodeRecord.

    Returns
    {
      "modes": {mode: {side: {"pos_cm", "ori_deg"} (T,) NaN outside subeps},
                       "chunk": (T,) int},
      "render_mode": mode whose G1 frames were rendered (modes[0]),
      "g1_frames": (T,) list of RGB arrays or None,
      "hands": {side: {"err_deg" (T,), "blocked" (T,)bool, "contact" (T,)bool,
                       "pen_mm" (T,), "pair_counts": {...},
                       "frames": (T,) list of RGB or None}},
      "subep_of": (T,) int,
    }
    T is the record length; ticks outside every sub-episode carry NaN/None.
    """
    if ik_iters is None:
        ik_iters = PipelineConfig().ik_iters
    modes = tuple(modes)
    for m in modes:
        assert m in ANCHOR_MODES, m
    render_mode = modes[0]
    T = rec.T

    backend = G1HandsBackend()        # mount from b_calib (see g1_hands)
    ik = DualArmIK(backend)
    base = backend.base_pose()

    result = {
        "modes": {m: {"chunk": np.full(T, -1, dtype=np.int32),
                      **{s: {"pos_cm": np.full(T, np.nan),
                             "ori_deg": np.full(T, np.nan)}
                         for s in ("left", "right")}} for m in modes},
        "render_mode": render_mode,
        "g1_frames": [None] * T,
        "hands": {},
        "subep_of": rec.subep_of(),
    }

    for i, se in enumerate(rec.subeps):
        sl = slice(se.start, se.end)
        pose_seq = {s: rec.pose[s][sl] for s in ("left", "right")}
        hand_seq = {s: rec.hand[s][sl] for s in ("left", "right")}
        for mode in modes:
            do_render = render and mode == render_mode
            fr = []
            arm, chunk = _replay_arms_subep(
                backend, ik, base, pose_seq, hand_seq, mode, rec.horizon,
                ik_iters, do_render, fr)
            mr = result["modes"][mode]
            mr["chunk"][sl] = chunk
            for s in ("left", "right"):
                mr[s]["pos_cm"][sl] = arm[s]["pos_cm"]
                mr[s]["ori_deg"][sl] = arm[s]["ori_deg"]
            if do_render:
                for j, f in enumerate(fr):
                    result["g1_frames"][se.start + j] = f
            if verbose:
                mp = {s: arm[s]["pos_cm"] for s in ("left", "right")}
                print(f"  [{rec.name} subep {i} | {mode}] "
                      + "  ".join(f"{s}: pos mean {mp[s].mean():.2f} cm "
                                  f"max {mp[s].max():.2f} cm"
                                  for s in ("left", "right")))

    for side in ("left", "right"):
        sim = HandSim(side)
        h = {"err_deg": np.full(T, np.nan), "blocked": np.zeros(T, bool),
             "contact": np.zeros(T, bool), "pen_mm": np.full(T, np.nan),
             "pair_counts": {}, "frames": [None] * T}
        for se in rec.subeps:
            sl = slice(se.start, se.end)
            cmds = rec.hand[side][sl]
            ts = rec.ticks_ns[sl]
            err, pen, pairs, other = sim.replay(cmds, ts)
            h["err_deg"][sl] = np.degrees(err.max(axis=1))
            h["blocked"][sl] = blocked_mask(err, cmds)
            h["contact"][sl] = other
            h["pen_mm"][sl] = pen
            for k, v in pairs.items():
                key = "+".join(k)
                h["pair_counts"][key] = h["pair_counts"].get(key, 0) + int(v)
            if render_hands:
                fr = _hand_frames(sim, cmds, ts)
                for j, f in enumerate(fr):
                    h["frames"][se.start + j] = f
        result["hands"][side] = h
        if verbose:
            ok = ~np.isnan(h["err_deg"])
            print(f"  [{rec.name} hand {side}] track err mean "
                  f"{np.nanmean(h['err_deg']):.1f} deg (hard>"
                  f"{np.degrees(ERR_HARD):.0f} deg on "
                  f"{int((h['err_deg'][ok] > np.degrees(ERR_HARD)).sum())} "
                  f"ticks), blocked {int(h['blocked'].sum())}, "
                  f"non-pinch contact {int(h['contact'].sum())}")

    return result
