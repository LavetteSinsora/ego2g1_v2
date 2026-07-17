"""Interactive live replay of one (sub)episode in a MuJoCo window.

macOS needs MuJoCo's own launcher (plain python cannot open the window):

    uv run mjpython -m ego2g1.data.dashboard.viewer episode_1 \
        [--source auto|dataset|workdir] [--anchor measured|ground-truth]

Keys: Space pause/resume, R restart, [ / ] slower / faster, G toggle the
ghost overlay (half-transparent robot frozen at the nominal ready pose -
compare where the episode actually starts vs where deployment would wake up).
Blue halo+dot = the stored (expected) flange pose; a red connector appears
when the achieved flange drifts > 5 mm from it. Loops forever; close the
window to exit.
"""

import argparse
import sys
import time

import numpy as np

from ...core import frames
from ...core.rot6d import vec9_to_se3
from ...kin.g1 import DualArmIK
from ...kin.g1_hands import G1HandsBackend
from .reader import open_reader


def make_ghost(backend, mujoco):
    """A frozen MjData at the nominal ready configuration + display options,
    for a half-transparent overlay. Rendering goes through mjv_addGeoms (the
    supported path for mesh geoms - mjv_initGeom only handles primitives);
    the caller recolors the appended geoms."""
    m = backend.model
    saved = backend.data.qpos.copy()
    backend.reset_nominal()
    ghost_data = mujoco.MjData(m)
    ghost_data.qpos[:] = backend.data.qpos
    mujoco.mj_forward(m, ghost_data)
    backend.set_qpos(saved)

    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[:3] = 1                # visual groups only (collision = 3)
    return ghost_data, opt


GHOST_RGBA = np.array([0.35, 0.75, 0.95, 0.3], np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episode")
    ap.add_argument("--source", default="auto",
                    choices=("auto", "dataset", "workdir"))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--anchor", default="measured",
                    choices=("measured", "ground-truth"))
    ap.add_argument("--ik-iters", type=int, default=5)
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer

    reader, _src = open_reader(args.source, args.dataset_root)
    rec = reader.load(args.episode)
    print(f"{rec.name}: {rec.T} ticks, {len(rec.subeps)} sub-episode(s), "
          f"H={rec.horizon}, anchor={args.anchor}")

    backend = G1HandsBackend()
    ik = DualArmIK(backend)

    def set_hands(t):
        for s in sides:
            backend.set_hand_cmds(s, rec.hand[s][t])
        backend.apply()
    base = backend.base_pose()
    sides = ("left", "right")
    # world-frame stored pose per tick per side
    W = {s: np.einsum("ij,tjk->tik", base, vec9_to_se3(rec.pose[s]))
         for s in sides}
    P = {s: vec9_to_se3(rec.pose[s]) for s in sides}

    state = {"paused": False, "restart": False, "dt": 1.0 / rec.fps,
             "ghost": False}

    def key_cb(keycode):
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
        elif keycode in (ord("r"), ord("R")):
            state["restart"] = True
        elif keycode in (ord("g"), ord("G")):
            state["ghost"] = not state["ghost"]
            print(f"  ghost (ready pose) {'ON' if state['ghost'] else 'off'}")
        elif keycode == ord("["):
            state["dt"] = min(state["dt"] * 1.5, 0.5)
        elif keycode == ord("]"):
            state["dt"] = max(state["dt"] / 1.5, 1 / 120.0)

    try:
        viewer = mujoco.viewer.launch_passive(backend.model, backend.data,
                                              key_callback=key_cb,
                                              show_left_ui=False,
                                              show_right_ui=False)
    except RuntimeError as e:
        raise SystemExit(
            f"could not open the viewer ({e}).\nOn macOS run under mjpython:\n"
            f"  uv run mjpython -m ego2g1.data.dashboard.viewer "
            f"{args.episode} --source {args.source}")

    print("Space=pause  R=restart  G=ghost ready-pose overlay  "
          "[=slower  ]=faster; close window to exit")
    ghost_data, ghost_opt = make_ghost(backend, mujoco)
    ghost_pert = mujoco.MjvPerturb()

    def emit_ghost(scn):
        """Append the ghost robot via mjv_addGeoms (correct mesh rendering),
        then recolor the appended geoms half-transparent."""
        n0 = scn.ngeom
        mujoco.mjv_addGeoms(backend.model, ghost_data, ghost_opt,
                            ghost_pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
        for i in range(n0, scn.ngeom):
            scn.geoms[i].rgba[:] = GHOST_RGBA
            scn.geoms[i].segid = -1

    with viewer:
        viewer.cam.lookat[:] = [0.25, 0.0, 1.0]
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 140, -15, 1.6

        while viewer.is_running():
            state["restart"] = False
            for i, se in enumerate(rec.subeps):
                if not viewer.is_running() or state["restart"]:
                    break
                print(f"  sub-episode {i} "
                      f"[ticks {se.start}:{se.end}, real_end={se.real_end}]")
                backend.reset_nominal()
                ik.config.update(backend.data.qpos.copy())
                ik.solve_static(W["left"][se.start], W["right"][se.start])
                set_hands(se.start)
                Ts = se.end - se.start
                tgt = {s: {0: W[s][se.start]} for s in sides}

                def plan(k0):
                    for s in sides:
                        anchor = (backend.flange_pose(s)
                                  if args.anchor == "measured"
                                  else W[s][se.start + k0])
                        inv0 = frames.se3_inv(P[s][se.start + k0])
                        K = min(rec.horizon, Ts - 1 - k0)
                        for k in range(1, K + 1):
                            tgt[s][k0 + k] = anchor @ (
                                inv0 @ P[s][se.start + k0 + k])

                plan(0)
                k = 0

                def draw_overlay(k):
                    scn = viewer.user_scn
                    scn.ngeom = 0
                    if state["ghost"]:
                        emit_ghost(scn)

                    def add_sphere(size, pos, rgba):
                        if scn.ngeom >= scn.maxgeom:
                            return
                        mujoco.mjv_initGeom(
                            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([size, 0, 0], np.float64),
                            np.asarray(pos, np.float64), np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
                        scn.ngeom += 1

                    for s in sides:
                        p_exp = W[s][se.start + k][:3, 3]
                        p_ach = backend.flange_pose(s)[:3, 3]
                        add_sphere(0.035, p_exp, (0.15, 0.4, 0.95, 0.3))
                        add_sphere(0.008, p_exp, (0.1, 0.3, 0.95, 1.0))
                        if (np.linalg.norm(p_ach - p_exp) > 0.005
                                and scn.ngeom < scn.maxgeom):
                            g = scn.geoms[scn.ngeom]
                            mujoco.mjv_initGeom(
                                g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                np.zeros(3), np.eye(3).flatten(),
                                np.array([0.95, 0.15, 0.1, 0.9], np.float32))
                            mujoco.mjv_connector(
                                g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004,
                                p_exp, p_ach)
                            scn.ngeom += 1

                while k < Ts and viewer.is_running() and not state["restart"]:
                    t_wall = time.time()
                    if state["paused"]:
                        draw_overlay(max(k - 1, 0))
                        viewer.sync()
                        time.sleep(0.03)
                        continue
                    ik.solve_tick(tgt["left"][k], tgt["right"][k],
                                  iters=args.ik_iters)
                    set_hands(se.start + k)
                    if k > 0 and k % rec.horizon == 0 and k < Ts - 1:
                        plan(k)

                    draw_overlay(k)
                    viewer.sync()
                    k += 1
                    time.sleep(max(0.0, state["dt"] - (time.time() - t_wall)))
                if viewer.is_running() and not state["restart"]:
                    time.sleep(0.6)     # hold between sub-episodes / loops

    rec.close()


if __name__ == "__main__":
    if sys.platform == "darwin" and "mjpython" not in sys.executable:
        print("NOTE: on macOS the MuJoCo viewer needs mjpython:\n"
              "  uv run mjpython -m ego2g1.data.dashboard.viewer "
              + " ".join(sys.argv[1:]))
    main()
