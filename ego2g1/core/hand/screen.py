"""Self-collision / feasibility screen: dynamic MuJoCo replay of retargeted
hand commands (migrated from pico2usable/hand2brainco/screen_replay.py,
keeping only HandSim + blocked_mask + thresholds; no CLI).

The hand is initialized at the first command and then every frame's command
is driven through the position servos with physics integrated between frames
(servos, force limits, distal couplings, and self-collision are all modeled).
Two failure modes surface that kinematic replay cannot see:

  - persistent tracking error: the commanded pose is blocked (finger pressed
    into another finger/palm) or the servo cannot keep up;
  - self-collision: which link pairs touch, how often, how deep.

Contact itself is not automatically bad -- thumb<->index / thumb<->middle pad
contact is the *goal* of the pinch snap rule. Those pairs are counted
separately from everything else; the primary infeasibility signal is large
steady tracking error, and non-pinch contact pairs are the diagnosis.
"""

import numpy as np

try:
    import mujoco
except ImportError:      # COUPLE and blocked_mask stay importable without
    mujoco = None        # mujoco; HandSim raises on construction instead

from .constants import ACTUATOR_NAME, MJCF_PATH, MOTOR_ORDER

ERR_SOFT = 0.10   # rad, ~5.7 deg: noticeable lag
ERR_HARD = 0.25   # rad, ~14 deg: command effectively unreachable
# "blocked" = hard error sustained BLOCK_FRAMES frames while the command is
# nearly static (chase lag during fast sweeps is a vendor servo-gain artifact:
# the thumb metacarpal actuator's frcrange +-0.5 with kv=10 cannot sustain the
# URDF-rated speeds in sim, so raw soft/hard counts overstate infeasibility)
BLOCK_FRAMES = 8          # ~0.2 s at the ~37.6 Hz tracking rate
BLOCK_CMD_STEP = 0.005    # max |dcmd|/frame still considered "static"
COUPLE = {"thumb": 1.0, "index": 1.155, "middle": 1.155, "ring": 1.155, "pinky": 1.155}
PINCH_PAIRS = {("index", "thumb"), ("middle", "thumb")}


class HandSim:
    def __init__(self, hand):
        self.hand = hand
        if mujoco is None:
            raise ImportError("HandSim needs mujoco (uv sync installs it)")
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH[hand]))
        self.data = mujoco.MjData(self.model)
        joint_to_motor = {v: k for k, v in ACTUATOR_NAME[hand].items()}
        # actuator index + driven-joint qpos address per MOTOR_ORDER entry
        self.act_idx = np.empty(6, dtype=int)
        self.qpos_adr = np.empty(6, dtype=int)
        self.ctrl_max = np.empty(6)
        for i in range(self.model.nu):
            jid = self.model.actuator(i).trnid[0]
            motor = joint_to_motor[self.model.joint(jid).name]
            m = MOTOR_ORDER.index(motor)
            self.act_idx[m] = i
            self.qpos_adr[m] = self.model.joint(jid).qposadr[0]
            self.ctrl_max[m] = self.model.actuator(i).ctrlrange[1]
        self.dist_adr = {
            f: int(self.model.joint(f"{hand}_{f}_distal_joint").qposadr[0]) for f in COUPLE
        }
        # classify geoms by finger via BODY name (left model geoms are unnamed)
        self.geom_finger = []
        for g in range(self.model.ngeom):
            body = self.model.body(self.model.geom_bodyid[g]).name
            self.geom_finger.append(
                next((f for f in COUPLE if f"_{f}_" in body), "palm")
            )

    def ctrl_of(self, cmd6):
        return np.clip(cmd6, 0.0, 1.0) * self.ctrl_max

    def replay(self, cmds, timestamps_ns):
        """cmds (T,6) in MOTOR_ORDER -> per-frame err/contact diagnostics."""
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        init = self.ctrl_of(cmds[0])
        data.qpos[self.qpos_adr] = init
        prox_of = dict(zip(MOTOR_ORDER, self.qpos_adr))
        for f, ratio in COUPLE.items():
            key = "thumb_flex" if f == "thumb" else f
            data.qpos[self.dist_adr[f]] = ratio * data.qpos[prox_of[key]]
        data.ctrl[self.act_idx] = init
        mujoco.mj_forward(model, data)

        dt = np.diff(np.asarray(timestamps_ns, dtype=np.float64)) * 1e-9
        frame_dt = float(np.median(dt)) if len(dt) else 1 / 30
        n_sub = max(1, round(frame_dt / model.opt.timestep))

        T = len(cmds)
        err = np.zeros((T, 6), dtype=np.float32)
        pen_mm = np.zeros(T, dtype=np.float32)
        other_contact = np.zeros(T, dtype=bool)   # any non-pinch pair touching
        pair_counts = {}
        for t in range(T):
            data.ctrl[self.act_idx] = self.ctrl_of(cmds[t])
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            err[t] = np.abs(data.qpos[self.qpos_adr] - data.ctrl[self.act_idx])
            worst = 0.0
            for c in data.contact[: data.ncon]:
                pen = -c.dist
                if pen <= 0:
                    continue
                pair = tuple(sorted((self.geom_finger[c.geom1], self.geom_finger[c.geom2])))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                if pair not in PINCH_PAIRS:
                    other_contact[t] = True
                worst = max(worst, pen)
            pen_mm[t] = worst * 1000
        return err, pen_mm, pair_counts, other_contact


def blocked_mask(err, cmds):
    """(T,) frames where some motor holds ERR_HARD for BLOCK_FRAMES while its
    command is static -- the signature of a physically blocked pose."""
    T = len(err)
    static = np.ones((T, 6), dtype=bool)
    static[1:] = np.abs(np.diff(cmds, axis=0)) < BLOCK_CMD_STEP
    cand = (err > ERR_HARD) & static
    out = np.zeros(T, dtype=bool)
    run = np.zeros(6, dtype=int)
    for t in range(T):
        run = np.where(cand[t], run + 1, 0)
        if (run >= BLOCK_FRAMES).any():
            out[t] = True
    return out
