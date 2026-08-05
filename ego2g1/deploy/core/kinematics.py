"""FK for the anchor + state, IK for the targets — the deploy face of ego2g1.kin.

Adapted from the old deploy's kinematics.py (third_party/openpi/ego2g1/deploy),
with two changes that ARE the point of the refactor:

  1. It builds on `ego2g1.kin` (G1Backend / DualArmIK) — the same model and
     solver that generate the training labels — instead of a vendored sim copy.
  2. The posture (null-space) task is retargeted to the PREVIOUS solution on
     EVERY tick, at cost 0.05. This is xr_teleoperate's `0.1·‖q − q_last‖²`
     smoothness cost transplanted into mink, and it is measured
     (docs/jitter_root_cause.md): worst-joint accel RMS drops from ~25 to
     4-7 rad/s² while EEF error stays ≤ ~1.7 cm max. The old deploy retargeted
     posture only once per chunk; episode_2 (elbow at its limit) shows why
     per-tick matters — the plain QP flails the wrist when a joint saturates,
     and the smoothness cost damps exactly that.

Facts inherited from training, none optional:
  * IK is 14 DOF (7/arm); waist and legs are frozen at EXACTLY 0 rad over a
    fixed pelvis. FK zeros the full qpos and writes only the arms — the real
    robot must be commanded to match (waist -> 0).
  * The flange is `*_ee_site` at the wrist_yaw_link origin, ZERO offset.
  * Poses live in the PELVIS frame. mink solves in the MJCF world, so targets
    are pre-multiplied by the (constant, fixed-base) base pose on the way in.
"""

import numpy as np

from ...core import layout, se3


class Kinematics:
    """Anchor/state FK and target IK against the training G1 model.

    Everything mujoco/mink is imported here, in __init__ — a joint-mode deploy
    never constructs this class and never pays for the imports.
    """

    def __init__(self, *, ik_iters: int = 25, fps: int = 30,
                 posture_cost: float = 0.05, collision_min_dist: float = 0.005,
                 filter_weights=(0.4, 0.3, 0.2, 0.1)):
        import mujoco

        from ...kin import g1 as _g1
        from ...kin.filters import JointFilter

        self._mujoco = mujoco
        self.ik_iters = int(ik_iters)
        self.dt = 1.0 / float(fps)
        # Output smoothing (zh parity). Pass filter_weights=() to disable.
        self._filter = JointFilter(filter_weights) if len(filter_weights) > 1 else None

        # One model, two MjData: the IK writes its solution into its own
        # backend and FK must never see that.
        model = mujoco.MjModel.from_xml_path(_g1.MODEL_XML)
        self._fk = _g1.G1Backend(model=model)
        self._ik_backend = _g1.G1Backend(model=model)
        # posture_cost=0.05: the temporal smoothness knob (see module docstring).
        self.ik = _g1.DualArmIK(self._ik_backend,
                                posture_cost=posture_cost,
                                collision_min_dist=collision_min_dist)

        self.arm_adr = np.concatenate(
            [self._fk.arm_qpos_adr["left"], self._fk.arm_qpos_adr["right"]]
        )
        if len(self.arm_adr) != layout.ARM_DOF:
            raise RuntimeError(
                f"expected {layout.ARM_DOF} arm DOF, model has {len(self.arm_adr)}")

        # Fixed base: constant, so compute once.
        self.base = self._fk.base_pose()
        self.base_inv = se3.se3_inv(self.base)

    # --- FK -----------------------------------------------------------------

    def _write_arm(self, backend, arm_q14) -> None:
        arm_q14 = np.asarray(arm_q14, dtype=np.float64)
        if arm_q14.shape != (layout.ARM_DOF,):
            raise ValueError(f"expected ({layout.ARM_DOF},) arm joints, got {arm_q14.shape}")
        backend.data.qpos[:] = 0.0   # waist AND legs at exactly 0, as in training
        backend.data.qpos[self.arm_adr] = arm_q14
        self._mujoco.mj_forward(backend.model, backend.data)

    def flange_poses(self, arm_q14) -> dict:
        """Measured arm joints -> {hand: (4,4)} flange pose in the PELVIS frame.

        This is the anchor. It is computed from MEASURED joints, never a
        commanded or stored pose — the whole action chunk composes onto it.
        """
        self._write_arm(self._fk, arm_q14)
        return {h: self.base_inv @ self._fk.flange_pose(h) for h in layout.HANDS}

    def state(self, arm_q14, hand_cmds: dict) -> np.ndarray:
        """Build the (30,) observation state the model expects.

        `hand_cmds` is the LAST COMMAND per hand — not encoder feedback.
        Training's state hand-block is the retargeted command (the recordings
        are human video; no robot executed them, so no encoders exist in the
        data). Encoders also stall against a grasped object and never reach the
        command — feeding them would push the model out of distribution exactly
        when it matters.
        """
        poses = self.flange_poses(arm_q14)
        out = np.zeros(layout.DIM, dtype=np.float32)
        for h in layout.HANDS:
            out[layout.EEF[h]] = se3.se3_to_vec9(poses[h])
            out[layout.HAND[h]] = np.clip(np.asarray(hand_cmds[h], dtype=np.float32), 0.0, 1.0)
        return out

    # --- IK -----------------------------------------------------------------

    def ground(self, arm_q14) -> None:
        """Re-seed the IK at the MEASURED configuration (once per chunk).

        That is what closes the loop: within a chunk the solver warm-starts
        from its previous solution; across chunks it must return to where the
        robot actually is or IK error accumulates as drift.
        """
        self._write_arm(self._ik_backend, arm_q14)
        self.ik.config.update(self._ik_backend.data.qpos.copy())
        self.ik.posture.set_target_from_configuration(self.ik.config)

    def solve(self, targets: dict) -> np.ndarray:
        """{hand: (4,4)} pelvis-frame flange targets -> (14,) arm joints.

        Warm-started from the previous solution; the posture task is then
        retargeted to THIS solution so the next tick's QP pays to move away
        from it — the per-tick smoothness cost (docs/jitter_root_cause.md,
        `resolve_smooth_cost=0.05` rows). Output is low-passed by the 4-tap
        JointFilter (zh parity) as the safety net for residual null-space
        wander.
        """
        q = self.ik.solve_tick(
            self.base @ np.asarray(targets["left"], dtype=np.float64),
            self.base @ np.asarray(targets["right"], dtype=np.float64),
            iters=self.ik_iters,
            dt=self.dt,
        )
        # posture <- this tick's solution: the temporal smoothness anchor.
        self.ik.posture.set_target_from_configuration(self.ik.config)
        q_arm = q[self.arm_adr].copy()
        return self._filter.add(q_arm) if self._filter is not None else q_arm

    def tracking_error(self, targets: dict) -> dict:
        """Post-solve flange position error per hand, metres. The honest signal
        for 'the IK could not reach that' — an unreachable target is silently
        approximated by the QP, never reported by it."""
        return {
            h: float(np.linalg.norm(
                self._ik_backend.flange_pose(h)[:3, 3]
                - (self.base @ np.asarray(targets[h], dtype=np.float64))[:3, 3]
            ))
            for h in layout.HANDS
        }

    def reset(self) -> None:
        """Clear the output filter — episode start / after an e-stop, so the
        first solutions are not blended with a stale trajectory."""
        if self._filter is not None:
            self._filter.clear()
