"""`umi_eef` mode: two wrist cameras + a live state history up, (H, 7)
anchor-relative rotvec chunks + a CONTINUOUS gripper down.

Serves `UmiTrainConfig` checkpoints (ego2g1/train/umi_transforms.py). One arm
acts; the other holds its pose and contributes only its camera, which is this
setup's stand-in for the head camera it does not have.

THE ANCHOR COMPOSITION IS DONE SERVER-SIDE, ON PURPOSE. This mode sends
ABSOLUTE poses -- `observation/pose_history` as (n_lags, 9) vec9 in the pelvis
frame -- and lets the server's own `UmiStateHistory` compute
`inv(T_anchor) @ T_lag` for every lag. That transform is literally the same
code that built the training labels, so the deploy-side geometry cannot drift
from the training-side geometry the way a re-implementation would. The only
thing this file computes about the history is WHICH SAMPLES, never what they
mean. (The inverse direction -- decoding the returned chunk -- has no such
option: the model emits deltas and somebody has to compose them onto the
anchor. That is `UmiEEFChunks._delta` below, and it is the exact inverse of
`UmiRelativeActions`.)

The anchor is the MEASURED flange pose at the observation tick, which is
exactly what training used (row 0 of the pose history is the anchor, and
`UmiRelativeActions` reads `poses[0]`). Deployment inherits the existing
capture-time anchoring: `build_observation` snapshots `arm_q` alongside the
camera read, `infer` uses that same snapshot, and the async strategies
compensate for inference latency by SKIPPING the elapsed slots rather than
re-anchoring (strategies.py). Nothing here changes that.

The idle arm is pinned twice over: its IK target is the latched hold pose, and
`_post_solve` overwrites its 7 joints with the latched joint values exactly. A
redundant arm has a null space, and "the context camera does not move" is not a
nice-to-have -- the server gives `base_0_rgb` random crop/rotate augmentation
precisely because that view's geometry is assumed independent of the action
labels (`UmiInputs`). An idle arm that wanders makes that assumption false.
"""

from __future__ import annotations

import logging

import numpy as np

from ...core import rotvec, se3, umi_layout
from .. import umi_history as _umi_history
from . import base, eef

logger = logging.getLogger(__name__)


class UmiEEFChunks(eef.EEFChunksBase):
    """(H, 7) anchor-relative rotvec chunks -> (H, 26) joint rows.

    Deploy-side inverse of `ego2g1.train.umi_transforms.UmiRelativeActions`:

        acting arm   delta_T = core.rotvec.vec6_to_se3(row[EEF6])  # [t(3), rotvec(3)]
                     target  = anchor[acting] @ delta_T
                 exactly inverting
                     delta_k = inv(T_anchor) @ T_target_k
                     row[k]  = [delta_k.t, mat_to_rotvec(delta_k.R)]

        idle arm     target = the LATCHED hold pose, expressed as a delta off
                     this tick's anchor so the shared pipeline can smooth and
                     solve it uniformly, then pinned exactly in `_post_solve`.

        gripper      ONE continuous dim, in RADIANS of Dex1 gear rotation --
                     the same quantity the training data stores and the model
                     emits natively. Executed VERBATIM: no rescaling, no
                     open/closed fraction, no `closed_pose` expansion. It lands
                     in slot 0 of the acting hand's block, which
                     `UnitreeExecutor._wire_row` maps to `kRightGripper`.
                     `PerSlotQuantizeActionsInverse` un-normalizes it back to
                     data units server-side, so what arrives here is already in
                     the units the robot wants.
    """

    mode = "umi_eef"
    chunk_dim = umi_layout.ACTION_DIM
    hands = umi_layout.HANDS          # both: the IK solves the pair together

    def __init__(self, kin=None, *, acting=umi_layout.ACTING_HAND,
                 idle_hold: str = "latch", **kwargs):
        super().__init__(kin, **kwargs)
        if idle_hold not in ("latch", "follow"):
            raise ValueError(f"idle_hold must be latch|follow, got {idle_hold!r}")
        # "latch"  freeze at the pose measured when the rollout starts. The
        #          context camera then CANNOT drift, which is what the server's
        #          base_0_rgb augmentation assumes. Use this for evaluation.
        # "follow" re-read the idle arm every chunk and command it back to
        #          where it currently is. It offers only the resistance needed
        #          to close its own tracking error, so you can push it into
        #          place by hand and it stays — the positioning workflow. It
        #          does NOT hold the view still to the same standard.
        self.idle_hold = idle_hold
        self.acting = acting
        self.idle = umi_layout.IDLE_HAND if acting == umi_layout.ACTING_HAND \
            else umi_layout.ACTING_HAND
        self._measured_idle_q: np.ndarray | None = None
        # Latched at the first convert() after a reset, from the MEASURED
        # joints — i.e. wherever the operator physically left the arm before
        # pressing Start. See `latch_idle`.
        self._idle_hold_q: np.ndarray | None = None
        self._idle_hold_T: np.ndarray | None = None
        # last commanded gripper, per hand; the history's gripper column is the
        # COMMAND stream (see UnitreeExecutor.ee_q's docstring)
        self.last_grip: dict[str, float] = {}

    # --- the idle arm --------------------------------------------------------

    def latch_idle(self, arm_q14, idle_grip: float = 0.0) -> None:
        """Freeze the idle arm at its currently MEASURED configuration.

        Called on the first tick after a reset, so the sequence an operator
        actually performs -- move the arm by hand, press Start -- lands the
        hold pose where they left it. Both the joint vector and its flange pose
        are kept: the pose drives the IK target (so collision avoidance sees a
        consistent arm), the joints are what `_post_solve` pins.

        `idle_grip` is the idle gripper's MEASURED value, held for the whole
        rollout. It has to be measured rather than defaulted: the policy never
        commands this gripper, and a Dex1 command of 0.0 rad is outside its
        travel (the data spans 1.20 .. 5.40) -- it would drive the idle gripper
        into a hard stop on the first tick.
        """
        from ...core import layout

        arm_q14 = np.asarray(arm_q14, dtype=np.float64)
        self._idle_hold_q = arm_q14[layout.ARM_SLICE[self.idle]].copy()
        self._idle_hold_T = self.kin.flange_poses(arm_q14)[self.idle].copy()
        self.last_grip[self.idle] = float(idle_grip)
        logger.info("umi_eef: idle arm (%s) latched at q=%s, gripper=%.4f",
                    self.idle, np.round(self._idle_hold_q, 4).tolist(),
                    float(idle_grip))

    @property
    def idle_latched(self) -> bool:
        """`follow` never needs a latch — it re-reads every chunk."""
        return self.idle_hold == "follow" or self._idle_hold_q is not None

    # --- the shared pipeline's hooks -----------------------------------------

    def convert(self, actions, arm_q14, hand_cmds):
        """Stash the measured joints, then run the shared pipeline.

        `follow` mode pins the idle arm to where it is RIGHT NOW rather than to
        the latch, and the shared base does not pass the measurement down to
        `_post_solve`.
        """
        from ...core import layout

        self._measured_idle_q = np.asarray(
            arm_q14, dtype=np.float64)[layout.ARM_SLICE[self.idle]].copy()
        return super().convert(actions, arm_q14, hand_cmds)

    def _delta(self, row, hand):
        if hand == self.acting:
            return rotvec.vec6_to_se3(row[umi_layout.EEF6])
        if self.idle_hold == "follow":
            # identity: target = anchor = the idle arm's MEASURED pose this
            # chunk, so pushing it by hand simply moves where it holds
            return np.eye(4)
        if self._idle_hold_T is None:
            raise RuntimeError(
                "umi_eef: the idle arm was never latched — call latch_idle() "
                "with the measured joints before converting a chunk")
        # Absolute hold pose, re-expressed as a delta off THIS tick's anchor so
        # the base's `target = anchor @ delta` composition reproduces it.
        return se3.se3_inv(self.last_anchor[hand]) @ self._idle_hold_T

    def _hand_block(self, row, hand):
        from ...core import layout

        block = np.zeros(layout.HAND_DIM, dtype=np.float64)
        if hand == self.acting:
            # verbatim: the model's own units, straight to the gripper
            block[0] = float(row[umi_layout.GRIP][0])
        elif hand in self.last_grip:
            # hold whatever the idle gripper was last commanded
            block[0] = float(self.last_grip[hand])
        self.last_grip[hand] = float(block[0])
        return block

    def _post_solve(self, q_arm):
        """Pin the idle arm's joints EXACTLY (see the module docstring).

        `latch` pins to the rollout-start pose; `follow` pins to this chunk's
        measurement. Either way the pin is exact, so the IK's null space can
        never wander that arm — the difference is only WHICH pose it holds.
        """
        from ...core import layout

        target = (self._measured_idle_q if self.idle_hold == "follow"
                  else self._idle_hold_q)
        if target is None:
            return q_arm
        out = np.asarray(q_arm, dtype=np.float64).copy()
        out[layout.ARM_SLICE[self.idle]] = target
        return out

    def _row_ok(self, row):
        from .. import actions as _actions
        return _actions.sanity_check_umi_action(row)

    def reset(self) -> None:
        """Episode start / re-arm: drop the causal filter state AND the latch,
        so the next tick re-reads where the operator has left the idle arm."""
        super().reset()
        self._idle_hold_q = None
        self._idle_hold_T = None
        self.last_grip = {}


class UmiPolicyAdapter:
    """`umi_eef` mode: `UmiTrainConfig` checkpoints.

    Up::

        observation/image_wrist            acting arm's wrist camera
        observation/image_context          idle arm's camera (workspace view)
        observation/pose_history      (n_lags, 9)  ABSOLUTE vec9, most recent first
        observation/gripper_history   (n_lags, 1)  radians, same lag grid
        observation/pose_history_is_pad (n_lags,)  which lags do not exist yet
        prompt                             str

    Down: (H, 7) -> `UmiEEFChunks` (OneEuroSE3 -> DualArmIK posture-tracks-last
    @ 0.05 -> JointFilter) -> (H, 26).

    RTC is NOT supported (the server refuses a prefix for these checkpoints
    too, `serve/policy.py`): the reanchor-prefix math for rotvec deltas under
    per-slot-quantized actions is a separate design, the same gap the
    relational config has.
    """

    mode = "umi_eef"

    def __init__(self, client, prompt: str = "", *, converter=None, kin=None,
                 ik_iters: int = 25, posture_cost: float = 0.05,
                 collision_min_dist: float = 0.005,
                 lag_ticks: tuple[int, ...] | None = None,
                 acting: str | None = None, idle_hold: str = "latch",
                 history_tol_ticks: float = 0.75):
        self._client = client
        self.prompt = prompt
        self.action_horizon = int(client.action_horizon)
        self.fps = int(client.fps)
        cfg = getattr(client, "metadata", {}).get("ego2g1", {})
        # The lag grid is the CHECKPOINT's, read off the handshake — never a
        # local default. A deploy-side grid that disagreed with the trained one
        # would feed correctly-shaped, wrongly-spaced history and the policy
        # would just be quietly worse.
        self.lag_ticks = tuple(lag_ticks if lag_ticks is not None
                               else cfg.get("lag_ticks", umi_layout.default_lag_ticks()))
        self.acting = acting or cfg.get("hand", umi_layout.ACTING_HAND)
        self._converter = converter or UmiEEFChunks(
            kin, acting=self.acting, idle_hold=idle_hold, fps=self.fps,
            ik_iters=ik_iters, posture_cost=posture_cost,
            collision_min_dist=collision_min_dist)
        self._kin = self._converter.kin
        self.history = _umi_history.PoseHistoryBuffer(
            self.lag_ticks, self.fps, tol_ticks=history_tol_ticks)
        self.last_history_len = 0
        # commanded / measured acting gripper, and the running worst gap
        self.last_grip_cmd = float("nan")
        self.last_grip_measured = float("nan")
        self.worst_grip_dev = 0.0

    def note_gripper(self, commanded: float, measured: float) -> None:
        """Record this tick's commanded vs measured acting gripper.

        Deliberately does NOT trip the watchdog. A large gap is the NORMAL
        state while holding an object (the command saturates at the closed
        limit, the jaws stop at the block's width), so a threshold here would
        fire on every successful grasp. It is a diagnostic — read it on the
        dashboard and in the recording, where a gap that stays near zero
        through a grasp phase means the gripper closed on nothing.
        """
        self.last_grip_cmd = float(commanded)
        self.last_grip_measured = float(measured)
        self.worst_grip_dev = max(self.worst_grip_dev,
                                  abs(float(commanded) - float(measured)))

    @property
    def last_tracking_error(self) -> float:
        return self._converter.last_tracking_error

    @property
    def converter(self) -> UmiEEFChunks:
        return self._converter

    def observe_tick(self, t: float, arm_q14, gripper: float,
                     idle_gripper: float = 0.0) -> None:
        """Record one control-rate sample and latch the idle arm if needed.

        Called from `UmiEEFMode.build_observation`, i.e. once per RUNNER tick —
        not per inference. The lag grid is spaced in control ticks, so a buffer
        filled at `inference_hz` could not resolve it at all.
        """
        if not self._converter.idle_latched:
            self._converter.latch_idle(arm_q14, idle_gripper)
        pose = self._kin.flange_poses(arm_q14)[self.acting]
        self.history.push(t, se3.se3_to_vec9(pose), gripper)

    def infer(self, request: dict) -> dict:
        arm_q = np.asarray(request["arm_q"], dtype=np.float64)
        if request.get("enable_rtc") and request.get("prev_action_chunk") is not None:
            raise NotImplementedError(
                "RTC is not implemented for umi_eef mode — the reanchor-prefix "
                "math for rotvec deltas under per-slot-quantized actions is a "
                "separate design (the server refuses the prefix too)")

        obs = {k: v for k, v in request.items()
               if k.startswith("observation/")}
        obs["prompt"] = request.get("prompt", self.prompt)
        missing = [k for k in ("observation/image_wrist", "observation/image_context",
                               "observation/pose_history", "observation/gripper_history")
                   if obs.get(k) is None]
        if missing:
            raise ValueError(
                f"umi_eef request is missing {missing}. Both cameras and the pose "
                "history are required every tick — see UmiEEFMode.build_observation.")
        self.last_history_len = int(request.get("history_len", 0))

        out = self._client.infer_obs(obs)
        state = np.asarray(obs["observation/pose_history"], dtype=np.float64)
        return eef.convert_with_diagnostics(self, out, state, arm_q, {})

    def reset(self) -> None:
        """Re-arm: the world moved on across the gap, so the causal filters,
        the idle-arm latch AND the pose history all go. A history spanning a
        pause would read as an enormous velocity at the exact moment the arm is
        stationary (see umi_history's module docstring)."""
        self._converter.reset()
        self.history.clear()
        self.last_history_len = 0
        self.last_grip_cmd = float("nan")
        self.last_grip_measured = float("nan")
        self.worst_grip_dev = 0.0


class UmiEEFMode(base.DeployMode):
    name = "umi_eef"
    supports_rtc = False
    # the UMI dataset has no absolute joint/hand start posture to ramp to
    supports_reset_to_episode = False
    # Dex1: one gripper motor per hand, commanded in radians — NOT Brainco's
    # 6-motor [0,1] block. `UnitreeExecutor` reads this to pick the wire
    # layout and the command limits.
    robot_type = "unitree_g1_dex1"

    def build_adapter(self, client, args, fps: int):
        cfg = client.metadata.get("ego2g1", {})
        for key in ("lag_ticks", "n_lags"):
            if key not in cfg:
                raise ValueError(
                    f"action_mode=umi_eef needs the server to advertise {key!r} "
                    "in its ego2g1 metadata (serve/policy.py's UmiTrainConfig "
                    "branch). The connected checkpoint does not — is it really a "
                    "UmiTrainConfig checkpoint?")
        return UmiPolicyAdapter(
            client, args.prompt, ik_iters=args.ik_iters,
            posture_cost=args.posture_cost,
            collision_min_dist=args.collision_min_dist,
            idle_hold=getattr(args, "idle_hold", "latch"))

    def build_camera(self, args):
        """Both wrist cameras. By default they come off the robot's own
        image_server — the same host and the same client the head camera uses,
        so a normal run needs no new flags."""
        from ..umi_camera import build_camera_pair
        return build_camera_pair(args)

    def build_observation(self, executor, camera, last_hands, prompt,
                          adapter=None) -> dict:
        """Push this tick into the history, then read the lag grid back out.

        Both halves happen HERE because this is the only hook the runner calls
        every control tick (loop step 1). `adapter.infer` runs at
        `inference_hz`, which is far too coarse for a 3-tick lag grid.
        """
        import time as _time

        arm_q = executor.arm_q()
        acting = adapter.acting if adapter is not None else umi_layout.ACTING_HAND
        idle = umi_layout.IDLE_HAND if acting == umi_layout.ACTING_HAND \
            else umi_layout.ACTING_HAND
        # The gripper the history carries is the COMMAND stream, seeded from
        # the encoder only for the very first sample of a rollout — see
        # UnitreeExecutor.ee_q. The idle gripper is read every tick but only
        # consumed at latch time, and it MUST be measured: the policy never
        # commands it, and 0.0 rad is outside a Dex1's travel.
        # Free: arm_q() above already captured it (UnitreeExecutor caches the
        # end-effector half of the same read).
        measured = executor.ee_q()
        grip = last_hands.get(acting)
        if grip is None:
            grip = measured[acting]
        if adapter is not None:
            adapter.observe_tick(_time.monotonic(), arm_q, float(grip),
                                 idle_gripper=float(measured[idle]))
            # Commanded vs measured. The policy is fed the COMMAND (the
            # training signal saturates at 1.20 when closed, so the encoder is
            # out of distribution there — see UnitreeExecutor.ee_q). But the
            # gap between them is the only grasp-success signal this setup has:
            # closed on the block and closed on nothing both read 1.20 to the
            # policy, and only the encoder can tell them apart.
            adapter.note_gripper(float(grip), float(measured[acting]))
            sampled = adapter.history.sample()
        else:
            sampled = {}

        acting_rgb = context_rgb = None
        if camera is not None:
            pair = getattr(camera, "read_pair", None)
            acting_rgb, context_rgb = pair() if pair is not None else (
                camera.read(), camera.read())
        return {"arm_q": arm_q,
                "observation/image_wrist": acting_rgb,
                "observation/image_context": context_rgb,
                "prompt": prompt,
                **sampled}

    def initial_hand_state(self) -> dict:
        """No commanded gripper yet — `build_observation` falls back to the
        measured encoder for the rollout's first history sample, which is the
        only moment the encoder is the honest source."""
        return {}

    def hand_state_from_row(self, row, adapter) -> dict:
        """The gripper value just EXECUTED, recovered from the row rather than
        from the converter's internals: by pop time the async strategies may
        have blended or reindexed which chunk slot this row came from (the same
        argument `gripper_calib.frac_from_command` makes for relation_eef).
        Slot 0 of each hand block is the Dex1 command."""
        from .. import actions as _actions
        from ...core import layout

        return {h: float(np.asarray(row)[_actions.HAND[h]][0])
                for h in layout.HANDS}

    def telemetry_extras(self, adapter) -> dict | None:
        conv = getattr(adapter, "converter", None)
        if conv is None:
            return None
        cmd = float(getattr(adapter, "last_grip_cmd", float("nan")))
        meas = float(getattr(adapter, "last_grip_measured", float("nan")))
        return {
            # discriminator: the page renders one card per policy family off
            # this, and the relation card's JS would throw on our shape
            "kind": "umi",
            "history_len": int(getattr(adapter, "last_history_len", 0)),
            "n_lags": len(getattr(adapter, "lag_ticks", ())),
            "lag_ticks": list(getattr(adapter, "lag_ticks", ())),
            "idle_hand": conv.idle,
            "idle_latched": bool(conv.idle_latched),
            "acting_hand": conv.acting,
            "last_grip": {k: round(v, 4) for k, v in conv.last_grip.items()},
            # what the policy asked for vs where the jaws actually are; a gap
            # is the expected state while holding something (see note_gripper)
            "grip_cmd": None if cmd != cmd else round(cmd, 4),
            "grip_measured": None if meas != meas else round(meas, 4),
            "grip_dev": None if (cmd != cmd or meas != meas) else round(abs(cmd - meas), 4),
            "grip_dev_worst": round(float(getattr(adapter, "worst_grip_dev", 0.0)), 4),
        }

    def record_tick(self, adapter, recorder, step: int, since_t: float) -> float:
        """How much history the policy actually got this tick.

        Worth a per-tick record rather than a summary: a run whose history is
        chronically short is a paced-loop problem (dropped ticks fail the
        buffer's nearest-sample tolerance), and it is invisible in the loss,
        the tracking error, or the video."""
        recorder.log("umi_history", step=step,
                     history_len=int(getattr(adapter, "last_history_len", 0)),
                     n_lags=len(getattr(adapter, "lag_ticks", ())))
        recorder.log("umi_gripper", step=step,
                     commanded=float(getattr(adapter, "last_grip_cmd", float("nan"))),
                     measured=float(getattr(adapter, "last_grip_measured", float("nan"))))
        return since_t


UMI_EEF = UmiEEFMode()
base.register(UMI_EEF)
