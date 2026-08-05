"""`relation_eef` mode: 56-dim live-perception relation state up, (H, 14)
anchor-relative rotvec chunks + binary grippers down
(docs/relation_deploy_plan.md). Everything relation-specific that used to
live as `if relation_mode:` branches in runner.py is HERE:

  * `build_adapter` — the fail-loud three-artifact config check + the live
    detector/depth/perception wiring (was `runner._build_relation_adapter`);
  * `build_observation` — stereo pair + scalar last-commanded gripper
    fraction (was `runner._observe_relation` AND the probe half of
    `runner._build_probe`, which had duplicated it);
  * `hand_state_from_row` — recovering the scalar open/closed fraction from
    the executed (6,)-motor command (was `runner._hand_frac_from_command`);
  * `record_tick` — draining latch/hand-closed transitions + the per-tick
    `percept` debug snapshot into the recorder (was runner step 4b);
  * `telemetry_extras` / `build_relation_telemetry` — the dashboard's
    overlay/status panel data (was `runner._relation_telemetry`).

Perception imports stay LAZY (inside build_adapter): a joint/relative_eef
deploy must never pay for cv2/DINO/SAM2 just by importing this package —
same discipline as perception/__init__.py.
"""

from __future__ import annotations

import numpy as np

from ...core import relation_layout
from .. import gripper_calib as _gripper_calib
from . import base


class RelationEEFMode(base.DeployMode):
    name = "relation_eef"
    supports_rtc = False           # docs/relation_deploy_plan.md §8
    # the relation dataset schema has no absolute joint/hand start posture
    # to ramp to — see reset_to_episode's refusal message in runner.py
    supports_reset_to_episode = False

    def build_adapter(self, client, args, fps: int):
        """Load task config + calibration, cross-check against the server,
        wire a real detector/depth perception stack into
        `RelationPolicyAdapter`.

        Fails loud, naming exactly what's missing, the moment `relation_eef`
        is selected and any of `--task-config`/`--stereo-calib`/
        `--camera-extrinsic` is absent — same "fail loud before it can
        silently mis-serve" philosophy as `ego2g1/train/stamp.py`'s
        `check_supported` and `perception/task_config.py`'s own
        `validate_against_server_metadata`."""
        from .. import policy_adapter as _policy_adapter

        missing = []
        if not args.task_config:
            missing.append("--task-config")
        if not args.stereo_calib:
            missing.append("--stereo-calib")
        if not args.camera_extrinsic:
            missing.append("--camera-extrinsic")
        if missing:
            raise ValueError(
                f"action_mode=relation_eef needs {', '.join(missing)} (the "
                "connected checkpoint's server control_mode advertises "
                "'relation_eef' — see ego2g1/deploy/client.py's handshake). "
                "relation_eef mode drives LIVE perception (object detection + "
                "stereo depth + hand-relative geometry, "
                "docs/relation_deploy_plan.md §5) every tick and refuses to "
                "guess a task config, stereo calibration, or camera extrinsic "
                "silently: --task-config is a YAML for "
                "ego2g1.deploy.perception.task_config.load_task_config, "
                "--stereo-calib is a .npz for "
                "ego2g1.deploy.perception.depth.StereoCalibration.load, and "
                "--camera-extrinsic is a .npz (key 'T_pelvis_camera') produced "
                "by ego2g1.deploy.perception.touch_calib.solve_camera_extrinsic. "
                "joint/relative_eef modes never need any of these.")

        from ..perception.config import RelationPerceptionConfig
        from ..perception.depth import StereoCalibration, StereoSGBMDepthSource
        from ..perception.detector import GroundingDinoSam2Detector
        from ..perception.relation_perception import RelationPerception
        from ..perception.task_config import (
            load_task_config,
            validate_against_server_metadata,
        )
        import logging
        logger = logging.getLogger(__name__)

        task_config = load_task_config(args.task_config)
        validate_against_server_metadata(task_config, client.metadata["ego2g1"])
        calib = StereoCalibration.load(args.stereo_calib)
        # touch_calib/handeye_calib's shared save convention: the primary
        # T_pelvis_camera key plus provenance (method, solved_iso, residual)
        # -- log which solver produced the loaded extrinsic instead of
        # trusting an anonymous matrix (docs/deploy_refactor_plan.md §6.2).
        extrinsic = np.load(args.camera_extrinsic)
        T_pelvis_camera = extrinsic["T_pelvis_camera"]
        provenance = {k: extrinsic[k].item() for k in
                      ("method", "solved_iso", "rms_residual_m",
                       "translation_std_m")
                      if k in extrinsic.files}
        if provenance:
            logger.info("camera extrinsic %s: %s", args.camera_extrinsic,
                        provenance)
        else:
            logger.warning("camera extrinsic %s carries no provenance keys "
                           "(pre-refactor calibration file?) — re-solve to "
                           "record method/date/residual", args.camera_extrinsic)

        # every tuning knob in one owner, YAML-loadable + recorded (§6.1)
        pcfg = RelationPerceptionConfig.load(args.perception_config)
        detector = GroundingDinoSam2Detector()
        depth_source = StereoSGBMDepthSource(calib, **pcfg.sgbm)
        perception = RelationPerception(
            task_config, detector, depth_source, calib, T_pelvis_camera,
            fps=fps,
            detector_period_ticks=pcfg.detector_period_ticks,
            orientation_period_ticks=pcfg.orientation_period_ticks,
            latch_config=pcfg.latch_config(),
            tracker_kwargs=pcfg.tracker)

        adapter = _policy_adapter.make_adapter(
            "relation_eef", client, args.prompt, ik_iters=args.ik_iters,
            posture_cost=args.posture_cost,
            collision_min_dist=args.collision_min_dist,
            perception=perception)
        # merged into meta.json by runner.main's build_meta call — a replayed
        # session must know the thresholds + calibration that produced it
        adapter.recorder_meta = {
            "perception_config": pcfg.as_dict(),
            "camera_extrinsic_provenance": provenance or None,
            "task_config_path": str(args.task_config),
            "stereo_calib_path": str(args.stereo_calib),
            "camera_extrinsic_path": str(args.camera_extrinsic),
        }
        return adapter

    def build_observation(self, executor, camera, last_hands, prompt) -> dict:
        """`RelationPolicyAdapter`'s `perception=` path needs BOTH camera
        eyes (for `StereoSGBMDepthSource`) plus the last-commanded gripper
        FRACTION per hand, not the proprio modes' single eye + (6,)-vector
        hand command. The single `image` is still sent — the model itself
        takes exactly ONE egocentric image either way (camera.py's own
        docstring); only the PERCEPTION input differs."""
        arm_q = executor.arm_q()
        image = camera.read() if camera is not None else None
        rgb_left = rgb_right = None
        if camera is not None:
            stereo = camera.read_stereo()
            if stereo is not None:
                rgb_left, rgb_right = stereo
        hand_cmds_last = {h: float(last_hands[h]) for h in relation_layout.HANDS}
        return {"arm_q": arm_q,
                # kept for RelationPolicyAdapter.infer's unconditional
                # `request["hand_cmds"]` read (unused by
                # RelativeEEFRotvecChunks.convert — see actions.py — so the
                # scalar shape here is harmless); hand_cmds_last is the one
                # perception.observe() actually consumes.
                "hand_cmds": dict(hand_cmds_last),
                "hand_cmds_last": hand_cmds_last,
                "image": image,
                "rgb_left": rgb_left,
                "rgb_right": rgb_right,
                "prompt": prompt}

    def initial_hand_state(self) -> dict:
        # a scalar fraction per hand (0=open..1=closed, the same convention
        # RelativeEEFRotvecChunks decodes / gripper_calib inverts), NOT the
        # proprio modes' (6,)-motor-vector — there is no "hand block" in the
        # 56-dim relation state, only a rounded grasp bit.
        return {h: 0.0 for h in relation_layout.HANDS}

    def hand_state_from_row(self, row, adapter) -> dict:
        """Recover the scalar open/closed fraction from the just-EXECUTED
        (6,)-motor command — see `gripper_calib.frac_from_command`'s
        docstring for why this inverts the executed row rather than reading
        the converter's internal per-chunk `frac` directly. `adapter` is a
        `RelationPolicyAdapter` (or test double exposing the same
        `closed_pose` property)."""
        from .. import actions as _actions
        return {h: _gripper_calib.frac_from_command(
                    row[_actions.HAND[h]], adapter.closed_pose[h])
                for h in relation_layout.HANDS}

    def discard_probe_state(self, adapter) -> None:
        """The probe's one inference also ran a real DINO/SAM2 pass and
        seeded trackers/latches (`adapter.reset()` owns only the action
        converter and does not touch perception). An arbitrary amount of
        real time can pass between the probe and the operator actually
        pressing Start; discard that seeded state ONCE here rather than
        trusting it into the live rollout. Deliberately not repeated on
        every later Pause -> Start (see RelationPerception.reset()'s
        docstring)."""
        super().discard_probe_state(adapter)
        perception = getattr(adapter, "perception", None)
        if perception is not None:
            perception.reset()

    def telemetry_extras(self, adapter) -> dict | None:
        """Built from the RECORDED `percept` shape (debug_snapshot) via
        `ui.telemetry.relation_panel` — the same builder the replay dashboard
        uses, so live and replay panels cannot diverge. None until the first
        perception tick has run (the page shows "n/a")."""
        perception = getattr(adapter, "perception", None)
        if perception is None or getattr(adapter, "last_percept", None) is None:
            return None
        from ..ui.telemetry import relation_panel
        return relation_panel(perception.debug_snapshot(include_masks=False),
                              perception.recent_events())

    def record_tick(self, adapter, recorder, step, since_t: float) -> float:
        """Drain new latch/hand-closed transitions (RelationPerception's own
        bounded event log, for the dashboard's timeline) into events.jsonl —
        reuses the existing recorder mechanism, no new file format — then
        log the full perception debug state for this tick (masks only on the
        ticks they actually refreshed — debug_snapshot()'s own default): the
        detections/tracked poses/latch geometry the live overlay draws from,
        which an offline reviewer (replay_dashboard.py) needs to show the
        same picture the live page does."""
        perception = getattr(adapter, "perception", None)
        if perception is None:
            return since_t
        for ev in perception.recent_events(since_t=since_t):
            kind = "latch" if ev["kind"] == "latch" else "hand_state"
            # ev's own "kind"/"t" would collide with log()'s positional
            # `kind` / self-stamped "t" — keep the original event time
            # distinctly as "event_t".
            fields = {k: v for k, v in ev.items() if k not in ("kind", "t")}
            recorder.log(kind, event_t=ev["t"], **fields)
            since_t = ev["t"]
        recorder.log("percept", step=step, **perception.debug_snapshot())
        return since_t


RELATION_EEF = RelationEEFMode()
base.register(RELATION_EEF)
