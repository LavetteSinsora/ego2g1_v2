"""`relative_eef` mode: 30-dim FK proprio state up, (H, 30) anchor-relative
vec9 chunks down, through the measured jitter-fix pipeline
(actions.RelativeEEFChunks). The current ego2g1 checkpoints."""

from __future__ import annotations

from . import base


class RelativeEEFMode(base.ProprioModeBase):
    name = "relative_eef"
    supports_rtc = True
    supports_reset_to_episode = True

    def build_adapter(self, client, args, fps: int):
        from .. import policy_adapter as _policy_adapter
        return _policy_adapter.make_adapter(
            "relative_eef", client, args.prompt, ik_iters=args.ik_iters,
            posture_cost=args.posture_cost,
            collision_min_dist=args.collision_min_dist)


RELATIVE_EEF = RelativeEEFMode()
base.register(RELATIVE_EEF)
