"""`joint` mode: the model space IS the executor space (ZH-style) — absolute
joint chunks pass through validated, never IK'd. The slot a future
joint-space ego2g1 policy drops into."""

from __future__ import annotations

from . import base


class JointMode(base.ProprioModeBase):
    name = "joint"
    supports_rtc = True            # joint rows are already model space
    supports_reset_to_episode = True

    def build_adapter(self, client, args, fps: int):
        from .. import policy_adapter as _policy_adapter
        return _policy_adapter.make_adapter("joint", client, args.prompt)


JOINT = JointMode()
base.register(JOINT)
