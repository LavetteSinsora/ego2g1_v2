"""Offline SAM 3 + Orient Anything V2 extraction over recorded episodes.

Separate from `ego2g1/deploy/` on purpose. Deploy is the ONLINE path and is
bound by a 221 ms round on a shared 4090; this is an offline experiment with
no latency budget at all, and the whole point is to spend that freedom on
capabilities the deploy loop cannot have (see `sam3_offline.py`).

What it shares with deploy is deliberate and narrow: the slot mapping, the
visibility gates, the crop geometry and the angle decode all come from
`ego2g1.deploy.perception.v2`. Forking those is how an experiment ends up
measuring something the robot will never run.
"""
