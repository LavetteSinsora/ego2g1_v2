"""Shared leaf helpers for the deploy package.

Deliberately dependency-free (stdlib only; the DDS import is lazy and only
happens when a caller actually initializes DDS) so that diagnostic tools can
import these without dragging in the runner's tyro Args, perception config,
or the vendored robot stack. Before this module existed, four replay tools
imported `precise_wait` FROM runner.py — paying the whole runner import for
a 10-line timing helper — and the DDS bootstrap block was copy-pasted seven
times across check.py/measure_rate.py/sniff_lowcmd.py/executor.py.
"""

from __future__ import annotations

import time


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    """Sleep coarsely, spin the last `slack_time`. Verbatim from
    unitree_deploy.robot_devices.robots_devices_utils.precise_wait (local copy
    so deploy logic imports without the vendored package). time.sleep
    overshoots by milliseconds (worse on macOS); the last ~1 ms is spun so
    ticks land where the 500 Hz interpolator expects them."""
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass


def dds_init(iface: str | None = None, domain: int = 0) -> None:
    """Initialize the unitree DDS channel factory, on `iface` if given, else
    joining the default domain (None deliberately does NOT pass an empty
    string through — the sdk treats the no-arg call differently).

    The sdk's init is a SINGLETON: the first call wins, later calls are
    no-ops. That is why `UnitreeExecutor.connect()` calls this BEFORE the
    vendor's own `ChannelFactoryInitialize(0)` — ours must land first for
    the interface choice to take effect. Callers that want best-effort
    semantics (executor.py) wrap this in their own try/except; the check/
    diagnostic rungs let it raise, since a wrong iface should fail loud
    there."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if iface:
        ChannelFactoryInitialize(domain, iface)
    else:
        ChannelFactoryInitialize(domain)
