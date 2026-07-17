"""TeleopLoop against a fake robot: the wiring, the clutch, and the failure modes.

The transforms are proven in test_cancellation and by `check replay`. What is left to
test is the part that only exists at runtime -- that engaging does not jump, that
disengaging actually stops the robot, and that losing the operator does the right thing
rather than the convenient thing.
"""

import threading
import time

import numpy as np
import pytest

from tools.teleop._vendor.eg.deploy.kinematics import Kinematics
from tools.teleop.loop import TeleopConfig, TeleopLoop
from tools.teleop.retarget import SIDES, TeleopRetargeter
from tools.teleop.source import Hdf5Source

EPISODE = "data/put_bottle_in_box_ego/episode_1.hdf5"


class FakeDDS:
    """A robot that tracks perfectly and remembers everything it was told."""

    def __init__(self, arm0):
        self._arm = np.asarray(arm0, dtype=np.float64).copy()
        self.sent, self.hands_sent = [], []
        self.damped = False
        self._lock = threading.Lock()

    def arm_q(self):
        with self._lock:
            return self._arm.copy()

    def lowstate_age(self):
        return 0.001

    def send_arm(self, q, *, waist=None):
        if self.damped:
            return
        with self._lock:
            self._arm = np.asarray(q, dtype=np.float64).copy()
            self.sent.append(self._arm.copy())

    def send_hands(self, cmds):
        if self.damped:
            return
        with self._lock:
            self.hands_sent.append(np.concatenate([cmds[h] for h in SIDES]))

    def damp(self):
        self.damped = True


class StaleSource:
    """A tracker that stops. `age()` grows without bound and `latest()` freezes --
    exactly what a backgrounded browser tab or a slept headset looks like."""

    def __init__(self, sample):
        self._s = sample
        self.frozen_at = time.monotonic()

    def start(self):
        pass

    def latest(self):
        return self._s

    def age(self):
        return time.monotonic() - self.frozen_at

    def close(self):
        pass


def _rig(source, *, control_hz=60.0, **cfg):
    src = source
    B = {s: np.eye(4) for s in SIDES}
    rt = TeleopRetargeter(B, rate_limit=False)
    rt.calibrate({s: Hdf5Source(EPISODE).pose[s][:60] for s in SIDES})

    kin = Kinematics()
    q0 = np.zeros(14)
    dds = FakeDDS(q0)
    loop = TeleopLoop(TeleopConfig(control_hz=control_hz, **cfg),
                      dds=dds, kinematics=kin, source=src, retargeter=rt)
    return loop, dds, kin


@pytest.fixture
def src():
    s = Hdf5Source(EPISODE, loop=True)
    s.start()
    return s


def test_idle_does_not_move_the_robot(src):
    """Before engaging, the operator's hands must have no effect whatsoever."""
    loop, dds, _ = _rig(src)
    loop.start()
    time.sleep(0.3)
    loop.stop()

    assert not loop.engaged
    assert loop.stats["ticks"] == 0
    # The emitter still runs (it must -- stopping the publisher is not a stop), but every
    # frame it sends is the seeded hold pose.
    assert dds.sent, "arm emitter is not running"
    assert np.allclose(np.stack(dds.sent), dds.sent[0], atol=1e-12), \
        "robot moved while IDLE"


def test_engage_does_not_jump(src):
    """The first commanded pose after engaging must be where the arm already is."""
    loop, dds, kin = _rig(src)
    loop.start()
    time.sleep(0.2)
    q_before = dds.arm_q()

    assert loop.engage()
    time.sleep(0.15)
    loop.stop()

    # The delta is identity at the anchor, so the first knot solves back to the same
    # joints. Allow only the clamp's own step, not a lurch.
    q_first = dds.sent[len(dds.sent) // 2]
    assert np.abs(q_first - q_before).max() < 0.05, \
        f"engaging lurched the arm by {np.abs(q_first - q_before).max():.3f} rad"


def test_disengage_freezes_the_robot(src):
    """Space must actually stop the robot, not just stop reading the tracker."""
    loop, dds, _ = _rig(src)
    loop.start()
    time.sleep(0.1)
    loop.engage()
    time.sleep(0.4)
    assert loop.stats["ticks"] > 0, "never moved while engaged"

    loop.disengage()
    time.sleep(0.05)          # let the queued lookahead drain
    n = len(dds.sent)
    time.sleep(0.3)
    after = np.stack(dds.sent[n:])
    loop.stop()

    assert not loop.engaged
    assert np.abs(after - after[0]).max() < 1e-9, "robot kept moving after disengage"


def test_dead_tracker_disengages_rather_than_damping():
    """A tracker that goes silent should drop to IDLE, NOT damp.

    Holding is right for a blink; a prolonged loss means the operator is gone. The safe
    response is a held IDLE that needs a deliberate re-engage -- the robot keeps its pose
    and does not go limp. Damp is reserved for genuine faults, not a glance away.
    """
    base = Hdf5Source(EPISODE)
    loop, dds, _ = _rig(StaleSource(base.at(30)),
                        stream_hold_s=0.05, stream_disengage_s=0.2)
    loop.start()
    time.sleep(0.1)
    loop.engage()
    assert loop.engaged

    deadline = time.monotonic() + 3.0
    while loop.engaged and time.monotonic() < deadline:
        time.sleep(0.05)
    n = len(dds.sent)
    time.sleep(0.2)
    after = np.stack(dds.sent[n:])
    loop.stop()

    assert not loop.engaged, "silent tracker did not disengage to IDLE"
    assert not loop.watchdog.tripped, "a mere tracker loss must not damp"
    assert not dds.damped, "robot was damped on a tracker loss (should only hold)"
    assert np.abs(after - after[0]).max() < 1e-9, "robot kept moving after disengage"
