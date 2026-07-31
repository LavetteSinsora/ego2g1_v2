"""The tcp-orientation bring-up rung (docs/relation_deploy_plan.md §4.4):
offline, no dataset, no robot — only the MuJoCo model, same cost fk/ik
already pay. Mirrors test_deploy_replay_mujoco.py's `test_check_runs_headless`
convention: call the rung's function directly and assert on captured stdout,
since check.py's other rungs (fk/ik/camera/hand-sweep) have no dedicated
tests of their own today — this is the first, matching that rigor level
rather than building a heavier harness for just the new rung.
"""

import re

from ego2g1.deploy.check import tcp_orientation


def test_tcp_orientation_runs_headless_and_covers_both_hands(capsys):
    tcp_orientation()  # must complete without error, no dataset/robot needed
    out = capsys.readouterr().out

    # both hands present, at least at the neutral/ready config
    assert "left" in out and "right" in out
    assert "ready (NOMINAL_ARM_QPOS" in out

    # the flange rotation matrix rows are printed (rotation-matrix-shaped
    # numeric output) for the ready config
    assert "flange rotation matrix" in out
    assert "flange translation" in out

    # the per-axis "flange local axis -> nearest pelvis axis" restatement is
    # present for all three axes
    for axis in ("+X", "+Y", "+Z"):
        assert f"local {axis}" in out

    # a 3x3-shaped block of floats appears at least once (a real rotation
    # matrix got printed, not just prose)
    triplets = re.findall(r"\[\s*-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\s*\]", out)
    assert len(triplets) >= 3

    # the honesty caveats required by the task are actually in the output,
    # not just in code comments
    assert "NOT a palm/hand-mount frame" in out
    assert "No automatic PASS/FAIL" in out
