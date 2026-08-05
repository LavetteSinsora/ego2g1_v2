"""Phase-5 refactor pin (docs/deploy_refactor_plan.md §9 task 12): every
documented `python -m ego2g1.deploy.<name>` entrypoint still resolves through
the layout-move shims, and the shim modules expose the real module's names
(sys.modules-swap + globals mirror).
"""

import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

ENTRYPOINTS = [
    "runner", "check", "replay_dataset", "replay_diag", "replay_record",
    "replay_mujoco", "replay_dashboard", "replay_relation_openloop",
    "measure_rate", "sniff_lowcmd",
]


@pytest.mark.parametrize("name", ENTRYPOINTS)
def test_entrypoint_help_resolves(name):
    proc = subprocess.run(
        [sys.executable, "-m", f"ego2g1.deploy.{name}", "--help"],
        cwd=REPO, capture_output=True, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode()[-2000:]


def test_shim_exposes_real_module():
    import ego2g1.deploy.core.runner as real
    from ego2g1.deploy import runner as shim
    assert shim is real                      # sys.modules swap took effect
    from ego2g1.deploy.runner import DeployRunner  # noqa: F401
    import ego2g1.deploy.session as sess_shim
    import ego2g1.deploy.core.session as sess_real
    assert sess_shim.ExecutorSession is sess_real.ExecutorSession
