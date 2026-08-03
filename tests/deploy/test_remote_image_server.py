"""ensure_running's decision logic (docs/relation_deploy_plan.md's calibration
capture workflow): already-running detection, candidate-path search, launch,
and the start-timeout/not-found error paths -- all against a FAKE SSH client
(no real robot, no real network, no real paramiko connection), since the
actual SSH transport is exactly the part this module cannot control and
should not need real hardware to verify the DECISIONS it makes around.
"""

from unittest import mock

import pytest

from ego2g1.deploy import remote_image_server as ris


class _FakeSSHClient:
    """Records every command run against it; `responses` maps an exact
    command string to what `exec_command`'s stdout should read back."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = dict(responses or {})
        self.commands: list[str] = []
        self.closed = False

    def exec_command(self, command: str):
        self.commands.append(command)
        stdout_text = self.responses.get(command, "")
        stdin = mock.Mock()
        stdout = mock.Mock()
        stdout.read.return_value = stdout_text.encode()
        stderr = mock.Mock()
        stderr.read.return_value = b""
        return stdin, stdout, stderr

    def close(self):
        self.closed = True


def _patch_connect(monkeypatch, fake_client):
    monkeypatch.setattr(ris, "_connect", lambda *a, **kw: fake_client)


class TestAlreadyRunning:
    def test_skips_search_and_launch_when_already_running(self, monkeypatch):
        fake = _FakeSSHClient({"bash -ic 'pgrep -f image_server.py'": "12345"})
        _patch_connect(monkeypatch, fake)

        was_running = ris.ensure_running("robot-host")

        assert was_running is True
        assert fake.closed is True
        # only the is_running check ran -- no path search, no launch command
        assert len(fake.commands) == 1


class TestLaunching:
    def test_finds_first_existing_candidate_and_launches_it(self, monkeypatch):
        path0, path1 = ris.DEFAULT_CANDIDATE_PATHS[0], ris.DEFAULT_CANDIDATE_PATHS[1]
        responses = {
            "bash -ic 'pgrep -f image_server.py'": "",              # not running (checked twice: before + after launch)
            f"bash -ic 'test -f {path0}/image_server.py && echo FOUND'": "",  # first candidate: absent
            f"bash -ic 'test -f {path1}/image_server.py && echo FOUND'": "FOUND",  # second: present
        }
        fake = _FakeSSHClient(responses)
        # after the launch command runs, is_running must report success
        call_count = {"n": 0}
        original_exec = fake.exec_command

        def exec_with_state(command):
            if command == "bash -ic 'pgrep -f image_server.py'":
                call_count["n"] += 1
                if call_count["n"] > 1:  # first call (pre-launch) already consumed
                    stdin, stdout, stderr = original_exec(command)
                    stdout.read.return_value = b"6789"
                    return stdin, stdout, stderr
            return original_exec(command)

        fake.exec_command = exec_with_state
        _patch_connect(monkeypatch, fake)
        monkeypatch.setattr(ris.time, "sleep", lambda _s: None)

        was_running = ris.ensure_running("robot-host", candidate_paths=(path0, path1))

        assert was_running is False
        launch_cmds = [c for c in fake.commands if "nohup python image_server.py" in c]
        assert len(launch_cmds) == 1
        assert path1 in launch_cmds[0]
        assert "disown" in launch_cmds[0]
        assert f"conda activate {ris.DEFAULT_CONDA_ENV}" in launch_cmds[0]


class TestErrors:
    def test_raises_clearly_when_no_candidate_path_exists(self, monkeypatch):
        fake = _FakeSSHClient({"bash -ic 'pgrep -f image_server.py'": ""})
        _patch_connect(monkeypatch, fake)

        with pytest.raises(ris.RemoteImageServerError, match="not found"):
            ris.ensure_running("robot-host", candidate_paths=("/nowhere/at/all",))

    def test_raises_clearly_when_process_never_appears_after_launch(self, monkeypatch):
        path0 = ris.DEFAULT_CANDIDATE_PATHS[0]
        fake = _FakeSSHClient({
            "bash -ic 'pgrep -f image_server.py'": "",  # never comes up, ever
            f"bash -ic 'test -f {path0}/image_server.py && echo FOUND'": "FOUND",
        })
        _patch_connect(monkeypatch, fake)
        monkeypatch.setattr(ris.time, "sleep", lambda _s: None)

        with pytest.raises(ris.RemoteImageServerError, match="does not appear"):
            ris.ensure_running("robot-host", candidate_paths=(path0,), start_timeout=0.01)

    def test_wraps_ssh_connection_failure(self, monkeypatch):
        def _boom(*a, **kw):
            raise OSError("connection refused")

        monkeypatch.setattr(ris, "_connect", _boom)

        with pytest.raises(ris.RemoteImageServerError, match="could not SSH"):
            ris.ensure_running("robot-host")


class TestHeadCameraIntegration:
    def test_head_camera_calls_ensure_running_when_auto_start_enabled(self, monkeypatch):
        from ego2g1.deploy.camera import HeadCamera

        calls = []
        monkeypatch.setattr(
            "ego2g1.deploy.remote_image_server.ensure_running",
            lambda host, **kw: calls.append((host, kw)) or True,
        )

        # Stop right after the auto-start call by making the real ImageClient
        # import fail loudly and distinctly -- proves ensure_running() ran
        # BEFORE that import, without needing a real unitree_deploy/robot.
        import sys
        fake_module = mock.Mock()
        fake_module.ImageClientCameraConfig = mock.Mock(side_effect=RuntimeError("stop-here"))
        monkeypatch.setitem(sys.modules, "unitree_deploy.robot_devices.cameras.configs", fake_module)

        cam = HeadCamera(host="robot-host", auto_start_server=True)
        with pytest.raises(RuntimeError, match="stop-here"):
            cam.connect()

        assert calls == [("robot-host", {})]

    def test_head_camera_skips_auto_start_by_default(self, monkeypatch):
        from ego2g1.deploy.camera import HeadCamera

        called = []
        monkeypatch.setattr(
            "ego2g1.deploy.remote_image_server.ensure_running",
            lambda host, **kw: called.append(host),
        )

        import sys
        fake_module = mock.Mock()
        fake_module.ImageClientCameraConfig = mock.Mock(side_effect=RuntimeError("stop-here"))
        monkeypatch.setitem(sys.modules, "unitree_deploy.robot_devices.cameras.configs", fake_module)

        cam = HeadCamera(host="robot-host")  # auto_start_server defaults False
        with pytest.raises(RuntimeError, match="stop-here"):
            cam.connect()

        assert called == []  # ensure_running must never have been called
