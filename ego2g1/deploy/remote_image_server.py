"""Auto-start `image_server.py` on the robot board over SSH.

The head camera is wired directly to the robot board's own USB bus (see
`ego2g1/deploy/camera.py`'s module docstring) -- there is no way to read it
from the deploy PC without SOME process on that board reading the physical
device and forwarding frames over the network. That fact doesn't go away.
What this module removes is the OPERATOR having to manually SSH in and
launch that process every session (the procedure `docs/robot.md` documents
as a manual step, and the one that's bitten calibration capture repeatedly:
the server not already running, or `image_server.py` living at a different
path than expected) -- `ensure_running(...)` does the same SSH-in-and-launch
by itself, so `check camera`/`check stereo-capture`/a real deployment run
can all just... work, the same way they'd need this running either way.

Candidate paths and credentials below are what was actually found/used on
the real robot board during bring-up (2026-08-03, `find / -iname
image_server.py`) -- multiple copies exist; the OTA backup snapshots under
`/unitree/ota/backup/...` are deliberately excluded from the candidate list
(old firmware backups, not meant to be run directly). If the robot's
filesystem changes again, update `DEFAULT_CANDIDATE_PATHS`, not the callers.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

DEFAULT_SSH_USER = "unitree"
DEFAULT_SSH_PASSWORD = "123"   # docs/robot.md's own documented lab default
DEFAULT_CONDA_ENV = "tv"       # docs/robot.md: "conda activate tv"

# In the order to try. The first is the one that's actually under the home
# directory in what looks like a live "service" install; the rest are
# fallbacks found on the same search.
DEFAULT_CANDIDATE_PATHS = (
    "/home/unitree/unitree_eai_environment/service/teleimager/src/teleimager",
    "/unitree/module/unitree_eai/xr_teleoperate/teleop/teleimager/src/teleimager",
    "/unitree/module/unitree_eai/unitree_eai_environment/service/teleimager/src/teleimager",
    "/unitree/module/unitree_eai/unitree_lerobot/unitree_lerobot/eval_robot/image_server",
)


class RemoteImageServerError(RuntimeError):
    """Could not confirm/start image_server.py on the robot board."""


def _connect(host: str, ssh_user: str, ssh_password: str, timeout: float):
    import paramiko  # lazy: joint/relative_eef deploys never pay for this

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=ssh_user, password=ssh_password, timeout=timeout)
    return client


def _run(client, command: str) -> str:
    """Run one command over an existing SSH connection, return stripped stdout.

    `bash -ic '...'` (interactive shell) rather than a plain exec_command --
    conda's own activation hook lives in `.bashrc`, which a non-interactive
    SSH exec channel does NOT source by default; `-i` forces the same
    initialization a human's own interactive login shell gets, which is the
    well-known trick for getting `conda activate` to work over paramiko.
    """
    _stdin, stdout, _stderr = client.exec_command(f"bash -ic {command!r}")
    return stdout.read().decode().strip()


# Matches every known way this thing gets launched: the plain script
# (`python image_server.py`, contains "image_server.py" literally), the
# package-module form `_launch_command` uses for a proper package
# (`python -m teleimager.image_server` -- no ".py" anywhere, but
# "image_server" is still a substring), and the installed console-script
# entry point (`teleimager-server`, a hyphenated name that contains
# NEITHER of the other two substrings at all).
#
# Found the hard way on real hardware (2026-08-03): the ORIGINAL pattern
# here was the literal string "image_server.py" -- which does not match
# "python -m teleimager.image_server" at all. That meant a call to
# ensure_running() that successfully started the server via the module
# form still concluded (wrongly) that the launch had failed or timed out,
# since is_running() couldn't see it -- leaving that process alive and
# bound to its ZMQ port forever, invisible to every later check. Every
# subsequent launch attempt (automated or by hand, any invocation style)
# then failed with `zmq.error.ZMQError: Address already in use`, with no
# indication that the real problem was an ORPHANED earlier process, not
# a config or hardware issue. If you're reading this after hitting that
# error again: `pgrep -fa 'image_server|teleimager-server'` to find it,
# `pkill -f 'image_server|teleimager-server'` to clear it, before retrying.
_PROCESS_PATTERN = "image_server|teleimager-server"


def is_running(client) -> bool:
    """True if any known invocation of the image server is already alive."""
    return bool(_run(client, f"pgrep -f '{_PROCESS_PATTERN}'"))


def _find_first_existing_path(client, candidate_paths) -> str | None:
    for path in candidate_paths:
        if _run(client, f"test -f {path}/image_server.py && echo FOUND") == "FOUND":
            return path
    return None


def _launch_command(client, path: str, conda_env: str) -> str:
    """The correct invocation for `{path}/image_server.py` -- NOT always a
    plain `cd {path} && python image_server.py`.

    Found the hard way (2026-08-03): the xr_teleoperate-style install
    (`.../teleimager/src/teleimager/image_server.py`) uses a RELATIVE import
    (`from .image_client import ...`) -- it is a MODULE inside the
    `teleimager` PACKAGE, not a standalone script, and running it directly
    strips that package context (`ImportError: attempted relative import
    with no known parent package`, straight from
    `/tmp/image_server_autostart.log` on the robot). The fix is to run it AS
    a module (`python -m teleimager.image_server`) from `path`'s PARENT
    directory, which is exactly what makes a relative import resolve.

    Detected via `__init__.py`'s presence in `path` (a real package marker),
    not assumed from directory naming alone -- a candidate without one
    (e.g. `unitree_lerobot`'s own `image_server` folder, a different
    codebase with no evidence of the same issue) keeps the plain
    direct-script invocation.
    """
    is_package = _run(client, f"test -f {path}/__init__.py && echo YES") == "YES"
    if is_package:
        parent, pkg_name = path.rsplit("/", 1)
        run = f"python -m {pkg_name}.image_server"
        cwd = parent
    else:
        run = "python image_server.py"
        cwd = path
    # nohup + disown + redirect: survives this SSH connection closing, just
    # like a human leaving a terminal open achieves, without needing one.
    return (
        f"conda activate {conda_env} && cd {cwd} && "
        f"nohup {run} > /tmp/image_server_autostart.log 2>&1 < /dev/null & disown"
    )


def ensure_running(
    host: str,
    *,
    ssh_user: str = DEFAULT_SSH_USER,
    ssh_password: str = DEFAULT_SSH_PASSWORD,
    candidate_paths=DEFAULT_CANDIDATE_PATHS,
    conda_env: str = DEFAULT_CONDA_ENV,
    connect_timeout: float = 10.0,
    start_timeout: float = 15.0,
) -> bool:
    """Make sure `image_server.py` is running on `host`, starting it over SSH
    if it isn't. Returns True if it was ALREADY running (nothing done), False
    if this call just started it.

    Raises `RemoteImageServerError` if: SSH itself fails (bad host/creds/
    network -- a real problem, not papered over), no candidate path has
    `image_server.py` on this robot (the filesystem changed again -- update
    `DEFAULT_CANDIDATE_PATHS`), or the process never appears in the process
    list within `start_timeout` seconds of being launched (check
    `/tmp/image_server_autostart.log` on the robot board for what it printed).
    """
    try:
        client = _connect(host, ssh_user, ssh_password, connect_timeout)
    except Exception as exc:  # noqa: BLE001 -- surfaced as one clear error type
        raise RemoteImageServerError(
            f"could not SSH to {ssh_user}@{host}: {exc}"
        ) from exc

    try:
        if is_running(client):
            logger.info("image_server already running on %s", host)
            return True

        path = _find_first_existing_path(client, candidate_paths)
        if path is None:
            raise RemoteImageServerError(
                f"image_server.py not found on {host} at any of "
                f"{list(candidate_paths)} -- it may have moved again; "
                "SSH in and run `find / -iname image_server.py` to locate "
                "it, then add the path to DEFAULT_CANDIDATE_PATHS."
            )

        logger.info("starting image_server.py at %s on %s (conda env %r)",
                    path, host, conda_env)
        _run(client, _launch_command(client, path, conda_env))

        t0 = time.monotonic()
        while time.monotonic() - t0 < start_timeout:
            time.sleep(1.0)
            if is_running(client):
                logger.info("image_server started successfully on %s", host)
                return False
        raise RemoteImageServerError(
            f"started image_server.py at {path} on {host}, but it does not "
            f"appear in the process list after {start_timeout}s -- check "
            "/tmp/image_server_autostart.log on the robot board for errors "
            "(e.g. the wrong conda env, a missing camera device, a port "
            "already in use)."
        )
    finally:
        client.close()
