# envs/mac-dev.sh — the macOS dev/data machine (extraction, offline IK, sim, tests).
# MUST be sourced, not executed:   source envs/mac-dev.sh
#
# What machine this is and why each line exists:
#   * This is a NORMAL machine: the venv is uv-managed (Python 3.12, `uv sync`
#     creates .venv from the lockfile). The profile installs nothing — it only
#     activates the venv and fixes PATH.
#   * ffmpeg lives INSIDE the venv (.venv/bin/ffmpeg, a symlink to the
#     imageio-ffmpeg binary). Pipeline stages that shell out to `ffmpeg`
#     (LeRobot video encoding) must find that copy first — hence the PATH prefix.
#     The system has no ffmpeg to fall back on.
#   * Interactive MuJoCo viewers need `.venv/bin/mjpython` — on macOS the viewer
#     must own the main-thread GUI loop, which plain python cannot. Offscreen
#     rendering works under plain python with no MUJOCO_GL set.
#   * Big downloads (torch/jax/mujoco wheels) require the user's proxy app to be
#     ON — direct connections drop large wheels even from mirrors. Nothing to
#     export here (the app sets the macOS system proxy); this is a reminder for
#     when `uv sync` stalls.
#   * git ssh to GitHub is blocked on this network — push via https remotes.

_EGO2G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

if [ ! -f "$_EGO2G1_ROOT/.venv/bin/activate" ]; then
    echo "ERROR: no .venv at $_EGO2G1_ROOT — run \`uv sync\` first (proxy app ON)" >&2
    return 1 2>/dev/null || exit 1
fi

source "$_EGO2G1_ROOT/.venv/bin/activate"
# venv bin first so `ffmpeg` resolves to the imageio-ffmpeg copy (see header)
export PATH="$_EGO2G1_ROOT/.venv/bin:$PATH"
cd "$_EGO2G1_ROOT"

echo "mac-dev: venv=$_EGO2G1_ROOT/.venv  ffmpeg=$(command -v ffmpeg || echo MISSING)"
echo "mac-dev: interactive MuJoCo viewers -> use mjpython, not python"

# CycloneDDS C library for building cyclonedds / unitree_sdk2py (deploy group):
# built from source once (docs/deps-deploy.md); the sdist build needs this var.
[ -d "$HOME/cyclonedds/install" ] && export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
