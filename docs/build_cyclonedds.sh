#!/usr/bin/env bash
# docs/build_cyclonedds.sh — one-time CycloneDDS C library build, INSIDE the
# repo (.cyclonedds/), for machines where `uv sync --group deploy` (or
# `--group perception`) fails to find/build cyclonedds==0.10.2. See
# docs/deps-deploy.md's "CycloneDDS on Linux" section for WHY this happens
# (no matching manylinux wheel for this box's Python/libc/architecture) and
# what each step below actually does.
#
# Usage (from the ego2g1_v2 repo root, or from anywhere — paths are computed
# from this script's own location):
#   bash docs/build_cyclonedds.sh
#   source envs/4090.sh                        # picks up CYCLONEDDS_HOME
#   uv sync --group train --group deploy       # or --group perception
#
# Safe to re-run — exits early if .cyclonedds/install already has the library.
# Delete .cyclonedds/ to force a from-scratch rebuild.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/.cyclonedds/src"
BUILD="$ROOT/.cyclonedds/build"
INSTALL="$ROOT/.cyclonedds/install"
BRANCH="releases/0.10.x"   # matches the cyclonedds==0.10.2 Python binding pin

echo "building CycloneDDS into $INSTALL ..."

if [ -f "$INSTALL/lib/cmake/CycloneDDS/CycloneDDSConfig.cmake" ]; then
    echo "already built at $INSTALL — nothing to do (delete .cyclonedds/ to force a rebuild)"
    exit 0
fi

command -v cmake >/dev/null 2>&1 || {
    echo "ERROR: cmake not found. Install a C toolchain first, e.g.:" >&2
    echo "  sudo apt-get install -y build-essential cmake   # Debian/Ubuntu" >&2
    exit 1
}
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || {
    echo "ERROR: no C compiler found. Install one first, e.g.:" >&2
    echo "  sudo apt-get install -y build-essential   # Debian/Ubuntu" >&2
    exit 1
}

# 1. Fetch the CycloneDDS C library's source at the release matching the
#    Python binding's pin (cyclonedds==0.10.2, pyproject.toml's deploy
#    group) — this is the real DDS message-bus implementation the robot
#    talks; the Python package is a thin wrapper that needs it compiled
#    and reachable, not a copy of it.
if [ ! -d "$SRC" ]; then
    git clone -b "$BRANCH" https://github.com/eclipse-cyclonedds/cyclonedds "$SRC"
else
    echo "source already present at $SRC, skipping clone"
fi

# 2. Generate the actual build files (Makefiles, on Linux) from the
#    project's CMakeLists.txt. -B/-S keep the build's scratch/object files
#    ($BUILD) out of the source tree ($SRC) — the standard "out-of-source
#    build" cmake pattern. -DCMAKE_INSTALL_PREFIX tells the LATER install
#    step where the finished library should end up; nothing is compiled by
#    this line yet.
cmake -B "$BUILD" -S "$SRC" -DCMAKE_INSTALL_PREFIX="$INSTALL"

# 3. Actually compile the C sources into object files, link them into the
#    shared library (libddsc.so), and copy the result — library, headers,
#    and a CycloneDDSConfig.cmake package file other build systems use to
#    find it — into $INSTALL/{lib,include,lib/cmake/CycloneDDS}. This is
#    the slow step (real C compilation).
cmake --build "$BUILD" --target install

echo
echo "done — $INSTALL now holds the built CycloneDDS library."
echo "next:"
echo "  source envs/4090.sh                       # auto-exports CYCLONEDDS_HOME"
echo "  uv sync --group train --group deploy      # or --group perception"
