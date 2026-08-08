# envs/ppu-extract.sh — DATA EXTRACTION / perception-v2 session on the PPU box
# (16x Alibaba/T-Head PPU-ZW810E). MUST be sourced, not executed:
#     source envs/ppu-extract.sh
#
# Sibling of envs/ppu-train.sh — read that header first for the machine-profile
# auto-sourcing and for why these are profiles rather than `uv sync` groups.
#
# WHY A SEPARATE VENV FROM TRAINING. Not a preference: pyproject.toml declares
# it under [tool.uv] conflicts —
#     [{group = "train"}, {group = "perception-v2"}]
# `train` pulls the vendored openpi, which pins transformers==4.53.2 exactly;
# SAM 3 (Sam3Model) landed in transformers 5.0.0. The two cannot coexist in one
# environment, so they get one venv each.
#
# NUMPY DIFFERS FROM THE TRAINING VENV, deliberately. There is no JAX here, so
# nothing forces numpy>=2 (PPU jax 0.7.2 does, which is what drags the training
# venv above 2.0). Staying at <2 matches the container image, which was built at
# numpy 1.26 — so the image's compiled packages (sklearn, pandas, scipy) work
# untouched, with none of the "numpy.dtype size changed, Expected 96 got 88"
# shadowing that envs/ppu-train.sh needs.
#
# REBUILD RECIPE (the venv is machine state; it is not in uv.lock):
#     uv venv --python 3.12 --system-site-packages $WS_HOME/venvs/extract
#     uv export --frozen --no-dev --group perception-v2 --no-hashes \
#         --no-emit-package torch --no-emit-package torchvision \
#         --no-emit-package torchcodec > reqs.txt
#     grep -vE "$DROP" reqs.txt > reqs.ppu.txt          # DROP table below
#     uv pip install --no-deps --python $WS_HOME/venvs/extract/bin/python -r reqs.ppu.txt
#
#   `--frozen` is load-bearing. Without it `uv export` RE-RESOLVES against the
#   PPU index, picks torch 2.11.0 (whose build shim finds no artifact for
#   cp312/ubuntu2404/cu129/sdk2.1.0), and dies trying to build it.
#
#   `--no-deps` is load-bearing too: a uv export is already the complete
#   transitive closure, so there is nothing to resolve — which is what stops uv
#   from noticing that something declares `torch` and attempting that sdist.
#
#   DROP='^(-e \./third_party/unitree_deploy|unitree-sdk2py @|cyclonedds==|casadi==|pin==|triton==|nvidia-[a-z0-9_-]*==)'
#
#     unitree_deploy / unitree-sdk2py / cyclonedds / casadi / pin
#         Robot-side, pulled in only because perception-v2 does
#         `include-group = "deploy"`. Safe to drop: ego2g1/deploy/__init__.py,
#         deploy/perception/__init__.py and deploy/perception/v2/__init__.py are
#         all docstring-only, so importing perception.v2.orientation_v2 touches
#         no robot package. cyclonedds would additionally need a CycloneDDS C
#         library build this box does not have.
#     triton / nvidia-*
#         torch's platform tail. The PPU index OVERRIDES these names rather than
#         proxying upstream, so the lockfile's PyPI pins do not exist there. The
#         image's torch ships with its own matching set.
#
# Verified 2026-08-08: numpy 1.26.4, torch 2.9.0 / 16 devices,
# transformers 5.14.1; Sam3Model, ego2g1.deploy.perception.v2.orientation_v2 and
# data_extraction.extract all import.
#
# Override per machine:
#   EGO2G1_VENV=/path/to/.venv EGO2G1_DATA=/path/to/data source envs/ppu-extract.sh

_EGO2G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# Machine profile, by convention one level up from the checkout. Only when
# WS_HOME is unset, so pre-sourcing it or re-sourcing this file is idempotent.
EGO2G1_MACHINE_PROFILE="${EGO2G1_MACHINE_PROFILE:-$(dirname "$_EGO2G1_ROOT")/env/paths.sh}"
if [ -z "${WS_HOME:-}" ] && [ -f "$EGO2G1_MACHINE_PROFILE" ]; then
    source "$EGO2G1_MACHINE_PROFILE"
fi

EGO2G1_VENV="${EGO2G1_VENV:-${WS_HOME:-$HOME}/venvs/extract}"

if [ ! -f "$EGO2G1_VENV/bin/activate" ]; then
    echo "ERROR: no venv at $EGO2G1_VENV (set EGO2G1_VENV=/path/to/.venv)" >&2
    return 1 2>/dev/null || exit 1
fi

source "$EGO2G1_VENV/bin/activate"

# Repo root ONLY. Unlike ppu-train.sh we do NOT put third_party/openpi/src on
# the path: openpi's transformers==4.53.2 pin is the entire reason this venv is
# separate, and nothing in the extraction path imports openpi.
export PYTHONPATH="$_EGO2G1_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Episodes and the recompute cache live on the shared filesystem, not in the
# checkout — see ego2g1/core/paths.py.
export EGO2G1_DATA="${EGO2G1_DATA:-${WS_HOME:-$HOME}/data}"
export EGO2G1_WORK="${EGO2G1_WORK:-${WS_HOME:-$HOME}/cache/work}"

export UV_CONSTRAINT="${UV_CONSTRAINT:-$_EGO2G1_ROOT/envs/ppu-extract-constraints.txt}"
export PIP_CONSTRAINT="$UV_CONSTRAINT"

cd "$_EGO2G1_ROOT"

# verification: ego2g1 must resolve to THIS checkout; torch must see the
# accelerators (it comes from the IMAGE, not this venv); transformers must be
# >=5 or SAM 3 is simply absent; numpy must stay <2 to match the image's
# compiled extensions.
EGO2G1_ROOT="$_EGO2G1_ROOT" python - <<'PY'
import os
import pathlib
import numpy
import torch
import transformers
import ego2g1

root = pathlib.Path(os.environ["EGO2G1_ROOT"]).resolve()
src = pathlib.Path(ego2g1.__file__).resolve()
print(f"ego2g1      : {src}")
if not src.is_relative_to(root):
    print(f"WARNING: ego2g1 does NOT resolve to this checkout ({root}) — PYTHONPATH shadowing failed")
print(f"numpy       : {numpy.__version__}")
print(f"torch       : {torch.__version__}  devices: {torch.cuda.device_count()}")
print(f"transformers: {transformers.__version__}")
if int(transformers.__version__.split(".")[0]) < 5:
    print("WARNING: transformers <5 — SAM 3 (Sam3Model) is not available in this env")
if numpy.__version__.startswith("2"):
    print("WARNING: numpy 2.x here will break the image's numpy-1.26 compiled packages")
if torch.cuda.device_count() == 0:
    print("WARNING: torch sees no devices — is the venv --system-site-packages?")
PY
