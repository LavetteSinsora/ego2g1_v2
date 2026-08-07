# envs/ppu-train.sh — TRAINING session setup on the PPU box (16x Alibaba/T-Head
# PPU-ZW810E NPUs behind a JAX CUDA12-compat shim: backend reports "gpu",
# plugin xla_cuda12). MUST be sourced, not executed:   source envs/ppu-train.sh
#
# MACHINE PROFILE — auto-sourced. Machine state (WS_HOME, the HOME redirect,
# cache locations, no_proxy, UV_DEFAULT_INDEX) is not repo business: it
# describes the pod and the account, not this project, so it lives OUTSIDE the
# checkout. But since the checkout sits inside the workspace, this profile can
# find it by convention at ../env/paths.sh and source it when WS_HOME is unset.
# Guarded by file existence, so it is a silent no-op on the Mac and robot PC.
# Override with EGO2G1_MACHINE_PROFILE=/path/to/paths.sh, or pre-source it
# yourself and this step is skipped.
#
# Why this is a PROFILE and not a `uv sync --group train` — two facts that
# cannot go in uv.lock without breaking the Mac and the robot PC:
#   1. The PPU package index publishes torch/torchvision as SDISTS ONLY (no
#      wheels). The venv therefore cannot be self-contained: it is created with
#      `uv venv --system-site-packages` and inherits the container image's
#      prebuilt PPU torch. Anything that makes uv resolve torch itself triggers
#      a source build that fails under build isolation — install such packages
#      with --no-deps and supply their dependency list by hand (this is how
#      lerobot goes in).
#   2. openpi's own pyproject pins jax[cuda12] from PyPI, which is the NVIDIA
#      build and falls back to CPU here. openpi is installed --no-deps and JAX
#      comes from the PPU index instead.
#
# Rebuild kit if the venv ever breaks:
#     envs/ppu-train-constraints.txt   (policy: what must never change)
#     $WS_HOME/env/ppu-train.freeze.txt (record: what was installed)
#
# NOTE ON VIDEO DECODING: torchcodec is deliberately NOT installed (it cannot
# load its native library on PPU). Training configs must therefore set
# video_backend="pyav" — see ego2g1/train/config.py.
#
# Override per machine:
#   EGO2G1_VENV=/path/to/.venv EGO2G1_DATA=/path/to/data source envs/ppu-train.sh

_EGO2G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# Machine profile, by convention one level up from the checkout. Only when
# WS_HOME is unset, so pre-sourcing it or re-sourcing this file is idempotent.
EGO2G1_MACHINE_PROFILE="${EGO2G1_MACHINE_PROFILE:-$(dirname "$_EGO2G1_ROOT")/env/paths.sh}"
if [ -z "${WS_HOME:-}" ] && [ -f "$EGO2G1_MACHINE_PROFILE" ]; then
    source "$EGO2G1_MACHINE_PROFILE"
fi

EGO2G1_VENV="${EGO2G1_VENV:-${WS_HOME:-$HOME}/venvs/train}"

if [ -z "${WS_HOME:-}" ]; then
    echo "NOTE: no machine profile found at $EGO2G1_MACHINE_PROFILE — using" >&2
    echo "      HOME-relative defaults. On the PPU box that is wrong (HF_HOME," >&2
    echo "      TMPDIR, no_proxy, UV_DEFAULT_INDEX); set EGO2G1_MACHINE_PROFILE." >&2
fi

if [ ! -f "$EGO2G1_VENV/bin/activate" ]; then
    echo "ERROR: no venv at $EGO2G1_VENV (set EGO2G1_VENV=/path/to/.venv)" >&2
    return 1 2>/dev/null || exit 1
fi

source "$EGO2G1_VENV/bin/activate"

# repo root first (ego2g1), then the openpi submodule's src (openpi), so both
# resolve to THIS checkout and shadow anything installed in the venv. PREPEND —
# do not clobber what the machine profile may have set.
export PYTHONPATH="$_EGO2G1_ROOT:$_EGO2G1_ROOT/third_party/openpi/src${PYTHONPATH:+:$PYTHONPATH}"

# Datasets and the recompute cache live on the shared filesystem, NOT in the
# checkout — see ego2g1/core/paths.py. Without these, data_dir() points at an
# empty <repo>/data and work_dir() writes into a pull-only checkout.
export EGO2G1_DATA="${EGO2G1_DATA:-${WS_HOME:-$HOME}/data}"
export EGO2G1_WORK="${EGO2G1_WORK:-${WS_HOME:-$HOME}/cache/work}"

# Guard the proven PPU core (jax pins, numpy window, torch) against accidental
# resolution drift from any install made in a training shell.
export UV_CONSTRAINT="${UV_CONSTRAINT:-$_EGO2G1_ROOT/envs/ppu-train-constraints.txt}"
export PIP_CONSTRAINT="$UV_CONSTRAINT"

# XLA's first compile of train_step takes many minutes, and the PPU additionally
# autotunes operators. Persist both so a restart does not pay that cost again.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${WS_HOME:-$HOME}/cache/xla}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null

# Training value. ppu-serve.sh lowers this AND pins one device for inference.
# Do NOT add XLA_PYTHON_CLIENT_PREALLOCATE=false or ALLOCATOR=platform — both
# segfault this NPU's driver on the first infer call.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

cd "$_EGO2G1_ROOT"

# verification: ego2g1 AND openpi must resolve to THIS checkout; jax must see
# the accelerators (backend "gpu" here means the PPUs via the CUDA-compat
# shim); torch comes from the IMAGE, so check it separately; numpy must stay in
# the window jax requires, because the image's compiled packages were built
# against numpy 1.26 and mismatches surface as "numpy.dtype size changed".
EGO2G1_ROOT="$_EGO2G1_ROOT" python - <<'PY'
import os
import pathlib
import numpy
import jax
import torch
import openpi
import ego2g1

root = pathlib.Path(os.environ["EGO2G1_ROOT"]).resolve()
for name, mod in (("ego2g1", ego2g1), ("openpi", openpi)):
    src = pathlib.Path(mod.__file__).resolve()
    print(f"{name:11s}: {src}")
    if not src.is_relative_to(root):
        print(f"WARNING: {name} does NOT resolve to this checkout ({root}) — PYTHONPATH shadowing failed")
print(f"numpy      : {numpy.__version__}")
print(f"jax backend: {jax.default_backend()}  devices: {jax.device_count()}")
print(f"torch      : {torch.__version__}  devices: {torch.cuda.device_count()}")
if jax.default_backend() == "cpu":
    print("WARNING: jax is on CPU — the PPU plugin did not load; training will not use the accelerators")
if torch.cuda.device_count() == 0:
    print("WARNING: torch sees no devices — the image's PPU torch is not visible (--system-site-packages?)")
PY
