# envs/ppu-train.sh — TRAINING session setup on the PPU box (16x Alibaba/T-Head
# PPU-ZW810E NPUs behind a JAX CUDA12-compat shim: backend reports "gpu",
# plugin xla_cuda12). MUST be sourced, not executed:   source envs/ppu-train.sh
#
# Why this is a PROFILE and not a `uv sync --group train`: the venv on this box
# is BORROWED — $HOME/openpi/.venv carries the NPU runtime libs
# (site-packages/lib/libacext*.so) and the accelerator-matched jax build. uv
# cannot create or manage that venv, so the lockfile story stops at the
# machine's edge and this file takes over. Nothing is ever written to the
# borrowed venv.
#
# What it does:
#   1. activates the venv that has the accelerator (PPU) packages
#   2. puts THIS checkout on PYTHONPATH so `import ego2g1` resolves here and
#      `import openpi` resolves to third_party/openpi/src, shadowing the
#      borrowed venv's own editable openpi install
#   3. sets the XLA memory fraction used for training
#
# Override the venv location per machine:
#   EGO2G1_VENV=/path/to/.venv source envs/ppu-train.sh

EGO2G1_VENV="${EGO2G1_VENV:-$HOME/openpi/.venv}"
_EGO2G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

if [ ! -f "$EGO2G1_VENV/bin/activate" ]; then
    echo "ERROR: no venv at $EGO2G1_VENV (set EGO2G1_VENV=/path/to/.venv)" >&2
    return 1 2>/dev/null || exit 1
fi

source "$EGO2G1_VENV/bin/activate"
# repo root first (ego2g1), then the openpi submodule's src (openpi)
export PYTHONPATH="$_EGO2G1_ROOT:$_EGO2G1_ROOT/third_party/openpi/src"
# extra packages the shared venv lacks (e.g. gcsfs), installed via
# `pip install --target ~/pypath-extra <pkg>` so the venv itself stays untouched
if [ -d "$HOME/pypath-extra" ]; then
    export PYTHONPATH="$PYTHONPATH:$HOME/pypath-extra"
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
cd "$_EGO2G1_ROOT"

# verification: ego2g1 AND openpi must resolve to THIS checkout; jax must see
# accelerators (backend "gpu" here means the PPUs via the CUDA-compat shim)
EGO2G1_ROOT="$_EGO2G1_ROOT" python - <<'PY'
import os
import pathlib
import jax
import openpi
import ego2g1

root = pathlib.Path(os.environ["EGO2G1_ROOT"]).resolve()
for name, mod in (("ego2g1", ego2g1), ("openpi", openpi)):
    src = pathlib.Path(mod.__file__).resolve()
    print(f"{name:11s}: {src}")
    if not src.is_relative_to(root):
        print(f"WARNING: {name} does NOT resolve to this checkout ({root}) — PYTHONPATH shadowing failed")
print(f"jax backend: {jax.default_backend()}  devices: {jax.device_count()}")
if jax.default_backend() == "cpu":
    print("WARNING: jax is on CPU — training will not use the accelerators")
PY
