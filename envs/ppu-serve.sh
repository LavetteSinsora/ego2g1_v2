# envs/ppu-serve.sh — INFERENCE session setup on the PPU box: pin serving to a
# SINGLE device so it stops occupying all 16 cards. MUST be sourced:
#   source envs/ppu-serve.sh
#
# The "PPU" box is 16 Alibaba/T-Head PPU-ZW810E NPUs (name=PPU-ZW810E), exposed
# to JAX through a CUDA-compatible shim (backend reports "gpu", plugin
# xla_cuda12). Same borrowed-venv story as ppu-train.sh — a profile, not a uv
# group, because uv cannot manage that venv.
#
# Why this exists:
#   ppu-train.sh is the TRAINING profile: XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
#   preallocates ~90% on EVERY visible device, and JAX auto-detects all 16.
#   Serving one pi0.5 replica only needs one card, so the other 15 are wasted.
#   This wrapper hides the rest BEFORE sourcing the training profile, so JAX
#   sees exactly one device.
#
# IMPORTANT — allocator flags: XLA_PYTHON_CLIENT_PREALLOCATE=false and
#   ALLOCATOR=platform are NVIDIA-oriented and have segfaulted this NPU's driver
#   on the first infer call (crash lands on the 2nd websocket connection, after
#   checkpoint load). Do NOT set them here. We keep the SAME allocator as the
#   (working) training profile and only (a) restrict the device and
#   (b) optionally lower the memory fraction.
#
# Usage:
#   source envs/ppu-serve.sh                        # device 0, mem 0.9
#   EGO2G1_PPU=15 source envs/ppu-serve.sh          # device 15
#   EGO2G1_PPU=15 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 source envs/ppu-serve.sh
#
# EGO2G1_PPU_VAR must be the env var this NPU actually honors. Confirmed on this
# box: CUDA_VISIBLE_DEVICES works (=0 -> jax.device_count()==1; unset -> 16);
# PPU_VISIBLE_DEVICES / ALINPU_VISIBLE_DEVICES are ignored. Re-probe with:
#   CUDA_VISIBLE_DEVICES=0 python -c "import jax; print(jax.device_count())"

: "${EGO2G1_PPU:=0}"
: "${EGO2G1_PPU_VAR:=CUDA_VISIBLE_DEVICES}"   # <-- set to whatever the probe proves works

# devices: expose exactly one card to JAX.
export "$EGO2G1_PPU_VAR"="$EGO2G1_PPU"

# memory: keep the training allocator (proven stable on this NPU); just cap the
# fraction. ppu-train.sh reads this with `:-0.9`, so an override here wins. With
# one device visible, 0.9 sits on ONE card instead of all 16 — lower it to share.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

echo "ppu-serve: ${EGO2G1_PPU_VAR}=${EGO2G1_PPU}  MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}  (allocator: default)"

# reuse the training profile for venv + PYTHONPATH + verification print
# (its trailing python prints `devices: N` — expect 1, not 16).
source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/ppu-train.sh"
