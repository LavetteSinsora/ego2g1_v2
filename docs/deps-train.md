# deps: train + serve

- `openpi` — path dep on the `third_party/openpi` submodule (editable), pinned
  at fork commit `9b73ae37` (= upstream + zero src changes). Brings jax/flax
  at ITS pins — never bump jax independently (0.10 breaks openpi's flax scan).
- Everything else (numpy, lerobot pin, datasets pin) comes from the root set.

PPU box exception: no uv there — `envs/ppu-train.sh` overlays the borrowed
`$HOME/openpi/.venv` and puts this repo + the submodule's `src/` on
PYTHONPATH. `CUDA_VISIBLE_DEVICES` pins a card; the NVIDIA allocator flags
(`PREALLOCATE=false`, `ALLOCATOR=platform`) segfault the NPU driver.
