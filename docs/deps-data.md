# deps: core + kin + data (the default `uv sync` set)

All in the root `[project.dependencies]`:

- `numpy`, `scipy`, `h5py`, `pillow`, `imageio[ffmpeg]` — core math, hdf5
  episodes, video io (ffmpeg binary ships inside imageio-ffmpeg; s005 shells
  out to it, envs/mac-dev.sh prefixes PATH accordingly)
- `mujoco>=3.10`, `mink`, `qpsolvers[daqp]` — kin: the G1 model + QP IK
- `datasets==3.6.0` — HARD PIN: ≥5 breaks the pinned lerobot API AND writes
  parquet the training env cannot read back (regenerate, don't re-read)
- `lerobot @ git+…@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` — openpi's exact
  rev (was s005_write_lerobot.INSTALL_CMD)
- `pandas` — dashboard + teleop_import
