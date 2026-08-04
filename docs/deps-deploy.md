# deploy dependencies

What `uv sync --group deploy` should end up meaning. Recorded here (not merged
into pyproject.toml by the deploy port — integration owns that file).

## Path dep (vendored)

```
unitree_deploy @ third_party/unitree_deploy        # editable path dep
```

Install today, from the repo root:

```bash
VIRTUAL_ENV=$PWD/../.venv uv pip install -e third_party/unitree_deploy
```

Vendored from the old repo's `unitree-deploy-main/` (Unitree's deploy stack,
BSD-3-Clause, v0.0.5) minus `unitree_deploy.egg-info/` and `__pycache__/`. The
copy already carries the Python-3.12 fix (mutable `np.ndarray`/`list`
dataclass defaults → `field(default_factory=...)` in
`unitree_deploy/robot_devices/arm/configs.py`);
`tests/test_deploy_vendored.py` pins it. The 52 MB of
`robot_devices/assets/g1/` (URDF + meshes) is load-bearing: gravity
compensation (`g1_arm_ik.solve_tau` = `pin.rnea`) reads
`g1_body29_hand14.urdf`.

Its pyproject declares `lerobot==0.4.1`, `pyrealsense2`, `meshcat`,
`matplotlib`, `logging_mp`, `opencv-python`, `tyro`, `draccus`, `mujoco` —
heavier than what the deploy layer actually imports (arm + brainco + image
client + env need: numpy, scipy, torch, draccus, opencv, logging_mp, tyro).
Installing with `--no-deps` and adding the short list below is the leaner
path if `lerobot==0.4.1` fights the repo's own lerobot pin — it did not in
the current `.venv`, which installs it editable with deps unresolved.

## Not on PyPI

```
unitree_sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python
```

DDS transport (`ChannelFactoryInitialize`, LowCmd_/LowState_ IDL, CRC). Pulls
in `cyclonedds` — already in the repo `.venv`. Note the repo memory: this
box pushes/pulls GitHub via https only.

## Binary/conda-grade wheels

```
pin        # pinocchio — gravity-comp rnea in unitree_deploy/robot_devices/arm/g1_arm_ik.py
casadi     # the vendored IK variant imports it at module level
```

Both must go into the uv venv itself (`VIRTUAL_ENV=$PWD/.venv uv pip install
pin casadi`), NOT a conda pip — see memory `unitree-deploy-env-setup.md` for
the failure mode.

## Rest of the deploy group

```
torch              # unitree_deploy's send_action takes tensors (already in .venv)
opencv-python      # recorder mp4 sink, check camera rung
pandas + pyarrow   # replay_dataset reads LeRobot parquet directly
tyro               # every CLI entry point
openpi-client      # websocket PolicyClient (from the openpi fork's packages/openpi-client)
draccus, logging_mp  # unitree_deploy internals
```

`ego2g1.deploy` itself needs mujoco/mink only in `relative_eef` mode
(kinematics.py imports them lazily); a joint-mode robot PC can skip them.

## Machine parameters (env/CLI, documented defaults)

| what | default | where |
|---|---|---|
| DDS network interface | none (join default domain) | `--network-interface` (runner, replay_*, check) |
| robot subnet | 192.168.123.x, deploy box on-subnet | docs/deploy.md |
| head camera host | 192.168.123.164 | `--camera-host` (runner, check camera) |
| policy server | 127.0.0.1:8000 | `--host/--port` |

## CycloneDDS on macOS

`unitree_sdk2py` needs `cyclonedds==0.10.2`, which has no macOS wheel — it
builds against the CycloneDDS C library. This machine has it built from
source at `~/cyclonedds/install` (C lib 0.10.2, matching the binding pin):

```bash
CYCLONEDDS_HOME="$HOME/cyclonedds/install" uv sync --group deploy
```

A fresh Mac needs that one-time build first:

```bash
git clone -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds ~/cyclonedds
cmake -B ~/cyclonedds/build -S ~/cyclonedds -DCMAKE_INSTALL_PREFIX=~/cyclonedds/install
cmake --build ~/cyclonedds/build --target install
```

## CycloneDDS on Linux (no matching wheel for this box)

Linux normally gets a manylinux wheel for `cyclonedds==0.10.2` (the bundled C
library, no source build needed) — that's why only macOS was called out
above. But the wheel is published for specific CPython versions/architectures
only; a box with an unsupported Python minor version, an unusual libc, or an
offline `uv sync` with no cached wheel hits the exact same barrier as macOS
(uv falls back to a source build, which fails without the C library and
`CYCLONEDDS_HOME` present). The fix is identical, just without needing Xcode's
compiler stand-in — install a C toolchain + cmake first if the box doesn't
have one (`sudo apt-get install -y build-essential cmake` on Debian/Ubuntu;
the repo's own `.venv/bin/cmake`, pulled in as a build dep of pinocchio/cmeel,
does NOT substitute for a system C compiler).

**Build it INSIDE the repo, not `$HOME`** — `.cyclonedds/` at the repo root is
already reserved for exactly this (`.gitignore` has carried the entry since
before this section did): a machine meant to be copied/USB-transplanted
wholesale (`envs/4090.sh`'s use case) should have every non-lockfile
dependency living under the checkout, not scattered into `$HOME` where a
directory copy won't follow it.

Copy-paste, from the repo root — or just run `docs/build_cyclonedds.sh`
(same commands, idempotent, checks for cmake/a compiler first, see its own
comments for what each step does):

```bash
bash docs/build_cyclonedds.sh
```

equivalent to, spelled out:

```bash
_ROOT="$(pwd)"   # run from the ego2g1_v2 repo root
git clone -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds "$_ROOT/.cyclonedds/src"
cmake -B "$_ROOT/.cyclonedds/build" -S "$_ROOT/.cyclonedds/src" -DCMAKE_INSTALL_PREFIX="$_ROOT/.cyclonedds/install"
cmake --build "$_ROOT/.cyclonedds/build" --target install
CYCLONEDDS_HOME="$_ROOT/.cyclonedds/install" uv sync --group deploy   # or --group perception
```

What each line actually does:
1. **`git clone`** — downloads CycloneDDS's own C source at the `0.10.x`
   release branch, matching the `cyclonedds==0.10.2` Python binding pinned in
   `pyproject.toml`. CycloneDDS itself is the real DDS message bus the robot
   speaks (`rt/lowcmd`/`rt/lowstate` etc.) — the Python `cyclonedds` package
   is a thin wrapper around it, and needs this actual C library compiled and
   reachable to do anything.
2. **`cmake -B build -S src -DCMAKE_INSTALL_PREFIX=...`** — CMake doesn't
   compile anything by itself; it's a build-system *generator*. It reads
   CycloneDDS's `CMakeLists.txt`, checks what compiler/tools this specific
   machine has, and writes out the actual low-level build files (Makefiles,
   on Linux) into `-B`'s directory. `-S` is the source tree; keeping the
   generated build files in a separate `-B` directory ("out-of-source
   build") is the standard cmake pattern — it keeps the checked-out source
   pristine and makes a clean rebuild just `rm -rf` the build dir. `-D` sets
   a CMake variable — here, `CMAKE_INSTALL_PREFIX` tells the *later* install
   step where the finished library should land; nothing is compiled yet at
   this line.
3. **`cmake --build build --target install`** — this is the step that
   actually invokes the generated Makefiles: compiles every `.c` file into
   object code, links it into the shared library (`libddsc.so`), then
   (because the target is `install`, not the default `all`) copies the
   library, its headers, and a `CycloneDDSConfig.cmake` package-descriptor
   file into `CMAKE_INSTALL_PREFIX`'s `lib/`, `include/`, and
   `lib/cmake/CycloneDDS/` subdirectories. This is the slow step — real C
   compilation, not a download.
4. **`CYCLONEDDS_HOME=... uv sync ...`** — sets the environment variable the
   Python `cyclonedds` package's own build script reads to find the
   already-compiled library/headers/CMake config from step 3, instead of
   trying to fetch or build its own copy. `uv sync` then resolves and
   installs everything else pinned in the requested groups too.

`envs/4090.sh` already exports `CYCLONEDDS_HOME` (and `LD_LIBRARY_PATH`) from
this exact path if it exists, so a re-sync later (`source envs/4090.sh &&
uv sync ...`) needs no flag repeated.

**The one thing repo-locality does NOT fix**: the compiled `cyclonedds`
extension bakes in an rpath to `.cyclonedds/install`'s *absolute* path at
build time (same as the Mac's own note below). Building at
`/home/alice/ego2g1_v2/.cyclonedds/install` and then copying the whole
directory to `/opt/robot/ego2g1_v2` breaks that rpath — the extension will
`dlopen` fail even though the `.so` files are sitting right there, just under
a different absolute path than the one baked in at build time. Two ways
around it, in order of preference:
1. **Build ON the final machine, at the final path** — copy the bare source
   (no `.venv`, no `.cyclonedds` — both already gitignored) over first, land
   it at the path it will actually run from, then run the build + `uv sync`
   there. This is the same advice as building the whole venv on-target rather
   than copying one built elsewhere (`envs/4090.sh`'s header).
2. If you must move a fully-built tree afterward, `envs/4090.sh` also puts
   `.cyclonedds/install/lib` on `LD_LIBRARY_PATH` as a fallback — the dynamic
   linker consults `LD_LIBRARY_PATH` before an rpath recorded as `DT_RUNPATH`
   (the modern default), so this recovers a moved install in the common case,
   but is not guaranteed if the library was linked with the older
   `DT_RPATH` instead. Don't rely on this over (1).

If `uv sync` still fails after building, the error is no longer the
wheel/build problem — capture the exact `uv sync` output (last ~20 lines)
before chasing further; it's likely something else (e.g. `unitree_sdk2py`'s
own git install, or a genuinely offline box with no GitHub access at all).

The built wheel keeps an rpath to that install dir — don't delete it.
