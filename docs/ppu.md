# The PPU box: what it is, how to set up on it, and what will bite you

Everything below was learned by doing it, on 2026-08-07/08, on a box we now own.
A setup PDF was handed over by a coworker; it is a useful bug list and an
unreliable layout guide — it was a log from a *different* machine, with paths
under `/home/zhonghao` that violate this cluster's storage rules and a
`~/.bashrc` step that is both unnecessary and non-durable here. Where this
document and that PDF disagree, this document was measured.

Companion files: [envs/ppu-train.sh](../envs/ppu-train.sh),
[envs/ppu-extract.sh](../envs/ppu-extract.sh) and their constraints files carry
the same facts in executable form. [environments.md](environments.md) is the
cross-machine story; this is the PPU-specific one.

---

## 1. Background: what you are actually logging into

### The accelerator

**PPU is a chip, not a server.** PPU-ZW810E is Alibaba's in-house AI
accelerator, sold as the PAI **PG1** instance type. Our box has **16 of them,
96 GB each** (`98304MiB` in `nvidia-smi`) — four times a 4090's VRAM, so the
memory budgeting in [perception_v2_pipeline.md](perception_v2_pipeline.md) is
simply not a constraint here.

**It pretends to be CUDA.** The PPU SDK ships a CUDA-compatible runtime at
`/usr/local/PPU_SDK/CUDA_SDK/`. That is why `nvidia-smi` works, why
`CUDA_VISIBLE_DEVICES` is the device-pinning variable, and why JAX reports
`backend: gpu` with `CudaDevice(...)`. The giveaway that it is really a PPU is
`ALINPU INFO: name=PPU-ZW810E` in the logs, and `ppu-smi` alongside `nvidia-smi`.

**This is the single fact the whole setup turns on.** Wheels containing compiled
GPU code are built for specific silicon. `jax` and `torch` from PyPI are NVIDIA
programs; they install fine here and then fall back to **CPU, silently**. The
accelerator packages must come from Alibaba's own index.

### The machine

You SSH into a **PAI DSW instance**, which is a **Kubernetes pod** (the hostname
`dsw-941440-7dc8df8769-lnwlk` is a pod name), running

    training-xpu-pytorch:2.1.0-torch2.9.0-ubuntu24.04-cuda12.9-py312

Consequences:

* **The container filesystem is not durable.** `apt` installs, `/root`, and
  anything outside `/mnt/*` do not survive a restart. Do not build on them.
* **The image already exports the entire PPU runtime** — `PPU_SDK`, `PPU_HOME`,
  `CUDA_HOME`, `CUDA_PATH`, `PJRT_DEVICE=CUDA`, `PATH`, `LD_LIBRARY_PATH`. The
  PDF's step 3 (writing these into `~/.bashrc`) is unnecessary *and* would be
  wiped.
* **The image already ships a working PPU torch 2.9.0** that sees all 16 cards.
  You never install torch. See §4 for why you *cannot*.
* **Everyone logs in as `root`.** `/root` is shared with ~20 coworkers:
  `/root/.cache` holds their HuggingFace, uv and pip caches, and `/root/.bashrc`
  is their login shell too. **Never edit either.** Isolation on this cluster is
  by directory convention, not by account.

### Storage

| mount | what | size | use for |
|---|---|---|---|
| `/mnt/cpfs` | BMCPFS parallel filesystem | 10 T, **~1.1 T free, shared** | everything you actively work on |
| `/mnt/oss` | `ossfs2` FUSE mount over OSS | 512 T | durable archive |
| `/` | container rootfs | 5 T | scratch only; **ephemeral** |

`/mnt/workspace` is the *same* BMCPFS volume as `/mnt/cpfs`, mounted twice —
there is no second persistent tier.

### Networking

* An HTTP proxy runs at **`127.0.0.1:7890`** (a coworker's Clash-style tunnel).
  It is what makes GitHub and HuggingFace reachable. It can disappear without
  warning, so depend on it only for genuinely external hosts.
* **Aliyun hosts must bypass it** or they hang. Set
  `no_proxy=...,.aliyuncs.com,.aliyun.com,...`. A `curl` that returns `000`
  through the proxy is a routing problem, not a missing package.
* HuggingFace goes through `HF_ENDPOINT=https://hf-mirror.com`.
* `gs://` URLs (openpi's `pi05_base` weights) are slow — see §6.

### The package indexes

Three URLs, two of which look confusingly similar:

| URL | what it is |
|---|---|
| `pypi.org` | PyPI, the default public index |
| `mirrors.aliyun.com/pypi/simple/` | a **mirror** of PyPI — same packages, faster in China |
| `aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu129/simple/` | a **different index**: PPU builds |

Call the third one **the PPU index**. It carries PPU-compiled builds under the
*same names and version numbers* as PyPI — there is a `jax==0.7.2` on each and
they are different binaries. It also proxies ordinary PyPI packages, so it can
serve as the sole index… except for a handful of names it **overrides** rather
than proxies (`torch`, `torchvision`, `triton`, `nvidia-*`), where only PPU
versions exist.

The image sets `PIP_INDEX_URL` to it. **`uv` does not read `PIP_INDEX_URL`** —
set `UV_DEFAULT_INDEX` too, or `pip install jax` and `uv pip install jax` fetch
from different servers and give you different binaries.

---

## 2. Your directory

### The rule for deciding where a file lives

Organise by **lifetime × how you would get it back**, not by topic:

| class | example | recovery | where |
|---|---|---|---|
| **A. source** | the checkout | `git clone`, seconds | CPFS, in the repo |
| **B. downloaded immutables** | SAM 3, `pi05_base`, wheels | re-download, minutes–hours | CPFS, **shared cache outside the repo** |
| **C. generated** | LeRobot datasets, checkpoints | recompute, hours–days of PPU time | CPFS to write, **archive to OSS** |
| **D. irreplaceable** | raw recordings | you can't | OSS is the master; CPFS holds a working copy |

CPFS-vs-OSS then collapses to two *independent* questions:

* *Does a PPU process read this at speed during a job?* → it must be on CPFS.
* *Would losing it cost more than an afternoon?* → it must have a copy on OSS.

Raw episodes answer yes to both, which is why they live in both places. **CPFS
is not a backup** — it is a small, contended scratch tier in front of a 512 T
archive. At ~1.1 T free shared between ~20 people, keep resident checkpoints to
a couple hundred GB and push to OSS as they land.

### Layout

```
/mnt/cpfs/<initials>/            # the team convention: hrk, gzr, lzb, hxy …
├── ego2g1_v2/                   # A — the checkout. Pull-only; author on the Mac.
├── venvs/{train,extract}/       # machine state, OUTSIDE the repo on purpose
├── env/                         # machine profile + credentials backup
├── cache/{hf,openpi,uv,pip,torch,xla,xdg}/   # B
├── data/{raw_hdf5,lerobot_datasets}/          # C/D — EGO2G1_DATA points here
├── runs/ego2g1_v2/              # C — checkpoints, logs, assets
└── (TMPDIR is /tmp/<initials> on the roomy rootfs, NOT on CPFS)

oss://<bucket>/<initials>/       # D + archived C
```

The property worth preserving: **`rm -rf ego2g1_v2` costs you nothing.** No
weights, no checkpoints, no venv. If that stops being true, something is in the
wrong place.

The venv sits outside the repo deliberately: it costs an hour to build, it is
machine state rather than repo state (§4), and keeping it out means you can
delete and re-clone the checkout freely.

### The machine profile

`$WS_HOME/env/paths.sh` is **not** in the repo — it describes the pod and the
account, not this project, and would be identical for any other project you ran
here. It sets `WS_HOME`, redirects `HOME` into the workspace, points every cache
at CPFS, sets `TMPDIR` to the rootfs, `no_proxy`, `HF_ENDPOINT`, and
`UV_DEFAULT_INDEX`.

**Redirecting `HOME` is the mechanism that keeps you out of everyone's way.**
Tools that hardcode `~/.rtccache` or `~/.cache` follow you into the workspace,
with zero effect on other sessions. The alternative — symlinking `/root/.cache`
— would hijack every coworker's cache into your quota and move their existing
files. Do not do that.

The repo's `envs/ppu-*.sh` profiles **auto-source** `../env/paths.sh` when
`WS_HOME` is unset, which works because the checkout lives inside the workspace.
Guarded by file existence, so it is a no-op on the Mac.

### Moving data: `ossutil`, not `rsync`

`rsync` is a POSIX tool; `/mnt/oss` is FUSE over HTTP. Its delta-transfer
algorithm needs to *read the destination*, which means downloading the whole
object; its temp-file-then-rename becomes a server-side copy; mtimes are
approximate so its skip heuristic misfires; and it is single-stream.

`ossutil` speaks the OSS API directly: parallel transfers, multipart upload,
resumable, CRC64 verification. Use it for anything large. Keep the mount for
browsing. A checkpoint directory is immutable once written, so you can archive
step-5000 while step-10000 is still training.

```bash
ossutil64 cp -r -u /mnt/cpfs/<you>/runs/... oss://<bucket>/<you>/... --jobs 8 --parallel 8
```

Check whether the endpoint in `~/.ossutilconfig` is the `-internal` one — the
public endpoint costs egress and is slower. Note those are **shared team
credentials**; don't copy them onto a laptop.

---

## 3. Running things

One command per activity, from anywhere:

```bash
source /mnt/cpfs/<you>/ego2g1_v2/envs/ppu-train.sh     # training
source /mnt/cpfs/<you>/ego2g1_v2/envs/ppu-extract.sh   # data extraction / perception-v2
```

Each prints a verification block: where `ego2g1`/`openpi` resolve, numpy, jax
backend and device count, torch device count. **Read it.** It is the cheapest
detection of a silently wrong environment.

### Training

The repo's documented recipe (`uv sync --group train`, then `uv run python -m …`)
is **Mac-only**. On this box it is actively destructive — see §4. The
translation is *source the profile, then plain `python -m`*:

```bash
python -m pytest tests/train/test_umi.py tests/train/test_relation.py -q
python -m ego2g1.train.compute_norm_stats --umi --video-backend pyav \
    --assets-base-dir /mnt/cpfs/<you>/runs/ego2g1_v2/assets
python -m ego2g1.train.train --umi \
    --num-train-steps 20 --warmup-steps 5 --no-wandb-enabled --exp-name smoke --overwrite \
    --video-backend pyav \
    --weight-loader-params-path /mnt/cpfs/<you>/cache/openpi/openpi-assets/checkpoints/pi05_base/params \
    --assets-base-dir     /mnt/cpfs/<you>/runs/ego2g1_v2/assets \
    --checkpoint-base-dir /mnt/cpfs/<you>/runs/ego2g1_v2
```

Three flags are effectively mandatory on this box:

* **`--video-backend pyav`** — `torchcodec` is deliberately not installed
  ("Could not load libtorchcodec" on PPU); see
  [config.py](../ego2g1/train/config.py) `video_backend`.
* **`--checkpoint-base-dir`** — the default `./checkpoints` writes tens of GB
  **into the pull-only checkout**, where a future `git clean -xdf` deletes them.
* **`--assets-base-dir`** — same reasoning for norm stats. Whatever you choose,
  `compute_norm_stats` and `train` must get the **same** value, or training
  cannot find the stats it just computed.

Run the 20-step shakeout before a 30k-step run. Expect several minutes of
apparent hang at step 0: that is XLA compiling `train_step` plus PPU operator
autotune. `Saved_file: ~/.rtccache/...` is the good sign.

### Extraction

```bash
source envs/ppu-extract.sh
python -m data_extraction.extract \
    --episodes "$EGO2G1_DATA/raw_hdf5/ego2g1/<task>/episode_2.hdf5" \
    --prompts "red block,yellow block,black pen holder" \
    --out-dir /mnt/cpfs/<you>/runs/data_extraction/out
```

Start with one `.hdf5`, not a directory: two full SAM 3 passes plus ~1800
orientation crops per 610-frame episode.

### Surviving a disconnect

The DSW image ships a `tmux` **wrapper** at `/etc/dsw/runtime/export_bin/tmux`
that execs a runtime component which is not installed, so `tmux` fails even
after `apt install tmux`. Use `/usr/bin/tmux` explicitly, or `hash -r` if bash
cached the wrapper. `nohup … > log 2>&1 &` plus `tail -f` works with no tmux at
all. Note that **none of these survive a pod restart** — only frequent
`--save-interval` plus archiving to OSS does.

---

## 4. Virtual environments, and every gotcha we hit

### Why there are two, and why neither is `uv sync`-managed

[pyproject.toml](../pyproject.toml) declares them conflicting:

```toml
conflicts = [ [{group = "train"}, {group = "perception-v2"}] ]
```

`train` pulls the vendored openpi, which pins `transformers==4.53.2`; SAM 3's
`Sam3Model` landed in transformers 5.0.0. **They cannot share an environment.**

| | `venvs/train` | `venvs/extract` |
|---|---|---|
| profile | `envs/ppu-train.sh` | `envs/ppu-extract.sh` |
| numpy | **2.x** (`>=2.0,<2.8`) | **1.26.4** (`<2`) |
| transformers | 4.53.2 | 5.14.1 |
| jax | 0.7.2 PPU, 16 devices | — |
| torch | the image's 2.9.0 | the image's 2.9.0 |

The numpy split is not an inconsistency; see the numpy gotcha below.

### `uv` gotchas, in the order they will bite you

**`uv sync` and `uv run` are forbidden on this box.** Both target
`<repo>/.venv`, not your real venv, and both resolve from `uv.lock`, which is a
*PyPI* lockfile — so they install NVIDIA torch and `jax[cuda12]`. Worst case
they *succeed*, and you get a plausible venv where `jax.default_backend()` is
`cpu` and training runs at a fraction of the speed. Plus ~10 GB on a 90 %-full
filesystem.

**`uv` ignores `PIP_INDEX_URL`.** Set `UV_DEFAULT_INDEX` explicitly or the two
tools disagree about what `jax` is.

**`uv export` re-resolves unless you pass `--frozen`.** Without it, export goes
to the network, picks the newest `torch` on the PPU index (2.11.0), tries to
build that sdist merely to read its metadata, and dies. `--frozen` makes export
what it claims to be: a pure lockfile→requirements.txt conversion.

**Constraints do not apply to `uv export`.** `UV_CONSTRAINT` governs
`uv pip install`/`uv sync`. A `torch==2.9.0` constraint will not stop export
choosing 2.11.0.

**`uv pip install` targets the *active* venv.** Always pass
`--python /path/to/venvs/<name>/bin/python` when installing into the other one.
Dropping `transformers>=5` into the training venv breaks openpi.

**`--system-site-packages` satisfies imports, not resolution.** The venv can
`import torch` from the image, but uv plans installs as if the venv were empty —
so any dependency naming `torch` makes uv try to install it. That is why
`lerobot` must go in with `--no-deps` and its dependency list supplied by hand.

**`--no-deps` is correct when installing a lockfile export**, because an export
is already the complete transitive closure. There is nothing to resolve, so uv
never gets the chance to reach for a torch sdist.

**PEP 440 local versions sort, and the highest wins.** `jax==0.7.2` also matches
`0.7.2+v0.1.0.ppu2.1.0.oe` and `.ce`; the resolver picks whichever sorts highest.
That is how a source-only `torchvision` `.ce` build got chosen. **Pin full
version strings** for anything from the PPU index.

**Torch is not installable from the index.** Its "sdists" are downloader shims
that fetch a prebuilt artifact matching python/platform/cuda/SDK; for some
versions no artifact exists and you get *"No installation package compatible
with the current environment"*. Use the image's torch. Corollary: `triton` and
`nvidia-*` are torch's tail, the index **overrides** those names, and the
lockfile's upstream pins do not exist there — drop them from any export.

### The numpy split

PPU `jax==0.7.2` **requires `numpy>=2.0`**, which contradicts openpi's `numpy<2`
pin (hence openpi is installed `--no-deps`). Measured: torch 2.9.0 works fine at
numpy 2.5.1 on this box.

The price: the image's compiled packages were built against numpy 1.26, so any
of them using numpy's C API raises

    ValueError: numpy.dtype size changed, Expected 96 from C header, got 88

at import. The fix is to shadow them with a venv copy built for numpy 2 —
**`scikit-learn` and `pandas`** were the two that mattered. Most of the image's
46 numpy-declaring packages are pure Python, Rust, or never imported
(`vllm`, `deepspeed`, `ray`, `xformers`). The failure is **loud, not silent**,
which is what makes this survivable.

The extraction venv has no JAX, so it stays at numpy 1.26 and none of this
applies — which is why the two venvs deliberately differ.

### Constraints files: policy vs record

Two artifacts, two jobs:

* **`envs/ppu-*-constraints.txt`** — *policy*, hand-written, version-controlled.
  A constraints file installs nothing; it caps what a resolver may choose, and
  only for packages something is already installing (so it is safe to list
  packages a venv does not use). Applied automatically via `UV_CONSTRAINT` /
  `PIP_CONSTRAINT` in the profiles, so it holds for every future install without
  remembering a flag.
* **`$WS_HOME/env/ppu-*.freeze.txt`** — *record*, generated by `uv pip freeze`
  after a change that worked. Never edited.

Together they are the rebuild kit. Note the freeze does **not** contain torch —
that comes from the image — which is exactly why the constraints file names it.

### Profile state leaks between sessions

`source` runs in your current shell, so variables outlive the script. Using
`${EGO2G1_VENV:-default}` in both profiles meant sourcing one after the other
silently re-activated the *first* venv. Worse, a leaked `UV_CONSTRAINT` would
carry `numpy<2` into a training shell, where jax cannot satisfy it.

Fixed by giving each profile its own override name
(`EGO2G1_TRAIN_VENV` / `EGO2G1_EXTRACT_VENV`), setting `UV_CONSTRAINT`
unconditionally, `deactivate`-ing any active venv first, and rebuilding
`PYTHONPATH` from a captured base instead of prepending.

**The cheapest check that you are where you think you are is the prompt
prefix** — `(train)`, `(extract)`, or nothing. "No venv active" fails as
`ModuleNotFoundError`, which reads like a missing dependency rather than a
missing environment.

---

## 5. Recipes

### Building the training venv

```bash
uv venv --python 3.12 --system-site-packages $WS_HOME/venvs/train
source $WS_HOME/venvs/train/bin/activate
uv pip install jax==0.7.2 jaxlib==0.7.2 jax-cuda12-plugin==0.7.2 jax-cuda12-pjrt==0.7.2
uv pip install "numpy>=2.0,<2.8" "scipy<1.18"
uv pip install --no-deps -e third_party/openpi
uv pip install --no-deps -e third_party/openpi/packages/openpi-client
uv pip install <openpi's runtime deps by hand>        # NOT its pyproject: it pins jax[cuda12]
uv pip install --no-deps "lerobot @ git+https://github.com/huggingface/lerobot@0cf8648…"
uv pip install <lerobot's deps minus torch/torchvision/torchcodec>
uv pip install scikit-learn pandas               # shadow the image's numpy-1.26 builds
```

Gate on the import gauntlet (`jax`, `flax`, `orbax.checkpoint`, `openpi_client`,
`openpi.training.config`, `lerobot`, `datasets`, `pandas`, `sklearn`, `torch`,
`ego2g1.train.config`) and on `jax.default_backend() == "gpu"` with 16 devices.

### Building the extraction venv

```bash
uv venv --python 3.12 --system-site-packages $WS_HOME/venvs/extract
uv export --frozen --no-dev --group perception-v2 --no-hashes \
    --no-emit-package torch --no-emit-package torchvision --no-emit-package torchcodec \
    > "$TMPDIR/reqs.txt" && mv "$TMPDIR/reqs.txt" $WS_HOME/env/extract-reqs.txt
DROP='^(-e \./third_party/unitree_deploy|unitree-sdk2py @|cyclonedds==|casadi==|pin==|triton==|nvidia-[a-z0-9_-]*==)'
grep -vE "$DROP" $WS_HOME/env/extract-reqs.txt > $WS_HOME/env/extract-reqs.ppu.txt
uv pip install --no-deps --python $WS_HOME/venvs/extract/bin/python \
    -r $WS_HOME/env/extract-reqs.ppu.txt
```

The robot five (`unitree_deploy`, `unitree-sdk2py`, `cyclonedds`, `casadi`,
`pin`) arrive only via `include-group = "deploy"` and are safe to drop:
`ego2g1/deploy/__init__.py`, `deploy/perception/__init__.py` and
`deploy/perception/v2/__init__.py` are all docstring-only, so importing
`perception.v2.orientation_v2` touches no robot package. `cyclonedds` would
additionally need a C library build this box does not have.

Write to `$TMPDIR` and `mv` on success: `>` truncates the target *before* the
command runs, so a failed export leaves you with an empty file.

### Getting `pi05_base` without waiting all day

openpi's default `gs://openpi-assets/checkpoints/pi05_base/params` is ~11.6 GiB
of Orbax shards fetched serially from Google Cloud Storage through the proxy.
The bucket is public, so parallel HTTPS is far faster:

```bash
DEST=$WS_HOME/cache/openpi/openpi-assets/checkpoints/pi05_base   # where openpi caches it
curl -s "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_base/&maxResults=5000" \
  | python3 -c "import sys,json;[print(i['name']) for i in json.load(sys.stdin).get('items',[])]" > "$TMPDIR/pi05.list"
fetch() { rel="${1#checkpoints/pi05_base/}"; out="$DEST/$rel"; [ -s "$out" ] && return 0
          mkdir -p "$(dirname "$out")"; curl -sf --retry 3 -o "$out" "https://storage.googleapis.com/openpi-assets/$1"; }
export -f fetch DEST
xargs -a "$TMPDIR/pi05.list" -P 16 -I{} bash -c 'fetch "$@"' _ {}
```

Verify by comparing per-file sizes against the listing (a dropped connection
leaves a truncated file that a non-empty check passes), delete openpi's
abandoned `params.partial/` and `params.lock`, then **archive to OSS** so nobody
pays that cost again. Afterwards pass
`--weight-loader-params-path $DEST/params`: openpi's `maybe_download`
short-circuits on a scheme-less path, so the download never runs again.

### After a pod restart

Everything durable is on CPFS. Only the rootfs needs restoring — see
`$WS_HOME/bootstrap.sh`: copy `.ossutilconfig` back into `/root`, re-`apt
install tmux`. Do **not** let that script regenerate `paths.sh`; the file on
CPFS is the source of truth, not a generated artifact.

---

## 6. Things in this repo that assume a different machine

Known and unfixed at the time of writing:

* [train.py](../ego2g1/train/train.py) hardcodes
  `jax_compilation_cache_dir` to `~/.cache/jax` in all three entrypoints,
  overriding `JAX_COMPILATION_CACHE_DIR`. It works here only because `HOME` is
  redirected onto CPFS — by accident, not design.
* `checkpoint_base_dir` defaults to `./checkpoints` and `assets_base_dir` to
  `./assets`, both relative to the repo root. Pass the flags every time, or make
  the defaults environment-aware.
* [environments.md](environments.md) still describes this box as having a
  *borrowed* venv that "uv cannot build". That was the old shared box; we own
  this one and uv built both venvs.
