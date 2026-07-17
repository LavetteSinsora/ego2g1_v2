# Environments: three machines, one lockfile, two exceptions

The repo's promise is `uv sync` on a fresh machine. Two of our three machines can't
keep that promise — not because of this repo, but because their venvs are not ours to
manage. So the rule is: **dependencies live in the lockfile; machine state lives in
`envs/`** — one sourceable profile per machine role, each documenting why it exists.

| machine | role | profile | venv |
|---|---|---|---|
| the Mac | dev + data extraction + offline checks | `envs/mac-dev.sh` | uv-managed `.venv` (Python 3.12) — the normal case |
| the PPU box | pi0.5 training + policy serving | `envs/ppu-train.sh` / `envs/ppu-serve.sh` | **borrowed** `$HOME/openpi/.venv` |
| the robot PC | `ego2g1.deploy` on the G1-D | `envs/robot.sh` | **stacked** `.venv-deploy` on a donor venv |

## uv groups (the normal machines)

```bash
uv sync                     # core + kin + data — the default working set
uv sync --group train       # + jax/openpi training stack
uv sync --group deploy      # + DDS/unitree_deploy robot stack
uv sync --group teleop      # + vuer WebXR teleop (tools/teleop)
```

On the Mac, that's the whole story; `envs/mac-dev.sh` only activates the venv, puts
`.venv/bin` first on PATH (ffmpeg comes from imageio-ffmpeg *inside* the venv — stages
that shell out to ffmpeg must find that copy), and reminds you of the two local quirks:
interactive MuJoCo viewers need `mjpython`, and big wheel downloads need the proxy app
ON (direct connections drop them; git ssh to GitHub is blocked — push via https).

## The PPU box exception

The box is 16× Alibaba/T-Head PPU-ZW810E NPUs exposed to JAX through a CUDA12-compat
shim (backend reports `gpu`). The only venv that can drive them is the pre-existing
`$HOME/openpi/.venv`, which carries the NPU runtime libs — **uv cannot build that venv,
so no lockfile group can describe this machine.** That is exactly why `envs/` are
profiles: `ppu-train.sh` borrows the venv read-only and shadows its openpi with this
checkout via PYTHONPATH; `ppu-serve.sh` additionally pins one device.

The three facts that will bite you if forgotten (details in the profiles' headers):

- device pinning is `CUDA_VISIBLE_DEVICES` (`EGO2G1_PPU=15 source envs/ppu-serve.sh`);
  the PPU/ALINPU-named variants are ignored. Unpinned, JAX grabs all 16 cards.
- **never** set `XLA_PYTHON_CLIENT_PREALLOCATE=false` or `ALLOCATOR=platform` — both
  segfault this NPU's driver on the first infer call. Keep the default allocator.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` is the working training value; lower it to
  share a card, don't change the allocator.

## The robot PC exception

`envs/robot.sh` — the machine is *on* the robot's 192.168.123.x subnet and its venv is
stacked (a dedicated `.venv-deploy` created from a donor venv's interpreter, `.pth`-linked
to the donor's site-packages, with only the genuinely missing packages pip-installed).
The full recipe is in the profile header; the robot facts are in [robot.md](robot.md).

## The `datasets==3.6.0` pin

`datasets` **must stay 3.6.0** (pinned in `pyproject.toml`), for two independent reasons:

1. newer `datasets` breaks the pinned lerobot API — extraction crashes;
2. worse, it **writes parquet the training env can't read back**. A dataset written
   under a newer `datasets` looks fine on disk and then fails at training time.

If a dataset was ever written under the wrong version, **regenerate it** — do not try
to re-read or convert it (`docs/datasets.md` has the regeneration commands).
