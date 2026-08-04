# envs/4090.sh — one box running BOTH ego2g1.serve (policy) and ego2g1.deploy
# (executor + relation_eef perception) on a real NVIDIA RTX 4090.
# MUST be sourced, not executed:   source envs/4090.sh
#
# What machine this is and why each line exists:
#   * Unlike the PPU box, this is a NORMAL machine as far as uv is concerned —
#     a real CUDA GPU, so `uv sync` resolves openpi's `jax[cuda12]==0.5.3` pin
#     directly and correctly for this hardware. No borrowed/stacked venv, no
#     envs/ppu-*.sh device-pinning dance. If `.venv` doesn't exist yet:
#       uv sync --group train --group deploy            # joint/relative_eef
#       uv sync --group train --group perception        # + relation_eef (SAM2/DINO)
#     Prefer running `uv sync` ON this machine over copying a `.venv` built
#     elsewhere via USB — .venv/bin/* scripts embed an absolute shebang path
#     at creation time, so a venv built at a different absolute path (or for
#     a no-GPU box, where jaxlib still resolves the same cuda12 wheels but
#     torch's default extras may differ) is a real way to get subtle breakage
#     that "it ran uv sync fine over there" doesn't catch.
#   * GPU SHARING: serve (JAX) and, in relation_eef mode, the perception
#     detector (torch — GroundingDinoSam2Detector defaults to `device="cuda"`
#     if available) both land on this ONE card. JAX preallocates ~90% of VRAM
#     by default, which starves torch's allocator. UNLIKE the PPU's NPU
#     (where disabling preallocation segfaults the driver on first infer —
#     envs/ppu-serve.sh), on a real NVIDIA card turning preallocation off is
#     the standard, safe fix — so this profile sets it, opposite of
#     ppu-serve.sh's rule. If VRAM is still tight, pass
#     GroundingDinoSam2Detector(device="cpu") explicitly instead of fighting
#     allocator flags further.
#   * DDS/robot network: same facts as envs/robot.sh (this box now plays that
#     role too) — NIC carrying the 192.168.123.x address, domain 0, head
#     camera + image_server SSH target. Auto-detected below; override with
#     EGO2G1_IFACE if the guess is wrong (e.g. deploy traffic rides a second
#     NIC/USB-ethernet dongle rather than the box's main interface).
#   * relation_eef defaults: EGO2G1_TASK_CONFIG/EGO2G1_STEREO_CALIB/
#     EGO2G1_CAMERA_EXTRINSIC below feed runner.py's --task-config/
#     --stereo-calib/--camera-extrinsic (ego2g1/deploy/runner.py's Args reads
#     these exact env vars as its field defaults) — set once here instead of
#     on every invocation; an explicit flag on the command line still wins.
#     Only exported when the file actually exists, so a fresh checkout with
#     no task config yet still gets runner.py's normal fail-loud "missing
#     --task-config" error instead of pointing at a dangling path.
#   * CycloneDDS: `cyclonedds==0.10.2` normally gets a manylinux wheel on
#     Linux (unlike macOS, which has none at all — docs/deps-deploy.md), but
#     an unsupported Python minor version/libc/offline sync still falls back
#     to a source build that needs the C library + CYCLONEDDS_HOME, same as
#     the Mac. If `uv sync --group deploy`/`--group perception` fails on this
#     box, build it once into `.cyclonedds/` AT THE REPO ROOT, not `$HOME`
#     (docs/deps-deploy.md's "CycloneDDS on Linux" section has the exact
#     commands) — this machine is meant to be copied/USB-transplanted whole,
#     and a dependency living under `$HOME` doesn't travel with the checkout.
#     This profile picks up the result automatically below. Building at one
#     absolute path and then copying the tree to a DIFFERENT absolute path
#     still breaks the compiled extension's baked-in rpath — see that same
#     doc section for why, and prefer building on the final machine at its
#     final path over copying a pre-built tree.
#   * HuggingFace access: GroundingDinoSam2Detector pulls grounding-dino-tiny
#     + the SAM2 checkpoint from huggingface.co at construction time
#     (perception/detector.py) — if that connection is as unreliable as
#     GitHub was, HF_ENDPOINT below repoints every huggingface_hub/transformers
#     download at the hf-mirror.com mirror (the standard workaround for
#     exactly this). HF_HOME keeps the downloaded weights repo-local
#     (.hf_cache/, NOT $HOME) for the same self-containment reason as
#     .cyclonedds/ above. If the mirror is ever the one that's down instead,
#     override with `HF_ENDPOINT=https://huggingface.co source envs/4090.sh`
#     (an already-exported value here always wins, see the `:-` below).
#   * Still-open prerequisites this profile does NOT solve for you (see the
#     relation_eef checklist): the task-config YAML itself (copy
#     ego2g1/deploy/perception/task_config.example.yaml to
#     ego2g1/deploy/perception/task_config.yaml and edit it — this profile
#     picks that path up automatically once it exists), BRAINCO_CLOSED_POSE in
#     ego2g1/deploy/gripper_calib.py (still the np.ones(6) placeholder until
#     you run `check.py hand-jog`), and confirming stereo_calib.npz/
#     camera_calib.npz at the repo root are current for today's camera mount.

_EGO2G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

if [ ! -f "$_EGO2G1_ROOT/.venv/bin/activate" ]; then
    echo "ERROR: no .venv at $_EGO2G1_ROOT — run \`uv sync --group train --group deploy\`" \
         "(add --group perception for relation_eef) first" >&2
    return 1 2>/dev/null || exit 1
fi

source "$_EGO2G1_ROOT/.venv/bin/activate"
cd "$_EGO2G1_ROOT"

# --- GPU sharing: serve (JAX) must not preallocate the whole card -----------
# Opposite of envs/ppu-serve.sh's rule — that one is specific to the T-Head
# PPU NPU driver, not a general CUDA fact. Lower MEM_FRACTION further if
# GroundingDINO+SAM2 still OOMs with serve already warm.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"

# --- robot network map (defaults = our G1-D; override per machine) ---------
export ROBOT_IP="${ROBOT_IP:-192.168.123.164}"          # robot board: ssh + image_server
export ROBOT_USER="${ROBOT_USER:-unitree}"              # ssh unitree@192.168.123.164 (password 123)
export EGO2G1_CAMERA_HOST="${EGO2G1_CAMERA_HOST:-$ROBOT_IP}"  # --camera-host
export EGO2G1_DDS_DOMAIN="${EGO2G1_DDS_DOMAIN:-0}"      # --domain

# --iface = the NIC holding our 192.168.123.x address. Auto-detect; override
# with EGO2G1_IFACE=<nic> if the guess is wrong.
if [ -z "${EGO2G1_IFACE:-}" ]; then
    if command -v ip >/dev/null 2>&1; then
        EGO2G1_IFACE="$(ip -4 -o addr 2>/dev/null | awk '/192\.168\.123\./{print $2; exit}')"
    else
        EGO2G1_IFACE="$(ifconfig 2>/dev/null | awk '/^[a-zA-Z0-9]+[:.]?/{i=$1; sub(":$","",i)} /inet 192\.168\.123\./{print i; exit}')"
    fi
    EGO2G1_IFACE="${EGO2G1_IFACE:-eth0}"
fi
export EGO2G1_IFACE

# CycloneDDS C library — repo-local (.cyclonedds/, NOT $HOME) so the whole
# checkout stays self-contained/copyable. Only set if it was actually built
# here (see the header comment / docs/deps-deploy.md); needed at `uv sync`
# build time, harmless to export otherwise since a wheel install never reads
# it. LD_LIBRARY_PATH is a defensive fallback ONLY — if this tree was built at
# a different absolute path and then copied here, it may recover the compiled
# extension's rpath (DT_RUNPATH honors LD_LIBRARY_PATH first), but the
# supported path is building fresh at this exact location (see the doc).
if [ -d "$_EGO2G1_ROOT/.cyclonedds/install" ]; then
    export CYCLONEDDS_HOME="$_EGO2G1_ROOT/.cyclonedds/install"
    export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
fi

# --- relation_eef defaults: runner.py's Args reads these three verbatim ----
# (task_config/stereo_calib/camera_extrinsic default_factory= os.environ.get(...),
# ego2g1/deploy/runner.py). Only set when the file exists; joint/relative_eef
# runs never look at these regardless.
[ -f "$_EGO2G1_ROOT/ego2g1/deploy/perception/task_config.yaml" ] && \
    export EGO2G1_TASK_CONFIG="$_EGO2G1_ROOT/ego2g1/deploy/perception/task_config.yaml"
[ -f "$_EGO2G1_ROOT/stereo_calib.npz" ] && \
    export EGO2G1_STEREO_CALIB="$_EGO2G1_ROOT/stereo_calib.npz"
[ -f "$_EGO2G1_ROOT/camera_calib.npz" ] && \
    export EGO2G1_CAMERA_EXTRINSIC="$_EGO2G1_ROOT/camera_calib.npz"

# --- HuggingFace: mirror + repo-local cache (relation_eef's GroundingDINO/SAM2) ---
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$_EGO2G1_ROOT/.hf_cache}"

echo "4090: venv=$_EGO2G1_ROOT/.venv  iface=$EGO2G1_IFACE  domain=$EGO2G1_DDS_DOMAIN  camera=$EGO2G1_CAMERA_HOST"
echo "4090: hf_endpoint=$HF_ENDPOINT  hf_home=$HF_HOME"
echo "4090: cyclonedds=${CYCLONEDDS_HOME:-<none — build into .cyclonedds/ per docs/deps-deploy.md if uv sync fails on cyclonedds>}"
echo "4090: relation_eef defaults — task-config=${EGO2G1_TASK_CONFIG:-<none, add ego2g1/deploy/perception/task_config.yaml>}"
echo "4090:   stereo-calib=${EGO2G1_STEREO_CALIB:-<missing>}  camera-extrinsic=${EGO2G1_CAMERA_EXTRINSIC:-<missing>}"
echo "4090: jax preallocate=off mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION (leaving room for torch/SAM2 on the same card)"
python - <<'PY'
import importlib.util
missing = [m for m in ("jax", "torch") if importlib.util.find_spec(m) is None]
if missing:
    print(f"4090: WARNING — {missing} not importable in this venv; run the uv sync groups noted at the top of this file")
PY
echo "4090: SAFETY — G1-D has no balance controller: stand/suspended, remote in hand (docs/robot.md)"
echo "4090: relation_eef checklist — task-config YAML authored? BRAINCO_CLOSED_POSE measured (check.py hand-jog)? stereo_calib.npz/camera_calib.npz current?"
