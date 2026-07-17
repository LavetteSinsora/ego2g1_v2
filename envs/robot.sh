# envs/robot.sh — the deploy machine. MUST be sourced:   source envs/robot.sh
#
# What machine this is and why each line exists:
#   * It sits ON the robot's 192.168.123.x subnet — it IS the "robot PC" in the
#     bring-up rung ladder. No gateway/bridge/relay: DDS is direct
#     subscribe/publish on the robot's domain, so `check listen` should see
#     rt/lowstate immediately.
#   * DDS needs the NIC that carries the 192.168.123.x address (--iface) and
#     domain 0 (--domain). EGO2G1_IFACE is auto-detected below; override it if
#     the guess is wrong.
#   * The head-camera image_server runs on the robot board at 192.168.123.164
#     (ZMQ) — that is --camera-host, and also the ssh target
#     (unitree@192.168.123.164, password 123) for starting image_server or the
#     lift tool. See docs/robot.md for the full network map.
#   * The venv here is STACKED, not uv-managed (another reason envs/ are
#     profiles): this machine already has a donor venv with the heavy compiled
#     deps (numpy/mujoco/jax...), the network is slow, and we must never write
#     into the donor. Recipe below.
#
# ---------------------------------------------------------------------------
# Venv stacking recipe (one-time setup; documented here because it is machine
# state, not lockfile state):
#
#   DONOR=~/openpi-zh/.venv          # any healthy venv with the compiled deps
#   ROOT=<this repo>
#
#   # 1. create .venv-deploy FROM the donor's interpreter — same python build,
#   #    so compiled packages inherited below are ABI-compatible
#   "$DONOR/bin/python" -m venv "$ROOT/.venv-deploy"
#
#   # 2. a .pth in the NEW venv's site-packages (never the donor's) appends the
#   #    donor's site-packages + this repo root to sys.path. This is the
#   #    project-scoped alternative to `export PYTHONPATH=...`, which leaks
#   #    into whatever you cd to next in the same shell — a real footgun.
#   SITE_DONOR=$("$DONOR/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
#   SITE_NEW=$("$ROOT/.venv-deploy/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
#   printf '%s\n%s\n' "$SITE_DONOR" "$ROOT" > "$SITE_NEW/ego2g1_paths.pth"
#
#   # 3. pip-install ONLY the genuinely missing pieces into .venv-deploy
#   #    (normal pip — it's its own venv; the donor stays untouched)
#   source "$ROOT/.venv-deploy/bin/activate"
#   pip install mink "qpsolvers[daqp]" pandas
#   pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git   # not on PyPI
#
#   # verify: ego2g1 resolves to this repo, compiled deps come from the donor
#   python -c "import ego2g1, numpy, mujoco; print(ego2g1.__file__)"
#
# Accepted tradeoff: pip may re-install small transitive deps (e.g. scipy) into
# .venv-deploy instead of reusing the donor's copy — pip's resolver only sees
# what is registered in the active venv. Extra disk, no shadowing risk.
# ---------------------------------------------------------------------------

_EGO2G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

if [ -f "$_EGO2G1_ROOT/.venv-deploy/bin/activate" ]; then
    source "$_EGO2G1_ROOT/.venv-deploy/bin/activate"
else
    echo "NOTE: no .venv-deploy yet — build it with the recipe in envs/robot.sh" >&2
fi

# --- robot network map (defaults = our G1-D; override per machine) ---
export ROBOT_IP="${ROBOT_IP:-192.168.123.164}"          # robot board: ssh + image_server
export ROBOT_USER="${ROBOT_USER:-unitree}"              # ssh unitree@192.168.123.164 (password 123)
export EGO2G1_CAMERA_HOST="${EGO2G1_CAMERA_HOST:-$ROBOT_IP}"  # --camera-host / check camera --host
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

cd "$_EGO2G1_ROOT"
echo "robot: iface=$EGO2G1_IFACE domain=$EGO2G1_DDS_DOMAIN camera=$EGO2G1_CAMERA_HOST ssh=$ROBOT_USER@$ROBOT_IP"
echo "robot: SAFETY — G1-D has no balance controller: stand/suspended, remote in hand (docs/robot.md)"
