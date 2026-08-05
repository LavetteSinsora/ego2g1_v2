"""Phase-0 refactor pins (docs/deploy_refactor_plan.md §9 tasks 1-2).

Three previously convention-held assumptions, now executable:

  * `_util.precise_wait` is importable WITHOUT the runner module — the whole
    point of extracting it (four replay tools used to pay the full runner
    import for a 10-line timing helper).
  * `fast_crc.crc32_mpeg2_words` really is CRC-32/MPEG-2 — checked against an
    independent bit-loop reference implemented HERE (not against the sdk's
    `_crc_py`, so the pin holds on machines without unitree_sdk2py too).
  * `executor.damp()`'s reach into the vendored `G1_29_ArmController`
    internals (stop_event, msg, crc, lowcmd_publisher, lowstate_buffer) —
    the e-stop's load-bearing assumption — matches the vendored source.
"""

import importlib
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest

from ego2g1.deploy import _util, fast_crc

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_precise_wait_importable_without_runner():
    """Importing a replay tool (which needs precise_wait) must not drag in
    the runner module. Run in a fresh interpreter so this session's imports
    don't mask a regression."""
    code = (
        "import sys\n"
        "import ego2g1.deploy.replay_dataset\n"
        "import ego2g1.deploy.replay_diag\n"
        "assert 'ego2g1.deploy.runner' not in sys.modules, "
        "'replay tools should not import the runner'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_precise_wait_waits_until_t_end():
    t = {"now": 0.0}

    def clock():
        t["now"] += 0.0004
        return t["now"]

    start = clock()
    _util.precise_wait(start + 0.01, time_func=clock)
    assert t["now"] >= start + 0.01


def _crc32_mpeg2_reference(words) -> int:
    """Independent CRC-32/MPEG-2 (poly 0x04C11DB7, init 0xFFFFFFFF,
    unreflected, no final xor) over uint32 words MSB-first — the textbook
    bit loop, deliberately NOT sharing any code with fast_crc."""
    crc = 0xFFFFFFFF
    for w in np.asarray(words, dtype=np.uint64).astype(np.uint32):
        for byte in int(w).to_bytes(4, "big"):
            crc ^= byte << 24
            for _ in range(8):
                crc = ((crc << 1) ^ 0x04C11DB7 if crc & 0x80000000 else crc << 1) \
                      & 0xFFFFFFFF
    return crc


def test_fast_crc_bit_identity_against_reference():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 47, 249):   # 249 words ≈ a LowCmd_ payload
        words = rng.integers(0, 2**32, size=n, dtype=np.uint64)
        assert fast_crc.crc32_mpeg2_words(words) == _crc32_mpeg2_reference(words)
    # known-answer vector: CRC-32/MPEG-2("123456789") == 0x0376E6E7, packed
    # into words MSB-first with the check string padded to a word multiple
    words = np.frombuffer(b"123456789\x00\x00\x00", dtype=">u4").astype(np.uint64)
    assert fast_crc.crc32_mpeg2_words(words) == _crc32_mpeg2_reference(words)


def test_fast_crc_install_idempotent_and_patches():
    pytest.importorskip("unitree_sdk2py")
    from unitree_sdk2py.utils.crc import CRC

    fast_crc.install()
    first_py = CRC._crc_py
    fast_crc.install()   # second call must be a no-op, not a double wrap
    assert CRC._crc_py is first_py
    words = list(np.random.default_rng(1).integers(0, 2**32, size=12, dtype=np.uint64))
    assert CRC._crc_py(None, words) == _crc32_mpeg2_reference(words)


def test_damp_internals_pinned_to_vendored_source():
    """damp() (executor.py) reaches for G1_29_ArmController.{stop_event, msg,
    crc, lowcmd_publisher, lowstate_buffer}. Pin those names against the
    vendored source text — no DDS needed, and a vendored-tree upgrade that
    renames any of them fails HERE instead of silently disarming the e-stop
    (executor's own AttributeError guard is the runtime backstop; this is
    the offline one)."""
    src_path = (REPO / "third_party" / "unitree_deploy" / "unitree_deploy"
                / "robot_devices" / "arm" / "g1_arm.py")
    src = src_path.read_text()
    m = re.search(r"class G1_29_ArmController\b.*?(?=\nclass |\Z)", src, re.S)
    assert m, "G1_29_ArmController not found in vendored g1_arm.py"
    body = m.group(0)
    for attr in ("stop_event", "msg", "crc", "lowcmd_publisher", "lowstate_buffer"):
        assert re.search(rf"self\.{attr}\b", body), (
            f"vendored G1_29_ArmController no longer assigns self.{attr}; "
            "executor.damp() is not armed against this copy")


def test_dds_init_helper_is_the_only_factory_call_site():
    """The 7×-duplicated ChannelFactoryInitialize block was collapsed into
    _util.dds_init; keep it that way."""
    deploy = REPO / "ego2g1" / "deploy"
    offenders = []
    for p in deploy.rglob("*.py"):
        if p.name == "_util.py":
            continue
        if "import ChannelFactoryInitialize" in p.read_text():
            offenders.append(p.name)
    assert not offenders, f"raw ChannelFactoryInitialize call in {offenders}; use _util.dds_init"
