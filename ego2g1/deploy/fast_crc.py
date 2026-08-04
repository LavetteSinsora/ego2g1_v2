"""Native-speed LowCmd CRC on macOS (and any platform without Unitree's .so) —
AND a same-speed fallback on Linux boxes whose unitree_sdk2py install is
missing its compiled crc_amd64.so/crc_aarch64.so (a broken/partial install,
not a platform gap: CRC.__init__ does an unguarded `ctypes.CDLL(...)` when
`platform.system() == "Linux"`, so a missing file is an unrecoverable crash
in the constructor with no fallback of its own -- confirmed by reading the
installed source; there is no flag or env var upstream to opt out of it).

unitree_sdk2py ships compiled CRC only for Linux; everywhere else it falls
back to a pure-Python bit loop that costs ~1 ms per LowCmd — 39–50% of the
500 Hz executor's 2 ms tick budget, inside the GIL, competing with the hand
thread and DDS callbacks. On ZH's Linux robot PC this cost never existed,
which made it an invisible machine-level difference between "their deploy is
smooth" and "ours judders" when driving from a Mac.

The fix: Unitree's CRC is CRC-32/MPEG-2 (poly 0x04C11DB7, unreflected). A
reflected-CRC identity lets zlib's C implementation compute it:

    mpeg2(msg) = bitrev32( zlib.crc32(bitrev_bytes(msg)) ^ 0xFFFFFFFF )

with byte-wise bit reversal done by bytes.translate (also C). Verified EXACT
against unitree_sdk2py's own `_crc_py` on 50 random word lists; measured
5.5 us vs 1.04 ms (187x) on the dev Mac.

`install()` monkeypatches three things on the CRC class, call it before
creating the executor (already done in `executor.py`'s `connect()`):
  - `_crc_py`        the non-Linux pure-Python path -- always replaced.
  - `__init__`        wrapped so a missing native library on Linux (OSError
                      from ctypes.CDLL) is caught instead of propagating.
                      The packFmt* attributes are already set by the ORIGINAL
                      __init__ by the time it reaches the ctypes line, so a
                      caught failure there still leaves a usable instance;
                      `crc_lib` is left unset/None to signal "no native lib"
                      to the patched _crc_ctypes below. CRC is a Singleton
                      whose __new__ caches the instance but does NOT skip
                      re-running __init__ on later CRC() calls (confirmed:
                      unitree_sdk2py/utils/singleton.py's Singleton.__init__
                      is a no-op `pass`, only CRC's own __init__ does real
                      work) -- so this is idempotent across retries.
  - `_crc_ctypes`     falls back to the same zlib path when `crc_lib` is
                      unavailable; unchanged (uses the real native lib) when
                      it loaded fine, so a working install's behavior and
                      performance are untouched.
"""

import zlib

import numpy as np

_REV = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))


def _bitrev32(x: int) -> int:
    x = ((x & 0x55555555) << 1) | ((x >> 1) & 0x55555555)
    x = ((x & 0x33333333) << 2) | ((x >> 2) & 0x33333333)
    x = ((x & 0x0F0F0F0F) << 4) | ((x >> 4) & 0x0F0F0F0F)
    return ((x << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) | (x >> 24)) & 0xFFFFFFFF


def crc32_mpeg2_words(words) -> int:
    """CRC-32/MPEG-2 over a sequence of uint32 words, MSB-first per word —
    bit-identical to unitree_sdk2py CRC._crc_py."""
    raw = np.asarray(words, dtype=">u4").tobytes()
    return _bitrev32(zlib.crc32(raw.translate(_REV)) ^ 0xFFFFFFFF)


_installed = False


def install() -> None:
    """Replace unitree_sdk2py's pure-Python CRC fallback with the zlib path,
    AND make a missing native .so on Linux degrade to the same path instead
    of crashing (see module docstring)."""
    global _installed
    if _installed:
        return
    from unitree_sdk2py.utils.crc import CRC

    import ctypes

    _orig_init = CRC.__init__

    def _safe_init(self):
        try:
            _orig_init(self)
        except OSError:
            # crc_amd64.so/crc_aarch64.so missing (broken/partial install) --
            # packFmt* attrs are already set (assigned before the ctypes.CDLL
            # call in the original __init__, so this catch doesn't lose
            # them); crc_lib stays unset, _crc_ctypes below checks for that.
            self.crc_lib = None

    def _crc_ctypes_with_fallback(self, data):
        crc_lib = getattr(self, "crc_lib", None)
        if crc_lib is None:
            return crc32_mpeg2_words(data)
        uint32_array = (ctypes.c_uint32 * len(data))(*data)
        return crc_lib.crc32_core(uint32_array, len(data))

    CRC.__init__ = _safe_init
    CRC._crc_py = lambda self, data: crc32_mpeg2_words(data)
    CRC._crc_ctypes = _crc_ctypes_with_fallback
    _installed = True
