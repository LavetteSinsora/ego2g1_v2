"""Native-speed LowCmd CRC on macOS (and any platform without Unitree's .so).

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

`install()` monkeypatches `CRC._crc_py`; call it before creating the
executor. Harmless on Linux (the native branch is taken before _crc_py).
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
    """Replace unitree_sdk2py's pure-Python CRC fallback with the zlib path."""
    global _installed
    if _installed:
        return
    from unitree_sdk2py.utils.crc import CRC

    CRC._crc_py = lambda self, data: crc32_mpeg2_words(data)
    _installed = True
