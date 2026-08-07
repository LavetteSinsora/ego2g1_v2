"""Rotational symmetry order (`alpha`) from Orient Anything V2.

WHAT ALPHA IS
    Not a network head. `val_fit_alpha` is a post-hoc fit that upstream runs
    on the SIGMOID of the azimuth logits — the 360-bin distribution, not its
    argmax. It fits a von Mises density `exp(k cos(a (x - mu)))` for
    `a in {1, 2, 4}`, keeps the best R^2, and then applies a per-alpha
    confidence floor on the concentration `kappa` and on R^2:

        a = 1  needs kappa >= 0.60 and R^2 >= 0.45
        a = 2  needs kappa >= 0.50 and R^2 >= 0.45
        a = 4  needs kappa >= 0.25 and R^2 >= 0.45
        otherwise -> 0

    So the value is the number of orientations about the vertical axis that
    look identical to the model:

        1   one unambiguous front — a mug with a handle, a shoe
        2   two-fold — a book, a rectangular box: 180 deg apart is the same
        4   four-fold — a cube, a plain cylinder-ish holder: 90 deg apart
        0   NO CONFIDENT CALL. The distribution was too flat or fit too badly
            to claim any symmetry. This is a real and common outcome and must
            never be silently rounded to 1.

    It is a property of the OBJECT, so a well-behaved episode should return
    the same non-zero alpha frame after frame. Watching it flicker is the
    diagnostic: an object whose alpha jumps between 1 and 2 has an azimuth
    distribution the model cannot commit to, and any rotation read off that
    frame is worth less than its confidence suggests.

WHY IT IS WORTH THE COST
    This is the measurement plan Q8 asks for — "symmetry group per object:
    cube, identity, or from Orient V2's `ref_alpha_pred` at seed then
    frozen?" — and §5.2 explicitly defers it, because online the fit is a
    scipy `curve_fit` per crop (up to three) and the group must be frozen
    once at seed anyway. Offline there is no budget and no reason to freeze:
    running it on every frame is what turns the question into data.

    The symmetry group is not decoration. `orientation.py`'s snap picks the
    representative of `measured @ S` nearest the previous frame, and with the
    wrong group a stationary object's rotation jumps between equivalent
    matrices — or, with a group that is too large, two genuinely different
    poses collapse to one.

IMPORTED, NOT REIMPLEMENTED
    `val_fit_alpha` lives in the Orient Anything V2 checkout, which
    `OrientAnythingV2.__init__` already puts on `sys.path`. The thresholds
    above are load-bearing and undocumented upstream, so a local port would
    drift from whatever the repo actually does at the version you cloned.
    Unavailable-import is a survivable degradation, not an error.
"""

from __future__ import annotations

import contextlib
import io
import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ALPHA_MEANING", "SymmetryFitter"]

ALPHA_MEANING = {
    0: "no confident symmetry call",
    1: "1-fold (one unambiguous front)",
    2: "2-fold (180 deg ambiguity)",
    4: "4-fold (90 deg ambiguity)",
}


class SymmetryFitter:
    """Wraps upstream `val_fit_alpha`, batched, quiet, and fail-soft."""

    def __init__(self, *, enabled: bool = True):
        self._fn = None
        self.available = False
        self.reason = "disabled"
        if not enabled:
            return
        try:
            from utils.app_utils import val_fit_alpha       # noqa: PLC0415
        except Exception as exc:                            # noqa: BLE001
            self.reason = f"import failed: {exc!r}"
            logger.warning(
                "val_fit_alpha unavailable (%s) — symmetry order will be "
                "written as -1 (unknown). Orientation itself is unaffected.",
                self.reason)
            return
        self._fn = val_fit_alpha
        self.available = True
        self.reason = ""

    def __call__(self, distribution: np.ndarray) -> np.ndarray:
        """(B, 360) sigmoid azimuth distribution -> (B,) int8 in {0,1,2,4}.

        Returns -1 for every row when the fit is unavailable or throws, so a
        consumer can always tell "not measured" from "measured as 0", which
        is a genuine outcome meaning "no confident symmetry call".
        """
        d = np.asarray(distribution, dtype=np.float64)
        if d.ndim == 1:
            d = d[None, :]
        if not self.available:
            return np.full(len(d), -1, dtype=np.int8)
        try:
            # `val_fit_alpha` prints its fit parameters on every row. Over a
            # few thousand crops that is the loudest thing in the log and
            # hides everything worth reading. It also MUTATES its input in
            # place (`y_noise /= ...`), so it gets a copy.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = self._fn(d.copy())
            values = np.asarray(
                out.detach().cpu().numpy() if hasattr(out, "detach") else out)
            return np.rint(values).astype(np.int8).reshape(-1)
        except Exception as exc:                            # noqa: BLE001
            logger.warning("val_fit_alpha raised %r on a batch of %d — those "
                           "rows are unknown (-1)", exc, len(d))
            return np.full(len(d), -1, dtype=np.int8)


def summarise(alpha: np.ndarray) -> dict:
    """What an episode concluded about one object's symmetry.

    `mode` is the answer you would freeze at seed; `agreement` is how much
    the episode actually supported it. Low agreement with a high rate means
    the model is not committing, and the frozen group would be a coin flip.
    """
    a = np.asarray(alpha)
    known = a[a >= 0]
    if known.size == 0:
        return {"mode": None, "agreement": 0.0, "measured_frames": 0}
    vals, counts = np.unique(known, return_counts=True)
    top = int(vals[int(np.argmax(counts))])
    return {
        "mode": top,
        "meaning": ALPHA_MEANING.get(top, "?"),
        "agreement": round(float(counts.max() / known.size), 4),
        "measured_frames": int(known.size),
        "histogram": {int(v): int(c) for v, c in zip(vals, counts)},
    }
