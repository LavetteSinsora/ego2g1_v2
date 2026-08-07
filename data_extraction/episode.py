"""Reading a raw Pico episode HDF5 — the egocentric video and nothing else.

Same files `tools/teleop/_vendor/de/common/episode.py` reads for wrist replay,
but this needs almost none of what that needs: no body tracking, no control
grid, no frame conversions. Just frames, intrinsics, timestamps, and whatever
the recorder wrote about which objects are in the scene.

Frames are decoded ON DEMAND rather than held. A 610-frame episode is ~560 MB
of uint8 RGB and it is needed exactly twice — once to feed SAM 3, once to cut
orientation crops — while the JPEG bytes are ~50 MB and decode in about a
millisecond. Holding the decoded array would be the single largest allocation
in the pipeline for no gain.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["RawEpisode", "load_episode", "find_episodes"]

# Which eye. `images_left_jpeg` is the legacy alias and the eye the training
# extraction used, so it is the default and the one any comparison must use.
EYES = {"left": "camera/images_left_jpeg", "right": "camera/images_right_jpeg"}


@dataclasses.dataclass
class RawEpisode:
    """One episode, opened lazily.

    Holds the JPEG byte strings (cheap) and decodes on `frame()`. `close()` is
    a no-op — the HDF5 file is read fully into memory at load and released, so
    a long extraction never holds an open handle across a model call.
    """

    path: Path
    name: str
    eye: str
    height: int
    width: int
    K: np.ndarray                    # (3, 3) intrinsics for `eye`
    timestamps_ns: np.ndarray        # (F,) camera clock
    attrs: dict
    _jpegs: list = dataclasses.field(repr=False, default_factory=list)
    # The OTHER eye, kept only so stereo depth is possible. None when the
    # recording has no second eye, which is a survivable degradation: masks,
    # boxes and orientation all still extract, only depth is unavailable.
    _jpegs_other: list | None = dataclasses.field(repr=False, default=None)
    K_other: np.ndarray | None = None
    extrinsics: dict = dataclasses.field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self._jpegs)

    @property
    def has_stereo(self) -> bool:
        return (self._jpegs_other is not None
                and len(self._jpegs_other) == len(self._jpegs)
                and self.K_other is not None
                and {"left", "right"} <= set(self.extrinsics))

    def frame(self, i: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB, the eye this extraction is built on."""
        return _decode(self._jpegs[i])

    def frame_other(self, i: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB, the opposite eye. Stereo partner of `frame`."""
        if self._jpegs_other is None:
            raise RuntimeError(f"{self.name} has no second eye")
        return _decode(self._jpegs_other[i])

    def frames(self):
        for i in range(self.n_frames):
            yield i, self.frame(i)

    # -- what the recorder thinks is in the scene ---------------------------

    def recorded_prompts(self) -> dict[str, str]:
        """`{object key: detector prompt}` from the episode's own attrs, or {}.

        The recorder writes `object_prompts_json` like
        `{"block": "block .", "holder": "holder ."}` — note the trailing " .",
        which is GroundingDINO's phrase separator and wrong for SAM 3.
        `normalize_prompt` strips it downstream and warns; this returns the
        raw strings so the warning fires with the real config text.

        Returned as a suggestion only. The recorded roster is frequently
        COARSER than the task: this episode's own instruction names a red
        block, a yellow block and a pen holder while the prompt map has two
        entries. Pass `--prompts` to say what you actually want tracked.
        """
        raw = self.attrs.get("object_prompts_json")
        if not raw:
            return {}
        try:
            return {str(k): str(v) for k, v in json.loads(raw).items()}
        except (ValueError, TypeError):
            logger.warning("%s: object_prompts_json is not JSON: %r",
                           self.name, raw)
            return {}

    @property
    def task_instruction(self) -> str:
        return str(self.attrs.get("task_instruction", ""))

    @property
    def anchor_object(self) -> str:
        """The recorder's anchor key.

        Training's anchor is `obj_keys[0]` and it is the only object that
        keeps Orient Anything's raw rotation; every other object's rotation is
        CONSTRUCTED relative to it (`orientation_v2
        .compose_relational_rotation`). Recorded here so an extraction can be
        checked against the roster order it was labelled under.
        """
        return str(self.attrs.get("anchor_object", ""))


def _decode(blob: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)


def load_episode(path, *, eye: str = "left", stereo: bool = True) -> RawEpisode:
    import h5py

    if eye not in EYES:
        raise ValueError(f"eye must be one of {sorted(EYES)}, got {eye!r}")
    other = "right" if eye == "left" else "left"
    path = Path(path)
    with h5py.File(path, "r") as f:
        dset = EYES[eye]
        if dset not in f:
            raise KeyError(f"{path} has no {dset!r} — is this a stereo Pico "
                           f"episode? (found: {sorted(f['camera'].keys())})")
        jpegs = [bytes(b) for b in f[dset][:]]
        size = f[f"camera/image_size_{eye}"][:] if f"camera/image_size_{eye}" in f \
            else f["camera/image_size"][:]
        kkey = f"camera/K_{eye}"
        K = np.asarray(f[kkey if kkey in f else "camera/K"][:], dtype=np.float64)
        ts = np.asarray(f["camera/timestamps_ns"][:], dtype=np.int64)
        attrs = {k: (v.item() if isinstance(v, np.generic) else v)
                 for k, v in f.attrs.items()}

        jpegs_other, K_other, extr = None, None, {}
        if stereo and EYES[other] in f:
            jpegs_other = [bytes(b) for b in f[EYES[other]][:]]
            ok = f"camera/K_{other}"
            if ok in f:
                K_other = np.asarray(f[ok][:], dtype=np.float64)
            for side in ("left", "right"):
                key = f"camera/extrinsics_{side}"
                if key in f:
                    extr[side] = np.asarray(f[key][:], dtype=np.float64)
            # `stereo_valid` is the recorder's own per-frame pairing flag. A
            # run with unpaired frames would silently match a left image to a
            # right one from a different instant, and SGBM would return
            # plausible nonsense rather than fail.
            if "camera/stereo_valid" in f:
                sv = np.asarray(f["camera/stereo_valid"][:], dtype=bool)
                if not sv.all():
                    logger.warning(
                        "%s: %d/%d frames have stereo_valid=False. Depth on "
                        "those frames pairs images from different instants — "
                        "they are recorded as invalid.",
                        path.name, int((~sv).sum()), sv.size)
                extr["stereo_valid"] = sv

    width, height = int(size[0]), int(size[1])
    if len(jpegs) != len(ts):
        # Never seen, but a silent off-by-one here would misalign every
        # timestamp in the output by a frame and look like tracker lag.
        raise ValueError(f"{path}: {len(jpegs)} frames but {len(ts)} camera "
                         f"timestamps")
    if jpegs_other is not None and len(jpegs_other) != len(jpegs):
        logger.warning("%s: %d %s frames vs %d %s — stereo disabled",
                       path.name, len(jpegs), eye, len(jpegs_other), other)
        jpegs_other = None
    return RawEpisode(
        path=path,
        name=f"{path.parent.name}/{path.stem}",
        eye=eye,
        height=height, width=width, K=K, timestamps_ns=ts,
        attrs=attrs, _jpegs=jpegs,
        _jpegs_other=jpegs_other, K_other=K_other, extrinsics=extr,
    )


def find_episodes(path) -> list[Path]:
    """One .hdf5 file, or every .hdf5 under a directory, sorted naturally.

    Sorted by the trailing integer where there is one, so `episode_9` comes
    before `episode_10` — lexical order would interleave runs in a way that
    makes a partial extraction hard to reason about.
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)

    def key(p: Path):
        digits = "".join(c for c in p.stem if c.isdigit())
        return (int(digits) if digits else 0, p.stem)

    files = sorted(path.glob("*.hdf5"), key=key)
    if not files:
        raise FileNotFoundError(f"no .hdf5 files under {path}")
    return files
