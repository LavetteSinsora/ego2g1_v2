"""One owner for every perception-v2 tuning knob (plan §7).

Same contract as the v1 `../config.py` it replaces: load from one YAML,
default to exactly the values the constructors use, reject unknown keys
loudly, and embed the loaded config verbatim in every recording's `meta.json`
so a replayed session knows the thresholds that produced it.

Two v1 keys are GONE rather than carried forward:

    detector_period_ticks      meaningless under T1. Perception is
    orientation_period_ticks   free-running: there is no cadence to set,
                               because a round starts the instant the last one
                               ends. Leaving them loadable would let an
                               operator set a number that silently does
                               nothing.

Everything that replaced them is expressed in SECONDS or METRES, never ticks
or samples (S2). The loop rate varies, so any constant in samples means a
different thing from one minute to the next.

YAML shape — every key optional, omitted keys keep the default:

    sam3:
      prune: true
      dtype: bfloat16
    visibility:                # sam3_source.VisibilityConfig
      min_det_score: 0.5
      min_area_fraction: 0.35
    orient:                    # orientation_v2.OrientAnythingV2
      size: 336
      cast_weights: true
      crop_pad: 0.15
    convention:                # orientation_v2.OrientationConvention
      azimuth_sign: -1.0
    tracker:                   # object_tracker.ObjectTracker kwargs
      max_speed_m_s: 1.5
      history_s: 6.0
    latch:                     # latch.LatchConfig
      confirm_displacement_m: 0.04
      divergence_gate: crop
    sgbm:                      # ../depth.StereoSGBMDepthSource kwargs
      num_disparities: 128
    timing:
      headroom: 1.15
      max_d: 20
"""

from __future__ import annotations

import dataclasses
import pathlib

from .latch import LatchConfig
from .orientation_v2 import OrientationConvention
from .sam3_source import VisibilityConfig

__all__ = ["PerceptionV2Config", "OrientConfig", "Sam3Config", "TimingConfig"]

_TOP_KEYS = {"sam3", "visibility", "orient", "convention", "tracker", "latch",
             "sgbm", "timing"}
# The two v1 keys that must fail loudly rather than be ignored: an operator
# who sets one is expressing an intent the design no longer has a way to obey.
_RETIRED = {
    "detector_period_ticks":
        "perception is free-running (T1); there is no detector cadence",
    "orientation_period_ticks":
        "orientation runs in-loop on usable crops only (R2/S1), not on a timer",
}


@dataclasses.dataclass(frozen=True)
class Sam3Config:
    repo: str = "facebook/sam3"
    dtype: str = "bfloat16"
    prune: bool = True


@dataclasses.dataclass(frozen=True)
class OrientConfig:
    """R2's levers. `size` is the big one: cost scales with `(size/14)^2`, so
    518 -> 336 is ~0.42x the work and 518 -> 252 ~0.24x, and the plan's §2.3
    establishes that rough orientation is sufficient. `cast_weights` is the
    VRAM lever — upstream keeps fp32 parameters and leans on internal
    autocast, which is why a 5 GB checkpoint occupies 10.2 GB."""

    enabled: bool = True
    repo_dir: str = "third_party/Orient-Anything-V2"
    checkpoint: str | None = None
    size: int = 518
    cast_weights: bool = False
    crop_pad: float = 0.15
    # Training runs rembg before inference (do_rm_bkg=True). We composite the
    # SAM 3 mask instead — a better segmentation, same kind of image.
    background: str = "white"
    # Training's anchor is obj_keys[0]; None means "the first roster entry".
    anchor_id: str | None = None


@dataclasses.dataclass(frozen=True)
class TimingConfig:
    headroom: float = 1.15
    max_d: int = 20
    # How long a replan waits for the in-flight round before falling back to
    # the last completed snapshot (T3, §4.3). A multiple of the measured round
    # time rather than an absolute, so it tracks the free-running rate.
    replan_timeout_rounds: float = 1.5


@dataclasses.dataclass(frozen=True)
class PerceptionV2Config:
    sam3: Sam3Config = dataclasses.field(default_factory=Sam3Config)
    visibility: VisibilityConfig = dataclasses.field(default_factory=VisibilityConfig)
    orient: OrientConfig = dataclasses.field(default_factory=OrientConfig)
    convention: OrientationConvention = dataclasses.field(
        default_factory=OrientationConvention)
    tracker: dict = dataclasses.field(default_factory=dict)
    latch: LatchConfig = dataclasses.field(default_factory=LatchConfig)
    sgbm: dict = dataclasses.field(default_factory=dict)
    timing: TimingConfig = dataclasses.field(default_factory=TimingConfig)

    @classmethod
    def load(cls, path=None) -> "PerceptionV2Config":
        if path is None:
            return cls()
        import yaml

        raw = yaml.safe_load(pathlib.Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")

        retired = sorted(set(raw) & set(_RETIRED))
        if retired:
            raise ValueError(
                f"{path}: {retired} no longer exist in perception v2. "
                + "; ".join(f"{k}: {_RETIRED[k]}" for k in retired))
        unknown = set(raw) - _TOP_KEYS
        if unknown:
            raise ValueError(
                f"{path}: unknown key(s) {sorted(unknown)} "
                f"(known: {sorted(_TOP_KEYS)}) — a typo'd knob must not "
                "silently keep the default")

        return cls(
            sam3=_strict(Sam3Config, raw.get("sam3"), path, "sam3"),
            visibility=_strict(VisibilityConfig, raw.get("visibility"), path,
                               "visibility"),
            orient=_strict(OrientConfig, raw.get("orient"), path, "orient"),
            convention=_strict(OrientationConvention, raw.get("convention"),
                               path, "convention"),
            tracker=dict(raw.get("tracker") or {}),
            latch=_strict(LatchConfig, raw.get("latch"), path, "latch"),
            sgbm=dict(raw.get("sgbm") or {}),
            timing=_strict(TimingConfig, raw.get("timing"), path, "timing"),
        )

    def as_dict(self) -> dict:
        """JSON-safe form for meta.json embedding.

        `convention` is in here on purpose and is the most important field to
        record: a rotation expressed in the wrong canonical frame is still a
        valid rotation, so nothing downstream can detect it after the fact.
        The only way to reinterpret an old recording is to know which
        convention produced it.
        """
        return dataclasses.asdict(self)


def _strict(cls, raw, path, section):
    """Build a frozen dataclass from a YAML block, rejecting unknown fields.

    Same philosophy as `task_config.validate_against_server_metadata`: a
    misspelled threshold that silently keeps its default is worse than a
    crash, because the operator believes they retuned something they did not.
    """
    if not raw:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: '{section}' must be a mapping, got {raw!r}")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown '{section}' key(s) "
                         f"{sorted(unknown)} (known: {sorted(known)})")
    return cls(**raw)
