"""RelationPerceptionConfig: one owner for every perception tuning knob
(docs/deploy_refactor_plan.md §6.1).

Before this module, `detector_period_ticks`, `orientation_period_ticks`, the
latch thresholds, the tracker gating, and the SGBM parameters were all
constructor-only — retuning `latch_distance_m` on hardware meant a source
edit. Now they load from one YAML (`--perception-config` /
`$EGO2G1_PERCEPTION_CONFIG`), default to exactly the values the constructors
always used, and the loaded config is embedded verbatim into every
recording's `meta.json` (via the adapter's `recorder_meta`, see
`modes/relation_eef.py`) so a replayed session knows the thresholds that
produced it.

YAML shape (every key optional; omitted keys keep the constructor default):

    detector_period_ticks: 15        # null -> ~2 Hz derived from fps
    orientation_period_ticks: 6
    latch:                           # perception/latch.py LatchConfig fields
      latch_distance_m: 0.05
      confirm_window_ticks: 12
    tracker:                         # perception/tracker.py ObjectTracker kwargs
      min_residual_m: 0.01
    sgbm:                            # perception/depth.py StereoSGBMDepthSource kwargs
      num_disparities: 128
      block_size: 5

Unknown keys fail loud at load (a typo'd knob must not silently keep the
default — same philosophy as task_config.py's strict validation).
"""

from __future__ import annotations

import dataclasses
import pathlib

from .latch import LatchConfig

_TOP_KEYS = {"detector_period_ticks", "orientation_period_ticks",
             "latch", "tracker", "sgbm"}


@dataclasses.dataclass(frozen=True)
class RelationPerceptionConfig:
    detector_period_ticks: int | None = None   # None -> round(fps / 2), ~2 Hz
    orientation_period_ticks: int = 6
    latch: dict = dataclasses.field(default_factory=dict)     # LatchConfig overrides
    tracker: dict = dataclasses.field(default_factory=dict)   # ObjectTracker kwargs
    sgbm: dict = dataclasses.field(default_factory=dict)      # StereoSGBMDepthSource kwargs

    @classmethod
    def load(cls, path: str | pathlib.Path | None) -> "RelationPerceptionConfig":
        """Load from YAML, or the all-defaults config for `path is None`."""
        if path is None:
            return cls()
        import yaml

        raw = yaml.safe_load(pathlib.Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")
        unknown = set(raw) - _TOP_KEYS
        if unknown:
            raise ValueError(
                f"{path}: unknown perception-config key(s) {sorted(unknown)} "
                f"(known: {sorted(_TOP_KEYS)}) — a typo'd knob must not "
                "silently keep the default")
        latch = dict(raw.get("latch") or {})
        known_latch = {f.name for f in dataclasses.fields(LatchConfig)}
        bad = set(latch) - known_latch
        if bad:
            raise ValueError(f"{path}: unknown latch key(s) {sorted(bad)} "
                             f"(known: {sorted(known_latch)})")
        return cls(
            detector_period_ticks=raw.get("detector_period_ticks"),
            orientation_period_ticks=int(raw.get("orientation_period_ticks", 6)),
            latch=latch,
            tracker=dict(raw.get("tracker") or {}),
            sgbm=dict(raw.get("sgbm") or {}),
        )

    def latch_config(self) -> LatchConfig:
        return LatchConfig(**self.latch)

    def as_dict(self) -> dict:
        """JSON-safe form for meta.json embedding."""
        return dataclasses.asdict(self)
