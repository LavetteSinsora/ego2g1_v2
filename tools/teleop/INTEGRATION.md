# De-vendoring `_vendor/` into `ego2g1.*`

`tools/teleop` was ported from the old repo's `human_hand_teleoperate/` as-is,
`_vendor/` included. The vendor dir exists because the old package had to run on a
robot PC without the rest of the repo; in v2 the same code lives in the one importable
package, so every vendored module has (or will have) a real home. This file makes the
removal mechanical: one import rewrite per row, no thinking required.

Until that happens, **do not edit anything under `_vendor/`** — `tests/test_vendor_drift.py`
hashes every vendored file against `_vendor/MANIFEST.json` and fails on any byte change.
That guarantee (teleop retarget ≡ the code that made the training labels, verified to
2.2e-16 in `tests/test_cancellation.py`) is exactly what de-vendoring must preserve:
after the rewrite, the equivalence check is `sha256(ego2g1 home) == MANIFEST src_sha`
for files ported verbatim, plus a green `tests/test_cancellation.py`.

## Module map

| vendored module | original source (old repo) | `ego2g1` home | home exists? |
|---|---|---|---|
| `_vendor/de/common/frames.py` | `data_extraction/common/frames.py` | `ego2g1.core.frames` | yes |
| `_vendor/de/common/episode.py` | `data_extraction/common/episode.py` | `ego2g1.core.episode` | yes |
| `_vendor/de/hand/constants.py` | `data_extraction/hand/constants.py` | `ego2g1.core.hand.constants` | **no — create `ego2g1/core/hand/`** |
| `_vendor/de/hand/fk_tables.py` | `data_extraction/hand/fk_tables.py` | `ego2g1.core.hand.fk_tables` | no (same) |
| `_vendor/de/hand/retarget.py` | `data_extraction/hand/retarget.py` | `ego2g1.core.hand.retarget` | no (same) |
| `_vendor/de/assets/revo2/*` (fk npz, hand MJCF, meshes) | `data_extraction/assets/revo2/` | repo `assets/revo2/` | yes |
| `_vendor/eg/chunk_math.py` | `third_party/openpi/ego2g1/chunk_math.py` | `ego2g1.core.chunk_math` | yes |
| `_vendor/eg/common/layout.py` | `.../ego2g1/common/layout.py` | `ego2g1.core.layout` | yes |
| `_vendor/eg/common/se3.py` | `.../ego2g1/common/se3.py` | `ego2g1.core.se3` | yes |
| `_vendor/eg/deploy/dds.py` | `.../ego2g1/deploy/dds.py` | `ego2g1.deploy.dds` | deploy port owns it |
| `_vendor/eg/deploy/kinematics.py` | `.../ego2g1/deploy/kinematics.py` | `ego2g1.deploy.kinematics` | deploy port owns it |
| `_vendor/eg/deploy/ramp.py` | `.../ego2g1/deploy/ramp.py` | `ego2g1.deploy.ramp` | deploy port owns it |
| `_vendor/eg/deploy/safety.py` | `.../ego2g1/deploy/safety.py` | `ego2g1.deploy.safety` | deploy port owns it |
| `_vendor/eg/deploy/trajectory.py` | `.../ego2g1/deploy/trajectory.py` | `ego2g1.deploy.trajectory` | deploy port owns it |
| `_vendor/eg/deploy/_g1_sim/sim/g1.py` | `data_extraction/sim/g1.py` | `ego2g1.kin.g1` | yes |
| `_vendor/eg/deploy/_g1_sim/common/frames.py` | `data_extraction/common/frames.py` (2nd copy) | `ego2g1.core.frames` | yes |
| `_vendor/eg/deploy/_g1_sim/assets/unitree_g1/*` | `data_extraction/assets/unitree_g1/` | repo `assets/unitree_g1/` | yes |

Two notes on the map:

- There is **no `_vendor/de/sim/`**: the G1 kinematic sim was vendored as
  `_vendor/eg/deploy/_g1_sim/` (built fresh from `data_extraction/sim/`, the way the old
  deploy package did). Its home is `ego2g1.kin` regardless of the vendor path.
- `_g1_sim/common/frames.py` is a byte-identical duplicate of `de/common/frames.py`; both
  rows collapse into the single `ego2g1.core.frames`.

## The one out-of-package import

`sim.py` (`--sim`, the dynamic MuJoCo stand-in for the robot) does

```python
from data_extraction.sim.g1_hands import build_g1_hands_spec
```

— an import into the **old repo**, kept unchanged by the port. It is the only line in
`tools/teleop` that reaches outside the package + `_vendor`. Future home:
`ego2g1.kin.g1_hands` (already present at `ego2g1/kin/g1_hands.py`); flip the import when
de-vendoring and verify the composite model builds identically. Until then `--sim` only
runs where the old `data_extraction/` is importable.

## Procedure

```bash
# 1 — per row above, rewrite the import (example):
#     from ._vendor.de.common import frames   ->   from ego2g1.core import frames
# 2 — asset paths: point fk_tables/hand-MJCF loaders at assets/ instead of _vendor/de/assets
# 3 — prove nothing moved: the retarget must still cancel the pipeline exactly
python -m pytest tools/teleop/tests/test_cancellation.py tools/teleop/tests/test_smoothing.py
# 4 — delete _vendor/, MANIFEST.json, _vendor/_build.py and tests/test_vendor_drift.py
#     (the drift test's job is done once the modules have one canonical copy)
```

`_vendor/_build.py` and the `python -m tools.teleop._vendor._build` rebuild command in the
README are old-repo machinery (they copy from `data_extraction/` / `third_party/openpi/ego2g1/`);
they die with `_vendor/` and are not worth porting.
