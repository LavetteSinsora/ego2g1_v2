"""Weight loader that tolerates ego2g1-only modules in the param tree.

Stock `CheckpointWeightLoader` calls
    _merge_params(loaded_params, params, missing_regex=".*lora.*")
and `_merge_params` keeps only (a) loaded keys that exist in the reference tree
and (b) reference keys whose path FULLMATCHES `missing_regex`. Everything else
in the reference tree is dropped from the result.

pi05_base contains no `relation_encoder` and no `grasp_head`, and neither path
matches `.*lora.*`, so under the stock loader those freshly-initialized params
are silently discarded and `init_train_state` then fails on a structure
mismatch -- or, worse, a future refactor makes it not fail and the encoder
trains from nothing. Widening the regex is the whole fix.

`_merge_params` is reused rather than reimplemented so its merge semantics
cannot drift from upstream; importing a private from openpi has precedent in
this package (`_checkpoints._split_params` in train.py).
"""

import dataclasses
import re

import numpy as np

import openpi.models.model as _model
import openpi.shared.download as download
from openpi.training.weight_loaders import WeightLoader
from openpi.training.weight_loaders import _merge_params  # noqa: PLC2701

# Param-path prefixes that ego2g1 adds on top of a stock pi0/pi05 tree. These
# never exist in a pretrained checkpoint, so they must always come from the
# freshly initialized reference tree.
EGO2G1_MODULE_PATTERNS = (".*lora.*", ".*relation_encoder.*", ".*grasp_head.*")

MISSING_REGEX = "|".join(f"({p})" for p in EGO2G1_MODULE_PATTERNS)


@dataclasses.dataclass(frozen=True)
class Ego2G1CheckpointWeightLoader(WeightLoader):
    """Stock CheckpointWeightLoader + ego2g1's own modules kept from the
    reference tree. Identical to stock for a config with neither module."""

    params_path: str

    def load(self, params):
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        merged = _merge_params(loaded_params, params, missing_regex=MISSING_REGEX)
        _assert_no_dropped_params(params, merged)
        return merged


def _flat_keys(tree) -> set:
    import flax.traverse_util

    return set(flax.traverse_util.flatten_dict(tree, sep="/"))


def _assert_no_dropped_params(reference, merged) -> None:
    """The guard the stock loader lacks: every reference param must survive.

    Without this, a newly added module whose name is not in
    EGO2G1_MODULE_PATTERNS is dropped silently and the failure surfaces much
    later as an opaque sharding/structure error.
    """
    missing = _flat_keys(reference) - _flat_keys(merged)
    if missing:
        pattern = re.compile(MISSING_REGEX)
        unmatched = sorted(k for k in missing if not pattern.fullmatch(k))
        raise ValueError(
            f"{len(missing)} params in the model tree were dropped by the weight loader "
            f"(e.g. {sorted(missing)[:5]}). Add their module name to "
            f"ego2g1.train.weight_loader.EGO2G1_MODULE_PATTERNS. "
            f"Paths not matching the current regex: {unmatched[:5]}"
        )
