"""Carry object-relation vectors into the model without editing src/openpi.

The relational state schema (docs/datasets.md, red_block_in_pen_holder_ego) needs
a per-object R^18 hand-object relation matrix to reach `embed_prefix`, where a
learned encoder turns each object into one prompt token embedding. Stock
`openpi.models.model.Observation` has no field for it, and `src/openpi` stays
bit-stock -- so, exactly as `gemma_patch` does for per-token adaRMS, this module
rebinds two module-level symbols behind a source-fingerprint guard.

Two symbols, and the second one is the subtle one:

1. `Observation` -> `RelationObservation`, adding a `relations` field. Safe to
   rebind because NO module in openpi does `from ...model import Observation`;
   every site goes through `_model.Observation`, resolved at call time. The two
   construction sites that matter are `training/data_loader.py` (train) and
   `policies/policy.py` (serve), both via `from_dict`.

2. `preprocess_observation` -> a wrapper. Stock builds a FRESH Observation from
   an explicit field list (model.py, `return Observation(images=..., state=...,
   ...)`), so under the rebind it constructs our subclass but leaves `relations`
   at its default None -- silently dropping the relations before embed_prefix
   ever runs. It is called at the top of compute_loss and every sample_actions
   variant, so missing this would look like "the encoder gets zeros" rather than
   like an error. The wrapper CALLS stock and re-attaches the field; it never
   copies the body, so upstream changes to image augmentation cannot drift from
   us and the guard only has to cover the return contract.

Re-derive the pinned digests with:
    python -m ego2g1.train.observation_patch
"""

import dataclasses
import hashlib
import inspect

from flax import struct

import openpi.models.model as _model
import openpi.shared.array_typing as at

# sha256 over inspect.getsource of the stock symbols this module wraps/replaces.
# If these fail, re-read the new stock code, confirm the assumptions in the
# module docstring still hold, then update the digests.
_STOCK_FINGERPRINTS = {
    "Observation": "f63f422374d6daea2a1cc3cc1478c72eedc09cb1376635f2984bb28bdf31e78c",
    "preprocess_observation": "55c024a8606d598994fbb5d101625bdb4011deea2fef61b4c2208d6ab8b37b5d",
}

# Captured at first import, before apply() can have rebound them.
_STOCK_OBSERVATION = _model.Observation
_STOCK_PREPROCESS = _model.preprocess_observation


class StockSourceChangedError(RuntimeError):
    """openpi's model.py no longer matches the code this patch was derived from."""


@at.typecheck
@struct.dataclass
class RelationObservation(_STOCK_OBSERVATION):
    """Stock Observation + the per-object hand-object relation matrix.

    `relations` is (*b, n, d), already normalized by the time it reaches the
    model. What n and d MEAN depends on the config, and the last axis is
    deliberately NOT pinned here:

      - relation configs (red_block_in_pen_holder_ego): n = n_objects, d = 18 —
        each object's pose in the LEFT TCP frame (vec9) concatenated with its
        pose in the RIGHT TCP frame (vec9). Z-scored with stats pooled ACROSS
        objects, so a shared encoder sees one consistent scale and the prompt's
        object order can be shuffled freely.
      - the UMI config reuses this same channel for STATE HISTORY: n = n_lags
        (history_lags + 1), d = history_dim (7). See UmiTrainConfig, which
        passes n_objects=n_lags / relation_dim=history_dim on purpose rather
        than renaming the injection fields.

    `EgoRelationModel.relation_dim` is the single source of truth for d — it is
    what `inputs_spec` builds the encoder against. Hardcoding 18 in this
    annotation instead made every UMI batch fail beartype at `from_dict`.

    Re-decorating with @struct.dataclass registers the subclass as its own flax
    pytree node with the full field list, so jax.tree.map / sharding / donation
    all see `relations` as an ordinary leaf. The field needs a default because
    every inherited field after `state` has one.
    """

    relations: at.Float[_model.ArrayT, "*b n d"] | None = None

    @classmethod
    def from_dict(cls, data):
        # Stock from_dict mutates data["image"] in place (uint8 -> [-1,1]) and
        # reads a fixed key set; delegate for all of that, then attach ours.
        base = super().from_dict(data)
        return dataclasses.replace(base, relations=data.get("relations"))


def _preprocess_observation(rng, observation, *, train: bool = False, image_keys=None, image_resolution=None):
    """Stock preprocess_observation, with `relations` carried through.

    Signature mirrors stock by keyword so callers are unaffected. Stock is
    called unmodified; only the dropped field is restored.
    """
    kwargs = {}
    if image_keys is not None:
        kwargs["image_keys"] = image_keys
    if image_resolution is not None:
        kwargs["image_resolution"] = image_resolution
    out = _STOCK_PREPROCESS(rng, observation, train=train, **kwargs)
    relations = getattr(observation, "relations", None)
    if relations is None:
        return out
    return dataclasses.replace(out, relations=relations)


def _fingerprint(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def verify_fingerprints() -> None:
    for name, obj in [("Observation", _STOCK_OBSERVATION), ("preprocess_observation", _STOCK_PREPROCESS)]:
        actual = _fingerprint(obj)
        expected = _STOCK_FINGERPRINTS[name]
        if actual != expected:
            raise StockSourceChangedError(
                f"openpi.models.model.{name} source digest {actual} != pinned {expected}. "
                "Upstream model.py changed: re-check the assumptions in "
                "ego2g1/train/observation_patch.py (especially that "
                "preprocess_observation still rebuilds Observation from an explicit "
                "field list), re-run tests/train/test_observation_patch.py, then "
                "update the pinned digest."
            )


def apply() -> None:
    """Idempotently rebind model.Observation / model.preprocess_observation.

    Must run before any Observation is constructed -- i.e. before the data
    loader yields its first batch and before model construction. ego2g1.train.model
    applies it at import, and all ego2g1 entrypoints import models through there.
    """
    if getattr(_model, "_ego2g1_relation_patch", False):
        return
    verify_fingerprints()
    _model.Observation = RelationObservation
    _model.preprocess_observation = _preprocess_observation
    _model._ego2g1_relation_patch = True  # noqa: SLF001


def is_applied() -> bool:
    return bool(getattr(_model, "_ego2g1_relation_patch", False))


if __name__ == "__main__":
    for _name, _obj in [("Observation", _STOCK_OBSERVATION), ("preprocess_observation", _STOCK_PREPROCESS)]:
        print(f'    "{_name}": "{_fingerprint(_obj)}",')
