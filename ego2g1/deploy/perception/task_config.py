"""Deploy-side object task configuration for `relation_eef` mode (§5.1).

`EgoRelationTrainConfig` checkpoints (`ego2g1/train/config.py`) are trained
against a FIXED, ORDERED tuple of object categories
(`train_config.objects`/`.object_prompt_names`/`.n_objects`) baked into the
checkpoint and re-advertised over the wire in
`policy.metadata["ego2g1"]` by `ego2g1/serve/policy.py`'s `create_policy`
(see docs/relation_deploy_plan.md §4.2). This module is the DEPLOY-side
mirror: what the operator wants the (not-yet-built, Phase 2) live
perception pipeline to actually go detect for a given task/run, in the
order the connected checkpoint expects.

Two moving parts:

  * `ObjectSpec`/`DeployTaskConfig` + `load_task_config` -- a small,
    human-edited YAML file describing the objects for this deploy session
    (instance id, category, detector prompt, graspability) and which hands
    are in play. Shape deliberately mirrors
    `data_extraction_zh/src/ego_relation/configs/default.yaml`'s
    `task.objects` block (instance_id/category/prompt/graspable) so a task
    config can be ported by copy-paste rather than reinvented -- but this
    module does NOT import anything from that project (it is a separate uv
    project; ego2g1_v2 must not depend on it).

  * `validate_against_server_metadata` -- a fail-loud guard, run once at
    connect time, that refuses to proceed if the operator's local task
    config doesn't match what the connected checkpoint actually expects.
    Same "fail loud before it can silently mis-serve" philosophy as
    `ego2g1/train/stamp.py`'s `check_supported` and
    `ego2g1/train/dataset.py`'s `assert_relation_dataset_compatible`: a
    mismatched object order here would silently feed the model an object's
    geometry under the wrong slot, which is a *worse* failure than crashing
    (it degrades quietly instead of loudly), so this is deliberately as
    strict and unforgiving as those two guards.

`object_prompt_names` is intentionally NOT validated here: it is a
training-time prompt-wording detail (what short name gets glued into the
instruction text) that the deploy operator has no reason to have to
reproduce exactly. `hands` is also not validated against server metadata:
it is a deploy-local choice (which of the robot's two hands are physically
in play for this session), not something the checkpoint asserts.
"""

from __future__ import annotations

import dataclasses
import pathlib

import yaml


@dataclasses.dataclass(frozen=True)
class ObjectSpec:
    """One object the live perception pipeline should track for this task.

    `category` MUST match one of `train_config.objects` POSITIONALLY --
    i.e. `DeployTaskConfig.objects[i].category` is expected to equal the
    connected checkpoint's `train_config.objects[i]`, in order. See
    `validate_against_server_metadata`.
    """

    instance_id: str
    category: str
    detector_prompt: str
    graspable: bool = True


@dataclasses.dataclass(frozen=True)
class DeployTaskConfig:
    """The full set of objects (in checkpoint order) plus which hands are live.

    `objects`' order MUST match the checkpoint's `train_config.objects`
    order -- this is the same fixed, ordered contract `RelationPrompt`
    expects server-side (docs/relation_deploy_plan.md §3.3), just declared
    on the deploy side of the wire.
    """

    objects: tuple[ObjectSpec, ...]
    hands: tuple[str, ...] = ("left", "right")


def _object_spec_from_dict(d: dict) -> ObjectSpec:
    try:
        return ObjectSpec(
            instance_id=str(d["instance_id"]),
            category=str(d["category"]),
            detector_prompt=str(d["detector_prompt"]),
            graspable=bool(d.get("graspable", True)),
        )
    except KeyError as e:
        raise ValueError(f"task config object entry {d!r} is missing required key {e}") from e


def load_task_config(path) -> DeployTaskConfig:
    """Load a `DeployTaskConfig` from a small YAML file.

    Expected shape (mirrors `data_extraction_zh`'s `task.objects` block,
    field-for-field except `prompt` -> `detector_prompt` to match
    `ObjectSpec`'s own field name):

        objects:
          - instance_id: obj1
            category: "black, metal pen holder"
            detector_prompt: "a black, metal pen holder ."
            graspable: false
          - instance_id: obj2
            category: "red cube"
            detector_prompt: "a red cube ."
            graspable: true
        hands: [left, right]   # optional, defaults to ("left", "right")
    """
    path = pathlib.Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "objects" not in raw:
        raise ValueError(f"task config {path} must be a mapping with an 'objects' list, got {raw!r}")

    objects_raw = raw["objects"]
    if not isinstance(objects_raw, list) or not objects_raw:
        raise ValueError(f"task config {path}'s 'objects' must be a non-empty list, got {objects_raw!r}")

    objects = tuple(_object_spec_from_dict(entry) for entry in objects_raw)
    hands = tuple(raw["hands"]) if "hands" in raw else ("left", "right")
    return DeployTaskConfig(objects=objects, hands=hands)


def validate_against_server_metadata(task_config: DeployTaskConfig, server_ego2g1_metadata: dict) -> None:
    """Refuse to proceed if `task_config` doesn't match the connected checkpoint.

    Checks, in order:
      1. the server metadata actually advertises a relation-mode object list
         (`n_objects`/`objects` keys present -- their absence means the
         connected checkpoint isn't an `EgoRelationTrainConfig` at all, e.g.
         the operator pointed a `relation_eef` deploy at an old 30-dim
         checkpoint);
      2. `len(task_config.objects) == server_ego2g1_metadata["n_objects"]`;
      3. `task_config.objects[i].category == server_ego2g1_metadata["objects"][i]`
         for every `i`, in order.

    Deliberately as strict and unforgiving as `ego2g1/train/stamp.py`'s
    `check_supported`: a silently mismatched object order would feed the
    model an object's geometry under the wrong slot, which is worse than a
    crash. `object_prompt_names` is NOT checked (prompt wording is a
    training-time detail, not a deploy-operator concern) and `hands` is NOT
    checked (deploy-local, not asserted by the checkpoint).

    Raises:
        ValueError: with a message naming exactly what mismatched, on any
            failure.
    """
    if "n_objects" not in server_ego2g1_metadata or "objects" not in server_ego2g1_metadata:
        raise ValueError(
            "server metadata does not advertise relation-mode object info "
            "('n_objects'/'objects' missing from metadata['ego2g1']) -- the "
            "connected checkpoint is likely not an EgoRelationTrainConfig "
            f"checkpoint. Got metadata keys: {sorted(server_ego2g1_metadata.keys())}"
        )

    server_objects = tuple(server_ego2g1_metadata["objects"])
    server_n = server_ego2g1_metadata["n_objects"]
    task_categories = tuple(obj.category for obj in task_config.objects)

    if len(task_categories) != server_n:
        raise ValueError(
            f"task config has {len(task_categories)} object(s) "
            f"{task_categories}, but the connected checkpoint expects "
            f"n_objects={server_n} (objects={server_objects}). Fix the "
            "task config's 'objects' list to match the checkpoint before "
            "starting."
        )

    if task_categories != server_objects:
        mismatches = [
            (i, got, want)
            for i, (got, want) in enumerate(zip(task_categories, server_objects))
            if got != want
        ]
        raise ValueError(
            f"task config object order/categories {task_categories} do not "
            f"match the connected checkpoint's train_config.objects "
            f"{server_objects}. Mismatched position(s) (index, task_config "
            f"category, checkpoint category): {mismatches}. `category` must "
            "match train_config.objects POSITIONALLY -- reorder or rename "
            "the task config's 'objects' entries to match."
        )
