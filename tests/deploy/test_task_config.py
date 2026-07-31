"""ego2g1.deploy.perception.task_config: YAML round-trip + server-metadata cross-check.

Two halves:
  * round-trip: a small YAML fixture -> `load_task_config` -> `DeployTaskConfig`
    with the right objects/order/fields.
  * `validate_against_server_metadata`: matching case passes silently; wrong
    count and wrong order/category each raise `ValueError` with a message
    that names what mismatched -- same "fail loud before it can silently
    mis-serve" philosophy as `ego2g1/train/stamp.py`'s `check_supported`.
"""

import pytest

from ego2g1.deploy.perception.task_config import (
    DeployTaskConfig,
    ObjectSpec,
    load_task_config,
    validate_against_server_metadata,
)

YAML_FIXTURE = """
objects:
  - instance_id: obj1
    category: "black, metal pen holder"
    detector_prompt: "a black, metal pen holder ."
    graspable: false
  - instance_id: obj2
    category: "red cube"
    detector_prompt: "a red cube ."
    graspable: true
  - instance_id: obj3
    category: "yellow cube"
    detector_prompt: "a yellow cube ."
    graspable: true
hands: [left, right]
"""


# --------------------------------------------------------------------------
# round-trip
# --------------------------------------------------------------------------


def test_load_task_config_roundtrip(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(YAML_FIXTURE)

    cfg = load_task_config(path)

    assert isinstance(cfg, DeployTaskConfig)
    assert cfg.hands == ("left", "right")
    assert len(cfg.objects) == 3

    assert cfg.objects[0] == ObjectSpec(
        instance_id="obj1",
        category="black, metal pen holder",
        detector_prompt="a black, metal pen holder .",
        graspable=False,
    )
    assert cfg.objects[1] == ObjectSpec(
        instance_id="obj2",
        category="red cube",
        detector_prompt="a red cube .",
        graspable=True,
    )
    assert cfg.objects[2] == ObjectSpec(
        instance_id="obj3",
        category="yellow cube",
        detector_prompt="a yellow cube .",
        graspable=True,
    )

    # order is preserved exactly as written, not resorted/deduped
    assert tuple(o.category for o in cfg.objects) == (
        "black, metal pen holder",
        "red cube",
        "yellow cube",
    )


def test_load_task_config_defaults_hands_when_omitted(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(
        """
objects:
  - instance_id: obj1
    category: "red cube"
    detector_prompt: "a red cube ."
"""
    )
    cfg = load_task_config(path)
    assert cfg.hands == ("left", "right")
    assert cfg.objects[0].graspable is True  # ObjectSpec default


def test_load_task_config_missing_objects_key_raises(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text("hands: [left, right]\n")
    with pytest.raises(ValueError, match="objects"):
        load_task_config(path)


def test_load_task_config_missing_required_object_field_raises(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(
        """
objects:
  - instance_id: obj1
    category: "red cube"
"""
    )
    with pytest.raises(ValueError, match="detector_prompt"):
        load_task_config(path)


# --------------------------------------------------------------------------
# validate_against_server_metadata
# --------------------------------------------------------------------------


def _task_config(categories):
    return DeployTaskConfig(
        objects=tuple(
            ObjectSpec(instance_id=f"obj{i}", category=c, detector_prompt=f"a {c} .")
            for i, c in enumerate(categories)
        )
    )


def test_validate_matching_config_passes_silently():
    cfg = _task_config(["black, metal pen holder", "red cube", "yellow cube"])
    server_meta = {
        "objects": ("black, metal pen holder", "red cube", "yellow cube"),
        "object_prompt_names": ("pen holder", "red cube", "yellow cube"),
        "n_objects": 3,
    }
    validate_against_server_metadata(cfg, server_meta)  # must not raise


def test_validate_ignores_object_prompt_names_mismatch():
    """object_prompt_names is a training-time wording detail, not checked."""
    cfg = _task_config(["black, metal pen holder", "red cube", "yellow cube"])
    server_meta = {
        "objects": ("black, metal pen holder", "red cube", "yellow cube"),
        "object_prompt_names": ("totally different wording", "x", "y"),
        "n_objects": 3,
    }
    validate_against_server_metadata(cfg, server_meta)  # must not raise


def test_validate_wrong_count_raises():
    cfg = _task_config(["red cube", "yellow cube"])
    server_meta = {
        "objects": ("black, metal pen holder", "red cube", "yellow cube"),
        "n_objects": 3,
    }
    with pytest.raises(ValueError, match=r"2.*n_objects=3|n_objects=3.*2"):
        validate_against_server_metadata(cfg, server_meta)


def test_validate_wrong_order_raises_naming_mismatch():
    cfg = _task_config(["red cube", "black, metal pen holder", "yellow cube"])
    server_meta = {
        "objects": ("black, metal pen holder", "red cube", "yellow cube"),
        "n_objects": 3,
    }
    with pytest.raises(ValueError) as excinfo:
        validate_against_server_metadata(cfg, server_meta)
    msg = str(excinfo.value)
    # the message should name the actual mismatched categories, not just
    # raise a bare/unlabeled exception
    assert "red cube" in msg
    assert "black, metal pen holder" in msg


def test_validate_wrong_category_raises():
    cfg = _task_config(["black, metal pen holder", "red cube", "BLUE cube"])
    server_meta = {
        "objects": ("black, metal pen holder", "red cube", "yellow cube"),
        "n_objects": 3,
    }
    with pytest.raises(ValueError, match="BLUE cube"):
        validate_against_server_metadata(cfg, server_meta)


def test_validate_non_relation_server_metadata_raises():
    """Old 30-dim checkpoint metadata has no n_objects/objects at all."""
    cfg = _task_config(["red cube"])
    server_meta = {"control_mode": "relative_eef", "hands": ("left", "right")}
    with pytest.raises(ValueError, match="EgoRelationTrainConfig"):
        validate_against_server_metadata(cfg, server_meta)
