"""relation_layout's slices must agree with EgoRelationTrainConfig, provably.

Not eyeballed: this imports the real training config class and checks its
`gripper_dims`/`action_dim_actual`/`state_dim`+2 properties against the
layout module's constants and slices.
"""

import numpy as np

from ego2g1.core import relation_layout as rl


def _config():
    from ego2g1.train.config import EgoRelationTrainConfig
    return EgoRelationTrainConfig()


def test_action_dim_matches_config():
    cfg = _config()
    assert rl.ACTION_DIM == cfg.action_dim_actual == 14


def test_gripper_slices_match_config_gripper_dims():
    cfg = _config()
    expected = set(cfg.gripper_dims)
    got = set()
    for h in rl.HANDS:
        got.update(range(rl.GRIP[h].start, rl.GRIP[h].stop))
    assert got == expected, (got, expected)


def test_eef6_and_grip_slices_partition_the_action_vector_exactly():
    covered = np.zeros(rl.ACTION_DIM, dtype=bool)
    for h in rl.HANDS:
        for sl in (rl.EEF6[h], rl.GRIP[h]):
            assert not covered[sl].any(), f"overlap at hand {h} slice {sl}"
            covered[sl] = True
    assert covered.all(), "action layout has uncovered dims"


def test_grip_dims_are_at_the_tail_in_hands_order():
    cfg = _config()
    # EgoRelationTrainConfig.gripper_dims is a contiguous tail range; relation_layout
    # must place hands onto it in the same HANDS order the config assumes.
    assert list(cfg.gripper_dims) == list(range(12, 14))
    assert rl.GRIP["left"] == slice(12, 13)
    assert rl.GRIP["right"] == slice(13, 14)


def test_relation_state_dim_matches_config_state_dim_plus_grasp_binaries():
    cfg = _config()
    # RelationPrompt's `state` output holds the grasp binaries only (n_hands,);
    # the relation_dim*n_objects part is `relations`, kept separate. The wire
    # format RelationPolicyAdapter's caller must build concatenates both, so
    # relation_layout.RELATION_STATE_DIM is state_dim + n_hands.
    assert rl.RELATION_STATE_DIM == cfg.state_dim + len(cfg.hands)
    assert rl.RELATION_STATE_DIM == 56
    assert rl.N_OBJECTS == cfg.n_objects
    assert rl.RELATION_DIM_PER_OBJECT_PER_HAND * rl.N_HANDS == cfg.relation_dim


def test_split_action_row_roundtrips():
    row = np.arange(14, dtype=np.float64)
    parts = rl.split_action_row(row)
    for h in rl.HANDS:
        np.testing.assert_array_equal(parts[h]["eef6"], row[rl.EEF6[h]])
        np.testing.assert_array_equal(parts[h]["grip"], row[rl.GRIP[h]])
    rebuilt = np.concatenate(
        [row[rl.EEF6["left"]], row[rl.EEF6["right"]],
         row[rl.GRIP["left"]], row[rl.GRIP["right"]]]
    )
    np.testing.assert_array_equal(rebuilt, row)
