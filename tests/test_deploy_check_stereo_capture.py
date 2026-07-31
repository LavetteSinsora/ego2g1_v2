"""`_next_pair_index` (ego2g1.deploy.check): auto-numbering for
`stereo-capture`'s left_NNN.png/right_NNN.png pairs (docs/relation_deploy_plan.md
§9 task 6b). Pure filesystem-glob logic -- no camera/robot needed."""

from ego2g1.deploy.check import _next_pair_index


def test_empty_dir_starts_at_zero(tmp_path):
    assert _next_pair_index(tmp_path) == 0


def test_picks_one_past_the_highest_existing_index(tmp_path):
    (tmp_path / "left_000.png").touch()
    (tmp_path / "right_000.png").touch()
    (tmp_path / "left_001.png").touch()
    (tmp_path / "right_001.png").touch()
    assert _next_pair_index(tmp_path) == 2


def test_fills_a_gap_from_a_manually_deleted_pair(tmp_path):
    (tmp_path / "left_000.png").touch()
    (tmp_path / "right_000.png").touch()
    # pair 1 deleted by hand; pair 2 still present
    (tmp_path / "left_002.png").touch()
    (tmp_path / "right_002.png").touch()
    assert _next_pair_index(tmp_path) == 1


def test_ignores_unrelated_files(tmp_path):
    (tmp_path / "left_000.png").touch()
    (tmp_path / "right_000.png").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "checkerboard_9x6_20mm.png").touch()  # no trailing _NNN
    assert _next_pair_index(tmp_path) == 1


def test_three_digit_zero_padding_still_sorts_numerically_not_lexically(tmp_path):
    for i in range(11):
        (tmp_path / f"left_{i:03d}.png").touch()
        (tmp_path / f"right_{i:03d}.png").touch()
    assert _next_pair_index(tmp_path) == 11
