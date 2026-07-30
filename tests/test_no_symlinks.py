"""This repo must stand alone: no symlink may point anywhere, ever.

The v1 monorepo grew symlinks into sibling checkouts (the last one:
data/put_bottle_in_box_ego -> ../data/...); each breaks the moment the repo is
cloned on its own. Raw episodes live OUTSIDE the repo and are reached through
PipelineConfig.episodes_dir (ego2g1/data/config.py) — or copied under data/
(git-ignored). Never symlinked.
"""

import pathlib

# Trees we don't own (or that tooling generates) are exempt.
_SKIP = {".git", ".venv", "third_party", "__pycache__", "node_modules"}


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def test_no_symlinks_anywhere():
    offenders = []
    stack = [repo_root()]
    while stack:
        d = stack.pop()
        for p in d.iterdir():
            if p.name in _SKIP:
                continue
            if p.is_symlink():
                offenders.append(p)
            elif p.is_dir():
                stack.append(p)
    assert not offenders, (
        "symlinks are forbidden (this repo must clone standalone); found:\n  "
        + "\n  ".join(str(p) for p in offenders)
        + "\nraw data belongs outside the repo — point episodes_dir at it "
          "(ego2g1/data/config.py) or copy it under data/."
    )
