# Datasets: naming, the three knobs, regeneration

## Encode the data source in the name

We now have two ways to produce episodes for the same task — egocentric human
recordings (Pico hand tracker) and robot teleop (`tools/teleop`) — and a dataset that
doesn't say which it is will eventually be trained on as the wrong thing. Policy:
**the source goes in the `repo_id`**:

```
put_bottle_in_box_ego       # egocentric human recordings
put_bottle_in_box_teleop    # robot teleop recordings
```

The extraction `repo_id` **names the output folder** (`data/lerobot_datasets/<repo_id>`),
so naming the repo_id names the dataset on disk — there is no second rename step.
History note: the original `put_bottle_in_box` (old repo) was ego-only and was *not*
renamed, to keep existing checkpoints resolving their norm stats; v2 raw data already
follows the policy (`data/put_bottle_in_box_ego`).

## Three name knobs, deliberately distinct

They were all the string `put_bottle_in_box` by convention only, which hid the fact
that they are independent. Keep them distinct in your head:

| knob | lives in | names | when to change |
|---|---|---|---|
| extraction `repo_id` | `ego2g1/data/config.py` | the output dataset folder | per extraction — encode `_ego`/`_teleop` |
| raw episodes dir | top-level `data/` | `source_episode` prefixes baked into `extraction_meta.json` (what `val_real_episodes` must match) | with the raw recording batch |
| training-config `repo_id` | `ego2g1/train/config.py` | the **norm-stats assets dir** (`assets/<config>/<repo_id>/`) | almost never — **existing checkpoints depend on it** |

The trap is the third knob: a checkpoint resolves its normalization stats by the
training config's `repo_id`, so "renaming the dataset" there orphans every existing
checkpoint. Re-extracting under the same training `repo_id` overwrites norm stats —
safe only if raw data and config are byte-identical.

## Where data lives

Raw recordings and generated datasets sit at top-level `data/` (git-ignored):

```
data/put_bottle_in_box_ego/        # raw Pico HDF5 episodes
data/lerobot_datasets/<repo_id>/   # generated LeRobot datasets
```

## Regeneration

Datasets are cheap to regenerate and expensive to debug — when in doubt (and always
after a `datasets`-version accident, see [environments.md](environments.md)),
regenerate rather than re-read:

```bash
# full pipeline: raw HDF5 -> LeRobot dataset (folder named by repo_id)
uv run python -m ego2g1.data.run_pipeline --set episodes_dir=$PWD/data/put_bottle_in_box_ego

# just the shared wrist->flange calibration used by teleop (--B)
uv run python -m ego2g1.data.run_pipeline --through b_calib --set episodes_dir=$PWD/data/put_bottle_in_box_ego
```

(Stage names and overrides: `uv run python -m ego2g1.data.run_pipeline --help`;
pipeline details are the data port's `docs/data.md`.)
