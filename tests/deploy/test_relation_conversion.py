"""The relation_eef action-mode conversion: (H, 14) rotvec chunks -> (H, 26) joints.

Two halves:
  * fast, synthetic, precise (no dataset, no train-group dependency) --
    identity chunk stays at the anchor, a known pure-translation delta moves
    the flange by exactly that amount, gripper interpolation is exact.
  * real-dataset ground truth: build the EXACT (H, 14) chunk
    `RelativeEEFRotvecActions` would have produced for a real recorded
    episode window, feed it through `RelativeEEFRotvecChunks.convert`, and
    check the resulting joint trajectory tracks tightly and stays smooth --
    mirrors tests/test_deploy_conversion.py's accel-RMS-style check for the
    older relative_eef mode.
"""

import numpy as np
import pytest

from ego2g1.core import layout, relation_layout

FPS = 30


@pytest.fixture(scope="module")
def kin_smooth():
    from ego2g1.deploy.kinematics import Kinematics
    return Kinematics(ik_iters=25, fps=FPS, posture_cost=0.05)


def _nominal_q0():
    """The same comfortable, reachable posture tests/test_deploy_conversion.py
    uses as its synthetic anchor: shoulders clear of the torso, elbows bent."""
    q0 = np.zeros(layout.ARM_DOF)
    q0[1], q0[8] = 0.15, -0.15
    q0[3], q0[10] = 0.3, 0.3
    return q0


def _accel_rms_worst(q, fps=FPS):
    a = np.diff(np.asarray(q), n=2, axis=0) * fps * fps
    return float(np.sqrt((a ** 2).mean(axis=0)).max())


# --------------------------------------------------------------------------
# fast, synthetic, no dataset needed
# --------------------------------------------------------------------------


def test_identity_chunk_stays_at_anchor(kin_smooth):
    from ego2g1.deploy.actions import ARM, RelativeEEFRotvecChunks

    conv = RelativeEEFRotvecChunks(kin=kin_smooth, fps=FPS)
    conv.reset()
    q0 = _nominal_q0()
    chunk = np.zeros((10, relation_layout.ACTION_DIM))
    chunk[:, relation_layout.GRIP["left"]] = -1.0    # open
    chunk[:, relation_layout.GRIP["right"]] = -1.0
    out = conv.convert(chunk, q0, {h: np.zeros(6) for h in layout.HANDS})
    anchor = kin_smooth.flange_poses(q0)
    fk = kin_smooth.flange_poses(out[-1, ARM])
    for h in layout.HANDS:
        assert np.linalg.norm(fk[h][:3, 3] - anchor[h][:3, 3]) < 0.01
    assert conv.last_tracking_error < 0.005


def test_pure_translation_moves_flange_by_expected_amount(kin_smooth):
    """A constant +5cm local-X delta on the left hand only, held for 1s (30
    slots, letting the One-Euro/JointFilter transient settle): the final
    flange position must land at anchor + anchor_R @ [0.05, 0, 0], and the
    untouched right hand must stay at its own anchor."""
    from ego2g1.deploy.actions import ARM, HAND, RelativeEEFRotvecChunks

    conv = RelativeEEFRotvecChunks(kin=kin_smooth, fps=FPS)
    conv.reset()
    q0 = _nominal_q0()
    H = 30
    chunk = np.zeros((H, relation_layout.ACTION_DIM))
    dx = 0.05
    chunk[:, relation_layout.EEF6["left"]] = np.array([dx, 0.0, 0.0, 0.0, 0.0, 0.0])
    chunk[:, relation_layout.GRIP["left"]] = -1.0
    chunk[:, relation_layout.GRIP["right"]] = -1.0

    out = conv.convert(chunk, q0, {h: np.zeros(6) for h in layout.HANDS})
    anchor = kin_smooth.flange_poses(q0)
    fk_last = kin_smooth.flange_poses(out[-1, ARM])

    expected_left = anchor["left"][:3, 3] + anchor["left"][:3, :3] @ np.array([dx, 0.0, 0.0])
    assert np.linalg.norm(fk_last["left"][:3, 3] - expected_left) < 0.003
    assert np.linalg.norm(fk_last["right"][:3, 3] - anchor["right"][:3, 3]) < 0.003
    # open gripper -> zero motor command on both hands (26-dim executor-row
    # HAND slices from actions.py, NOT core.layout.HAND -- that dict slices
    # the old, incompatible 30-dim state/action vector)
    np.testing.assert_allclose(out[-1, HAND["left"]], 0.0, atol=1e-9)
    np.testing.assert_allclose(out[-1, HAND["right"]], 0.0, atol=1e-9)


def test_gripper_interpolation_open_mid_closed(kin_smooth):
    """raw model-space grip in {-1, 0, +1} -> frac {0, 0.5, 1} * closed_pose.
    Uses an explicit, non-default closed_pose so the test also exercises the
    "editing gripper_calib.py is the only required change" contract."""
    from ego2g1.deploy.actions import HAND, RelativeEEFRotvecChunks

    closed_pose = {"left": np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.5], dtype=np.float32),
                   "right": np.ones(6, dtype=np.float32)}
    conv = RelativeEEFRotvecChunks(kin=kin_smooth, fps=FPS, closed_pose=closed_pose)
    conv.reset()
    q0 = _nominal_q0()
    chunk = np.zeros((3, relation_layout.ACTION_DIM))
    chunk[0, relation_layout.GRIP["left"]] = -1.0   # fully open -> frac 0
    chunk[1, relation_layout.GRIP["left"]] = 0.0    # half -> frac 0.5
    chunk[2, relation_layout.GRIP["left"]] = 1.0    # fully closed -> frac 1
    chunk[:, relation_layout.GRIP["right"]] = -1.0

    out = conv.convert(chunk, q0, {h: np.zeros(6) for h in layout.HANDS})
    np.testing.assert_allclose(out[0, HAND["left"]], 0.0 * closed_pose["left"], atol=1e-6)
    np.testing.assert_allclose(out[1, HAND["left"]], 0.5 * closed_pose["left"], atol=1e-6)
    np.testing.assert_allclose(out[2, HAND["left"]], 1.0 * closed_pose["left"], atol=1e-6)


def test_default_closed_pose_is_the_gripper_calib_placeholder(kin_smooth):
    from ego2g1.deploy import gripper_calib
    from ego2g1.deploy.actions import RelativeEEFRotvecChunks

    conv = RelativeEEFRotvecChunks(kin=kin_smooth, fps=FPS)   # no closed_pose override
    for h in layout.HANDS:
        np.testing.assert_array_equal(conv.closed_pose[h], gripper_calib.BRAINCO_CLOSED_POSE[h])
        np.testing.assert_array_equal(conv.closed_pose[h], np.ones(6, dtype=np.float32))


def test_bad_shape_and_nonfinite_are_refused(kin_smooth):
    from ego2g1.deploy.actions import RelativeEEFRotvecChunks

    conv = RelativeEEFRotvecChunks(kin=kin_smooth, fps=FPS)
    q0 = _nominal_q0()
    hand_cmds = {h: np.zeros(6) for h in layout.HANDS}
    with pytest.raises(ValueError):
        conv.convert(np.zeros((5, 13)), q0, hand_cmds)   # wrong width (not 14)
    with pytest.raises(ValueError):
        bad = np.zeros((5, relation_layout.ACTION_DIM))
        bad[0, 0] = np.nan
        conv.convert(bad, q0, hand_cmds)


def test_make_converter_and_make_adapter_dispatch_relation_eef(kin_smooth):
    from ego2g1.deploy.actions import RelativeEEFRotvecChunks, make_converter
    from ego2g1.deploy.policy_adapter import RelationPolicyAdapter, make_adapter

    conv = make_converter("relation_eef", kin=kin_smooth, fps=FPS)
    assert isinstance(conv, RelativeEEFRotvecChunks)
    assert conv.mode == "relation_eef"

    class _FakeClient:
        action_horizon = 6
        fps = FPS
        control_mode = "relation_eef"

        def infer(self, image, state, prompt, **kw):
            raise AssertionError("not called in this test")

    adapter = make_adapter("relation_eef", _FakeClient(), "task", kin=kin_smooth)
    assert isinstance(adapter, RelationPolicyAdapter)
    assert adapter.mode == "relation_eef"


class _FakeRelationClient:
    """Minimal PolicyClient stand-in: records what it was called with, and
    returns a fixed all-zero-delta (H, 14) chunk (identity EEF, open grippers)
    so downstream conversion can be checked precisely."""

    def __init__(self, horizon=5, control_mode="relation_eef"):
        self.action_horizon = horizon
        self.fps = FPS
        self.control_mode = control_mode
        self.last_call = None

    def infer(self, image, state, prompt, *, prev_chunk=None, d=0, n_prefix=None):
        self.last_call = {"image": image, "state": np.asarray(state), "prompt": prompt,
                          "prev_chunk": prev_chunk, "d": d, "n_prefix": n_prefix}
        chunk = np.zeros((self.action_horizon, relation_layout.ACTION_DIM), dtype=np.float32)
        chunk[:, relation_layout.GRIP["left"]] = -1.0
        chunk[:, relation_layout.GRIP["right"]] = -1.0
        return {"actions": chunk}


def test_relation_policy_adapter_infer_passes_relation_state_and_converts(kin_smooth):
    from ego2g1.deploy.actions import ARM, ROBOT_DIM
    from ego2g1.deploy.policy_adapter import RelationPolicyAdapter

    client = _FakeRelationClient()
    adapter = RelationPolicyAdapter(client, "task", kin=kin_smooth)
    q0 = _nominal_q0()
    hand_cmds = {h: np.zeros(6) for h in layout.HANDS}
    state = np.arange(relation_layout.RELATION_STATE_DIM, dtype=np.float32)

    out = adapter.infer({"arm_q": q0, "hand_cmds": hand_cmds, "image": None,
                         "prompt": "task", "relation_state": state})

    assert out["actions"].shape == (client.action_horizon, ROBOT_DIM)
    # relation state passed through byte-for-byte -- this adapter is a
    # thin pass-through, it must not transform it in any way
    np.testing.assert_array_equal(client.last_call["state"], state)
    assert client.last_call["prompt"] == "task"
    assert adapter.last_tracking_error == pytest.approx(adapter._converter.last_tracking_error)
    # identity-delta, open-gripper chunk -> stays at anchor, hands at 0
    anchor = kin_smooth.flange_poses(q0)
    fk = kin_smooth.flange_poses(out["actions"][-1, ARM])
    for h in layout.HANDS:
        assert np.linalg.norm(fk[h][:3, 3] - anchor[h][:3, 3]) < 0.01


def test_relation_policy_adapter_rejects_wrong_state_shape(kin_smooth):
    from ego2g1.deploy.policy_adapter import RelationPolicyAdapter

    adapter = RelationPolicyAdapter(_FakeRelationClient(), "task", kin=kin_smooth)
    q0 = _nominal_q0()
    hand_cmds = {h: np.zeros(6) for h in layout.HANDS}
    with pytest.raises(ValueError):
        adapter.infer({"arm_q": q0, "hand_cmds": hand_cmds, "image": None,
                       "prompt": "task", "relation_state": np.zeros(30)})


def test_relation_policy_adapter_rejects_rtc(kin_smooth):
    from ego2g1.deploy.policy_adapter import RelationPolicyAdapter

    adapter = RelationPolicyAdapter(_FakeRelationClient(), "task", kin=kin_smooth)
    q0 = _nominal_q0()
    hand_cmds = {h: np.zeros(6) for h in layout.HANDS}
    state = np.zeros(relation_layout.RELATION_STATE_DIM, dtype=np.float32)
    with pytest.raises(NotImplementedError):
        adapter.infer({"arm_q": q0, "hand_cmds": hand_cmds, "image": None,
                       "prompt": "task", "relation_state": state,
                       "enable_rtc": True, "prev_action_chunk": np.zeros((2, 26))})


# --------------------------------------------------------------------------
# real-dataset ground truth
# --------------------------------------------------------------------------

_REPO_ID_PARTS = ("ego2g1", "red_block_in_pen_holder_ego")
_ANCHORS = [
    ("episode_000000.parquet", 0),
    ("episode_000000.parquet", 132),
    ("episode_000001.parquet", 0),
    ("episode_000002.parquet", 0),
]
_H = 50   # EgoRelationTrainConfig.action_horizon


def _dataset_root():
    from ego2g1.core.paths import data_dir
    return data_dir() / "lerobot_datasets" / _REPO_ID_PARTS[0] / _REPO_ID_PARTS[1]


def _require_dataset():
    root = _dataset_root()
    if not (root / "data" / "chunk-000").exists():
        pytest.skip(f"real dataset not present at {root} (see docs/datasets.md)")
    return root


@pytest.fixture(scope="module")
def relation_actions_transform():
    from ego2g1.train.relation_transforms import RelativeEEFRotvecActions
    return RelativeEEFRotvecActions(hands=("left", "right"))


def _ground_truth_chunk(root, filename, t, transform, horizon=_H):
    """The EXACT (H, 14) chunk RelativeEEFRotvecActions builds for this real
    episode window, read parquet-only (mirrors
    ego2g1.train.dataset.relation_raw_action_chunks)."""
    import pandas as pd

    path = root / "data" / "chunk-000" / filename
    df = pd.read_parquet(path, columns=["action", "observation.action_reference_tcp"])
    act = np.stack(df["action"].to_numpy()).astype(np.float64)
    ref = np.stack(df["observation.action_reference_tcp"].to_numpy()).astype(np.float64)
    window = act[t:t + horizon]
    if len(window) < horizon:   # terminal repeat-pad, same rule as training's stats builder
        pad = np.repeat(window[-1:], horizon - len(window), axis=0)
        window = np.concatenate([window, pad], axis=0)
    out = transform({"action": window, "observation/action_reference_tcp": ref[t]})
    return out["actions"]


@pytest.mark.parametrize("filename,t", _ANCHORS)
def test_real_episode_ground_truth_chunk_tracks_tightly_and_smoothly(
    kin_smooth, relation_actions_transform, filename, t
):
    from ego2g1.deploy.actions import ARM, RelativeEEFRotvecChunks

    root = _require_dataset()
    gt_chunk = _ground_truth_chunk(root, filename, t, relation_actions_transform)
    assert gt_chunk.shape == (_H, relation_layout.ACTION_DIM)
    assert np.all(np.isfinite(gt_chunk))

    # a fixed, comfortable, FK-reachable synthetic anchor -- the ground-truth
    # deltas are relative quantities (docs/relation_deploy_plan.md §0's
    # "guiding fact"), so composing them onto ANY reachable anchor should
    # reproduce poses close to what real-robot execution would trace, as long
    # as the anchor is itself a sane posture (this one is verified, at these
    # specific anchors, to keep every target inside the reachable envelope --
    # see the tolerances below).
    q0 = _nominal_q0()
    conv = RelativeEEFRotvecChunks(kin=kin_smooth, fps=FPS)
    conv.reset()
    hand_cmds = {h: np.zeros(6) for h in layout.HANDS}
    out = conv.convert(gt_chunk, q0, hand_cmds)
    assert out.shape == (_H, 26)

    # IK tracking error: worst per-slot flange error, metres. Budget with
    # generous margin over the measured range on this dataset (max ~3 mm at
    # these anchors; see the scratch calibration in the task notes).
    assert conv.last_slot_errors.mean() < 0.006, (
        f"mean tracking error {conv.last_slot_errors.mean()*1000:.2f} mm too high")
    assert conv.last_tracking_error < 0.015, (
        f"max tracking error {conv.last_tracking_error*1000:.2f} mm too high")

    # smoothness: no absurd accel spikes (mirrors test_deploy_conversion.py's
    # accel-RMS-style check for the old relative_eef mode).
    rms = _accel_rms_worst(out[:, ARM])
    assert rms < 15.0, f"worst-joint accel RMS {rms:.1f} rad/s^2 too high"


def test_real_episode_gripper_dims_decode_to_binary_extremes(
    relation_actions_transform,
):
    """Ground-truth grip dims are exactly {-1, +1} (RelativeEEFRotvecActions:
    grip = stored_binary * 2 - 1), so the frac map must land on exactly {0, 1}
    modulo the closed_pose scale -- this is the wire-format assumption
    RelativeEEFRotvecChunks.convert relies on, checked against real data
    rather than only against hand-built synthetic rows."""
    root = _require_dataset()
    gt_chunk = _ground_truth_chunk(root, "episode_000000.parquet", 0, relation_actions_transform)
    grip = gt_chunk[:, 12:14]
    uniq = set(np.round(grip.reshape(-1), 6).tolist())
    assert uniq <= {-1.0, 1.0}, f"unexpected gripper values on real data: {uniq}"
