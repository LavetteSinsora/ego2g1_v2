"""s004: per-tick quality filters -> sub-episodes.

Every filter yields a (T,) bool mask (True = bad tick). Interior bad runs of
<= cfg.bridge_max_ticks are BRIDGED (the recording stays whole across short
blips: the flagged ticks' stored poses may appear inside action windows, but
they are excluded as datapoint anchors via the anchor_bad mask). The
remaining combined mask splits the recording into contiguous good runs; runs
shorter than cfg.min_subepisode_ticks (anchor + one full chunk) are dropped;
the run that contains the recording's final tick is marked episode_real_end
(only there is repeat-padding of action chunks semantically "hold pose" -
everywhere else the human kept moving and padding would be a lie).

Filters and their signal sources:
- gap/tracking-lost: s001 valid masks (bracketing gap > max_gap_ms, hand inactive)
- camera staleness:  s001 cam_gap_ms > cam_stale_ms
- wrist velocity:    finite-diff on the s001 grid, only across valid-valid
                     pairs (fills around invalid ticks would fake spikes)
- IK tracking error: s003_state per-tick error (inert for proprio_source=direct)
- self-penetration:  s003_state per-tick min self-clearance < min_self_clearance_m
                     (safety net behind the IK collision limit; inert for direct)
- hand blocked / non-pinch self-collision: dynamic Revo2 replay (hand/screen.py)
- hand retarget residual: s002_02, sustained >= hand_residual_min_run ticks
"""

import numpy as np

from ...core import frames
from .. import io


def _sustained(mask, min_run):
    """Keep only ticks inside a run of >= min_run consecutive True."""
    out = np.zeros_like(mask)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= min_run:
            out[i - run + 1:i + 1] = True
    return out


def _residual_bad(residual_m, cols, max_mm, hand_valid):
    """True where a gated finger's residual exceeds max_mm; an empty finger
    selection (residual filter disabled) flags nothing."""
    if not cols:
        return np.zeros(len(residual_m), dtype=bool)
    return (residual_m[:, cols].max(axis=1) * 1000 > max_mm) & hand_valid


def _velocity_masks(cfg, pos, quat, valid):
    T = len(pos)
    bad = np.zeros(T, dtype=bool)
    dt = 1.0 / cfg.control_hz
    for k in range(1, T):
        if not (valid[k] and valid[k - 1]):
            continue
        v = np.linalg.norm(pos[k] - pos[k - 1]) / dt
        w = frames.rot_geodesic_deg(frames.mat_from_quat(quat[k - 1]),
                                    frames.mat_from_quat(quat[k])) / dt
        if v > cfg.vel_max_m_s or w > cfg.ang_vel_max_deg_s:
            bad[k] = bad[k - 1] = True
    return bad


def _bridge_short_runs(bad, max_ticks):
    """Interior bad runs of <= max_ticks become good-for-splitting; returns
    (bad_after_bridging, bridged_mask). A run touching tick 0 or T-1 is an
    edge (plain trim), never bridged."""
    bad = bad.copy()
    bridged = np.zeros_like(bad)
    if max_ticks <= 0:
        return bad, bridged
    T = len(bad)
    k = 0
    while k < T:
        if bad[k]:
            j = k
            while j < T and bad[j]:
                j += 1
            if k > 0 and j < T and j - k <= max_ticks:
                bad[k:j] = False
                bridged[k:j] = True
            k = j
        else:
            k += 1
    return bad, bridged


def _split_runs(good, min_len, T):
    """-> (starts, ends_exclusive, real_end flags)"""
    starts, ends = [], []
    k = 0
    while k < T:
        if good[k]:
            j = k
            while j < T and good[j]:
                j += 1
            if j - k >= min_len:
                starts.append(k)
                ends.append(j)
            k = j
        else:
            k += 1
    real_end = [e == T for e in ends]
    return (np.array(starts, dtype=np.int32), np.array(ends, dtype=np.int32),
            np.array(real_end, dtype=bool))


def run_episode(cfg, ep_path):
    from ...core.hand.screen import HandSim, blocked_mask

    ep = ep_path.stem
    s001, _ = io.load_stage(cfg, ep, "s001")
    s003, _ = io.load_stage(cfg, ep, "s003_state")
    s002h, _ = io.load_stage(cfg, ep, "s002_02")
    T = len(s001["ticks_ns"])

    masks = {"bad_cam": s001["cam_gap_ms"] > cfg.cam_stale_ms}
    for side, pre in (("left", "l"), ("right", "r")):
        valid = s001[f"{pre}_valid"].astype(bool)
        masks[f"bad_gap_{pre}"] = ~valid
        masks[f"bad_vel_{pre}"] = _velocity_masks(
            cfg, s001[f"{pre}_pos"], s001[f"{pre}_quat"], valid)
        masks[f"bad_ik_{pre}"] = ((s003[f"ik_pos_cm_{pre}"] > cfg.ik_err_max_cm)
                                  | (s003[f"ik_ori_deg_{pre}"] > cfg.ik_err_max_deg)) & valid
        # self-penetration of the achieved configuration: physically
        # impossible at deployment, so any such tick is unconditionally bad.
        # With the IK collision limit on this is a safety net that should
        # never fire; legacy s003 outputs without the signal pass vacuously.
        clear_key = f"self_clear_m_{pre}"
        masks[f"bad_clear_{pre}"] = (
            (s003[clear_key] < cfg.min_self_clearance_m) & valid
            if clear_key in s003 else np.zeros(T, dtype=bool))

        hand_valid = s002h[f"hand_valid_{pre}"].astype(bool)
        if cfg.hand_blocked_filter or cfg.hand_contact_filter:
            sim = HandSim(side)
            err, _pen, _pairs, other_contact = sim.replay(
                s002h[f"hand_cmds_{pre}"], s001["ticks_ns"])
            masks[f"bad_hand_blocked_{pre}"] = (
                blocked_mask(err, s002h[f"hand_cmds_{pre}"])
                if cfg.hand_blocked_filter else np.zeros(T, dtype=bool))
            masks[f"bad_hand_contact_{pre}"] = (
                other_contact if cfg.hand_contact_filter else np.zeros(T, dtype=bool))
        else:
            masks[f"bad_hand_blocked_{pre}"] = np.zeros(T, dtype=bool)
            masks[f"bad_hand_contact_{pre}"] = np.zeros(T, dtype=bool)

        finger_order = ("thumb", "index", "middle", "ring", "pinky")
        cols = [finger_order.index(f) for f in cfg.hand_residual_fingers]
        res_bad = _residual_bad(s002h[f"hand_residual_{pre}"], cols,
                                cfg.hand_residual_max_mm, hand_valid)
        masks[f"bad_hand_residual_{pre}"] = _sustained(res_bad, cfg.hand_residual_min_run)

    per_side_filters = ("bad_gap", "bad_vel", "bad_ik", "bad_clear",
                        "bad_hand_blocked", "bad_hand_contact",
                        "bad_hand_residual")
    sides = ([s[0] for s in cfg.hands] if cfg.filter_hands_independently
             else ["l", "r"])
    bad_any = masks["bad_cam"].copy()
    for f in per_side_filters:
        for pre in sides:
            bad_any |= masks[f"{f}_{pre}"]
    masks["bad_any"] = bad_any

    bad_split, bridged = _bridge_short_runs(bad_any, cfg.bridge_max_ticks)
    starts, ends, real_end = _split_runs(~bad_split, cfg.min_subepisode_ticks, T)
    arrays = {**{k: v.astype(bool) for k, v in masks.items()},
              "anchor_bad": bridged.astype(bool),
              "subep_start": starts, "subep_end": ends, "subep_real_end": real_end}
    meta = {
        "ticks_total": int(T),
        "ticks_kept": int(sum(int(e - s) for s, e in zip(starts, ends))),
        "ticks_bridged": int(bridged.sum()),
        "n_subepisodes": int(len(starts)),
        "subepisodes": [{"start": int(s), "end": int(e), "real_end": bool(r)}
                        for s, e, r in zip(starts, ends, real_end)],
        "bad_counts": {k: int(v.sum()) for k, v in masks.items()},
    }
    print(f"  [{ep}] kept {meta['ticks_kept']}/{T} ticks in "
          f"{meta['n_subepisodes']} sub-episode(s), {meta['ticks_bridged']} bridged; "
          f"bad: {({k: c for k, c in meta['bad_counts'].items() if c} or 'none')}")
    return arrays, meta
