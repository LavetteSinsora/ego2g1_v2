"""Does Orient Anything V2 drift on a STATIC object? (plan §5.2, H3)

    uv run --group perception-v2 python -m \
        ego2g1.deploy.perception_v2_orient_stability --image penholder.png

    # the truthful version — a real stationary object, real sensor noise
    uv run --group perception-v2 python -m \
        ego2g1.deploy.perception_v2_orient_stability --camera-frames 300

THE QUESTION IS THREE QUESTIONS, AND ONLY THE THIRD MATTERS
"Show it the same pen holder twice — does the answer change?" has a trivial
answer and a useful one, and they are not the same question:

  A. the SAME TENSOR, forwarded twice. `VGGT_OriAny_Ref` is feedforward, in
     eval mode, argmaxed — no sampling, no state. This *should* be bit-exact,
     and it is measured only to prove the harness is not itself the noise
     source. A non-zero result here means nondeterministic CUDA kernels, and
     every later number would be uninterpretable.

  B. the same tensor at a DIFFERENT BATCH POSITION / batch size. bf16
     reductions are not associative and kernel selection is shape-dependent,
     so this is not guaranteed zero even though A is. It matters because the
     deploy roster changes size as slots gate in and out
     (`OrientAnythingV2.estimate` batches only the usable crops), so the pen
     holder is at batch index 2 one round and index 0 the next.

  C. a static object seen through a REAL CAMERA. Consecutive frames of a
     motionless object are never identical: sensor noise, auto-exposure, and
     a SAM 3 mask whose boundary breathes by a pixel or two. This is the only
     one that predicts deploy behaviour, and the answer is expected to be
     "yes, it drifts" — the interesting part is WHICH DEGREE OF FREEDOM.

WHY THE DOF DECOMPOSITION IS THE POINT
A pen holder is a body of revolution: its azimuth is *unidentifiable*, because
every azimuth renders the same image. The model has no way to abstain — it
emits an argmax over 360 bins regardless — so whatever asymmetric cue survives
the noise (a scratch, a shading gradient, one bright pixel) decides it.

The decode `R = Rz(ro) @ Rx(el) @ Ry(az)` isolates this perfectly: azimuth is
*exactly* a rotation about the object frame's own y-axis and moves nothing
else, algebraically, for any elevation and roll. So this bench reports the
wobble split two ways rather than as one number:

    barrel axis (object y)   the observable part — is the holder upright or
                             tilted, and which way does it lean
    spin about that axis     the unidentifiable part

A run where the y-axis holds to a degree while the full rotation swings tens
of degrees is not a broken model. It is the model correctly reporting that it
cannot see something that is genuinely not there — and it is the direct
justification for the axial symmetry snap (`perception/orientation.py`), which
freezes that DOF instead of letting it walk into the policy's state vector.

IT ALSO ANSWERS THE ARGMAX QUESTION FOR FREE
Every trial is decoded twice: once by argmax (what `_forward` ships today) and
once by circular expectation over the softmaxed bins. Argmax has a 1° floor
and flips between adjacent bins whenever the distribution is near-bimodal —
pure jitter on a motionless object. If the expectation column is materially
quieter, that is a few lines of free accuracy in both the deploy and the
extraction pipeline.

And the per-bin distributions are reported from a SINGLE forward: an azimuth
distribution that is flat or two-peaked is direct evidence of unidentifiability
that needs no repeats at all.
"""

from __future__ import annotations

import numpy as np

__all__ = ["main"]

_AZ, _EL, _RO = slice(0, 360), slice(360, 540), slice(540, 900)
_EL_OFFSET, _RO_OFFSET = -90.0, -180.0


# ============================================================================
# pure: decoding and circular statistics
# ============================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _circular_mean(deg: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mean direction, in degrees. Plain `np.mean` is wrong on a wrapped axis:
    it puts the average of 359° and 1° at 180°, i.e. exactly opposite."""
    r = np.radians(np.asarray(deg, dtype=np.float64))
    w = np.ones_like(r) if weights is None else np.asarray(weights, np.float64)
    return float(np.degrees(np.arctan2((w * np.sin(r)).sum(),
                                       (w * np.cos(r)).sum())))


def _circular_spread(deg: np.ndarray) -> tuple[float, float]:
    """(circular std, max deviation from the mean) in degrees.

    The std comes from the resultant length R as sqrt(-2 ln R), which diverges
    as R -> 0. Clamped at 180°, because past that point the number stops
    carrying information: a scatter that reaches 180° is indistinguishable
    from uniform, and uniform is exactly the finding — "this DOF is not
    determined by the image". A larger figure would only report how close the
    finite sample happened to land to a perfect cancellation.
    """
    deg = np.asarray(deg, dtype=np.float64)
    if deg.size < 2:
        return 0.0, 0.0
    r = np.radians(deg)
    resultant = np.hypot(np.cos(r).mean(), np.sin(r).mean())
    # `-2 * log(1.0)` is -0.0, and sqrt(-0.0) is -0.0 — which prints as
    # "-0.00°" for a perfectly constant angle and reads like a bug.
    std = float(np.degrees(np.sqrt(abs(-2.0 * np.log(max(resultant, 1e-12))))))
    dev = np.abs((deg - _circular_mean(deg) + 180.0) % 360.0 - 180.0)
    return min(std, 180.0), float(dev.max())


def _wrap180(d: np.ndarray) -> np.ndarray:
    """Signed difference on a wrapped axis. 359° and 1° are 2° apart, not
    358° — the distinction decides whether a stable azimuth is reported as
    stable."""
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


def decode_argmax(pose: np.ndarray) -> np.ndarray:
    """(N, >=900) logits -> (N, 3) of (az, el, ro) in degrees. Exactly what
    `orientation_v2.OrientAnythingV2._forward` does today, reimplemented in
    numpy so both decodes read off the same logits."""
    return np.stack([
        pose[:, _AZ].argmax(1).astype(np.float64),
        pose[:, _EL].argmax(1).astype(np.float64) + _EL_OFFSET,
        pose[:, _RO].argmax(1).astype(np.float64) + _RO_OFFSET,
    ], axis=1)


def decode_expectation(pose: np.ndarray) -> np.ndarray:
    """The same logits, decoded as a distribution mean instead of a mode.

    Azimuth and roll are CIRCULAR (they wrap), so they get a circular mean —
    a linear average of a distribution straddling the wrap point lands 180°
    from where it belongs. Elevation spans -90..+89 and does not wrap, so a
    plain weighted mean is correct there.
    """
    az_p, el_p, ro_p = (_softmax(pose[:, s]) for s in (_AZ, _EL, _RO))
    az_bins = np.arange(360, dtype=np.float64)
    el_bins = np.arange(180, dtype=np.float64) + _EL_OFFSET
    ro_bins = np.arange(360, dtype=np.float64) + _RO_OFFSET
    # `% 360` on a mean that lands a hair below zero returns 360.0, not 0.0,
    # which then reads as a distinct value in every table below. Fold it.
    az = np.array([_circular_mean(az_bins, p) % 360.0 for p in az_p])
    az[az >= 360.0 - 1e-9] = 0.0
    ro = np.array([_circular_mean(ro_bins, p) for p in ro_p])
    return np.stack([az, el_p @ el_bins, ro], axis=1)


def _entropy_bits(p: np.ndarray) -> float:
    return float(-(p * np.log2(np.clip(p, 1e-12, None))).sum())


# ============================================================================
# pure: how much did the FRAME move, and about which axis
# ============================================================================

def _geodesic_deg(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.degrees(np.arccos(
        np.clip((np.trace(A.T @ B) - 1.0) / 2.0, -1.0, 1.0))))


def axis_split(R_ref: np.ndarray, R: np.ndarray) -> tuple[float, float, float]:
    """(total, spin about the object y-axis, everything else), all degrees.

    `R_ref.T @ R` is the residual rotation expressed in the OBJECT frame, so
    when the only thing that moved is azimuth the residual is exactly Ry(daz)
    and its axis is [0, 1, 0] to machine precision. Projecting the residual's
    rotation vector onto y therefore separates the unidentifiable spin from
    real movement of the barrel axis, with no thresholding and no fitting.
    """
    total = _geodesic_deg(R_ref, R)
    residual = R_ref.T @ R
    # Rotation vector via the standard log map, guarded at theta -> 0 and pi
    # (the cube-symmetry case in `perception/orientation.py` hits the same
    # branch); at those angles the axis is either irrelevant or degenerate.
    theta = np.radians(total)
    if theta < 1e-9:
        return 0.0, 0.0, 0.0
    if abs(theta - np.pi) < 1e-6:
        return total, float("nan"), float("nan")
    axis = np.array([residual[2, 1] - residual[1, 2],
                     residual[0, 2] - residual[2, 0],
                     residual[1, 0] - residual[0, 1]]) / (2.0 * np.sin(theta))
    spin = abs(float(axis[1])) * total
    return total, spin, float(np.sqrt(max(total ** 2 - spin ** 2, 0.0)))


def barrel_axis_deg(R_ref: np.ndarray, R: np.ndarray) -> float:
    """Angle between the two frames' y-axes. For a body of revolution this is
    the ONLY part of the orientation the image actually determines, so it is
    reported separately from the full geodesic distance."""
    y0, y1 = R_ref[:, 1], R[:, 1]
    return float(np.degrees(np.arccos(np.clip(float(y0 @ y1), -1.0, 1.0))))


# ============================================================================
# input: one static image, one mask
# ============================================================================

def _load_image(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: str | None, shape: tuple[int, int]) -> np.ndarray:
    """A mask file (anything non-zero is object), or the whole frame.

    Falling back to the whole frame is a real fallback, not a degenerate one:
    if `--image` is already a tight crop of the pen holder on a plain
    background, the mask adds nothing. It IS wrong for a full scene, where the
    crop would then be the whole table — hence the warning at the call site.
    """
    if path is None:
        return np.ones(shape, dtype=bool)
    from PIL import Image
    m = np.asarray(Image.open(path).convert("L")) > 0
    if m.shape != shape:
        raise ValueError(f"mask is {m.shape}, image is {shape} — they must "
                         f"come from the same frame")
    return m


def _perturb(rgb: np.ndarray, mask: np.ndarray, rng, *, noise_dn: float,
             jitter_px: int, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """One camera-realistic redraw of a MOTIONLESS object.

    Three effects, because these are the three that a static scene actually
    produces and they enter the model by different routes:

      noise_dn   photon/read noise, in 8-bit counts. Enters as pixel values.
      gain       auto-exposure breathing, as a multiplicative factor. Enters
                 as a global intensity change, which a network trained with
                 photometric augmentation should shrug off — worth confirming.
      jitter_px  the SAM 3 mask boundary breathing, which moves the bbox and
                 therefore the crop WINDOW. This one is different in kind: it
                 changes framing and scale, not just pixel values, and
                 orientation is a global-shape judgement.

    This is a stand-in. `--camera-frames` measures the real thing and should
    be preferred whenever the hardware is in front of you; these knobs exist
    so the question is answerable from a saved PNG on a laptop.
    """
    out = rgb.astype(np.float32)
    if gain:
        out *= 1.0 + rng.uniform(-gain, gain)
    if noise_dn:
        out += rng.normal(0.0, noise_dn, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if jitter_px:
        dy, dx = rng.integers(-jitter_px, jitter_px + 1, 2)
        mask = np.roll(mask, (int(dy), int(dx)), axis=(0, 1))
    return out, mask


# ============================================================================
# the model, kept at arm's length so the logits survive
# ============================================================================

class _Head:
    """`OrientAnythingV2` with the argmax removed.

    `_forward` returns three decoded angles and drops the distributions. This
    bench needs the raw 900 logits — the shape of the azimuth distribution is
    half the evidence — so the forward is repeated here rather than the model
    being re-wrapped. Everything upstream of the logits (crop, preprocess,
    dtype, size) comes from the real `OrientAnythingV2` instance, so no
    preprocessing detail can silently differ from deploy.
    """

    def __init__(self, orient):
        import torch
        self._torch = torch
        self.orient = orient

    def logits(self, crops) -> np.ndarray:
        return self.logits_from_tensor(self.tensor(crops))

    def logits_from_tensor(self, batch) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            pose = self.orient.model(batch.unsqueeze(1))
        if pose.ndim == 3:                      # S>1 path; S=1 returns (B, D)
            pose = pose.reshape(pose.shape[0] * pose.shape[1], -1)
        return pose.float().cpu().numpy()

    def tensor(self, crops):
        from ego2g1.deploy.perception.v2.orientation_v2 import preprocess_crops
        return preprocess_crops(crops, self.orient.size).to(
            device=self.orient.model.get_device(), dtype=self.orient.dtype)


def _crop(rgb, mask, orient):
    from ego2g1.deploy.perception.v2.orientation_v2 import crop_from_mask
    c = crop_from_mask(rgb, mask, pad=orient.crop_pad,
                       background=orient.background)
    if c is None:
        raise ValueError("mask produced no usable crop (empty or < 8 px)")
    return c


# ============================================================================
# the three experiments
# ============================================================================

def exp_a_identical(head, crop, n: int) -> dict:
    """Same tensor, N forwards. Proves the harness is not the noise source."""
    batch = head.tensor([crop])
    rows = [head.logits_from_tensor(batch)[0] for _ in range(n)]
    L = np.stack(rows)
    return {"logits": L, "max_logit_delta": float(np.abs(L - L[0]).max()),
            "angles": decode_argmax(L)}


def exp_b_batch_position(head, crop, max_batch: int) -> dict:
    """The same crop at every position of every batch size up to `max_batch`.

    Padding with copies of the same crop is deliberate: it holds the CONTENT
    fixed so any difference is attributable to shape and reduction order
    alone, not to whatever else was in the batch.
    """
    out = []
    for b in range(1, max_batch + 1):
        L = head.logits(([crop] * b))
        for i in range(b):
            out.append((b, i, decode_argmax(L[i:i + 1])[0]))
    return {"rows": out,
            "angles": np.stack([a for _, _, a in out])}


def exp_c_realistic(head, rgb, mask, orient, n: int, rng, *, noise_dn,
                    jitter_px, gain, frames=None) -> dict:
    """The deploy question: a motionless object, redrawn n times.

    `frames`, when given, replaces the synthetic perturbation with real
    consecutive camera frames of the (motionless) object — strictly better
    evidence, since it contains whatever the sensor actually does rather than
    whatever this file guesses it does.
    """
    argmax_rows, expect_rows = [], []
    for i in range(n):
        if frames is not None:
            rgb_i, mask_i = frames[i], mask
        else:
            rgb_i, mask_i = _perturb(rgb, mask, rng, noise_dn=noise_dn,
                                     jitter_px=jitter_px, gain=gain)
        L = head.logits([_crop(rgb_i, mask_i, orient)])
        argmax_rows.append(decode_argmax(L)[0])
        expect_rows.append(decode_expectation(L)[0])
    return {"argmax": np.stack(argmax_rows),
            "expectation": np.stack(expect_rows)}


# ============================================================================
# reporting
# ============================================================================

def _angle_table(name: str, angles: np.ndarray) -> None:
    az, el, ro = angles[:, 0], angles[:, 1], angles[:, 2]
    az_std, az_dev = _circular_spread(az)
    ro_std, ro_dev = _circular_spread(ro)
    print(f"  {name:<14} {'spread (std)':>14} {'max deviation':>15} "
          f"{'distinct':>9}")
    print(f"    azimuth      {az_std:>13.2f}° {az_dev:>14.2f}° "
          f"{len(np.unique(np.round(az, 1))):>9}")
    print(f"    elevation    {el.std():>13.2f}° "
          f"{np.abs(el - el.mean()).max():>14.2f}° "
          f"{len(np.unique(np.round(el, 1))):>9}")
    print(f"    roll         {ro_std:>13.2f}° {ro_dev:>14.2f}° "
          f"{len(np.unique(np.round(ro, 1))):>9}")


def _frame_table(angles: np.ndarray, convention) -> dict:
    """Turn the angle triples into rotations and report where the wobble went.

    The reference is the CIRCULAR MEAN pose, not the first sample — with a
    freely-spinning azimuth the first sample is an arbitrary draw, and
    measuring everything against an arbitrary draw doubles the apparent
    spread.
    """
    from ego2g1.deploy.perception.v2.orientation_v2 import angles_to_matrix
    R = angles_to_matrix(angles[:, 0], angles[:, 1], angles[:, 2],
                         convention=convention)
    R_ref = angles_to_matrix(_circular_mean(angles[:, 0]) % 360.0,
                             float(angles[:, 1].mean()),
                             _circular_mean(angles[:, 2]),
                             convention=convention)
    total, spin, other, barrel = [], [], [], []
    for Ri in R:
        t, s, o = axis_split(R_ref, Ri)
        total.append(t)
        spin.append(s)
        other.append(o)
        barrel.append(barrel_axis_deg(R_ref, Ri))
    total, spin = np.array(total), np.array(spin)
    other, barrel = np.array(other), np.array(barrel)
    print(f"    {'':<24}{'median':>10}{'p95':>10}{'max':>10}")
    for label, v in (("full rotation", total),
                     ("  spin about barrel axis", spin),
                     ("  everything else", other),
                     ("barrel axis direction", barrel)):
        print(f"    {label:<24}{np.nanmedian(v):>9.2f}°{np.nanpercentile(v, 95):>9.2f}°"
              f"{np.nanmax(v):>9.2f}°")
    return {"total": total, "spin": spin, "other": other, "barrel": barrel}


def _distribution_report(pose: np.ndarray) -> dict:
    """One forward's three distributions: how peaked is each DOF?

    This is the cheapest evidence in the file and the only kind that needs no
    repeats at all. A classification head cannot say "I don't know", but it
    can only spread its mass — so an unidentifiable DOF shows up as a flat or
    multi-peaked distribution on a SINGLE forward, before any noise is added.

    Three columns, because peakedness fails in two different ways:

      mass ±10°   how much probability sits near the mode. A confident answer
                  concentrates; a uniform one puts ~6% of 360 bins in that
                  window and no more.
      entropy     quoted against that block's uniform ceiling, because "8.4
                  bits" means nothing until you know azimuth's ceiling is
                  log2(360) = 8.49. At the ceiling, the head is saying as
                  plainly as it can that the image does not fix the angle.
      2nd mode    the best bin at least 20° away from the mode, with its
                  probability relative to the peak. This is the bimodality
                  test, and it needs the 20° exclusion: the runner-up BIN is
                  always the mode's neighbour on any smooth distribution, so
                  a plain top-2 comparison reads 1° every time and detects
                  nothing. A second peak near 1.0 means the model is torn
                  between two poses, which is what makes argmax flip between
                  frames while the underlying distribution barely moves.
    """
    out = {}
    print("\n  Bin distributions from ONE forward "
          "(no repeats — this is about identifiability, not noise):")
    print(f"    {'':<12}{'top-1':>8}{'mass ±10°':>11}{'entropy':>10}"
          f"{'uniform':>9}{'2nd mode':>18}")
    for name, sl in (("azimuth", _AZ), ("elevation", _EL), ("roll", _RO)):
        p = _softmax(pose[sl])
        n = p.size
        mode = int(p.argmax())
        d = np.abs(np.arange(n) - mode)
        if name != "elevation":                        # wrapped axes
            d = np.minimum(d, n - d)
        near = float(p[d <= 10].sum())
        far = np.flatnonzero(d >= 20)
        if far.size:
            second = int(far[p[far].argmax()])
            rel = float(p[second] / max(p[mode], 1e-12))
            second_txt = f"{d[second]:>6.0f}° @ {rel:>6.3f}"
        else:
            second, rel, second_txt = -1, 0.0, "        n/a"
        print(f"    {name:<12}{p[mode]:>8.4f}{near:>11.3f}"
              f"{_entropy_bits(p):>9.2f}b{np.log2(n):>8.2f}b{second_txt:>18}")
        out[name] = {"top1": float(p[mode]), "mass_within_10deg": near,
                     "entropy": _entropy_bits(p), "ceiling": float(np.log2(n)),
                     "second_mode_deg": float(d[second]) if second >= 0 else None,
                     "second_mode_rel": rel}
    return out


def _verdict(a: dict, b: dict, c: dict, frames_were_real: bool) -> None:
    print("\n" + "=" * 72)
    print("VERDICT")

    print("\n  A. same tensor, repeated forwards")
    if a["max_logit_delta"] == 0.0:
        print("     Bit-identical. The model is deterministic, so every number "
              "below\n     is a property of the INPUT, not of the GPU.")
    else:
        print(f"     NOT bit-identical: logits move by up to "
              f"{a['max_logit_delta']:.2e}.\n"
              f"     Nondeterministic kernels are in play. If the decoded "
              f"angles below moved\n     too, the rest of this bench is "
              f"measuring the GPU as well as the image.")

    print("\n  B. same tensor, different batch size/position")
    d_b = b["angles"] - b["angles"][0]
    d_b[:, 0], d_b[:, 2] = _wrap180(d_b[:, 0]), _wrap180(d_b[:, 2])
    spread_b = float(np.abs(d_b).max()) if len(d_b) else 0.0
    if spread_b == 0.0:
        print("     Identical at every batch position. Roster size changing "
              "round to round\n     does not perturb the answer.")
    else:
        print(f"     Moves by up to {spread_b:.1f}° purely from batch shape. "
              f"This is real at\n     deploy — the batch contains only the "
              f"currently-usable crops, so its size\n     changes as slots "
              f"gate in and out.")

    src = "real camera frames" if frames_were_real else "synthetic perturbation"
    print(f"\n  C. static object, {src}")
    az_std, _ = _circular_spread(c["argmax"][:, 0])
    el_std = float(c["argmax"][:, 1].std())
    ro_std, _ = _circular_spread(c["argmax"][:, 2])
    spin_med = float(np.nanmedian(c["frames"]["spin"]))
    barrel_med = float(np.nanmedian(c["frames"]["barrel"]))

    if az_std > 5.0 * max(el_std, 0.1) and az_std > 5.0:
        print(f"     Azimuth wanders ({az_std:.1f}° std) while elevation holds "
              f"({el_std:.1f}°).\n"
              f"     This is the unidentifiable-DOF signature, and it is the "
              f"EXPECTED result\n     for a body of revolution: every azimuth "
              f"renders the same image, so the\n     argmax is decided by "
              f"noise. Nothing is broken.")
    elif az_std <= 2.0 and el_std <= 2.0 and ro_std <= 2.0:
        print(f"     Everything is stable (az {az_std:.1f}°, el {el_std:.1f}°, "
              f"ro {ro_std:.1f}°).\n"
              f"     Either this object has enough asymmetric texture to pin "
              f"its azimuth, or\n     the perturbation was too gentle to be "
              f"representative — check against\n     --camera-frames before "
              f"concluding the DOF is identifiable.")
    else:
        print(f"     az {az_std:.1f}°, el {el_std:.1f}°, ro {ro_std:.1f}° — "
              f"no clean split. All three\n     DOFs are moving, which is a "
              f"different (worse) problem than an\n     unidentifiable spin: "
              f"check the crop is actually framing the object.")

    print(f"\n     Of the wobble, {spin_med:.1f}° (median) is spin about the "
          f"barrel axis and the\n     barrel axis itself moves "
          f"{barrel_med:.1f}°.")
    if spin_med > 3.0 * max(barrel_med, 0.05):
        print("     The unidentifiable DOF carries essentially all of it. An "
              "axial symmetry\n     snap freezes exactly that component and "
              "leaves elevation/roll untouched:\n"
              "         M = R_ref.T @ R_meas;  theta* = atan2(M[2,0] - M[0,2], "
              "M[0,0] + M[2,2])\n"
              "         R_snapped = R_meas @ Ry(-theta*)\n"
              "     That is the whole fix, and it is exact rather than a "
              "filter.")
    else:
        print("     The barrel axis moves too — a symmetry snap will NOT fix "
              "this on its own,\n     because the part it cannot touch is "
              "also drifting.")

    print("\n  argmax vs circular expectation")
    for i, name in enumerate(("azimuth", "elevation", "roll")):
        f = _circular_spread if name != "elevation" else (
            lambda v: (float(v.std()), float(np.abs(v - v.mean()).max())))
        s_arg = f(c["argmax"][:, i])[0]
        s_exp = f(c["expectation"][:, i])[0]
        # Below the 1° bin width neither decode has anything left to resolve,
        # so a ratio between two sub-quantum numbers is noise about noise.
        verdict = ("both flat (< 1 bin)" if max(s_arg, s_exp) < 1.0
                   else "expectation quieter" if s_exp < 0.8 * s_arg
                   else "no material difference" if s_exp < 1.2 * s_arg
                   else "expectation NOISIER")
        print(f"    {name:<12}argmax {s_arg:>7.2f}°   expectation "
              f"{s_exp:>7.2f}°   -> {verdict}")
    print("     Expectation being quieter on a DOF the model can actually see "
          "is free\n     accuracy; on an unidentifiable DOF it just averages "
          "noise more smoothly and\n     means nothing.")


# ============================================================================
# entry point
# ============================================================================

def main(
    image: str | None = None,
    mask: str | None = None,
    camera_frames: int = 0,
    camera_host: str | None = None,
    fake_camera: bool = False,
    repeats: int = 100,
    determinism_repeats: int = 10,
    max_batch: int = 4,
    noise_dn: float = 2.0,
    jitter_px: int = 2,
    gain: float = 0.02,
    orient_size: int | None = None,
    orient_cast_weights: bool | None = None,
    background: str | None = None,
    perception_config: str | None = None,
    device: str | None = None,
    auto_download: bool = False,
    seed: int = 0,
) -> None:
    """Measure whether a static object's predicted orientation drifts.

    image: a saved frame (or a tight crop) of the motionless object.
    mask: object mask for that frame, anything non-zero. Omit only if `image`
        is already a tight crop — otherwise the crop becomes the whole scene
        and the measurement is about the wrong pixels.
    camera_frames: instead of perturbing one saved image, grab this many
        consecutive frames from the head camera. Point it at a motionless
        object and do not touch anything. This is the measurement that counts;
        the synthetic path exists so the question is answerable off-hardware.
    repeats: trials for experiment C. 100 is enough to see a wandering
        azimuth; use 500+ to characterise its distribution.
    noise_dn / jitter_px / gain: the synthetic perturbation, ignored when
        --camera-frames is used. Defaults are deliberately mild — roughly what
        a well-lit static scene produces. If the answer flips between mild and
        aggressive settings, that itself is the finding.
    """
    import os

    from ego2g1.deploy.perception.v2.config import PerceptionV2Config
    from ego2g1.deploy.perception_v2_e2e import _build_orient, dataclasses_replace

    if image is None and not camera_frames and not fake_camera:
        raise SystemExit(
            "give --image <static frame> or --camera-frames <n>.\n"
            "  --image is the offline path; --camera-frames measures a real\n"
            "  motionless object through the real sensor and is the one to "
            "trust.")

    cfg = PerceptionV2Config.load(perception_config)
    if orient_size is not None:
        cfg = dataclasses_replace(cfg, "orient", size=orient_size)
    if orient_cast_weights is not None:
        cfg = dataclasses_replace(cfg, "orient", cast_weights=orient_cast_weights)
    if background is not None:
        cfg = dataclasses_replace(cfg, "orient", background=background)

    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    print("=" * 72)
    if dev.startswith("cuda"):
        pr = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"device : {pr.name} ({pr.total_memory / 1024 ** 3:.1f} GB)")
    else:
        print(f"device : {dev}  -- CPU numbers do not represent the deploy box")
    print(f"orient : size={cfg.orient.size} cast_weights="
          f"{cfg.orient.cast_weights} background={cfg.orient.background!r} "
          f"crop_pad={cfg.orient.crop_pad}")
    print("=" * 72)

    frames = None
    close = None
    if camera_frames or fake_camera:
        from ego2g1.deploy.perception_v2_latency import _camera
        host = camera_host or os.environ.get("EGO2G1_CAMERA_HOST",
                                             "192.168.123.164")
        read, close = _camera(fake=fake_camera, host=host, w=640, h=480)
        n = camera_frames or repeats
        print(f"[setup] grabbing {n} frames — do not move the object")
        frames = [read()[0].copy() for _ in range(n)]
        rgb = frames[0]
        repeats = min(repeats, len(frames))
    else:
        rgb = _load_image(image)

    try:
        m = _load_mask(mask, rgb.shape[:2])
        if mask is None:
            print(f"[note] no --mask: cropping the WHOLE {rgb.shape[1]}x"
                  f"{rgb.shape[0]} frame. Correct only if this\n"
                  f"       image is already a tight crop of the object.")
        print(f"[setup] mask covers {100 * m.mean():.1f}% of the frame")

        orient = _build_orient(cfg, dev, auto_download)
        if orient is None:
            raise SystemExit("Orient Anything V2 unavailable — nothing to "
                             "measure. Re-run with --auto-download.")
        head = _Head(orient)
        crop = _crop(rgb, m, orient)
        print(f"[setup] crop {crop.size[0]}x{crop.size[1]} -> "
              f"{orient.size}x{orient.size}")

        print("\n" + "-" * 72)
        print(f"A. SAME TENSOR x{determinism_repeats}  (is the model itself "
              f"deterministic?)")
        print("-" * 72)
        a = exp_a_identical(head, crop, determinism_repeats)
        print(f"  max logit delta across runs: {a['max_logit_delta']:.3e}")
        _angle_table("decoded", a["angles"])
        _distribution_report(a["logits"][0])

        print("\n" + "-" * 72)
        print(f"B. SAME TENSOR, BATCH SIZES 1..{max_batch}  (does roster size "
              f"perturb it?)")
        print("-" * 72)
        b = exp_b_batch_position(head, crop, max_batch)
        base = b["angles"][0]
        for bs, idx, ang in b["rows"]:
            d = ang - base
            d[0], d[2] = _wrap180(d[0]), _wrap180(d[2])
            flag = "" if not np.abs(d).max() else "   <- differs"
            print(f"    batch={bs} pos={idx}   az {ang[0]:7.1f}° "
                  f"el {ang[1]:6.1f}° ro {ang[2]:7.1f}°{flag}")

        print("\n" + "-" * 72)
        src = (f"{repeats} REAL CAMERA FRAMES" if frames is not None
               else f"{repeats} PERTURBED REDRAWS "
                    f"(noise={noise_dn} DN, jitter=±{jitter_px} px, "
                    f"gain=±{100 * gain:.0f}%)")
        print(f"C. STATIC OBJECT, {src}")
        print("-" * 72)
        c = exp_c_realistic(head, rgb, m, orient, repeats, rng,
                            noise_dn=noise_dn, jitter_px=jitter_px, gain=gain,
                            frames=frames)
        _angle_table("argmax", c["argmax"])
        print()
        _angle_table("expectation", c["expectation"])
        print("\n  Where the wobble went (argmax decode, vs the mean pose):")
        c["frames"] = _frame_table(c["argmax"], cfg.convention)

        _verdict(a, b, c, frames is not None)
    finally:
        if close is not None:
            close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
