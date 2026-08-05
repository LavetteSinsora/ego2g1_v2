"""The relation_eef perception overlay renderer, fed the RECORDED `percept`
shape (docs/deploy_refactor_plan.md §5): DINO+SAM2 boxes/masks, the live
fast-tracker position per object, the latch's rigid-predicted pose while a
hand is CANDIDATE/LATCHED, and each hand's FK wrist projected through the
same calibration (a hand-eye sanity check independent of the detector — if
that marker doesn't sit on the visibly-real wrist, the extrinsics or K are
wrong, not the detection/tracking cascade).

One renderer for live and replay: `dashboard.Dashboard` feeds it the live
`RelationPerception.debug_snapshot()`, `replay_dashboard.ReplayLoop` feeds
it a recorded `percept` event — the same JSON shape by construction, so the
two views cannot diverge. cv2 is imported lazily (drawing happens on the
dashboard's own HTTP thread, never a hot one)."""

from __future__ import annotations

import base64

import numpy as np

PALETTE = [(66, 133, 244), (219, 68, 55), (15, 157, 88), (244, 160, 0),
           (171, 71, 188), (0, 172, 193)]         # per-instance_id, RGB
TRACK_COLOR = (255, 255, 255)   # live Kalman/OneEuro fast-tracker position
LATCH_COLOR = (255, 90, 0)      # GraspLatch's rigid-predicted pose
WRIST_COLOR = (0, 230, 230)     # FK wrist, projected through the calibration


def project_to_pixel(point_pelvis: np.ndarray, T_pelvis_camera: np.ndarray,
                     K: np.ndarray):
    """(3,) pelvis-frame point -> (u, v) pixel ints, or None if behind the
    camera. Inverse of `relation_perception.pixel_depth_to_camera_point`'s
    back-projection, through the same `T_pelvis_camera`/`K_left` convention
    (`point_pelvis = R @ point_camera + t` -> `point_camera = R.T @
    (point_pelvis - t)`)."""
    R = T_pelvis_camera[:3, :3]
    t = T_pelvis_camera[:3, 3]
    point_camera = R.T @ (np.asarray(point_pelvis, dtype=np.float64) - t)
    z = float(point_camera[2])
    if z <= 1e-6:
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = cx + point_camera[0] * fx / z
    v = cy + point_camera[1] * fy / z
    return int(round(u)), int(round(v))


def _decode_mask_png(b64: str) -> np.ndarray | None:
    import cv2

    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img > 127 if img is not None else None


def draw_perception_overlay(rgb: np.ndarray, snapshot: dict,
                            K: np.ndarray, T_pelvis_camera: np.ndarray,
                            flange_poses: dict | None = None,
                            masks: dict | None = None) -> np.ndarray:
    """Annotate `rgb` (copied, not mutated) from a `percept`-shaped
    `snapshot` ({"objects": {...}, "hands": {...}} —
    `RelationPerception.debug_snapshot()`'s own JSON).

    `masks`: optional {instance_id: (H, W) bool} overriding the snapshot's
    `mask_png_b64` fields — the live dashboard passes the detector's raw
    arrays to skip a pointless PNG encode/decode round-trip; replay decodes
    from the recording. `flange_poses`: {hand: (4, 4)} pelvis-frame FK, or
    None to skip the wrist markers."""
    import cv2

    img = np.ascontiguousarray(rgb).copy()
    objects = snapshot.get("objects") or {}
    for i, (instance_id, o) in enumerate(objects.items()):
        color = PALETTE[i % len(PALETTE)]
        mask = (masks or {}).get(instance_id)
        if mask is None and o.get("mask_png_b64"):
            mask = _decode_mask_png(o["mask_png_b64"])
        if mask is not None:
            layer = np.zeros_like(img)
            layer[np.asarray(mask, dtype=bool)] = color
            img = cv2.addWeighted(img, 1.0, layer, 0.4, 0.0)
        if o.get("box_xyxy") is not None:
            x0, y0, x1, y1 = (int(round(float(v))) for v in o["box_xyxy"])
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            conf = o.get("confidence")
            label = (f"{instance_id} {conf:.2f}" if conf is not None
                     else instance_id)
            cv2.putText(img, label, (x0, max(12, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for instance_id, o in objects.items():
        pose = o.get("tracked_pose")
        if pose is None:
            continue
        uv = project_to_pixel(np.asarray(pose)[:3, 3], T_pelvis_camera, K)
        if uv is not None:
            cv2.circle(img, uv, 5, TRACK_COLOR, -1, cv2.LINE_AA)
            cv2.putText(img, "tracked", (uv[0] + 7, uv[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TRACK_COLOR, 1,
                        cv2.LINE_AA)

    for hand, h in (snapshot.get("hands") or {}).items():
        rigid = h.get("rigid_pose")
        if rigid is None:
            continue
        uv = project_to_pixel(np.asarray(rigid)[:3, 3], T_pelvis_camera, K)
        if uv is not None:
            cv2.drawMarker(img, uv, LATCH_COLOR, cv2.MARKER_TILTED_CROSS,
                           14, 2, cv2.LINE_AA)
            cv2.putText(img, f"predicted ({hand})", (uv[0] + 9, uv[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, LATCH_COLOR, 1,
                        cv2.LINE_AA)

    if flange_poses is not None:
        for hand, pose in flange_poses.items():
            uv = project_to_pixel(np.asarray(pose)[:3, 3], T_pelvis_camera, K)
            if uv is not None:
                cv2.drawMarker(img, uv, WRIST_COLOR, cv2.MARKER_DIAMOND,
                               16, 2, cv2.LINE_AA)
                cv2.putText(img, f"FK wrist ({hand})", (uv[0] + 9, uv[1] + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, WRIST_COLOR, 1,
                            cv2.LINE_AA)
    return img
