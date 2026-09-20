from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

from src.doctor_v2.relative_mesh_style_gallery import render_holographic_duotone_mesh
from src.preprocess.image_preprocessor import LEFT_EYE, RIGHT_EYE, PreprocessResultV2


@dataclass(frozen=True, slots=True)
class RelativeFaceGeometryResult:
    overlay: np.ndarray
    face_scope: np.ndarray
    metrics: dict[str, float]


def _display_geometry(pre: PreprocessResultV2) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(pre.landmarks[:, :2], dtype=np.float32).copy()
    brow_y = float(min(points[105, 1], points[334, 1]))
    top_y = float(points[10, 1])
    forehead_height = max(brow_y - top_y, 1.0)
    weights = np.clip((brow_y - points[:, 1]) / forehead_height, 0.0, 1.0)
    points[:, 1] -= forehead_height * weights
    height, width = pre.analysis_image.shape[:2]
    points[:, 0] = np.clip(points[:, 0], 0.0, width - 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, height - 1.0)
    rounded = np.rint(points).astype(np.int32)
    scope = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(scope, cv2.convexHull(rounded[:468]), 255)
    return rounded, scope


def _jaw_metrics(landmarks: np.ndarray) -> dict[str, float]:
    jaw_indices = np.asarray(
        (234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
         377, 400, 378, 379, 365, 397, 288, 361, 323, 454),
        dtype=np.int32,
    )
    eye_left = np.mean(landmarks[np.asarray(LEFT_EYE, dtype=np.int32)], axis=0)
    eye_right = np.mean(landmarks[np.asarray(RIGHT_EYE, dtype=np.int32)], axis=0)
    scale = max(float(np.linalg.norm(eye_right - eye_left)), 1e-6)
    jaw = landmarks[jaw_indices] / scale
    segments = np.diff(jaw, axis=0)
    arc = np.linalg.norm(segments, axis=1)
    midpoint = int(np.where(jaw_indices == 152)[0][0])
    left_arc = float(np.sum(arc[:midpoint]))
    right_arc = float(np.sum(arc[midpoint:]))
    cross = segments[:-1, 0] * segments[1:, 1] - segments[:-1, 1] * segments[1:, 0]
    dot = np.sum(segments[:-1] * segments[1:], axis=1)
    turns = np.abs(np.arctan2(cross, dot))
    signs = np.sign(cross[np.abs(cross) > 1e-5])
    changes = int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    curve_p90 = float(np.percentile(turns, 90)) if turns.size else 0.0
    face_height = max(float(np.linalg.norm(landmarks[10] - landmarks[152])), 1e-6)
    return {
        "jaw_continuity_ratio": round(float(np.clip(1.0 - curve_p90 / (math.pi / 2.0), 0.0, 1.0)), 6),
        "jaw_curve_p90": round(curve_p90, 6),
        "jaw_curve_direction_changes": float(changes),
        "jaw_arc_asymmetry_ratio": round(abs(left_arc - right_arc) / max((left_arc + right_arc) * 0.5, 1e-6), 6),
        "jaw_width_to_face_height_ratio": round(float(np.linalg.norm(landmarks[234] - landmarks[454])) / face_height, 6),
    }


def _midface_metrics(pre: PreprocessResultV2, landmarks: np.ndarray) -> dict[str, float]:
    relative_z = pre.landmarks_relative_z
    if relative_z is None or np.asarray(relative_z).shape != (len(landmarks),):
        return {
            "midface_surface_continuity_ratio": 0.0,
            "midface_convex_concave_turns": 0.0,
            "midface_relative_relief_p90": 0.0,
        }
    eye_left = np.mean(landmarks[np.asarray(LEFT_EYE, dtype=np.int32)], axis=0)
    eye_right = np.mean(landmarks[np.asarray(RIGHT_EYE, dtype=np.int32)], axis=0)
    scale = max(float(np.linalg.norm(eye_right - eye_left)), 1e-6)
    face_height = max(float(np.linalg.norm(landmarks[10] - landmarks[152])), 1e-6)
    face_width = max(abs(float(landmarks[454, 0] - landmarks[234, 0])), 1e-6)
    eye_y = float((eye_left[1] + eye_right[1]) * 0.5)
    mouth_y = float(landmarks[13, 1])
    indices = np.asarray([
        index for index, (x, y) in enumerate(landmarks)
        if min(landmarks[234, 0], landmarks[454, 0]) + 0.08 * face_width <= x
        <= max(landmarks[234, 0], landmarks[454, 0]) - 0.08 * face_width
        and eye_y + 0.03 * face_height <= y <= mouth_y - 0.03 * face_height
    ], dtype=np.int32)
    if len(indices) < 24:
        return {"midface_surface_continuity_ratio": 0.0, "midface_convex_concave_turns": 0.0, "midface_relative_relief_p90": 0.0}
    xy = landmarks[indices] / scale
    local_z = (np.asarray(relative_z, dtype=np.float64)[indices] - float(np.asarray(relative_z)[1])) / scale
    design = np.column_stack((np.ones(len(xy)), xy[:, 0], xy[:, 1]))
    coefficients, *_ = np.linalg.lstsq(design, local_z, rcond=None)
    corrected = local_z - design @ coefficients
    lookup = {int(source): offset for offset, source in enumerate(indices)}
    deltas = [abs(float(corrected[lookup[a]] - corrected[lookup[b]])) for a, b in FACEMESH_TESSELATION if a in lookup and b in lookup]
    edge_p90 = float(np.percentile(deltas, 90)) if deltas else float(np.percentile(np.abs(corrected), 90))
    row_order = np.argsort(xy[:, 0])
    second = np.diff(corrected[row_order], n=2)
    signs = np.sign(second[np.abs(second) > 0.0015])
    return {
        "midface_surface_continuity_ratio": round(float(np.exp(-edge_p90 / 0.08)), 6),
        "midface_convex_concave_turns": float(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0.0,
        "midface_relative_relief_p90": round(float(np.percentile(np.abs(corrected), 90)), 6),
    }


def analyze_relative_face_geometry(pre: PreprocessResultV2) -> RelativeFaceGeometryResult | None:
    if pre.quality_status == "REJECT" or pre.landmarks_relative_z is None:
        return None
    if any(flag in pre.quality_flags for flag in ("EXTREME_POSE", "PARTIAL_FACE", "INVALID_FACE_GEOMETRY")):
        return None
    points, scope = _display_geometry(pre)
    overlay = render_holographic_duotone_mesh(
        pre.analysis_image,
        points,
        scope,
        pre.landmarks_relative_z,
        FACEMESH_TESSELATION,
    )
    landmarks = np.asarray(pre.landmarks, dtype=np.float64)
    return RelativeFaceGeometryResult(
        overlay=overlay,
        face_scope=scope,
        metrics={**_jaw_metrics(landmarks), **_midface_metrics(pre, landmarks)},
    )


__all__ = ["RelativeFaceGeometryResult", "analyze_relative_face_geometry"]
