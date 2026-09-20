from __future__ import annotations

"""Portable subset of the local Stage-3 landmark registration assessment.

Numerical method and limits come from doctor_v2/clinic_engineering.py.
No pixel warp is performed: the existing clinic provider owns image alignment.
"""

from dataclasses import dataclass
import hashlib
import json
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RegistrationThresholds:
    minimum_landmark_inlier_ratio: float = 0.90
    maximum_normalized_median_error: float = 0.015
    ransac_reprojection_threshold_normalized: float = 0.015
    minimum_scale: float = 0.90
    maximum_scale: float = 1.10
    maximum_absolute_rotation_degrees: float = 5.0
    minimum_rgb_face_hull_area_ratio: float = 0.08
    maximum_absolute_yaw_proxy_degrees: float = 28.0
    maximum_absolute_pitch_proxy_degrees: float = 22.0
    maximum_absolute_roll_proxy_degrees: float = 18.0
    minimum_landmark_count: int = 468


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def rgb_geometry_valid(points: np.ndarray, limits: RegistrationThresholds) -> bool:
    if len(points) < limits.minimum_landmark_count or not np.isfinite(points).all():
        return False
    left = points[[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]].mean(axis=0)
    right = points[[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]].mean(axis=0)
    if left[0] > right[0]:
        left, right = right, left
    roll = float(np.degrees(np.arctan2(*(right - left)[::-1])))
    cheek_left, cheek_right = sorted((float(points[234,0]), float(points[454,0])))
    half_width = max((cheek_right - cheek_left) * 0.5, 1e-6)
    yaw = 35.0 * (float(points[1,0]) - (cheek_left + cheek_right) * 0.5) / half_width
    eye_y = float((left[1] + right[1]) * 0.5)
    span = max(float(points[152,1]) - eye_y, 1e-6)
    pitch = 80.0 * ((float(points[1,1]) - eye_y) / span - 0.38)
    area = abs(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
    return bool(
        area >= limits.minimum_rgb_face_hull_area_ratio
        and abs(yaw) <= limits.maximum_absolute_yaw_proxy_degrees
        and abs(pitch) <= limits.maximum_absolute_pitch_proxy_degrees
        and abs(roll) <= limits.maximum_absolute_roll_proxy_degrees
    )


def assess_registration(role, source_points, rgb_points, source_shape, rgb_shape,
                        limits: RegistrationThresholds | None = None) -> dict:
    limits = limits or RegistrationThresholds()
    failed = {"status": "FAILED", "transform_sha256": None, "diagnostics": {}}
    if (source_shape != rgb_shape or source_points.shape != rgb_points.shape
            or len(source_points) < limits.minimum_landmark_count
            or not np.isfinite(source_points).all() or not np.isfinite(rgb_points).all()):
        return {**failed, "reason": "GEOMETRY_OR_LANDMARK_MISMATCH"}
    matrix, inliers = cv2.estimateAffinePartial2D(
        source_points.astype(np.float32), rgb_points.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=limits.ransac_reprojection_threshold_normalized,
        maxIters=2000, confidence=0.99, refineIters=10,
    )
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return {**failed, "reason": "TRANSFORM_UNAVAILABLE"}
    projected = cv2.transform(source_points.reshape(1, -1, 2), matrix)[0]
    diagonal = max(float(np.linalg.norm(np.ptp(rgb_points, axis=0))), 1e-8)
    error = float(np.median(np.linalg.norm(projected - rgb_points, axis=1)) / diagonal)
    ratio = float(np.mean(inliers.reshape(-1) > 0))
    scale = math.hypot(float(matrix[0,0]), float(matrix[0,1]))
    rotation = math.degrees(math.atan2(float(matrix[1,0]), float(matrix[0,0])))
    passed = (ratio >= limits.minimum_landmark_inlier_ratio
              and error <= limits.maximum_normalized_median_error
              and limits.minimum_scale <= scale <= limits.maximum_scale
              and abs(rotation) <= limits.maximum_absolute_rotation_degrees)
    diagnostics = dict(landmark_inlier_ratio=ratio,
                       normalized_median_reprojection_error=error,
                       similarity_scale=scale, rotation_degrees=rotation)
    return {
        "status": "PASSED" if passed else "FAILED",
        "transform_sha256": canonical_sha({
            "source": role.removesuffix("_M"),
            "matrix": [[round(float(v), 10) for v in row] for row in matrix],
        }) if passed else None,
        "diagnostics": diagnostics,
        "reason": "ENGINEERING_LANDMARK_REGISTRATION_PASSED" if passed else "TRANSFORM_UNSTABLE",
    }


class CaptureRegistration:
    """One process-local landmark model, reused across requests."""

    def __init__(self):
        self.model = None

    def landmarks(self, image):
        import mediapipe as mp
        from src.utils.model_loader import load_face_landmarker
        if self.model is None:
            self.model = load_face_landmarker()
        result = self.model.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        ))
        if len(result.face_landmarks) != 1:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray([(p.x, p.y) for p in result.face_landmarks[0]], dtype=np.float32)

    def close(self):
        if self.model is not None:
            self.model.close()
            self.model = None
