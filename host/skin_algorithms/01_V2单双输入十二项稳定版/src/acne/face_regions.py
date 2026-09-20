from __future__ import annotations

import cv2
import numpy as np


REGION_NAMES = [
    "forehead",
    "subject_left_cheek",
    "subject_right_cheek",
    "nose",
    "chin",
    "subject_left_jaw",
    "subject_right_jaw",
]


def _fill_hull(mask: np.ndarray, points: np.ndarray, value: int) -> None:
    points = points.astype(np.int32)
    if len(points) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(points), value)


def build_region_masks(landmarks_xy: np.ndarray, image_size: int = 1024) -> dict[str, np.ndarray]:
    """Build coarse acne reporting regions in standardized coordinates."""
    h = w = image_size
    regions = {name: np.zeros((h, w), dtype=np.uint8) for name in REGION_NAMES}

    xs = landmarks_xy[:, 0]
    ys = landmarks_xy[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    face_w = max(1.0, x_max - x_min)
    face_h = max(1.0, y_max - y_min)
    x_mid = float(landmarks_xy[1, 0]) if len(landmarks_xy) > 1 else (x_min + x_max) / 2.0

    y_forehead_bottom = y_min + 0.34 * face_h
    y_cheek_top = y_min + 0.27 * face_h
    y_cheek_bottom = y_min + 0.76 * face_h
    y_chin_top = y_min + 0.70 * face_h
    y_jaw_top = y_min + 0.62 * face_h
    nose_left = x_mid - 0.15 * face_w
    nose_right = x_mid + 0.15 * face_w
    cheek_inner_left = x_mid - 0.08 * face_w
    cheek_inner_right = x_mid + 0.08 * face_w

    def rect(name: str, x1: float, y1: float, x2: float, y2: float) -> None:
        cv2.rectangle(
            regions[name],
            (max(0, int(round(x1))), max(0, int(round(y1)))),
            (min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))),
            255,
            thickness=-1,
        )

    rect("forehead", x_min, y_min - 0.03 * face_h, x_max, y_forehead_bottom)
    rect("nose", nose_left, y_forehead_bottom - 0.05 * face_h, nose_right, y_chin_top)
    rect("subject_left_cheek", x_mid + 0.04 * face_w, y_cheek_top, x_max, y_cheek_bottom)
    rect("subject_right_cheek", x_min, y_cheek_top, x_mid - 0.04 * face_w, y_cheek_bottom)
    rect("chin", x_mid - 0.24 * face_w, y_chin_top, x_mid + 0.24 * face_w, y_max)
    rect("subject_left_jaw", x_mid + 0.18 * face_w, y_jaw_top, x_max, y_max)
    rect("subject_right_jaw", x_min, y_jaw_top, x_mid - 0.18 * face_w, y_max)

    return regions


def draw_region_debug(image_bgr: np.ndarray, region_masks: dict[str, np.ndarray], skin_mask: np.ndarray) -> np.ndarray:
    colors = {
        "forehead": (255, 210, 80),
        "subject_left_cheek": (80, 220, 255),
        "subject_right_cheek": (120, 255, 120),
        "nose": (255, 160, 120),
        "chin": (180, 160, 255),
        "subject_left_jaw": (90, 180, 255),
        "subject_right_jaw": (160, 220, 120),
    }
    debug = image_bgr.copy()
    overlay = image_bgr.copy()
    for name, mask in region_masks.items():
        valid = cv2.bitwise_and(mask, skin_mask)
        overlay[valid > 0] = colors[name]
        contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, contours, -1, colors[name], 2, cv2.LINE_AA)
    debug = cv2.addWeighted(overlay, 0.22, debug, 0.78, 0)
    return debug
