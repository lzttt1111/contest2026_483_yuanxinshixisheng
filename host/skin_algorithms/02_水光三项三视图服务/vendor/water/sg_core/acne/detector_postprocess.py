from __future__ import annotations
from contextlib import nullcontext
import csv
import json
from pathlib import Path
from typing import Any
import cv2
import numpy as np

def _mask_at(mask: np.ndarray, x: float, y: float, shape: tuple[int, int]) -> int:
    (target_h, target_w) = shape
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    px = int(max(0, min(target_w - 1, round(x))))
    py = int(max(0, min(target_h - 1, round(y))))
    return int(mask[py, px])

def filter_detections(detections: list[dict[str, Any]], skin_mask: np.ndarray, forbidden_mask: np.ndarray, image_shape: tuple[int, int], min_box_area_ratio: float=1e-05, max_box_area_ratio: float=0.08, iou_threshold: float=0.45, center_distance_px: float=6.0) -> list[dict[str, Any]]:
    (height, width) = image_shape
    image_area = float(height * width)
    kept: list[dict[str, Any]] = []
    for det in detections:
        (x1, y1, x2, y2) = [float(v) for v in det['bbox_xyxy']]
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            continue
        if area / image_area < min_box_area_ratio or area / image_area > max_box_area_ratio:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        if _mask_at(skin_mask, cx, cy, (height, width)) <= 0:
            continue
        if _mask_at(forbidden_mask, cx, cy, (height, width)) > 0:
            continue
        item = dict(det)
        item['center_xy'] = [round(cx, 4), round(cy, 4)]
        item['box_area'] = round(area, 4)
        kept.append(item)
    kept.sort(key=lambda item: float(item.get('confidence', 0.0)), reverse=True)
    nms_kept: list[dict[str, Any]] = []
    for det in kept:
        if all((_iou(det['bbox_xyxy'], existing['bbox_xyxy']) <= iou_threshold for existing in nms_kept)):
            nms_kept.append(det)
    deduped: list[dict[str, Any]] = []
    for det in nms_kept:
        (cx, cy) = det['center_xy']
        if all((_distance(cx, cy, *existing['center_xy']) >= center_distance_px for existing in deduped)):
            deduped.append(det)
    for (idx, det) in enumerate(deduped, start=1):
        det['id'] = idx
    return deduped

def _iou(a: list[float], b: list[float]) -> float:
    (ax1, ay1, ax2, ay2) = [float(v) for v in a]
    (bx1, by1, bx2, by2) = [float(v) for v in b]
    (ix1, iy1) = (max(ax1, bx1), max(ay1, by1))
    (ix2, iy2) = (min(ax2, bx2), min(ay2, by2))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
