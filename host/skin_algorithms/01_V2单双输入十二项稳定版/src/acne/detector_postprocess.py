from __future__ import annotations

from contextlib import nullcontext
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .artifact_policy import AcneArtifactPolicy, AcnePhaseTimings
from .face_regions import REGION_NAMES
from .image_io import write_image


def _mask_at(mask: np.ndarray, x: float, y: float, shape: tuple[int, int]) -> int:
    target_h, target_w = shape
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    px = int(max(0, min(target_w - 1, round(x))))
    py = int(max(0, min(target_h - 1, round(y))))
    return int(mask[py, px])


def filter_detections(
    detections: list[dict[str, Any]],
    skin_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    image_shape: tuple[int, int],
    min_box_area_ratio: float = 0.00001,
    max_box_area_ratio: float = 0.08,
    iou_threshold: float = 0.45,
    center_distance_px: float = 6.0,
) -> list[dict[str, Any]]:
    height, width = image_shape
    image_area = float(height * width)
    kept: list[dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
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
        item["center_xy"] = [round(cx, 4), round(cy, 4)]
        item["box_area"] = round(area, 4)
        kept.append(item)

    kept.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    nms_kept: list[dict[str, Any]] = []
    for det in kept:
        if all(_iou(det["bbox_xyxy"], existing["bbox_xyxy"]) <= iou_threshold for existing in nms_kept):
            nms_kept.append(det)

    deduped: list[dict[str, Any]] = []
    for det in nms_kept:
        cx, cy = det["center_xy"]
        if all(_distance(cx, cy, *existing["center_xy"]) >= center_distance_px for existing in deduped):
            deduped.append(det)
    for idx, det in enumerate(deduped, start=1):
        det["id"] = idx
    return deduped


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)


def add_original_coordinates(
    candidates: list[dict[str, Any]],
    crop_box_xyxy: tuple[int, int, int, int],
    square_offset_xy: tuple[int, int],
    resize_scale: float,
    original_shape_hw: tuple[int, int],
    input_mode: str,
    detection_scope: str,
) -> list[dict[str, Any]]:
    """Attach original-image coordinates while preserving standardized coordinates."""
    crop_x1, crop_y1, _, _ = crop_box_xyxy
    offset_x, offset_y = square_offset_xy
    original_h, original_w = original_shape_hw
    scale = max(float(resize_scale), 1e-8)

    def map_point(x: float, y: float) -> tuple[float, float]:
        original_x = x / scale + crop_x1 - offset_x
        original_y = y / scale + crop_y1 - offset_y
        original_x = max(0.0, min(float(original_w), original_x))
        original_y = max(0.0, min(float(original_h), original_y))
        return original_x, original_y

    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        bbox = item.get("bbox_xyxy") or item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            enriched.append(item)
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        ox1, oy1 = map_point(x1, y1)
        ox2, oy2 = map_point(x2, y2)
        center = item.get("center_xy")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        ocx, ocy = map_point(float(center[0]), float(center[1]))
        item["bbox_standardized_xyxy"] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
        item["center_standardized_xy"] = [round(float(center[0]), 4), round(float(center[1]), 4)]
        item["bbox_original_xyxy"] = [round(ox1, 4), round(oy1, 4), round(ox2, 4), round(oy2, 4)]
        item["center_original_xy"] = [round(ocx, 4), round(ocy, 4)]
        item["input_mode"] = input_mode
        item["detection_scope"] = detection_scope
        enriched.append(item)
    return enriched


def candidates_in_original_space(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return drawable candidate copies whose primary coordinates are original-image coordinates."""
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        bbox = candidate.get("bbox_original_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        item = dict(candidate)
        item["bbox_xyxy"] = [float(value) for value in bbox]
        center = item.get("center_original_xy")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            item["center_xy"] = [float(center[0]), float(center[1])]
        output.append(item)
    return output


def draw_boxes(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    canvas = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(float(v))) for v in det["bbox_xyxy"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return canvas


def draw_circles(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    canvas = image_bgr.copy()
    for det in detections:
        if "center_xy" in det:
            cx, cy = [float(v) for v in det["center_xy"]]
        else:
            x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
        radius = max(6, int(round(max(x2 - x1, y2 - y1) / 2.0)))
        cv2.circle(canvas, (int(round(cx)), int(round(cy))), radius, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    return canvas


def count_by_region(detections: list[dict[str, Any]], region_masks: dict[str, np.ndarray]) -> dict[str, int]:
    counts = {name: 0 for name in REGION_NAMES}
    for det in detections:
        if "center_xy" in det:
            cx, cy = [float(v) for v in det["center_xy"]]
        else:
            x1, y1, x2, y2 = [float(v) for v in det["bbox_xyxy"]]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        assigned = False
        for name in REGION_NAMES:
            mask = region_masks.get(name)
            if mask is None:
                continue
            h, w = mask.shape[:2]
            px = int(max(0, min(w - 1, round(cx))))
            py = int(max(0, min(h - 1, round(cy))))
            if int(mask[py, px]) > 0:
                counts[name] += 1
                det["region"] = name
                assigned = True
                break
        if not assigned:
            det["region"] = "unassigned"
    return counts


def save_detection_outputs(
    output_dir: str | Path,
    standardized_bgr: np.ndarray,
    raw_result: dict[str, Any],
    filtered_detections: list[dict[str, Any]],
    region_counts: dict[str, int] | None = None,
    region_analysis: dict[str, Any] | None = None,
    artifact_policy: AcneArtifactPolicy = AcneArtifactPolicy.FORMAL,
    timings: AcnePhaseTimings | None = None,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    paths = {
        "raw_detections_json": out_dir / "14_raw_detections.json",
        "filtered_detections_json": out_dir / "15_filtered_detections.json",
        "raw_boxes_image": out_dir / "16_raw_boxes.jpg",
        "circle_image": out_dir / "17_acne_candidate_circles.jpg",
        "detections_csv": out_dir / "18_detections.csv",
    }
    filtered_payload = {
        "status": "ok",
        "detections": filtered_detections,
        "count": len(filtered_detections),
        "region_counts": region_counts if region_counts is not None else {},
        "region_analysis": region_analysis or {
            "status": "skipped",
            "reason": "global_regions_not_available",
        },
        "empty_detections_are_valid": True,
        "label": "acne_candidate",
        "label_zh": "疑似痤疮病灶",
    }
    structured_scope = timings.measure_structured_persistence() if timings else nullcontext()
    with structured_scope:
        paths["raw_detections_json"].write_text(
            json.dumps(raw_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["filtered_detections_json"].write_text(
            json.dumps(filtered_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    image_scope = timings.measure_image_encoding() if timings else nullcontext()
    with image_scope:
        if artifact_policy.includes_debug_images:
            write_image(paths["raw_boxes_image"], draw_boxes(standardized_bgr, raw_result.get("detections", [])))
        write_image(paths["circle_image"], draw_circles(standardized_bgr, filtered_detections))
    csv_scope = timings.measure_structured_persistence() if timings else nullcontext()
    with csv_scope:
        with paths["detections_csv"].open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "label", "label_zh", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "region", "source"],
            )
            writer.writeheader()
            for det in filtered_detections:
                x1, y1, x2, y2 = det["bbox_xyxy"]
                cx, cy = det.get("center_xy", [(x1 + x2) / 2, (y1 + y2) / 2])
                writer.writerow(
                    {
                        "id": det.get("id", ""),
                        "label": det.get("label", "acne_candidate"),
                        "label_zh": det.get("label_zh", "疑似痤疮病灶"),
                        "confidence": det.get("confidence", ""),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": cx,
                        "cy": cy,
                        "region": det.get("region", "unassigned"),
                        "source": det.get("source", "full_face"),
                    }
                )
    return {key: str(path) for key, path in paths.items() if path.exists()}
