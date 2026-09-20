from __future__ import annotations

"""V0.1.2 综合色素空间融合层。

本模块不改变 Visible / UV / Brown 的原始检测结果和正式评分输入；Red
权重固定为零，只生成空间核查与解释证据。
"""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.detailed_metrics import (
    MEDICAL_REGION_LABELS,
    MEDICAL_REGION_ORDER,
    build_medical_report_regions,
)


def _binary(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value) > 0).astype(np.uint8) * 255


def _dilate(value: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.dilate(_binary(value), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents((_binary(mask) > 0).astype(np.uint8), 8)
    return max(0, int(count) - 1)


def _region_summary(mask: np.ndarray, analysis_mask: np.ndarray, landmarks: np.ndarray) -> list[dict[str, Any]]:
    regions = build_medical_report_regions(analysis_mask, landmarks)
    output: list[dict[str, Any]] = []
    for name in MEDICAL_REGION_ORDER:
        region = regions.regions[name]
        area = int(np.count_nonzero(region))
        feature = cv2.bitwise_and(_binary(mask), region)
        output.append({
            "region_id": name,
            "region_name": MEDICAL_REGION_LABELS[name],
            "available": bool(regions.available[name]),
            "valid_area_px": area,
            "feature_area_px": int(np.count_nonzero(feature)),
            "feature_area_ratio": float(np.count_nonzero(feature) / max(area, 1)),
            "component_count": _component_count(feature),
        })
    return output


def fuse_pigmentation(
    *,
    image: np.ndarray,
    analysis_mask: np.ndarray,
    landmarks: np.ndarray,
    visible_mask: np.ndarray,
    uv_mask: np.ndarray,
    brown_mask: np.ndarray,
    red_mask: np.ndarray,
) -> dict[str, Any]:
    valid = _binary(analysis_mask)
    visible = cv2.bitwise_and(_binary(visible_mask), valid)
    uv_all = cv2.bitwise_and(_binary(uv_mask), valid)
    brown = cv2.bitwise_and(_binary(brown_mask), valid)
    red = cv2.bitwise_and(_binary(red_mask), valid)

    # “UV中相对更突出”仅做空间差异解释：UV目标若不与扩张后的Visible
    # 匹配，则进入增强证据。它不替换UV正式四指标，也不改变shadow分。
    uv_enhanced = cv2.bitwise_and(uv_all, cv2.bitwise_not(_dilate(visible, 5)))
    red_brown = cv2.bitwise_and(red, _dilate(brown, 4))
    pure_red = cv2.bitwise_and(red, cv2.bitwise_not(_dilate(brown, 4)))

    region_union = cv2.bitwise_or(cv2.bitwise_or(visible, uv_all), brown)
    regions = _region_summary(region_union, valid, landmarks)
    priority = sorted(
        (row for row in regions if row["available"]),
        key=lambda row: (row["feature_area_ratio"], row["feature_area_px"]),
        reverse=True,
    )[:3]

    masks = {
        "visible_pigment": visible,
        "uv_all_pigment": uv_all,
        "uv_enhanced_pigment": uv_enhanced,
        "brown_pigment": brown,
        "red_brown_mixed": red_brown,
        "pure_red_suspected": pure_red,
    }
    valid_area = max(int(np.count_nonzero(valid)), 1)
    metrics = {
        name: {
            "area_px": int(np.count_nonzero(mask)),
            "area_ratio": float(np.count_nonzero(mask) / valid_area),
            "component_count": _component_count(mask),
        }
        for name, mask in masks.items()
    }
    return {
        "schema_version": "pigmentation_fusion_v0.1.2",
        "score_role": "shadow_v012",
        "formal_formula": "Visible 40% + UV 25% + Brown 35%",
        "red_weight": 0.0,
        "red_invariance_rule": "Red辅助Mask不得改变Visible/UV/Brown正式输入及综合色素分",
        "metrics": metrics,
        "priority_pigment_regions": priority,
        "region_distribution": regions,
        "masks": masks,
        "overlay": _overlay(image, masks),
    }


def _overlay(image: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    output = np.asarray(image).copy()
    colors = {
        "visible_pigment": (80, 210, 250),
        "uv_enhanced_pigment": (220, 70, 200),
        "brown_pigment": (40, 150, 220),
        "red_brown_mixed": (60, 30, 230),
        "pure_red_suspected": (30, 30, 255),
    }
    for name, color in colors.items():
        mask = masks[name] > 0
        if np.any(mask):
            output[mask] = np.clip(.45 * output[mask] + .55 * np.asarray(color), 0, 255).astype(np.uint8)
    return output


def save_pigmentation_fusion(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for index, name in enumerate((
        "visible_pigment", "uv_all_pigment", "uv_enhanced_pigment",
        "brown_pigment", "red_brown_mixed", "pure_red_suspected",
    ), start=1):
        path = target / f"{index:02d}_{name}.png"
        cv2.imwrite(str(path), result["masks"][name])
        paths[name] = str(path)
    overlay = target / "07_综合色素融合结果图.jpg"
    cv2.imwrite(str(overlay), result["overlay"])
    paths["overlay"] = str(overlay)
    metrics = target / "综合色素融合指标.json"
    document = {key: value for key, value in result.items() if key not in {"masks", "overlay"}}
    metrics.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths["metrics"] = str(metrics)
    return paths


__all__ = ["fuse_pigmentation", "save_pigmentation_fusion"]
