"""Report-only grouped overlays built from the legacy wrinkle regions.

This module never creates, removes, or reclassifies a wrinkle candidate.  It
only selects the region masks and assigned centre-lines already produced by
the balanced legacy wrinkle pipeline, then renders three uncluttered full-face
views for report sections 07, 08, and 09.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


REGION_GROUPS: dict[str, dict[str, Any]] = {
    "07": {
        "filename": "07_干燥性细纹全脸分区结果图.jpg",
        "region_keys": ("left_under_eye", "right_under_eye"),
        "region_names": ("左眼下细纹", "右眼下细纹"),
    },
    "08": {
        "filename": "08_稳定性线性皱纹全脸分区结果图.jpg",
        "region_keys": (
            "forehead",
            "glabella",
            "left_crow_feet",
            "right_crow_feet",
        ),
        "region_names": ("额头纹", "眉间纹", "左鱼尾纹", "右鱼尾纹"),
    },
    "09": {
        "filename": "09_结构性沟纹全脸分区结果图.jpg",
        "region_keys": (
            "left_nasolabial",
            "right_nasolabial",
            "left_marionette",
            "right_marionette",
        ),
        "region_names": ("左法令纹", "右法令纹", "左木偶纹", "右木偶纹"),
    },
}

REGION_OUTLINE_BGR = (222, 197, 65)
WRINKLE_LINE_BGR = (54, 255, 148)


def region_name_in_group(module_id: str, region_name: str) -> bool:
    """Return whether a legacy region belongs to one report support domain.

    The display overlay and every text/JSON projection use this same frozen
    mapping.  Exact names are intentional: fuzzy token matching previously
    allowed fish-tail or nasolabial metrics to leak into the wrong chapter.
    """

    definition = REGION_GROUPS.get(str(module_id))
    if definition is None:
        raise ValueError(f"未知皱纹报告模块: {module_id}")
    return str(region_name) in definition["region_names"]


def _normalize_region_defs(
    region_defs: Iterable[tuple[Any, ...]],
) -> dict[str, tuple[str, str, np.ndarray, np.ndarray, bool]]:
    normalized: dict[str, tuple[str, str, np.ndarray, np.ndarray, bool]] = {}
    for item in region_defs:
        if len(item) == 6:
            key, display_name, short_name, mask, line_mask, analysis_available = item
        elif len(item) == 5:
            key, display_name, short_name, mask, line_mask = item
            analysis_available = True
        elif len(item) == 4:
            key, display_name, short_name, mask = item
            line_mask = np.zeros_like(mask)
            analysis_available = True
        else:
            raise ValueError(f"皱纹分区定义长度异常: {len(item)}")
        normalized[str(key)] = (
            str(display_name),
            str(short_name),
            np.asarray(mask, dtype=np.uint8),
            np.asarray(line_mask, dtype=np.uint8),
            bool(analysis_available),
        )
    return normalized


def _overlay_lines(image: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    if not np.any(line_mask):
        return output
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    visible = cv2.dilate((line_mask > 0).astype(np.uint8), kernel, iterations=1) > 0
    color = np.empty_like(output)
    color[:] = WRINKLE_LINE_BGR
    blended = cv2.addWeighted(output, 0.08, color, 0.92, 0)
    output[visible] = blended[visible]
    return output


def _largest_component_for_outline(mask: np.ndarray) -> np.ndarray:
    """Drop tiny disconnected display fragments without changing evidence."""

    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def render_legacy_group_overlays(
    display_image: np.ndarray,
    region_defs: Iterable[tuple[Any, ...]],
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Render three full-face views without changing legacy evidence.

    Selected region contours remain visible even when their assigned legacy
    centre-line is empty.  No labels, scores, fills, or experimental Head
    decisions are drawn into the image.
    """

    if display_image.ndim != 3 or display_image.shape[2] != 3:
        raise ValueError("报告分区图要求BGR三通道图像")

    regions = _normalize_region_defs(region_defs)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}

    for module_id, definition in REGION_GROUPS.items():
        requested = tuple(definition["region_keys"])
        displayable = [key for key in requested if key in regions]
        available = [key for key in displayable if regions[key][4]]
        unavailable = [key for key in displayable if not regions[key][4]]
        missing = [key for key in requested if key not in regions]
        output = display_image.copy()
        line_union = np.zeros(display_image.shape[:2], dtype=np.uint8)
        selected_names: list[str] = []

        display_names: list[str] = []
        for key in displayable:
            display_name, _short_name, mask, line_mask, analysis_available = regions[key]
            display_names.append(display_name)
            if analysis_available:
                selected_names.append(display_name)
            display_mask = _largest_component_for_outline(mask)
            contours, _ = cv2.findContours(
                display_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(
                output,
                contours,
                -1,
                REGION_OUTLINE_BGR,
                2,
                cv2.LINE_AA,
            )
            line_union = np.maximum(line_union, line_mask)

        output = _overlay_lines(output, line_union)
        path = target / str(definition["filename"])
        if not cv2.imwrite(str(path), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"无法写入报告分区图: {path}")

        manifest[module_id] = {
            "path": str(path.resolve()),
            "filename": path.name,
            "selected_region_keys": list(available),
            "selected_region_names": selected_names,
            "display_region_keys": list(displayable),
            "display_region_names": display_names,
            "unavailable_region_keys": unavailable,
            "missing_region_keys": missing,
            "legacy_centerline_pixels": int(np.count_nonzero(line_union)),
            "image_size": {
                "width": int(output.shape[1]),
                "height": int(output.shape[0]),
            },
        }

    return manifest


__all__ = ["REGION_GROUPS", "region_name_in_group", "render_legacy_group_overlays"]
