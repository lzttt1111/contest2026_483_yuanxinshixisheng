"""Observation outlines over saved V3 lines; no detection or reference fitting."""
import numpy as np

VERSION = "v301-wrinkle-observation-frames-2"
STABLE_ROLES = {
    "forehead": ("额头纹", "FH"), "glabella": ("眉间纹", "GL"),
    "left_crow_feet": ("左鱼尾纹", "LCF"),
    "right_crow_feet": ("右鱼尾纹", "RCF"),
}


def definitions(arrays):
    from src.wrinkle.algorithms.wrinkle_detection_algorithm import filtered_region_indices, build_face_filter_masks, build_aesthetic_region_masks
    from .stage1_lines import _domains
    keys = filtered_region_indices(len(arrays["landmarks"]))
    masks = build_face_filter_masks(arrays["image"], arrays["landmarks"], keys, arrays["valid"])
    native = build_aesthetic_region_masks(arrays["landmarks"], keys, masks, arrays["skeleton"])
    # 07 and 09 retain their native frame roles. The four existing 08 roles
    # come from the same target domains as measurement, never the wide Z support.
    result = [item for item in native if item[0] not in STABLE_ROLES]
    domains = _domains(arrays, "08")
    for key, (label, short) in STABLE_ROLES.items():
        domain = domains.get(key, np.zeros_like(arrays["valid"], dtype=bool))
        mask = (np.asarray(domain) > 0).astype(np.uint8) * 255
        result.append((key, label, short, mask, np.zeros_like(mask), bool(mask.any())))
    return result


def draw(canvas, regions, module):
    import cv2
    from src.wrinkle.report_region_overlays import REGION_GROUPS, REGION_OUTLINE_BGR, _normalize_region_defs, _largest_component_for_outline
    result = canvas.copy()
    normalized = _normalize_region_defs(regions)
    pixels = 0
    for key in REGION_GROUPS[module]["region_keys"]:
        if key not in normalized:
            continue
        if module == "08":
            # Preserve every measured component and exclusion boundary.
            mask = (normalized[key][2] > 0).astype(np.uint8)
            retrieval = cv2.RETR_LIST
        else:
            mask = _largest_component_for_outline(normalized[key][2])
            retrieval = cv2.RETR_EXTERNAL
        contours, _ = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, REGION_OUTLINE_BGR, 2, cv2.LINE_AA)
        pixels += sum(len(c) for c in contours)
    if pixels == 0:
        raise ValueError("wrinkle observation outlines missing")
    return result
