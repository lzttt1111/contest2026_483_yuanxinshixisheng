# -*- coding: utf-8 -*-
"""Shared VISIA-like face regions and deterministic image helpers.

The regions in this module are an engineering approximation of the masks
visible in the supplied VISIA reports.  They are built from the aligned
MediaPipe landmarks and the Preprocessor V2 skin mask.  The same masks are
used for detection, metrics, and drawing so a displayed region can never
silently differ from the quantified region.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.filters import frangi

from src.preprocess.image_preprocessor import (
    LEFT_EYE,
    LIPS,
    RIGHT_EYE,
    build_feature_exclusion_masks,
)
from src.engines.visia_unified_contour import (
    build_unified_visia_scope,
    draw_unified_visia_contour,
)


REGION_ORDER = ("forehead", "left_cheek", "right_cheek", "nose", "chin")
REGION_LABELS = {
    "forehead": "额头",
    "left_cheek": "左脸颊",
    "right_cheek": "右脸颊",
    "nose": "鼻部",
    "chin": "下巴",
}

# OpenCV BGR.  This is close to the cyan boundary used by VISIA reports.
VISIA_BOUNDARY_COLOR = (222, 198, 65)
PUBLIC_BOUNDARY_SAFETY_MARGIN_PX = 7
PUBLIC_BOUNDARY_LINE_THICKNESS_PX = 3
DISPLAY_CLOSE_KERNEL = 25
DISPLAY_SMOOTH_SIGMA = 6.0
DISPLAY_CHAIKIN_ITERATIONS = 3
DISPLAY_MIN_CONTOUR_AREA = 600.0
DISPLAY_FACE_CLOSE_KERNEL = 31
# 展示用脸部椭圆的额头扩展。Face Mesh 顶点落在发际线以下，31px 会只
# 包住半个额头；145px 让顶线延伸到完整可见额头，同时仍由语义皮肤 Mask
# 限制，避免把头发或背景纳入展示区域。只影响蓝色分区线。
DISPLAY_FOREHEAD_EXTENSION_KERNEL = 185

NOSE_LANDMARKS = (
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 97, 98, 327, 326,
)
# MediaPipe ``FACEMESH_NOSE`` 外沿的有序点。它只用于绘制圆润的鼻部
# 展示线，不参与鼻部分析 Mask、候选筛选或量化。相比把全部鼻点做凸包，
# 这条路径会沿鼻梁两侧和鼻翼自然弯曲，避免出现生硬的三角形直线。
DISPLAY_NOSE_OUTLINE_LANDMARKS = (
    168, 193, 122, 196, 3, 51, 45, 220, 115, 48, 64, 98,
    97, 2, 326, 327, 294, 278, 344, 440, 275, 281, 248, 419,
    351, 417,
)
MOUTH_LANDMARKS = (61, 291, 0, 17)
BROW_LANDMARKS = (105, 334)
FOREHEAD_CURVE_LANDMARKS = (70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300)
MOUTH_CURVE_LANDMARKS = (61, 146, 91, 181, 17, 314, 321, 375, 291)
# MediaPipe Face Mesh outer oval, in contour order (not a convex hull).  It is
# only a display prior: real semantic skin still decides which part is visible.
FACE_OVAL_LANDMARKS = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
NASOLABIAL_PATHS = (
    (209, 203, 206, 61),
    (429, 423, 426, 291),
)

# 鼻底到上唇之间的固定胡须高风险区。这里只用于实例后处理，不参与
# 底图着色、局部背景估计、连续分数图或有效皮肤 Mask 的计算。
MOUSTACHE_UPPER_LANDMARKS = (98, 97, 2, 326, 327)
MOUSTACHE_LOWER_LANDMARKS = (291, 267, 0, 37, 61)



@dataclass(frozen=True)
class VisiaRegionSet:
    """One shared, mutually exclusive VISIA-like region definition."""

    analysis_mask: np.ndarray
    feature_exclusion_mask: np.ndarray
    regions: dict[str, np.ndarray]
    display_regions: dict[str, np.ndarray]
    partial_face: bool
    display_contour: np.ndarray | None = None
    display_separator: np.ndarray | None = None
    scope_mask: np.ndarray | None = None
    public_boundary_safety_mask: np.ndarray | None = None


def _unified_public_boundary_mask(
    shape: tuple[int, int],
    contour: np.ndarray | None,
    separator: np.ndarray | None,
) -> np.ndarray:
    line = np.zeros(shape, dtype=np.uint8)
    if contour is not None and len(contour) >= 4:
        cv2.polylines(
            line,
            [np.asarray(contour, dtype=np.int32)],
            True,
            255,
            PUBLIC_BOUNDARY_LINE_THICKNESS_PX,
            lineType=cv2.LINE_8,
        )
    if separator is not None and len(separator) >= 2:
        cv2.polylines(
            line,
            [np.asarray(separator, dtype=np.int32)],
            False,
            255,
            PUBLIC_BOUNDARY_LINE_THICKNESS_PX,
            lineType=cv2.LINE_8,
        )
    kernel_size = 2 * PUBLIC_BOUNDARY_SAFETY_MARGIN_PX + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(line, kernel)


def polygon_mask(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    indices: tuple[int, ...] | list[int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    valid = [index for index in indices if 0 <= index < len(landmarks)]
    if len(valid) < 3:
        return mask
    points = np.rint(landmarks[valid]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    return mask


def _chaikin_open_curve(points: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Smooth an open landmark path while preserving both endpoints."""
    curve = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(curve) < 2:
        return curve
    for _ in range(max(0, int(iterations))):
        first = 0.75 * curve[:-1] + 0.25 * curve[1:]
        second = 0.25 * curve[:-1] + 0.75 * curve[1:]
        smoothed = np.empty((2 * len(first) + 2, 2), dtype=np.float32)
        smoothed[0] = curve[0]
        smoothed[-1] = curve[-1]
        smoothed[1:-1:2] = first
        smoothed[2:-1:2] = second
        curve = smoothed
    return curve


def build_nasolabial_corridor_mask(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    analysis_mask: np.ndarray,
    *,
    radius_px: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return anatomical nasolabial corridors and their center lines.

    The wide corridor is only a search prior.  Callers should not remove the
    whole corridor because real spots and pores may occur there.
    """
    corridor = np.zeros(shape, dtype=np.uint8)
    centerline = np.zeros(shape, dtype=np.uint8)
    points = np.asarray(landmarks, dtype=np.float32)
    for indices in NASOLABIAL_PATHS:
        if any(index < 0 or index >= len(points) for index in indices):
            continue
        curve = _chaikin_open_curve(points[list(indices), :2], iterations=3)
        polyline = np.rint(curve).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            centerline,
            [polyline],
            False,
            255,
            thickness=5,
            lineType=cv2.LINE_AA,
        )
        cv2.polylines(
            corridor,
            [polyline],
            False,
            255,
            thickness=2 * max(1, int(radius_px)) + 1,
            lineType=cv2.LINE_AA,
        )
    valid = (analysis_mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(corridor, valid), cv2.bitwise_and(centerline, valid)


def build_nasolabial_shadow_mask(
    image: np.ndarray,
    landmarks: np.ndarray,
    analysis_mask: np.ndarray,
    *,
    radius_px: int = 25,
    output_margin_px: int = 5,
    restrict_frangi_to_corridor: bool = False,
) -> np.ndarray:
    """Detect the actual dark line inside both nasolabial corridors.

    Geometry limits the search area; a dark local residual and Frangi
    black-ridge response provide photometric support.  This avoids deleting
    the entire nose-to-mouth corridor and preserves compact real features.
    """
    corridor, centerline = build_nasolabial_corridor_mask(
        analysis_mask.shape,
        landmarks,
        analysis_mask,
        radius_px=radius_px,
    )
    valid = analysis_mask > 0
    search = corridor > 0
    if np.count_nonzero(search) < 20:
        return np.zeros_like(analysis_mask)

    l_channel = (
        cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        / 255.0
    )
    local_background = masked_gaussian(l_channel, analysis_mask, 12.0)
    dark_residual = np.maximum(local_background - l_channel, 0.0)
    dark_values = dark_residual[search]
    median = float(np.median(dark_values))
    mad = float(np.median(np.abs(dark_values - median)))
    dark_threshold = max(median + 1.35 * max(1.4826 * mad, 0.004), 0.012)

    filled = l_channel.copy()
    filled[~valid] = float(np.median(l_channel[valid])) if np.any(valid) else 0.5
    frangi_sigmas = (1.0, 2.0, 3.0, 5.0)
    if restrict_frangi_to_corridor:
        points = cv2.findNonZero((corridor > 0).astype(np.uint8))
        x, y, width, height = cv2.boundingRect(points)
        # skimage's Gaussian derivatives are truncated at four sigma.  Keep a
        # wider halo so values inside the anatomical corridor are unaffected
        # by the crop boundary.
        padding = int(np.ceil(4.0 * max(frangi_sigmas))) + 4
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(filled.shape[1], x + width + padding)
        y1 = min(filled.shape[0], y + height + padding)
        cropped_response = frangi(
            filled[y0:y1, x0:x1],
            sigmas=frangi_sigmas,
            black_ridges=True,
        ).astype(np.float32)
        line_response = np.zeros_like(filled, dtype=np.float32)
        line_response[y0:y1, x0:x1] = cropped_response
    else:
        line_response = frangi(
            filled,
            sigmas=frangi_sigmas,
            black_ridges=True,
        ).astype(np.float32)
    line_values = line_response[search]
    high = max(float(np.percentile(line_values, 98.5)), 1e-8)
    line_response = np.clip(line_response / high, 0.0, 1.0)

    candidate = (
        search
        & (dark_residual >= dark_threshold)
        & (line_response >= 0.16)
    ).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )

    # Retain only line fragments that stay close to the anatomical path.
    path_support = cv2.dilate(
        centerline,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8),
        connectivity=8,
    )
    shadow = np.zeros_like(candidate)
    for label in range(1, component_count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 8 or not np.any(component & (path_support > 0)):
            continue
        shadow[component] = 255
    if output_margin_px > 0 and np.count_nonzero(shadow):
        radius = int(output_margin_px)
        shadow = cv2.dilate(
            shadow,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius + 1, 2 * radius + 1),
            ),
        )
    return cv2.bitwise_and(shadow, (analysis_mask > 0).astype(np.uint8) * 255)


def build_moustache_feature_exclusion_mask(
    shape: tuple[int, int],
    landmarks: np.ndarray,
    analysis_mask: np.ndarray,
    *,
    margin_px: int = 12,
) -> np.ndarray:
    """Build a deterministic nose-base-to-upper-lip instance exclusion.

    Short moustache stubble is visually indistinguishable from pores or fine
    texture in a single white-light RGB photograph.  Product requirements
    therefore exclude this anatomical strip unconditionally.  The returned
    mask is intended only for deleting final feature centres and their marker
    masks; callers must not use it to alter the image, score map or skin mask.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.asarray(landmarks, dtype=np.float32)
    required = MOUSTACHE_UPPER_LANDMARKS + MOUSTACHE_LOWER_LANDMARKS
    if points.ndim != 2 or points.shape[1] < 2 or any(
        index < 0 or index >= len(points) for index in required
    ):
        return mask

    upper = points[list(MOUSTACHE_UPPER_LANDMARKS), :2]
    lower = points[list(MOUSTACHE_LOWER_LANDMARKS), :2]
    # Follow the anatomical upper and lower curves instead of using a broad
    # rectangle, then slightly dilate to cover stubble touching either edge.
    polygon = np.vstack([upper, lower]).astype(np.float32)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)], 255)
    if margin_px > 0:
        radius = int(margin_px)
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius + 1, 2 * radius + 1),
            ),
        )
    return cv2.bitwise_and(
        mask,
        (np.asarray(analysis_mask) > 0).astype(np.uint8) * 255,
    )


def _smooth_binary_mask(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth raster stair-steps without inventing pixels outside the mask."""
    binary = (mask > 0).astype(np.uint8) * 255
    if not np.count_nonzero(binary):
        return binary
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    soft = cv2.GaussianBlur(
        closed.astype(np.float32),
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return (soft >= 127.5).astype(np.uint8) * 255


def _display_face_envelope(skin_mask: np.ndarray) -> np.ndarray:
    """Build a smooth *display-only* envelope from the real skin silhouette.

    The analysis mask deliberately contains holes for eyes, lips, nostrils and
    hair.  It must stay that way for algorithms, but using a landmark convex
    hull for drawing created the rectangular forehead and off-canvas edge seen
    in the Spots overlay.  For drawing we close small segmentation gaps and
    fill only the largest real face contour.  No pixels from this envelope are
    used for detection or metrics.
    """
    binary = (skin_mask > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return binary
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (DISPLAY_FACE_CLOSE_KERNEL, DISPLAY_FACE_CLOSE_KERNEL),
        ),
    )
    points = cv2.findNonZero(closed)
    if points is None or len(points) < 3:
        return binary
    envelope = np.zeros_like(binary)
    # This is intentionally the deployed smooth-overlay construction: first
    # constrain the semantic pixels with the anatomical oval, then fill that
    # shape's hull.  It yields one clean forehead arc and facial outline;
    # tracing the raw semantic edge instead follows individual hair strands.
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(envelope, hull, 255)
    return envelope


def _ordered_face_oval_mask(
    shape: tuple[int, int], landmarks: np.ndarray
) -> np.ndarray:
    """Create the smooth anatomical face-oval prior without a convex hull."""
    mask = np.zeros(shape, dtype=np.uint8)
    valid = [index for index in FACE_OVAL_LANDMARKS if index < len(landmarks)]
    if len(valid) < 3:
        return mask
    points = np.rint(landmarks[valid, :2]).astype(np.int32)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    return mask


def _interpolated_landmark_curve(
    landmarks: np.ndarray,
    indices: tuple[int, ...],
    width: int,
    *,
    y_offset: float = 0.0,
) -> np.ndarray:
    """Return a stable per-column curve through the selected landmarks."""
    points = np.asarray(
        [
            landmarks[index, :2]
            for index in indices
            if 0 <= index < len(landmarks)
        ],
        dtype=np.float32,
    )
    if len(points) < 2:
        return np.zeros(width, dtype=np.float32)
    points = points[np.argsort(points[:, 0])]
    unique_x, unique_indices = np.unique(points[:, 0], return_index=True)
    unique_y = points[unique_indices, 1] + float(y_offset)
    if len(unique_x) < 2:
        return np.full(width, float(unique_y[0]), dtype=np.float32)
    curve = np.interp(
        np.arange(width, dtype=np.float32),
        unique_x,
        unique_y,
        left=float(unique_y[0]),
        right=float(unique_y[-1]),
    ).astype(np.float32)
    curve = cv2.GaussianBlur(curve.reshape(1, -1), (0, 0), sigmaX=12.0).ravel()
    return curve


def build_visia_regions(
    analysis_image: np.ndarray,
    skin_mask: np.ndarray,
    landmarks: np.ndarray,
    quality_flags: list[str] | tuple[str, ...] | None = None,
    *,
    include_chin: bool = True,
    mode: str = "full",
    feature_margin_px: int = 10,
) -> VisiaRegionSet:
    """Build smooth forehead, cheek, nose and chin analysis regions.

    ``mode="texture"`` follows the supplied reports and analyses cheeks only.
    ``mode="pores"`` uses forehead, cheeks and nose. ``mode="full"`` uses all
    five regions. A side face naturally contains fewer valid pixels because
    every region is intersected with the real Preprocessor V2 skin mask.
    """
    if analysis_image.shape[:2] != skin_mask.shape:
        raise ValueError("analysis_image and skin_mask shapes do not match")
    if landmarks.ndim != 2 or landmarks.shape[1] < 2 or len(landmarks) <= 334:
        raise ValueError("aligned MediaPipe face landmarks are required")

    height, width = skin_mask.shape
    base_skin = (skin_mask > 0).astype(np.uint8) * 255

    feature_masks = build_feature_exclusion_masks(
        analysis_image,
        landmarks[:, :2].astype(np.float32),
        use_photometric_nostrils=True,
    )
    exclusions = feature_masks["feature_exclusions"]
    if feature_margin_px > 0:
        radius = int(feature_margin_px)
        exclusions = cv2.dilate(
            exclusions,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius + 1, 2 * radius + 1),
            ),
        )
    safe_skin = cv2.bitwise_and(base_skin, cv2.bitwise_not(exclusions))

    yy, xx = np.indices((height, width))
    valid_y, valid_x = np.where(safe_skin > 0)
    if valid_y.size < 500:
        raise ValueError("valid skin region is too small")

    centre_x = int(
        np.clip(
            landmarks[1, 0] if len(landmarks) > 1 else np.median(valid_x),
            0,
            width - 1,
        )
    )
    brow_y = int(
        np.clip(
            min(landmarks[BROW_LANDMARKS[0], 1], landmarks[BROW_LANDMARKS[1], 1]),
            int(valid_y.min()),
            int(valid_y.max()),
        )
    )
    mouth_y = int(
        np.clip(
            np.mean(landmarks[list(MOUTH_LANDMARKS), 1]),
            int(valid_y.min()),
            int(valid_y.max()),
        )
    )

    nose = polygon_mask(skin_mask.shape, landmarks, list(NOSE_LANDMARKS))
    nose = cv2.dilate(
        nose,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    nose = cv2.bitwise_and(nose, safe_skin)

    # Detection follows ``safe_skin`` exactly.  The overlay uses an enlarged
    # anatomical oval: the explicit forehead extension reaches the hairline,
    # while the semantic skin silhouette keeps the line off hair/background.
    display_oval = _ordered_face_oval_mask(skin_mask.shape, landmarks)
    display_oval = cv2.dilate(
        display_oval,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                DISPLAY_FOREHEAD_EXTENSION_KERNEL,
                DISPLAY_FOREHEAD_EXTENSION_KERNEL,
            ),
        ),
    )
    display_safe = _display_face_envelope(
        cv2.bitwise_and(base_skin, display_oval)
    )

    skin = safe_skin > 0
    display_skin = display_safe > 0
    forehead = ((yy < brow_y) & skin).astype(np.uint8) * 255
    # The original prototype split the forehead with one constant y value,
    # producing the conspicuous ruler-straight cyan line.  For display only,
    # follow the real eyebrow/bridge curvature.  Analysis masks remain
    # unchanged, so this visual cleanup cannot alter any quantified count.
    forehead_curve = _interpolated_landmark_curve(
        landmarks,
        FOREHEAD_CURVE_LANDMARKS,
        width,
        y_offset=-10.0,
    )
    if not np.any(forehead_curve):
        forehead_curve = np.full(width, float(brow_y), dtype=np.float32)
    mouth_curve = _interpolated_landmark_curve(
        landmarks,
        MOUTH_CURVE_LANDMARKS,
        width,
        y_offset=14.0,
    )
    if not np.any(mouth_curve):
        mouth_curve = np.full(width, float(mouth_y), dtype=np.float32)
    mouth_curve = np.clip(mouth_curve, 0.0, float(height - 1))
    display_forehead = (
        (yy < forehead_curve[None, :]) & display_skin
    ).astype(np.uint8) * 255
    middle = (yy >= brow_y) & (yy <= mouth_y) & skin & (nose == 0)
    display_nose = polygon_mask(skin_mask.shape, landmarks, list(NOSE_LANDMARKS))
    display_nose = cv2.dilate(
        display_nose,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    display_nose = cv2.bitwise_and(display_nose, display_safe)
    display_middle = (
        (yy >= forehead_curve[None, :])
        & (yy <= mouth_curve[None, :])
        & display_skin
        & (display_nose == 0)
    )
    left_cheek = (middle & (xx < centre_x)).astype(np.uint8) * 255
    right_cheek = (middle & (xx >= centre_x)).astype(np.uint8) * 255
    display_left_cheek = (
        display_middle & (xx < centre_x)
    ).astype(np.uint8) * 255
    display_right_cheek = (
        display_middle & (xx >= centre_x)
    ).astype(np.uint8) * 255
    chin = ((yy > mouth_y) & skin).astype(np.uint8) * 255
    display_chin = ((yy > mouth_curve[None, :]) & display_skin).astype(np.uint8) * 255

    raw_regions = {
        "forehead": forehead,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
        "nose": nose,
        "chin": chin,
    }
    raw_display_regions = {
        "forehead": display_forehead,
        "left_cheek": display_left_cheek,
        "right_cheek": display_right_cheek,
        "nose": display_nose,
        "chin": display_chin,
    }
    flags = set(quality_flags or [])
    if "FOREHEAD_OCCLUDED" in flags:
        raw_regions["forehead"] = np.zeros_like(safe_skin)
        raw_display_regions["forehead"] = np.zeros_like(safe_skin)
    enabled = set(REGION_ORDER)
    if mode == "texture":
        enabled = {"left_cheek", "right_cheek"}
    elif mode == "pores":
        enabled = {"forehead", "left_cheek", "right_cheek", "nose"}
    elif mode != "full":
        raise ValueError(f"unknown VISIA region mode: {mode}")
    if not include_chin:
        enabled.discard("chin")

    regions: dict[str, np.ndarray] = {}
    display_regions: dict[str, np.ndarray] = {}
    occupied = np.zeros_like(safe_skin)
    display_occupied = np.zeros_like(safe_skin)
    # Nose is assigned first so it cannot also be counted as cheek.
    for name in ("nose", "forehead", "left_cheek", "right_cheek", "chin"):
        mask = raw_regions[name] if name in enabled else np.zeros_like(safe_skin)
        mask = _smooth_binary_mask(mask)
        mask = cv2.bitwise_and(mask, safe_skin)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(occupied))
        regions[name] = mask
        occupied = cv2.bitwise_or(occupied, mask)
        display_mask = (
            raw_display_regions[name]
            if name in enabled
            else np.zeros_like(safe_skin)
        )
        display_mask = _smooth_binary_mask(display_mask, sigma=3.0)
        display_mask = cv2.bitwise_and(display_mask, display_safe)
        display_mask = cv2.bitwise_and(
            display_mask, cv2.bitwise_not(display_occupied)
        )
        display_regions[name] = display_mask
        display_occupied = cv2.bitwise_or(display_occupied, display_mask)

    partial_face = "PARTIAL_FACE" in flags

    # The validated 2026-08-05 VISIA outline is now the single source of
    # truth for both display and quantification scope.  Detector formulas and
    # thresholds remain unchanged; this is a deterministic spatial gate.
    display_contour, scope_mask, display_separator = build_unified_visia_scope(
        np.asarray(landmarks, dtype=np.float32),
        skin_mask.shape,
        display_safe,
        partial_face=partial_face,
        left_visible_area=int(np.count_nonzero(raw_regions["left_cheek"])),
        right_visible_area=int(np.count_nonzero(raw_regions["right_cheek"])),
    )
    if display_contour is None or np.count_nonzero(scope_mask) < 500:
        # Geometry failure must not silently discard a valid face.  The old
        # mutually-exclusive masks remain as a deterministic fallback.
        scope_mask = display_occupied.copy()

    # Only targets inside the visible closed line may be detected, displayed
    # or counted.  The semantic skin/exclusion masks are still authoritative,
    # so filling the contour never adds eyes, lips, hair or background.
    occupied = np.zeros_like(safe_skin)
    display_occupied = np.zeros_like(safe_skin)
    for name in ("nose", "forehead", "left_cheek", "right_cheek", "chin"):
        clipped = cv2.bitwise_and(regions[name], scope_mask)
        clipped = cv2.bitwise_and(clipped, cv2.bitwise_not(occupied))
        regions[name] = clipped
        occupied = cv2.bitwise_or(occupied, clipped)

        display_clipped = cv2.bitwise_and(display_regions[name], scope_mask)
        display_clipped = cv2.bitwise_and(
            display_clipped,
            cv2.bitwise_not(display_occupied),
        )
        display_regions[name] = display_clipped
        display_occupied = cv2.bitwise_or(display_occupied, display_clipped)

    public_boundary_safety = _unified_public_boundary_mask(
        safe_skin.shape,
        display_contour,
        display_separator,
    )
    scope_mask = cv2.bitwise_and(
        scope_mask,
        cv2.bitwise_not(public_boundary_safety),
    )
    occupied = np.zeros_like(safe_skin)
    for name in ("nose", "forehead", "left_cheek", "right_cheek", "chin"):
        regions[name] = cv2.bitwise_and(regions[name], scope_mask)
        occupied = cv2.bitwise_or(occupied, regions[name])

    return VisiaRegionSet(
        analysis_mask=occupied,
        feature_exclusion_mask=exclusions,
        regions=regions,
        display_regions=display_regions,
        partial_face=partial_face,
        display_contour=display_contour,
        display_separator=display_separator,
        scope_mask=scope_mask,
        public_boundary_safety_mask=public_boundary_safety,
    )


def masked_gaussian(
    channel: np.ndarray,
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Gaussian background estimate that never samples outside the mask."""
    valid = (mask > 0).astype(np.float32)
    values = channel.astype(np.float32)
    numerator = cv2.GaussianBlur(
        values * valid,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    denominator = cv2.GaussianBlur(
        valid,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )
    result = numerator / np.maximum(denominator, 1e-6)
    result[valid == 0] = 0.0
    return result.astype(np.float32)


def regional_positive_z(
    values: np.ndarray,
    region_set: VisiaRegionSet,
    sigma_floor: float = 0.01,
) -> np.ndarray:
    """Positive robust Z score calculated independently in each face region."""
    output = np.zeros_like(values, dtype=np.float32)
    for mask in region_set.regions.values():
        valid = mask > 0
        samples = values[valid].astype(np.float32)
        if samples.size < 100:
            continue
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        scale = max(1.4826 * mad, float(sigma_floor))
        output[valid] = np.maximum(
            (values[valid].astype(np.float32) - median) / scale,
            0.0,
        )
    output[region_set.analysis_mask == 0] = 0.0
    return np.clip(output, 0.0, 20.0)


def robust_unit_map(
    values: np.ndarray,
    mask: np.ndarray,
    low: float = 5.0,
    high: float = 98.0,
) -> np.ndarray:
    valid = mask > 0
    output = np.zeros_like(values, dtype=np.float32)
    if np.count_nonzero(valid) < 100:
        return output
    lo, hi = np.percentile(values[valid], [low, high])
    if hi <= lo + 1e-6:
        return output
    output[valid] = np.clip(
        (values[valid].astype(np.float32) - float(lo)) / float(hi - lo),
        0.0,
        1.0,
    )
    return output


def assign_region(
    x: float,
    y: float,
    regions: dict[str, np.ndarray],
) -> str:
    px, py = int(round(x)), int(round(y))
    for name in ("nose", "forehead", "left_cheek", "right_cheek", "chin"):
        mask = regions.get(name)
        if (
            mask is not None
            and 0 <= py < mask.shape[0]
            and 0 <= px < mask.shape[1]
            and mask[py, px] > 0
        ):
            return name
    return "other"


def _chaikin_closed_curve(
    contour: np.ndarray,
    iterations: int = DISPLAY_CHAIKIN_ITERATIONS,
) -> np.ndarray:
    """Smooth a closed contour using the proven deployed RBX overlay method."""
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 4:
        return contour.astype(np.int32)
    sample_step = max(1, len(points) // 220)
    points = points[::sample_step]
    for _ in range(iterations):
        following = np.roll(points, -1, axis=0)
        first = 0.75 * points + 0.25 * following
        second = 0.25 * points + 0.75 * following
        smoothed = np.empty((len(points) * 2, 2), dtype=np.float32)
        smoothed[0::2] = first
        smoothed[1::2] = second
        points = smoothed
    return np.rint(points).astype(np.int32).reshape(-1, 1, 2)


def _clean_display_mask(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if not np.count_nonzero(binary):
        return binary
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (DISPLAY_CLOSE_KERNEL, DISPLAY_CLOSE_KERNEL),
        ),
    )
    soft = cv2.GaussianBlur(
        binary,
        (0, 0),
        sigmaX=DISPLAY_SMOOTH_SIGMA,
        sigmaY=DISPLAY_SMOOTH_SIGMA,
        borderType=cv2.BORDER_REFLECT,
    )
    return (soft >= 127.0).astype(np.uint8) * 255


def _draw_smooth_mask_contours(
    canvas: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    *,
    largest_only: bool,
) -> None:
    contours, _ = cv2.findContours(
        _clean_display_mask(mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    contours = [
        contour
        for contour in contours
        if abs(float(cv2.contourArea(contour))) >= DISPLAY_MIN_CONTOUR_AREA
    ]
    if not contours:
        return
    if largest_only:
        contours = [max(contours, key=lambda item: abs(cv2.contourArea(item)))]
    for contour in contours:
        smooth = _chaikin_closed_curve(contour)
        cv2.polylines(
            canvas,
            [smooth],
            True,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )


def _draw_smooth_nose_landmark_contour(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> bool:
    """Draw a display-only anatomical nose curve.

    Returns ``True`` when enough MediaPipe landmarks were available.  The
    caller can fall back to the legacy raster-mask outline otherwise.
    """
    points = np.asarray(landmarks, dtype=np.float32)
    valid = [
        index
        for index in DISPLAY_NOSE_OUTLINE_LANDMARKS
        if 0 <= index < len(points)
    ]
    if len(valid) < len(DISPLAY_NOSE_OUTLINE_LANDMARKS):
        return False
    contour = np.rint(points[valid, :2]).astype(np.int32).reshape(-1, 1, 2)
    smooth = _chaikin_closed_curve(contour, iterations=4)
    cv2.polylines(
        canvas,
        [smooth],
        True,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )
    return True


def draw_region_boundaries(
    image: np.ndarray,
    regions: dict[str, np.ndarray],
    color: tuple[int, int, int] = VISIA_BOUNDARY_COLOR,
    thickness: int = 3,
    *,
    partial_face: bool = False,
    nose_style: str = "hidden",
    landmarks: np.ndarray | None = None,
    closed_contour: np.ndarray | None = None,
    separator_contour: np.ndarray | None = None,
) -> np.ndarray:
    """Draw only smooth outer region contours.

    Eyes, lips and other feature exclusions remain active in the detection
    mask, but are deliberately not traced as extra cyan rings.  The nose is
    also hidden by default because its narrow triangular boundary distracts
    from the feature overlay.  Its analysis region remains fully active.  This
    keeps the user-facing partition legible while preserving the exact
    analysis scope.
    """
    if nose_style not in {"mask", "smooth", "hidden"}:
        raise ValueError(f"unknown nose boundary style: {nose_style}")

    output = image.copy()
    if closed_contour is not None and len(closed_contour) >= 4:
        return draw_unified_visia_contour(
            output,
            closed_contour,
            color,
            thickness,
            separator_contour,
        )
    if partial_face:
        # A strongly yawed/partial face does not have enough visible anatomy
        # for a meaningful five-zone partition.  Draw one truthful visible
        # envelope instead of several long triangular or vertical artefacts.
        visible = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in regions.values():
            visible = cv2.bitwise_or(visible, mask)
        _draw_smooth_mask_contours(
            output,
            visible,
            color,
            thickness,
            largest_only=True,
        )
        return output

    # Draw cheeks as one logical group. They remain two disconnected visible
    # components around the nose, but no duplicate centre seam is produced.
    groups = [
        regions.get("forehead"),
        cv2.bitwise_or(
            regions.get("left_cheek", np.zeros(image.shape[:2], np.uint8)),
            regions.get("right_cheek", np.zeros(image.shape[:2], np.uint8)),
        ),
        regions.get("chin"),
    ]
    for mask in groups:
        if mask is None or np.count_nonzero(mask) < 100:
            continue
        _draw_smooth_mask_contours(
            output,
            mask,
            color,
            thickness,
            largest_only=False,
        )

    if nose_style == "hidden":
        return output
    if (
        nose_style == "smooth"
        and landmarks is not None
        and _draw_smooth_nose_landmark_contour(
            output,
            landmarks,
            color,
            thickness,
        )
    ):
        return output

    nose_mask = regions.get("nose")
    if nose_mask is not None and np.count_nonzero(nose_mask) >= 100:
        _draw_smooth_mask_contours(
            output,
            nose_mask,
            color,
            thickness,
            largest_only=False,
        )
    return output


def compact_region_counts(
    locations: list[dict],
    partial_face: bool,
) -> tuple[list[str], list[int]]:
    counts = {name: 0 for name in REGION_ORDER}
    for item in locations:
        region = str(item.get("region", "other"))
        if region in counts:
            counts[region] += 1
    if partial_face:
        return ["总计"], [len(locations)]
    headers = ["总计"] + [REGION_LABELS[name] for name in REGION_ORDER]
    values = [len(locations)] + [counts[name] for name in REGION_ORDER]
    return headers, values
