from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
from typing import Any
import urllib.request

import cv2
import numpy as np
from typing_extensions import assert_never

from src.capture_profile import CaptureProfile, capture_profile_from_environment
from src.preprocess.profile_mask_policy import apply_profile_mask_policy
from .artifact_policy import AcneArtifactPolicy, AcnePhaseTimings
from .face_regions import build_region_masks, draw_region_debug
from .image_io import read_image_bgr, write_image
from .quality_control import assess_image_quality

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models" / "acne"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_multiclass_256x256/float32/"
    "latest/selfie_multiclass_256x256.tflite"
)

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
RIGHT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
NOSTRIL_CENTERS = [98, 327]
VISIBLE_REGION_NAMES = [
    "left_eye",
    "right_eye",
    "left_eyebrow",
    "right_eyebrow",
    "lips",
    "left_nostril",
    "right_nostril",
    "face_oval",
]


class FaceInputMode(str, Enum):
    FULL_FACE = "full_face"
    PARTIAL_FACE = "partial_face"
    SKIN_PATCH = "skin_patch"
    NO_FACE = "no_face"


class PreprocessError(RuntimeError):
    """Base preprocessing error."""


class NoFaceDetectedError(PreprocessError):
    """Raised when no face landmarks are detected."""


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 1024
    assets_dir: Path = MODEL_ROOT
    face_landmarker_path: Path = MODEL_ROOT / "face_landmarker.task"
    segmenter_path: Path = MODEL_ROOT / "selfie_multiclass_256x256.tflite"
    min_model_bytes: int = 100_000
    num_faces: int = 2
    crop_pad_x_ratio: float = 0.06
    crop_pad_top_ratio: float = 0.10
    crop_pad_bottom_ratio: float = 0.04
    eye_dilate: int = 3
    mouth_dilate: int = 2
    nostril_dilate: int = 2
    eyebrow_dilate: int = 1
    skin_patch_min_skin_ratio: float = 0.08


@dataclass
class LandmarkQuality:
    is_reliable: bool
    full_face_score: float
    partial_face_score: float
    inside_ratio: float
    face_bbox_ratio: float
    border_touch_ratio: float
    geometric_checks: dict[str, bool]
    visible_regions: dict[str, bool]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_reliable": self.is_reliable,
            "is_reliable_full_face": self.is_reliable,
            "full_face_score": self.full_face_score,
            "partial_face_score": self.partial_face_score,
            "inside_ratio": self.inside_ratio,
            "face_bbox_ratio": self.face_bbox_ratio,
            "border_touch_ratio": self.border_touch_ratio,
            "geometric_checks": self.geometric_checks,
            "visible_regions": self.visible_regions,
            "reasons": self.reasons,
        }


@dataclass
class PreprocessResult:
    original_bgr: np.ndarray
    standardized_bgr: np.ndarray
    masked_face_bgr: np.ndarray
    skin_mask: np.ndarray
    forbidden_mask: np.ndarray
    landmarks_xy: np.ndarray
    crop_box_xyxy: tuple[int, int, int, int]
    square_offset_xy: tuple[int, int]
    resize_scale: float
    metadata: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_model(path: Path, url: str, min_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= min_bytes:
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    logger.info("downloading MediaPipe model: %s", path.name)
    with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    size = tmp_path.stat().st_size
    if size < min_bytes:
        tmp_path.unlink(missing_ok=True)
        raise PreprocessError(f"downloaded model too small: {path.name}, {size} bytes")
    tmp_path.replace(path)


def ensure_mediapipe_models(config: PreprocessConfig) -> dict[str, Any]:
    _ensure_model(config.face_landmarker_path, LANDMARKER_URL, config.min_model_bytes)
    _ensure_model(config.segmenter_path, SEGMENTER_URL, config.min_model_bytes)
    return {
        "face_landmarker": {
            "path": str(config.face_landmarker_path),
            "bytes": config.face_landmarker_path.stat().st_size,
            "sha256": sha256_file(config.face_landmarker_path),
        },
        "segmenter": {
            "path": str(config.segmenter_path),
            "bytes": config.segmenter_path.stat().st_size,
            "sha256": sha256_file(config.segmenter_path),
        },
    }


def load_mediapipe_models(config: PreprocessConfig):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    ensure_mediapipe_models(config)
    seg_options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.segmenter_path)),
        output_category_mask=True,
    )
    lm_options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.face_landmarker_path)),
        num_faces=config.num_faces,
    )
    return (
        vision.ImageSegmenter.create_from_options(seg_options),
        vision.FaceLandmarker.create_from_options(lm_options),
        mp,
    )


def _points(landmarks_xy: np.ndarray, indices: list[int]) -> np.ndarray:
    return landmarks_xy[indices].astype(np.int32)


def _fill_poly(mask: np.ndarray, landmarks_xy: np.ndarray, indices: list[int], value: int = 255) -> None:
    pts = _points(landmarks_xy, indices)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], value)


def _largest_landmark_face(face_landmarks: list[Any], width: int, height: int) -> tuple[int, np.ndarray]:
    best_index = 0
    best_area = -1.0
    best_xy: np.ndarray | None = None
    for index, landmarks in enumerate(face_landmarks):
        xy = np.array([(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float32)
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        area = float((x2 - x1) * (y2 - y1))
        if area > best_area:
            best_index = index
            best_area = area
            best_xy = xy
    if best_xy is None:
        raise NoFaceDetectedError("no face landmarks detected")
    return best_index, best_xy


def _inside_points(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    return (
        (points[:, 0] >= 0)
        & (points[:, 0] < w)
        & (points[:, 1] >= 0)
        & (points[:, 1] < h)
    )


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def validate_polygon(
    points: np.ndarray,
    image_shape: tuple[int, int],
    min_inside_ratio: float = 0.85,
    min_area: float = 10.0,
    max_area_ratio: float = 0.25,
) -> bool:
    h, w = image_shape
    if len(points) < 3:
        return False
    inside = _inside_points(points, image_shape)
    if float(inside.mean()) < min_inside_ratio:
        return False
    if not inside.all():
        return False
    area = _polygon_area(points)
    if area < min_area:
        return False
    if area > (h * w * max_area_ratio):
        return False
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    if x2 <= x1 or y2 <= y1:
        return False
    return True


def _region_center(landmarks_xy: np.ndarray, indices: list[int]) -> np.ndarray:
    return landmarks_xy[indices].mean(axis=0)


def _region_visible(
    landmarks_xy: np.ndarray,
    indices: list[int],
    image_shape: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    min_inside_ratio: float = 0.9,
) -> bool:
    h, w = image_shape
    points = landmarks_xy[indices]
    min_area = h * w * min_area_ratio
    return validate_polygon(points, image_shape, min_inside_ratio, min_area, max_area_ratio)


def evaluate_landmark_quality(landmarks_xy: np.ndarray, image_shape: tuple[int, int]) -> LandmarkQuality:
    h, w = image_shape
    image_area = max(1, h * w)
    inside = _inside_points(landmarks_xy, image_shape)
    inside_ratio = float(inside.mean())
    inside_points = landmarks_xy[inside] if inside.any() else landmarks_xy
    x1, y1 = inside_points.min(axis=0)
    x2, y2 = inside_points.max(axis=0)
    bbox_w = max(0.0, float(x2 - x1))
    bbox_h = max(0.0, float(y2 - y1))
    face_bbox_ratio = float((bbox_w * bbox_h) / image_area)
    margin = max(4.0, min(h, w) * 0.015)
    border_hits = [
        x1 <= margin,
        y1 <= margin,
        x2 >= w - 1 - margin,
        y2 >= h - 1 - margin,
    ]
    border_touch_ratio = float(sum(border_hits) / len(border_hits))

    left_eye_center = _region_center(landmarks_xy, LEFT_EYE)
    right_eye_center = _region_center(landmarks_xy, RIGHT_EYE)
    left_brow_center = _region_center(landmarks_xy, LEFT_EYEBROW)
    right_brow_center = _region_center(landmarks_xy, RIGHT_EYEBROW)
    mouth_center = _region_center(landmarks_xy, MOUTH_OUTER)
    nose_center = landmarks_xy[[1, 2, 98, 327]].mean(axis=0)
    chin_y = float(landmarks_xy[152, 1])
    eye_center = (left_eye_center + right_eye_center) / 2.0
    brow_center = (left_brow_center + right_brow_center) / 2.0
    eye_distance = abs(float(right_eye_center[0] - left_eye_center[0]))

    left_eye_area = _polygon_area(landmarks_xy[LEFT_EYE])
    right_eye_area = _polygon_area(landmarks_xy[RIGHT_EYE])
    mouth_area = _polygon_area(landmarks_xy[MOUTH_OUTER])
    oval_area = _polygon_area(landmarks_xy[FACE_OVAL])
    eye_boxes_separate = max(landmarks_xy[LEFT_EYE][:, 0]) < min(landmarks_xy[RIGHT_EYE][:, 0]) or max(landmarks_xy[RIGHT_EYE][:, 0]) < min(landmarks_xy[LEFT_EYE][:, 0])

    order_ok = (
        float(brow_center[1]) < float(eye_center[1])
        < float(nose_center[1])
        < float(mouth_center[1])
        < chin_y
    )
    eyes_ok = (
        left_eye_area > image_area * 0.00003
        and right_eye_area > image_area * 0.00003
        and eye_distance > bbox_w * 0.18
        and eye_distance < bbox_w * 0.75
        and eye_boxes_separate
    )
    mouth_ok = mouth_area > image_area * 0.00008 and mouth_area < image_area * 0.05
    oval_ok = (
        validate_polygon(
            landmarks_xy[FACE_OVAL],
            image_shape,
            min_inside_ratio=0.92,
            min_area=image_area * 0.06,
            max_area_ratio=0.9,
        )
        and oval_area > image_area * 0.06
    )
    bbox_ok = 0.12 <= face_bbox_ratio <= 0.92 and bbox_w > w * 0.28 and bbox_h > h * 0.40
    border_ok = border_touch_ratio <= 0.5

    visible_regions = {
        "left_eye": _region_visible(landmarks_xy, LEFT_EYE, image_shape, 0.00003, 0.04),
        "right_eye": _region_visible(landmarks_xy, RIGHT_EYE, image_shape, 0.00003, 0.04),
        "left_eyebrow": _region_visible(landmarks_xy, LEFT_EYEBROW, image_shape, 0.00002, 0.04),
        "right_eyebrow": _region_visible(landmarks_xy, RIGHT_EYEBROW, image_shape, 0.00002, 0.04),
        "lips": _region_visible(landmarks_xy, MOUTH_OUTER, image_shape, 0.00008, 0.06),
        "left_nostril": bool(_inside_points(landmarks_xy[[98]], image_shape)[0] and bbox_w > 1),
        "right_nostril": bool(_inside_points(landmarks_xy[[327]], image_shape)[0] and bbox_w > 1),
        "face_oval": oval_ok,
    }

    geometric_checks = {
        "inside_ratio_ok": inside_ratio >= 0.94,
        "bbox_ok": bbox_ok,
        "border_ok": border_ok,
        "vertical_order_ok": order_ok,
        "eyes_ok": eyes_ok,
        "mouth_ok": mouth_ok,
        "face_oval_ok": oval_ok,
    }
    reasons = [name for name, ok in geometric_checks.items() if not ok]
    full_score = sum(1 for ok in geometric_checks.values() if ok) / len(geometric_checks)
    partial_visible_count = sum(
        int(visible_regions[name])
        for name in ["left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "lips", "left_nostril", "right_nostril"]
    )
    partial_score = min(1.0, partial_visible_count / 3.0)
    is_full_reliable = (
        full_score >= 0.85
        and inside_ratio >= 0.94
        and border_ok
        and bbox_ok
        and order_ok
        and eyes_ok
        and mouth_ok
        and visible_regions["left_eye"]
        and visible_regions["right_eye"]
        and visible_regions["lips"]
    )
    return LandmarkQuality(
        is_reliable=is_full_reliable,
        full_face_score=float(full_score),
        partial_face_score=float(partial_score),
        inside_ratio=inside_ratio,
        face_bbox_ratio=face_bbox_ratio,
        border_touch_ratio=border_touch_ratio,
        geometric_checks=geometric_checks,
        visible_regions=visible_regions,
        reasons=reasons,
    )


def _build_original_masks(
    image_shape: tuple[int, int],
    landmarks_xy: np.ndarray | None,
    face_skin_mask: np.ndarray,
    config: PreprocessConfig,
    input_mode: FaceInputMode,
    landmark_quality: LandmarkQuality | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_shape
    oval_mask = np.zeros((h, w), dtype=np.uint8)

    if (
        input_mode == FaceInputMode.FULL_FACE
        and landmarks_xy is not None
        and landmark_quality is not None
        and landmark_quality.visible_regions.get("face_oval", False)
    ):
        safe_oval_points = np.clip(_points(landmarks_xy, FACE_OVAL), [0, 0], [w - 1, h - 1])
        cv2.fillConvexPoly(oval_mask, cv2.convexHull(safe_oval_points), 255)

        pt_234 = landmarks_xy[234]
        pt_454 = landmarks_xy[454]
        x_left, x_right = sorted([float(pt_234[0]), float(pt_454[0])])
        side_y_left = float(pt_234[1] if pt_234[0] <= pt_454[0] else pt_454[1])
        side_y_right = float(pt_454[1] if pt_454[0] >= pt_234[0] else pt_234[1])
        face_top = float(landmarks_xy[:, 1].min())
        face_bottom = float(landmarks_xy[:, 1].max())
        face_h = max(1.0, face_bottom - face_top)
        forehead_top = max(0.0, face_top - 0.22 * face_h)
        forehead_poly = np.array(
            [
                [x_left, side_y_left],
                [max(0.0, x_left - 0.04 * (x_right - x_left)), forehead_top],
                [min(float(w - 1), x_right + 0.04 * (x_right - x_left)), forehead_top],
                [x_right, side_y_right],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(oval_mask, forehead_poly, 255)
        refined_skin = cv2.bitwise_and(face_skin_mask, oval_mask)
    else:
        refined_skin = face_skin_mask.copy()

    forbidden = np.zeros((h, w), dtype=np.uint8)
    if landmarks_xy is None or landmark_quality is None:
        return refined_skin, forbidden, oval_mask

    for region_name, indices, dilation in [
        ("left_eye", LEFT_EYE, config.eye_dilate),
        ("right_eye", RIGHT_EYE, config.eye_dilate),
        ("lips", MOUTH_OUTER, config.mouth_dilate),
        ("lips", MOUTH_INNER, config.mouth_dilate),
        ("left_eyebrow", LEFT_EYEBROW, config.eyebrow_dilate),
        ("right_eyebrow", RIGHT_EYEBROW, config.eyebrow_dilate),
    ]:
        if not landmark_quality.visible_regions.get(region_name, False):
            continue
        points = landmarks_xy[indices]
        if not validate_polygon(points, image_shape, min_inside_ratio=0.9, min_area=5.0, max_area_ratio=0.08):
            continue
        part = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(part, [points.astype(np.int32)], 255)
        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            part = cv2.dilate(part, kernel, iterations=1)
        forbidden[part > 0] = 255

    visible_points = landmarks_xy[_inside_points(landmarks_xy, image_shape)]
    if len(visible_points) > 0:
        face_width = max(1.0, float(visible_points[:, 0].max() - visible_points[:, 0].min()))
        nostril_radius = max(2, int(round(face_width * 0.018)))
        if config.nostril_dilate > 0:
            nostril_radius += int(config.nostril_dilate)
        for region_name, idx in [("left_nostril", 98), ("right_nostril", 327)]:
            if not landmark_quality.visible_regions.get(region_name, False):
                continue
            center = tuple(np.round(landmarks_xy[idx]).astype(int))
            if center[0] < 0 or center[0] >= w or center[1] < 0 or center[1] >= h:
                continue
            cv2.circle(forbidden, center, nostril_radius, 255, thickness=-1, lineType=cv2.LINE_AA)

    skin_mask = refined_skin.copy()
    skin_mask[forbidden > 0] = 0
    return skin_mask, forbidden, oval_mask


def _standardize_array(
    array: np.ndarray,
    crop_box: tuple[int, int, int, int],
    offset_xy: tuple[int, int],
    side: int,
    image_size: int,
    interpolation: int,
) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    off_x, off_y = offset_xy
    crop = array[y1:y2, x1:x2]
    if array.ndim == 2:
        canvas = np.zeros((side, side), dtype=array.dtype)
        canvas[off_y:off_y + crop.shape[0], off_x:off_x + crop.shape[1]] = crop
    else:
        canvas = np.zeros((side, side, array.shape[2]), dtype=array.dtype)
        canvas[off_y:off_y + crop.shape[0], off_x:off_x + crop.shape[1]] = crop
    return cv2.resize(canvas, (image_size, image_size), interpolation=interpolation)


def _standardize_points(
    points_xy: np.ndarray,
    crop_box: tuple[int, int, int, int],
    offset_xy: tuple[int, int],
    scale: float,
    image_size: int,
) -> np.ndarray:
    x1, y1, _, _ = crop_box
    off_x, off_y = offset_xy
    out = points_xy.copy()
    out[:, 0] = (out[:, 0] - x1 + off_x) * scale
    out[:, 1] = (out[:, 1] - y1 + off_y) * scale
    out[:, 0] = np.clip(out[:, 0], 0, image_size - 1)
    out[:, 1] = np.clip(out[:, 1], 0, image_size - 1)
    return out.astype(np.float32)


def original_to_standard_xy(
    points_xy: np.ndarray,
    crop_box_xyxy: tuple[int, int, int, int],
    square_offset_xy: tuple[int, int],
    resize_scale: float,
) -> np.ndarray:
    """Map original image coordinates into standardized 1024-style coordinates."""
    x1, y1, _, _ = crop_box_xyxy
    off_x, off_y = square_offset_xy
    points = np.asarray(points_xy, dtype=np.float32).copy()
    points[:, 0] = (points[:, 0] - x1 + off_x) * resize_scale
    points[:, 1] = (points[:, 1] - y1 + off_y) * resize_scale
    return points


def standard_to_original_xy(
    points_xy: np.ndarray,
    crop_box_xyxy: tuple[int, int, int, int],
    square_offset_xy: tuple[int, int],
    resize_scale: float,
) -> np.ndarray:
    """Map standardized coordinates back into original image coordinates."""
    x1, y1, _, _ = crop_box_xyxy
    off_x, off_y = square_offset_xy
    points = np.asarray(points_xy, dtype=np.float32).copy()
    points[:, 0] = points[:, 0] / resize_scale + x1 - off_x
    points[:, 1] = points[:, 1] / resize_scale + y1 - off_y
    return points


def _classify_input_mode(
    landmark_quality: LandmarkQuality | None,
    face_skin_mask: np.ndarray,
    config: PreprocessConfig,
) -> FaceInputMode:
    skin_ratio = float((face_skin_mask > 0).mean())
    if landmark_quality is None:
        return FaceInputMode.SKIN_PATCH if skin_ratio >= config.skin_patch_min_skin_ratio else FaceInputMode.NO_FACE
    if landmark_quality.is_reliable:
        return FaceInputMode.FULL_FACE
    if landmark_quality.partial_face_score >= 0.34 and skin_ratio >= config.skin_patch_min_skin_ratio:
        return FaceInputMode.PARTIAL_FACE
    if skin_ratio >= config.skin_patch_min_skin_ratio:
        return FaceInputMode.SKIN_PATCH
    return FaceInputMode.NO_FACE


def _fallback_skin_mask_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Conservative color fallback for close-up skin patches; does not infer face geometry."""
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    skin = (
        (y_channel > 40)
        & (cr_channel >= 133)
        & (cr_channel <= 180)
        & (cb_channel >= 77)
        & (cb_channel <= 135)
    ).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel, iterations=1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin = _fill_small_mask_holes(skin, max_hole_area_ratio=0.02)
    return skin


def _fill_small_mask_holes(mask: np.ndarray, max_hole_area_ratio: float = 0.005) -> np.ndarray:
    mask = ((mask > 0).astype(np.uint8) * 255)
    inv = cv2.bitwise_not(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = mask.shape[:2]
    max_area = h * w * max_hole_area_ratio
    filled = mask.copy()
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches_border and area <= max_area:
            filled[labels == label] = 255
    return filled


def _mode_capabilities(input_mode: FaceInputMode) -> dict[str, bool]:
    return {
        "grading_allowed": input_mode == FaceInputMode.FULL_FACE,
        "global_region_analysis_allowed": input_mode == FaceInputMode.FULL_FACE,
        "local_detection_allowed": input_mode in {
            FaceInputMode.FULL_FACE,
            FaceInputMode.PARTIAL_FACE,
            FaceInputMode.SKIN_PATCH,
        },
    }


def _draw_landmark_quality_debug(
    image_bgr: np.ndarray,
    landmarks_xy: np.ndarray | None,
    landmark_quality: LandmarkQuality | None,
    input_mode: FaceInputMode,
) -> np.ndarray:
    debug = image_bgr.copy()
    if landmarks_xy is not None:
        inside = _inside_points(landmarks_xy, image_bgr.shape[:2])
        for point, ok in zip(landmarks_xy, inside):
            x, y = np.round(point).astype(int)
            if 0 <= x < image_bgr.shape[1] and 0 <= y < image_bgr.shape[0]:
                color = (80, 220, 80) if ok and landmark_quality and landmark_quality.is_reliable else (40, 40, 255)
                cv2.circle(debug, (x, y), 1, color, -1, cv2.LINE_AA)
        if landmark_quality is not None:
            for name, indices in [
                ("left_eye", LEFT_EYE),
                ("right_eye", RIGHT_EYE),
                ("left_eyebrow", LEFT_EYEBROW),
                ("right_eyebrow", RIGHT_EYEBROW),
                ("lips", MOUTH_OUTER),
                ("face_oval", FACE_OVAL),
            ]:
                points = landmarks_xy[indices]
                if not _inside_points(points, image_bgr.shape[:2]).all():
                    continue
                color = (80, 220, 80) if landmark_quality.visible_regions.get(name, False) else (40, 40, 255)
                cv2.polylines(debug, [points.astype(np.int32)], True, color, 2, cv2.LINE_AA)
    cv2.putText(
        debug,
        f"mode: {input_mode.value}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        debug,
        f"mode: {input_mode.value}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (40, 180, 255),
        1,
        cv2.LINE_AA,
    )
    return debug


def _draw_visible_regions_debug(image_bgr: np.ndarray, forbidden_mask: np.ndarray) -> np.ndarray:
    debug = image_bgr.copy()
    overlay = image_bgr.copy()
    overlay[forbidden_mask > 0] = (0, 220, 255)
    debug = cv2.addWeighted(overlay, 0.35, debug, 0.65, 0)
    contours, _ = cv2.findContours(forbidden_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    return debug


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def preprocess_image(
    image_path: str | Path,
    config: PreprocessConfig,
    segmenter: Any,
    landmarker: Any,
    mp_module: Any,
) -> PreprocessResult:
    original = read_image_bgr(image_path)
    h, w = original.shape[:2]
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    quality = assess_image_quality(original)
    seg_result = segmenter.segment(mp_image)
    category_mask = seg_result.category_mask.numpy_view()
    if category_mask.ndim == 3:
        category_mask = np.squeeze(category_mask)
    face_skin_mask = ((category_mask == 3).astype(np.uint8) * 255)
    if face_skin_mask.shape[:2] != (h, w):
        face_skin_mask = cv2.resize(face_skin_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mediapipe_skin_ratio = float((face_skin_mask > 0).mean())
    skin_mask_source = "mediapipe_selfie_multiclass_face_skin"
    if mediapipe_skin_ratio < config.skin_patch_min_skin_ratio:
        fallback_skin_mask = _fallback_skin_mask_bgr(original)
        fallback_skin_ratio = float((fallback_skin_mask > 0).mean())
        if fallback_skin_ratio >= config.skin_patch_min_skin_ratio:
            face_skin_mask = fallback_skin_mask
            skin_mask_source = "conservative_ycrcb_skin_fallback"
    else:
        fallback_skin_ratio = None

    lm_result = landmarker.detect(mp_image)
    selected_face_index: int | None = None
    landmarks_original: np.ndarray | None = None
    landmark_quality: LandmarkQuality | None = None
    if lm_result.face_landmarks:
        selected_face_index, landmarks_original = _largest_landmark_face(lm_result.face_landmarks, w, h)
        landmark_quality = evaluate_landmark_quality(landmarks_original, (h, w))
    input_mode = _classify_input_mode(landmark_quality, face_skin_mask, config)
    if input_mode == FaceInputMode.NO_FACE:
        raise NoFaceDetectedError(f"no reliable face or skin patch detected: {image_path}")

    skin_original, forbidden_original, oval_mask = _build_original_masks(
        (h, w),
        landmarks_original,
        face_skin_mask,
        config,
        input_mode,
        landmark_quality,
    )
    y_coords, x_coords = np.where(skin_original > 0)
    if len(x_coords) == 0 or len(y_coords) == 0:
        raise PreprocessError(f"empty skin mask after refinement: {image_path}")

    x_min, x_max = int(x_coords.min()), int(x_coords.max()) + 1
    y_min, y_max = int(y_coords.min()), int(y_coords.max()) + 1
    face_w = max(1, x_max - x_min)
    face_h = max(1, y_max - y_min)
    pad_x = int(round(face_w * config.crop_pad_x_ratio))
    pad_top = int(round(face_h * config.crop_pad_top_ratio))
    pad_bottom = int(round(face_h * config.crop_pad_bottom_ratio))
    crop_box = (
        max(0, x_min - pad_x),
        max(0, y_min - pad_top),
        min(w, x_max + pad_x),
        min(h, y_max + pad_bottom),
    )
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]
    side = max(crop_w, crop_h)
    off_x = (side - crop_w) // 2
    off_y = (side - crop_h) // 2
    scale = config.image_size / float(side)

    standardized = _standardize_array(original, crop_box, (off_x, off_y), side, config.image_size, cv2.INTER_CUBIC)
    skin_mask = _standardize_array(skin_original, crop_box, (off_x, off_y), side, config.image_size, cv2.INTER_NEAREST)
    forbidden_mask = _standardize_array(forbidden_original, crop_box, (off_x, off_y), side, config.image_size, cv2.INTER_NEAREST)
    skin_mask = ((skin_mask > 127).astype(np.uint8) * 255)
    forbidden_mask = ((forbidden_mask > 127).astype(np.uint8) * 255)
    masked_face = cv2.bitwise_and(standardized, standardized, mask=skin_mask)
    if landmarks_original is not None:
        landmarks_standard = _standardize_points(landmarks_original, crop_box, (off_x, off_y), scale, config.image_size)
    else:
        landmarks_standard = np.zeros((0, 2), dtype=np.float32)

    capture_profile = capture_profile_from_environment()
    mask_contract_evidence = None
    match capture_profile:
        case CaptureProfile.INSTITUTION:
            pass
        case CaptureProfile.CONSUMER:
            applied = apply_profile_mask_policy(
                capture_profile,
                standardized,
                skin_mask,
                forbidden_mask,
                landmarks_standard,
                "aligned_1024_acne",
            )
            skin_mask = applied.skin_mask
            forbidden_mask = applied.forbidden_mask
            masked_face = cv2.bitwise_and(standardized, standardized, mask=skin_mask)
            mask_contract_evidence = applied.bundle.evidence()
        case unreachable:
            assert_never(unreachable)

    capabilities = _mode_capabilities(input_mode)
    if capabilities["global_region_analysis_allowed"] and len(landmarks_standard) > 0:
        region_masks = build_region_masks(landmarks_standard, config.image_size)
        region_areas = {
            name: int(cv2.bitwise_and(mask, skin_mask).sum() // 255)
            for name, mask in region_masks.items()
        }
    else:
        region_areas = {}

    metadata = {
        "source_path": str(image_path),
        "status": "ok",
        "input_mode": input_mode.value,
        **capabilities,
        "original_shape_hw": [int(h), int(w)],
        "standardized_shape_hw": [config.image_size, config.image_size],
        "face_count": len(lm_result.face_landmarks),
        "selected_face_index": selected_face_index,
        "crop_box_xyxy": list(crop_box),
        "square_side": int(side),
        "square_offset_xy": [int(off_x), int(off_y)],
        "resize_scale": float(scale),
        "skin_pixels": int(skin_mask.sum() // 255),
        "forbidden_pixels": int(forbidden_mask.sum() // 255),
        "skin_mask_source": skin_mask_source,
        "mediapipe_face_skin_ratio": mediapipe_skin_ratio,
        "fallback_skin_ratio": fallback_skin_ratio,
        "region_skin_pixels": region_areas,
        "landmark_quality": landmark_quality.to_dict() if landmark_quality is not None else {
            "is_reliable": False,
            "is_reliable_full_face": False,
            "full_face_score": 0.0,
            "partial_face_score": 0.0,
            "inside_ratio": 0.0,
            "face_bbox_ratio": 0.0,
            "border_touch_ratio": 0.0,
            "geometric_checks": {},
            "visible_regions": {name: False for name in VISIBLE_REGION_NAMES},
            "reasons": ["no_landmarks"],
        },
        "visible_regions": (
            landmark_quality.visible_regions
            if landmark_quality is not None
            else {name: False for name in VISIBLE_REGION_NAMES}
        ),
        "quality": quality,
        "coordinate_mapping": {
            "standard_x": "(original_x - crop_x1 + square_offset_x) * resize_scale",
            "standard_y": "(original_y - crop_y1 + square_offset_y) * resize_scale",
            "original_x": "standard_x / resize_scale + crop_x1 - square_offset_x",
            "original_y": "standard_y / resize_scale + crop_y1 - square_offset_y",
        },
        "masks": {
            "skin_mask": "255 means valid acne-analysis skin",
            "forbidden_mask": "255 means invalid/forbidden for acne candidate centers",
            "forbidden_policy": [
                "only reliable visible feature cores are included",
                "background and non-skin are represented by skin_mask, not forbidden_mask",
                "eye_and_sclera_core_if_visible",
                "lip_body_if_visible",
                "nostril_interior_if_visible",
                "eyebrow_hair_core_if_visible",
            ],
        },
        "reference_code_changes": {
            "removed": [
                "gray_texture_input",
                "gaussian_high_pass",
                "clahe_main_path",
                "black_hat",
                "skeletonization",
                "large_nose_or_jaw_exclusion",
            ],
            "kept_or_adapted": [
                "MediaPipe FaceLandmarker",
                "MediaPipe ImageSegmenter",
                "face-skin category 3",
                "face oval and forehead completion",
                "square centered crop",
                "1024 aligned outputs",
            ],
        },
    }
    if mask_contract_evidence is not None:
        metadata["capture_profile"] = capture_profile.value
        metadata["mask_contract"] = mask_contract_evidence
    metadata["model_files"] = ensure_mediapipe_models(config)
    metadata["_debug_original_landmarks_xy"] = landmarks_original
    metadata["_debug_original_forbidden_mask"] = forbidden_original
    metadata["_debug_input_mode"] = input_mode

    return PreprocessResult(
        original_bgr=original,
        standardized_bgr=standardized,
        masked_face_bgr=masked_face,
        skin_mask=skin_mask,
        forbidden_mask=forbidden_mask,
        landmarks_xy=landmarks_standard,
        crop_box_xyxy=crop_box,
        square_offset_xy=(off_x, off_y),
        resize_scale=scale,
        metadata=metadata,
    )


def save_preprocess_outputs(
    result: PreprocessResult,
    output_dir: str | Path,
    artifact_policy: AcneArtifactPolicy = AcneArtifactPolicy.FORMAL,
    timings: AcnePhaseTimings | None = None,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_path = out_dir / "00_input_quality.json"
    metadata_path = out_dir / "07_preprocess_metadata.json"
    landmark_quality_path = out_dir / "08_landmark_quality.json"
    debug_original_landmarks = result.metadata.get("_debug_original_landmarks_xy")
    input_mode = result.metadata.get("_debug_input_mode", FaceInputMode.NO_FACE)
    if isinstance(input_mode, str):
        input_mode = FaceInputMode(input_mode)
    paths = {
        "input_quality": quality_path,
        "original": out_dir / "01_original.jpg",
        "standardized": out_dir / "02_standardized.jpg",
        "masked_face": out_dir / "03_masked_face.jpg",
        "skin_mask": out_dir / "04_skin_mask.png",
        "forbidden_mask": out_dir / "05_forbidden_mask.png",
        "region_debug": out_dir / "06_region_debug.jpg",
        "preprocess_metadata": metadata_path,
        "landmark_quality": landmark_quality_path,
        "landmark_debug": out_dir / "09_landmark_debug.jpg",
        "visible_regions_debug": out_dir / "10_visible_regions_debug.jpg",
    }
    image_scope = timings.measure_image_encoding() if timings else nullcontext()
    with image_scope:
        write_image(paths["original"], result.original_bgr)
        write_image(paths["standardized"], result.standardized_bgr)
        write_image(paths["skin_mask"], result.skin_mask)
        write_image(paths["forbidden_mask"], result.forbidden_mask)
        if artifact_policy.includes_debug_images:
            if result.metadata.get("global_region_analysis_allowed") and len(result.landmarks_xy) > 0:
                region_masks = build_region_masks(result.landmarks_xy, result.standardized_bgr.shape[0])
                region_debug = draw_region_debug(result.standardized_bgr, region_masks, result.skin_mask)
            else:
                region_debug = result.standardized_bgr.copy()
                cv2.putText(
                    region_debug,
                    f"global regions disabled: {result.metadata['input_mode']}",
                    (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            landmark_debug = _draw_landmark_quality_debug(
                result.original_bgr,
                debug_original_landmarks,
                None if debug_original_landmarks is None else LandmarkQuality(**{
                    "is_reliable": result.metadata["landmark_quality"]["is_reliable"],
                    "full_face_score": result.metadata["landmark_quality"]["full_face_score"],
                    "partial_face_score": result.metadata["landmark_quality"]["partial_face_score"],
                    "inside_ratio": result.metadata["landmark_quality"]["inside_ratio"],
                    "face_bbox_ratio": result.metadata["landmark_quality"]["face_bbox_ratio"],
                    "border_touch_ratio": result.metadata["landmark_quality"]["border_touch_ratio"],
                    "geometric_checks": result.metadata["landmark_quality"]["geometric_checks"],
                    "visible_regions": result.metadata["landmark_quality"]["visible_regions"],
                    "reasons": result.metadata["landmark_quality"]["reasons"],
                }),
                input_mode,
            )
            visible_regions_debug = _draw_visible_regions_debug(result.standardized_bgr, result.forbidden_mask)
            write_image(paths["masked_face"], result.masked_face_bgr)
            write_image(paths["region_debug"], region_debug)
            write_image(paths["landmark_debug"], landmark_debug)
            write_image(paths["visible_regions_debug"], visible_regions_debug)
    serializable_metadata = dict(result.metadata)
    serializable_metadata.pop("_debug_original_landmarks_xy", None)
    serializable_metadata.pop("_debug_original_forbidden_mask", None)
    serializable_metadata.pop("_debug_input_mode", None)
    serializable_metadata["landmarks_xy"] = result.landmarks_xy.round(3).tolist()
    structured_scope = timings.measure_structured_persistence() if timings else nullcontext()
    with structured_scope:
        quality_path.write_text(
            json.dumps(result.metadata["quality"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metadata_path.write_text(
            json.dumps(_json_safe(serializable_metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        landmark_quality_path.write_text(
            json.dumps(
                _json_safe({
                    "input_mode": result.metadata["input_mode"],
                    **result.metadata["landmark_quality"],
                    "visible_regions": result.metadata["visible_regions"],
                    "grading_allowed": result.metadata["grading_allowed"],
                    "global_region_analysis_allowed": result.metadata["global_region_analysis_allowed"],
                    "local_detection_allowed": result.metadata["local_detection_allowed"],
                }),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return {key: str(path) for key, path in paths.items() if path.exists()}
