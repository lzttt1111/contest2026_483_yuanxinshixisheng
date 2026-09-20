"""DermaVision Preprocessor V2.

The V2 preprocessor creates two strictly separated image products:

* ``analysis_image`` keeps the decoded sRGB pixel values and only receives a
  global similarity transform. It is the only image intended for algorithms.
* ``display_image`` may have its background removed and is only intended for
  UI/report rendering.

No reference-face colour transfer, whitening, CLAHE, or display enhancement is
applied to ``analysis_image``.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import cv2
import mediapipe as mp
import numpy as np
from sg_core.preprocess.analysis_mask_bundle import AnalysisMaskBundle, build_analysis_mask_bundle, expand_common_hair_masks
from sg_core.capture_profile import CaptureProfile, capture_profile_from_environment
from sg_core.preprocess.consumer_quality import evaluate_consumer_quality
from typing_extensions import assert_never
from sg_core.utils.io_utils import cv_imread, cv_imwrite
from sg_core.utils.model_loader import load_face_landmarker, load_selfie_segmenter
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class PreprocessResultV2:
    """Unified input contract for all V2 skin-analysis algorithms.

    Images use OpenCV's BGR memory layout while retaining sRGB colour
    semantics. ``landmarks`` contains aligned pixel coordinates in the
    1024x1024 analysis coordinate system. Masks contain 255 for valid pixels
    and 0 elsewhere.
    """
    analysis_image: np.ndarray
    display_image: np.ndarray
    skin_mask: np.ndarray
    landmarks: np.ndarray
    face_transform_matrix: np.ndarray
    inverse_transform_matrix: np.ndarray
    quality_score: float
    quality_status: str
    quality_flags: list[str]
    quality_metrics: dict[str, float] = field(default_factory=dict)
    landmarks_relative_z: np.ndarray | None = None
    mask_bundle: AnalysisMaskBundle | None = None

class Config:
    """Preprocessor V2 geometry and quality defaults."""
    ANALYSIS_SIZE = 1024
    TARGET_LEFT_EYE = np.array([344.0, 400.0], dtype=np.float32)
    TARGET_RIGHT_EYE = np.array([680.0, 400.0], dtype=np.float32)
    CROP_PAD_X_RATIO = 0.05
    CROP_PAD_TOP_RATIO = 0.08
    CROP_PAD_BOTTOM_RATIO = 0.02
    PERSON_THRESHOLD = 0.5
    FACE_GEOMETRY_DILATE = 7
    FOREHEAD_EXTENSION_RATIO = 0.8
    FOREHEAD_TOP_HALF_WIDTH_RATIO = 0.38
    FEATURE_DILATE = 15
    EYEBROW_DILATE = 17
    NOSTRIL_MIN_RADIUS_X = 10
    NOSTRIL_MAX_RADIUS_X = 28
    NOSTRIL_MIN_RADIUS_Y = 7
    NOSTRIL_MAX_RADIUS_Y = 18
    HAIR_LOCAL_SIGMAS = (3.0, 6.0, 12.0)
    HAIR_LINE_LENGTHS = (9, 17, 31)
    HAIR_LINE_ANGLES = (0, 30, 60, 90, 120, 150)
    HAIR_DARK_Z = 2.8
    HAIR_BULK_Z = 4.0
    HAIR_MIN_SCALE_VOTES = 2
    HAIR_FOREHEAD_RESCUE_LINE_RESPONSE = 0.3
    HAIR_FOREHEAD_RESCUE_MIN_VOTES = 1
    HAIR_FOREHEAD_RESCUE_CLOSE_KERNEL = 7
    HAIR_MASK_DILATE = 5
    HAIR_STRAND_MAX_DOMAIN_RATIO = 0.35
    FACIAL_HAIR_DARK_Z = 3.2
    FACIAL_HAIR_DENSITY = 0.12
    FACIAL_HAIR_MIN_COMPONENT_AREA = 6
    FACIAL_HAIR_MAX_COMPONENT_AREA = 2000
    FACIAL_HAIR_MIN_ASPECT_RATIO = 2.2
    FACIAL_OCCLUSION_HIGHLIGHT_VALUE = 0.94
    FACIAL_OCCLUSION_HIGHLIGHT_MAX_SATURATION = 0.3
    FACIAL_OCCLUSION_HIGHLIGHT_SEARCH_KERNEL = 17
    HAIR_COLOR_L_MAD = 2.5
    HAIR_COLOR_MIN_L_DELTA = 12.0
    HAIR_COLOR_DE_MAD = 3.0
    HAIR_COLOR_MIN_L_FOR_DELTA_E = 4.0
    HAIR_COLOR_MIN_AREA = 20
    PASS_SCORE = 75.0
    WARNING_SCORE = 60.0
    MIN_FACE_AREA_RATIO = 0.08
    IDEAL_FACE_AREA_RATIO = 0.16
    MAX_FACE_AREA_RATIO = 0.72
    MIN_SKIN_AREA_RATIO = 0.08
    BLUR_WARNING_VARIANCE = 75.0
    BLUR_GOOD_VARIANCE = 180.0
    MAX_CLIPPED_RATIO = 0.18
    MAX_HIGHLIGHT_RATIO = 0.08
    POSE_WARNING_YAW = 15.0
    POSE_WARNING_PITCH = 12.0
    POSE_WARNING_ROLL = 10.0
    POSE_REJECT_YAW = 28.0
    POSE_REJECT_PITCH = 22.0
    POSE_REJECT_ROLL = 18.0
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSTRILS = [98, 327]

def _fill_landmark_feature(mask: np.ndarray, points: np.ndarray, indices: list[int], dilate: int) -> None:
    valid = [index for index in indices if index < len(points)]
    if len(valid) < 3:
        return
    local = np.zeros_like(mask)
    polygon = cv2.convexHull(np.rint(points[valid]).astype(np.int32))
    cv2.fillConvexPoly(local, polygon, 255)
    if dilate > 1:
        local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    mask[:] = cv2.bitwise_or(mask, local)

def build_nostril_exclusion_mask(analysis_image: np.ndarray, points: np.ndarray, use_photometric_refinement: bool=True) -> np.ndarray:
    """Build a scale-aware nostril aperture mask from geometry and dark pixels.

    Landmarks 98 and 327 sit near the outer nose base, so each seed is moved
    toward landmark 2 before looking for the local dark connected component.
    The search is deliberately restricted to two small lower-nose ROIs; this
    avoids removing genuine findings on the nose wings.
    """
    shape = analysis_image.shape[:2]
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) <= 327:
        return mask
    left_outer = points[98].astype(np.float32)
    right_outer = points[327].astype(np.float32)
    nose_base = points[2].astype(np.float32)
    span = max(float(np.linalg.norm(right_outer - left_outer)), 1.0)
    radius_x = int(np.clip(round(0.18 * span), Config.NOSTRIL_MIN_RADIUS_X, Config.NOSTRIL_MAX_RADIUS_X))
    radius_y = int(np.clip(round(0.11 * span), Config.NOSTRIL_MIN_RADIUS_Y, Config.NOSTRIL_MAX_RADIUS_Y))
    centres = (0.68 * left_outer + 0.32 * nose_base, 0.68 * right_outer + 0.32 * nose_base)
    lab_l = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2LAB)[:, :, 0] if use_photometric_refinement else None
    (height, width) = shape
    for centre_f in centres:
        centre = np.rint(centre_f).astype(np.int32)
        cx = int(np.clip(centre[0], 0, width - 1))
        cy = int(np.clip(centre[1], 0, height - 1))
        geometry = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(geometry, (cx, cy), (radius_x, radius_y), 0, 0, 360, 255, -1)
        mask = cv2.bitwise_or(mask, geometry)
        if not use_photometric_refinement:
            continue
        x0 = max(0, cx - 2 * radius_x)
        x1 = min(width, cx + 2 * radius_x + 1)
        y0 = max(0, cy - 2 * radius_y)
        y1 = min(height, cy + 2 * radius_y + 1)
        roi = lab_l[y0:y1, x0:x1]
        if roi.size < 25:
            continue
        median = float(np.median(roi))
        mad = float(np.median(np.abs(roi.astype(np.float32) - median)))
        threshold = median - max(6.0, 1.8 * 1.4826 * mad)
        dark = (roi.astype(np.float32) < threshold).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        seed = np.zeros_like(dark)
        local_centre = (cx - x0, cy - y0)
        cv2.ellipse(seed, local_centre, (max(3, radius_x // 2), max(2, radius_y // 2)), 0, 0, 360, 1, -1)
        (count, labels, stats, _) = cv2.connectedComponentsWithStats(dark, connectivity=8)
        for label in range(1, count):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area <= 3 or area > int(roi.size * 0.55):
                continue
            if np.any(component & (seed > 0)):
                local_mask = mask[y0:y1, x0:x1]
                local_mask[component] = 255
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

def build_feature_exclusion_masks(analysis_image: np.ndarray, points: np.ndarray, use_photometric_nostrils: bool=True) -> dict[str, np.ndarray]:
    """Return the shared facial-feature and nostril masks for all engines."""
    exclusions = np.zeros(analysis_image.shape[:2], dtype=np.uint8)
    _fill_landmark_feature(exclusions, points, LEFT_EYE, Config.FEATURE_DILATE)
    _fill_landmark_feature(exclusions, points, RIGHT_EYE, Config.FEATURE_DILATE)
    _fill_landmark_feature(exclusions, points, LEFT_EYEBROW, Config.EYEBROW_DILATE)
    _fill_landmark_feature(exclusions, points, RIGHT_EYEBROW, Config.EYEBROW_DILATE)
    _fill_landmark_feature(exclusions, points, LIPS, Config.FEATURE_DILATE)
    nostril_mask = build_nostril_exclusion_mask(analysis_image, points, use_photometric_refinement=use_photometric_nostrils)
    exclusions = cv2.bitwise_or(exclusions, nostril_mask)
    return {'feature_exclusions': exclusions, 'nostril_mask': nostril_mask}

def _masked_gaussian(channel: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    mask_f = (mask > 0).astype(np.float32)
    values = channel.astype(np.float32)
    numerator = cv2.GaussianBlur(values * mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    denominator = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    result = numerator / np.maximum(denominator, 1e-05)
    result[mask_f == 0] = 0.0
    return result.astype(np.float32)

def _robust_positive_z(values: np.ndarray, mask: np.ndarray, sigma_floor: float) -> np.ndarray:
    valid = values[mask > 0]
    if valid.size < 100:
        return np.zeros_like(values, dtype=np.float32)
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    sigma = max(1.4826 * mad, sigma_floor)
    z_score = np.maximum((values.astype(np.float32) - median) / sigma, 0.0)
    z_score[mask == 0] = 0.0
    return np.clip(z_score, 0.0, 20.0).astype(np.float32)

def _line_kernel(length: int, angle: int) -> np.ndarray:
    size = length if length % 2 else length + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    centre = (size - 1) * 0.5
    radius = centre - 1.0
    theta = np.deg2rad(float(angle))
    dx = radius * np.cos(theta)
    dy = radius * np.sin(theta)
    start = (int(round(centre - dx)), int(round(centre - dy)))
    end = (int(round(centre + dx)), int(round(centre + dy)))
    cv2.line(kernel, start, end, 1, 1, cv2.LINE_8)
    return kernel

def _plausible_strand_component(area: float, aspect_ratio: float, touches_boundary: bool, in_forehead: bool, domain_pixels: int) -> bool:
    maximum = max(2000.0, float(domain_pixels) * Config.HAIR_STRAND_MAX_DOMAIN_RATIO)
    if area > maximum:
        return False
    return bool(touches_boundary or (in_forehead and aspect_ratio >= 2.2) or (aspect_ratio >= 3.0 and area <= 2000.0))

def detect_visible_hair_masks(analysis_image: np.ndarray, valid_mask: np.ndarray, points: np.ndarray, feature_exclusions: np.ndarray | None=None) -> dict[str, np.ndarray]:
    """Detect bulk scalp hair, strands, and visible facial hair.

    The detector is deterministic and uses only adaptive, within-face image
    statistics. It is an occlusion detector, not a hair-colour classifier.
    """
    domain = (valid_mask > 0).astype(np.uint8) * 255
    if feature_exclusions is None:
        feature_exclusions = np.zeros_like(domain)
    analysis_domain = cv2.bitwise_and(domain, cv2.bitwise_not(feature_exclusions))
    if np.count_nonzero(analysis_domain) < 100:
        empty = np.zeros_like(domain)
        empty_f = empty.astype(np.float32)
        return {'bulk_hair_mask': empty.copy(), 'color_hair_mask': empty.copy(), 'strand_hair_mask': empty.copy(), 'facial_hair_mask': empty.copy(), 'hair_mask': empty.copy(), 'hair_line_response': empty_f.copy(), 'hair_scale_vote': empty.copy()}
    lab_image = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab_image[:, :, 0].astype(np.uint8)
    l_float = l_channel.astype(np.float32)
    (yy, xx) = np.indices(domain.shape)
    brow_y = int(min(points[105, 1], points[334, 1])) if len(points) > 334 else 400
    mouth_y = int(np.mean(points[[0, 17], 1])) if len(points) > 17 else 700
    centre_x = int(points[1, 0]) if len(points) > 1 else domain.shape[1] // 2
    if len(points) > 454:
        face_width = max(abs(float(points[454, 0] - points[234, 0])), 1.0)
    else:
        face_width = float(domain.shape[1]) * 0.6
    reference_mask = (analysis_domain > 0) & (yy > brow_y + 60) & (yy < mouth_y - 30) & (np.abs(xx - centre_x) < 0.42 * face_width)
    reference_values = lab_image[reference_mask]
    if reference_values.shape[0] < 100:
        reference_values = lab_image[analysis_domain > 0]
    skin_median = np.median(reference_values, axis=0)
    l_mad = float(np.median(np.abs(reference_values[:, 0] - skin_median[0])))
    delta_e_to_skin = np.linalg.norm(lab_image - skin_median, axis=2)
    reference_delta_e = delta_e_to_skin[reference_mask]
    if reference_delta_e.size < 100:
        reference_delta_e = delta_e_to_skin[analysis_domain > 0]
    delta_e_median = float(np.median(reference_delta_e))
    delta_e_mad = float(np.median(np.abs(reference_delta_e - delta_e_median)))
    l_dark = skin_median[0] - lab_image[:, :, 0]
    colour_hair_raw = ((l_dark > max(Config.HAIR_COLOR_MIN_L_DELTA, Config.HAIR_COLOR_L_MAD * 1.4826 * l_mad)) | (delta_e_to_skin > delta_e_median + Config.HAIR_COLOR_DE_MAD * 1.4826 * delta_e_mad) & (l_dark > Config.HAIR_COLOR_MIN_L_FOR_DELTA_E)) & (analysis_domain > 0) & (yy < brow_y + 30)
    colour_hair_raw = colour_hair_raw.astype(np.uint8) * 255
    colour_hair_raw = cv2.morphologyEx(colour_hair_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    max_dark_residual = np.zeros_like(l_float)
    for sigma in Config.HAIR_LOCAL_SIGMAS:
        background = _masked_gaussian(l_float, analysis_domain, sigma)
        max_dark_residual = np.maximum(max_dark_residual, np.maximum(background - l_float, 0.0))
    dark_z = _robust_positive_z(max_dark_residual, analysis_domain, 1.5)
    line_vote = np.zeros_like(domain, dtype=np.uint8)
    max_line_z = np.zeros_like(l_float)
    l_for_lines = l_channel.copy()
    l_for_lines[analysis_domain == 0] = int(np.clip(skin_median[0], 0, 255))
    for length in Config.HAIR_LINE_LENGTHS:
        scale_response = np.zeros_like(l_channel)
        for angle in Config.HAIR_LINE_ANGLES:
            response = cv2.morphologyEx(l_for_lines, cv2.MORPH_BLACKHAT, _line_kernel(length, angle))
            scale_response = np.maximum(scale_response, response)
        scale_z = _robust_positive_z(scale_response.astype(np.float32), analysis_domain, 1.0)
        scale_support = scale_z >= Config.HAIR_DARK_Z
        line_vote += scale_support.astype(np.uint8)
        max_line_z = np.maximum(max_line_z, scale_z)
    strand_raw = ((dark_z >= Config.HAIR_DARK_Z) & (line_vote >= Config.HAIR_MIN_SCALE_VOTES) & (analysis_domain > 0)).astype(np.uint8) * 255
    strand_raw = cv2.morphologyEx(strand_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    boundary = cv2.morphologyEx(domain, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    forehead_zone = np.zeros_like(domain)
    cv2.rectangle(forehead_zone, (0, 0), (domain.shape[1] - 1, int(np.clip(brow_y + 20, 0, domain.shape[0] - 1))), 255, -1)
    relaxed_forehead_strand = ((max_line_z / 8.0 >= Config.HAIR_FOREHEAD_RESCUE_LINE_RESPONSE) & (line_vote >= Config.HAIR_FOREHEAD_RESCUE_MIN_VOTES) & (analysis_domain > 0) & (forehead_zone > 0)).astype(np.uint8) * 255
    relaxed_forehead_strand = cv2.morphologyEx(relaxed_forehead_strand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.HAIR_FOREHEAD_RESCUE_CLOSE_KERNEL, Config.HAIR_FOREHEAD_RESCUE_CLOSE_KERNEL)))
    strand_candidates = cv2.bitwise_or(strand_raw, relaxed_forehead_strand)
    strand_hair = np.zeros_like(domain)
    domain_pixels = int(np.count_nonzero(analysis_domain))
    (contours, _) = cv2.findContours(strand_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 6.0:
            continue
        (x, y, width, height) = cv2.boundingRect(contour)
        aspect_ratio = max(width, height) / max(1, min(width, height))
        component = np.zeros_like(domain)
        cv2.drawContours(component, [contour], -1, 255, -1)
        touches_boundary = bool(np.any((component > 0) & (boundary > 0)))
        in_forehead = bool(np.any((component > 0) & (forehead_zone > 0)))
        keep = _plausible_strand_component(area, aspect_ratio, touches_boundary, in_forehead, domain_pixels)
        if not np.any((component > 0) & (strand_raw > 0)):
            keep = bool(touches_boundary and in_forehead and (aspect_ratio >= 2.2))
        if keep:
            strand_hair[component > 0] = 255
    bulk_raw = ((dark_z >= Config.HAIR_BULK_Z) & (analysis_domain > 0)).astype(np.uint8) * 255
    bulk_raw = cv2.morphologyEx(bulk_raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    (count, labels, stats, _) = cv2.connectedComponentsWithStats(bulk_raw, connectivity=8)
    bulk_hair = np.zeros_like(domain)
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 12 and np.any(component & (boundary > 0)):
            bulk_hair[component] = 255
    (count, labels, stats, _) = cv2.connectedComponentsWithStats(colour_hair_raw, connectivity=8)
    colour_hair = np.zeros_like(domain)
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < Config.HAIR_COLOR_MIN_AREA:
            continue
        if np.any(component & (boundary > 0)) or np.any(component & (strand_hair > 0)):
            colour_hair[component] = 255
    lower_zone = np.zeros_like(domain)
    if len(points) > 17:
        nose_bottom_y = int(np.clip(max(points[2, 1], points[94, 1]), 0, domain.shape[0] - 1))
        cv2.rectangle(lower_zone, (0, nose_bottom_y), (domain.shape[1] - 1, domain.shape[0] - 1), 255, -1)
    visible_dark_line = ((dark_z >= Config.FACIAL_HAIR_DARK_Z) & (line_vote >= Config.HAIR_MIN_SCALE_VOTES) & (analysis_domain > 0)).astype(np.uint8)
    density = cv2.boxFilter(visible_dark_line.astype(np.float32), cv2.CV_32F, (21, 21), normalize=True)
    facial_hair_raw = ((visible_dark_line > 0) & (density >= Config.FACIAL_HAIR_DENSITY) & (lower_zone > 0)).astype(np.uint8) * 255
    (component_count, component_labels, component_stats, _) = cv2.connectedComponentsWithStats(facial_hair_raw, connectivity=8)
    facial_hair = np.zeros_like(domain)
    hsv = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    specular = (hsv[:, :, 2] / 255.0 >= Config.FACIAL_OCCLUSION_HIGHLIGHT_VALUE) & (hsv[:, :, 1] / 255.0 <= Config.FACIAL_OCCLUSION_HIGHLIGHT_MAX_SATURATION)
    highlight_search_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.FACIAL_OCCLUSION_HIGHLIGHT_SEARCH_KERNEL, Config.FACIAL_OCCLUSION_HIGHLIGHT_SEARCH_KERNEL))
    for label in range(1, component_count):
        area = int(component_stats[label, cv2.CC_STAT_AREA])
        if not Config.FACIAL_HAIR_MIN_COMPONENT_AREA <= area <= Config.FACIAL_HAIR_MAX_COMPONENT_AREA:
            continue
        width = int(component_stats[label, cv2.CC_STAT_WIDTH])
        height = int(component_stats[label, cv2.CC_STAT_HEIGHT])
        aspect_ratio = max(width, height) / max(1, min(width, height))
        component = (component_labels == label).astype(np.uint8) * 255
        highlight_near_component = bool(np.any((cv2.dilate(component, highlight_search_kernel) > 0) & specular))
        if aspect_ratio < Config.FACIAL_HAIR_MIN_ASPECT_RATIO and (not highlight_near_component):
            continue
        facial_hair[component > 0] = 255
    combined = cv2.bitwise_or(bulk_hair, colour_hair)
    combined = cv2.bitwise_or(combined, strand_hair)
    combined = cv2.bitwise_or(combined, facial_hair)
    combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.HAIR_MASK_DILATE, Config.HAIR_MASK_DILATE)))
    combined = cv2.bitwise_and(combined, domain)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(feature_exclusions))
    return {'bulk_hair_mask': bulk_hair, 'color_hair_mask': colour_hair, 'strand_hair_mask': strand_hair, 'facial_hair_mask': facial_hair, 'hair_mask': combined, 'hair_line_response': np.clip(max_line_z / 8.0, 0.0, 1.0).astype(np.float32), 'hair_scale_vote': line_vote}

class ImagePreprocessor:
    """Preprocess photos while preserving the existing Pipeline interface."""

    def __init__(self, input_dir=None, output_dir=None, ref_image_path=None, capture_profile: CaptureProfile | None=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.legacy_ref_image_path = ref_image_path
        self.capture_profile = capture_profile or capture_profile_from_environment()
        self.face_landmarker = load_face_landmarker(num_faces=2, min_detection_confidence=0.5, min_presence_confidence=0.5)
        self.segmenter = load_selfie_segmenter()
        self.last_result: PreprocessResultV2 | None = None
        self.last_debug_masks: dict[str, np.ndarray] = {}
        self._compat_result: PreprocessResultV2 | None = None
        self._compat_display_id: int | None = None

    @staticmethod
    def _to_mp_image(img: np.ndarray) -> mp.Image:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    @staticmethod
    def _landmarks_to_pixels(face_landmarks, width: int, height: int) -> np.ndarray:
        return np.asarray([(lm.x * width, lm.y * height) for lm in face_landmarks], dtype=np.float32)

    @staticmethod
    def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return cv2.transform(points.reshape(1, -1, 2), matrix)[0].astype(np.float32)

    @staticmethod
    def _face_hull_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        if len(points) >= 3:
            hull = cv2.convexHull(np.rint(points).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        return mask

    @staticmethod
    def _status_from_score(score: float, flags: list[str]) -> tuple[float, str]:
        hard_failures = {'NO_FACE', 'MULTIPLE_FACES', 'INVALID_IMAGE', 'INVALID_FACE_GEOMETRY'}
        if hard_failures.intersection(flags):
            return (min(float(score), Config.WARNING_SCORE - 1.0), 'REJECT')
        severe_capture_failures = {'BLUR', 'UNDEREXPOSED', 'OVEREXPOSED'}
        if 'PARTIAL_FACE' in flags and score >= 45.0 and (not severe_capture_failures.intersection(flags)):
            return (float(score), 'WARNING')
        if score >= Config.PASS_SCORE:
            return (float(score), 'PASS')
        if score >= Config.WARNING_SCORE:
            return (float(score), 'WARNING')
        return (float(score), 'REJECT')

    def _fallback_result(self, img: np.ndarray, flags: list[str]) -> PreprocessResultV2:
        """Create a complete, non-analyzable result for hard failures."""
        size = Config.ANALYSIS_SIZE
        (height, width) = img.shape[:2]
        if height <= 0 or width <= 0:
            analysis = np.zeros((size, size, 3), dtype=np.uint8)
            matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        else:
            scale = min(size / width, size / height)
            tx = (size - width * scale) * 0.5
            ty = (size - height * scale) * 0.5
            matrix = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
            analysis = cv2.warpAffine(img, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        inverse = cv2.invertAffineTransform(matrix).astype(np.float32)
        (score, status) = self._status_from_score(0.0, flags)
        result = PreprocessResultV2(analysis_image=analysis, display_image=analysis.copy(), skin_mask=np.zeros((size, size), dtype=np.uint8), landmarks=np.empty((0, 2), dtype=np.float32), face_transform_matrix=matrix, inverse_transform_matrix=inverse, quality_score=score, quality_status=status, quality_flags=flags)
        empty_mask = np.zeros((size, size), dtype=np.uint8)
        self.last_debug_masks = {'semantic_skin_mask': empty_mask.copy(), 'face_geometry_mask': empty_mask.copy(), 'forehead_completion': empty_mask.copy(), 'feature_exclusions': empty_mask.copy(), 'nostril_mask': empty_mask.copy(), 'bulk_hair_mask': empty_mask.copy(), 'color_hair_mask': empty_mask.copy(), 'strand_hair_mask': empty_mask.copy(), 'facial_hair_mask': empty_mask.copy(), 'hair_mask': empty_mask.copy(), 'hair_line_response': empty_mask.astype(np.float32), 'hair_scale_vote': empty_mask.copy(), 'display_face_mask': empty_mask.copy()}
        self._attach_mask_bundle(result)
        self.last_result = result
        return result

    def _attach_mask_bundle(self, result: PreprocessResultV2) -> None:
        result._debug_masks = {key: value.copy() for (key, value) in self.last_debug_masks.items()}
        empty = np.zeros_like(result.skin_mask, dtype=np.uint8)
        hair = result._debug_masks.get('hair_mask', empty)
        facial_hair = result._debug_masks.get('facial_hair_mask', empty)
        profile = getattr(self, 'capture_profile', CaptureProfile.CONSUMER)
        (hair, facial_hair) = expand_common_hair_masks(hair, facial_hair)
        result.mask_bundle = build_analysis_mask_bundle(valid_skin_mask=result.skin_mask, hair_mask=hair, facial_hair_mask=facial_hair, feature_exclusion_mask=result._debug_masks.get('feature_exclusions', empty), nostril_mask=result._debug_masks.get('nostril_mask', empty), display_face_mask=result._debug_masks.get('display_face_mask', result.skin_mask), coordinate_space='aligned_1024_dermavision')

    @staticmethod
    def _eye_centres(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        eye_a = points[LEFT_EYE].mean(axis=0)
        eye_b = points[RIGHT_EYE].mean(axis=0)
        if eye_a[0] <= eye_b[0]:
            return (eye_a, eye_b)
        return (eye_b, eye_a)

    def _person_mask_original(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return the person segmentation mask in original image coordinates."""
        (height, width) = img.shape[:2]
        try:
            result = self.segmenter.segment(self._to_mp_image(img))
            if not result.confidence_masks:
                raise ValueError('segmenter returned no confidence mask')
            confidence = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
            if confidence.shape[:2] != (height, width):
                confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = (confidence > Config.PERSON_THRESHOLD).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            return (mask, True)
        except Exception as exc:
            print(f'⚠️ [Preprocessor V2] 人像分割回退到面部几何 Mask: {exc}')
            return (np.full((height, width), 255, dtype=np.uint8), False)

    def _similarity_transform(self, points: np.ndarray, image_shape: tuple[int, int], original_person_mask: np.ndarray | None=None) -> np.ndarray | None:
        """Return an original-orientation square-crop transform.

        Despite the legacy method name, this intentionally does not rotate the
        face or force both eyes onto template coordinates.  It follows the older
        validated preprocessor idea: compute the visible face extent, crop that
        region from the original image, place it on a square canvas, and resize
        to the analysis size.  Display and analysis images therefore remain
        natural and complete, while skin_mask carries algorithm exclusions.
        """
        (height, width) = image_shape
        if points.size == 0 or height <= 0 or width <= 0:
            return None
        base_face_mask = self._face_hull_mask((height, width), points)
        (geometry_mask, forehead_completion) = self._forehead_geometry((height, width), points, base_face_mask)
        if original_person_mask is None:
            original_person_mask = np.full((height, width), 255, dtype=np.uint8)
        semantic_face = cv2.bitwise_or((original_person_mask > 0).astype(np.uint8) * 255, forehead_completion)
        crop_basis = cv2.bitwise_and(semantic_face, geometry_mask)
        if np.count_nonzero(crop_basis) < 20:
            crop_basis = geometry_mask
        (y_coords, x_coords) = np.where(crop_basis > 0)
        if y_coords.size == 0 or x_coords.size == 0:
            return None
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        box_w = max(x_max - x_min + 1, 1)
        box_h = max(y_max - y_min + 1, 1)
        pad_x = int(round(box_w * Config.CROP_PAD_X_RATIO))
        pad_top = int(round(box_h * Config.CROP_PAD_TOP_RATIO))
        pad_bottom = int(round(box_h * Config.CROP_PAD_BOTTOM_RATIO))
        crop_x_min = max(0, x_min - pad_x)
        crop_x_max = min(width, x_max + pad_x + 1)
        crop_y_min = max(0, y_min - pad_top)
        crop_y_max = min(height, y_max + pad_bottom + 1)
        rect_w = max(crop_x_max - crop_x_min, 1)
        rect_h = max(crop_y_max - crop_y_min, 1)
        square_side = max(rect_w, rect_h)
        off_x = (square_side - rect_w) * 0.5
        off_y = (square_side - rect_h) * 0.5
        scale = float(Config.ANALYSIS_SIZE) / float(square_side)
        return np.array([[scale, 0.0, scale * (off_x - crop_x_min)], [0.0, scale, scale * (off_y - crop_y_min)]], dtype=np.float32)

    @staticmethod
    def _pose_proxies(points: np.ndarray) -> tuple[float, float, float]:
        """Estimate 2-D pose quality proxies in degree-like units.

        These values are capture-quality gates, not clinical head-pose
        measurements.
        """
        (left_eye, right_eye) = ImagePreprocessor._eye_centres(points)
        eye_vector = right_eye - left_eye
        roll = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))
        left_cheek_x = float(points[234, 0])
        right_cheek_x = float(points[454, 0])
        if left_cheek_x > right_cheek_x:
            (left_cheek_x, right_cheek_x) = (right_cheek_x, left_cheek_x)
        nose_x = float(points[1, 0])
        half_width = max((right_cheek_x - left_cheek_x) * 0.5, 1.0)
        face_mid_x = (left_cheek_x + right_cheek_x) * 0.5
        yaw = 35.0 * (nose_x - face_mid_x) / half_width
        eye_y = float((left_eye[1] + right_eye[1]) * 0.5)
        chin_y = float(points[152, 1])
        nose_y = float(points[1, 1])
        vertical_span = max(chin_y - eye_y, 1.0)
        nose_ratio = (nose_y - eye_y) / vertical_span
        pitch = 80.0 * (nose_ratio - 0.38)
        return (float(yaw), float(pitch), float(roll))

    def _quality_assessment(self, img: np.ndarray, points: np.ndarray, face_count: int) -> tuple[float, str, list[str]]:
        flags: list[str] = []
        if face_count > 1:
            flags.append('MULTIPLE_FACES')
        (height, width) = img.shape[:2]
        face_mask = self._face_hull_mask((height, width), points)
        face_pixels = face_mask > 0
        face_area_ratio = float(np.count_nonzero(face_pixels) / max(height * width, 1))
        score = 20.0
        if face_area_ratio < Config.MIN_FACE_AREA_RATIO:
            flags.append('FACE_TOO_SMALL')
            area_score = 15.0 * np.clip(face_area_ratio / Config.MIN_FACE_AREA_RATIO, 0.0, 1.0)
        elif face_area_ratio < Config.IDEAL_FACE_AREA_RATIO:
            area_score = 10.0 + 5.0 * ((face_area_ratio - Config.MIN_FACE_AREA_RATIO) / (Config.IDEAL_FACE_AREA_RATIO - Config.MIN_FACE_AREA_RATIO))
        elif face_area_ratio <= Config.MAX_FACE_AREA_RATIO:
            area_score = 15.0
        else:
            flags.append('FACE_TOO_LARGE')
            area_score = 15.0 * np.clip((1.0 - face_area_ratio) / (1.0 - Config.MAX_FACE_AREA_RATIO), 0.0, 1.0)
        score += float(area_score)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_variance = float(np.var(laplacian[face_pixels])) if np.any(face_pixels) else 0.0
        blur_score = 15.0 * np.clip(blur_variance / Config.BLUR_GOOD_VARIANCE, 0.0, 1.0)
        if blur_variance < Config.BLUR_WARNING_VARIANCE:
            flags.append('BLUR')
        score += float(blur_score)
        face_gray = gray[face_pixels]
        dark_ratio = float(np.mean(face_gray < 20)) if face_gray.size else 1.0
        bright_ratio = float(np.mean(face_gray > 240)) if face_gray.size else 1.0
        clipped_ratio = dark_ratio + bright_ratio
        exposure_score = 15.0 * (1.0 - np.clip(clipped_ratio / Config.MAX_CLIPPED_RATIO, 0.0, 1.0))
        if dark_ratio > 0.12:
            flags.append('UNDEREXPOSED')
        if bright_ratio > 0.12:
            flags.append('OVEREXPOSED')
        score += float(exposure_score)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        highlight = (hsv[:, :, 2] > 245) & (hsv[:, :, 1] < 45) & face_pixels
        highlight_ratio = float(np.count_nonzero(highlight) / max(np.count_nonzero(face_pixels), 1))
        highlight_score = 10.0 * (1.0 - np.clip(highlight_ratio / Config.MAX_HIGHLIGHT_RATIO, 0.0, 1.0))
        if highlight_ratio > Config.MAX_HIGHLIGHT_RATIO:
            flags.append('EXCESSIVE_HIGHLIGHT')
        score += float(highlight_score)
        (yaw, pitch, roll) = self._pose_proxies(points)
        pose_fraction = max(abs(yaw) / Config.POSE_REJECT_YAW, abs(pitch) / Config.POSE_REJECT_PITCH, abs(roll) / Config.POSE_REJECT_ROLL)
        score += 20.0 * (1.0 - np.clip(pose_fraction, 0.0, 1.0))
        if abs(yaw) > Config.POSE_WARNING_YAW:
            flags.append('POSE_YAW')
        if abs(pitch) > Config.POSE_WARNING_PITCH:
            flags.append('POSE_PITCH')
        if abs(roll) > Config.POSE_WARNING_ROLL:
            flags.append('POSE_ROLL')
        if abs(yaw) > Config.POSE_REJECT_YAW or abs(pitch) > Config.POSE_REJECT_PITCH or abs(roll) > Config.POSE_REJECT_ROLL:
            flags.append('EXTREME_POSE')
            flags.append('PARTIAL_FACE')
        nose_x = int(np.clip(points[1, 0], 0, width - 1))
        (yy, xx) = np.indices((height, width))
        left_values = gray[face_pixels & (xx < nose_x)]
        right_values = gray[face_pixels & (xx >= nose_x)]
        if left_values.size and right_values.size:
            illumination_delta = abs(float(np.median(left_values)) - float(np.median(right_values)))
        else:
            illumination_delta = 255.0
        score += 5.0 * (1.0 - np.clip(illumination_delta / 45.0, 0.0, 1.0))
        if illumination_delta > 35.0:
            flags.append('UNEVEN_LIGHTING')
        score = float(np.clip(score, 0.0, 100.0))
        (score, status) = self._status_from_score(score, flags)
        return (round(score, 2), status, flags)

    def _person_mask(self, img: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, bool]:
        size = Config.ANALYSIS_SIZE
        (original_mask, ok) = self._person_mask_original(img)
        aligned_mask = cv2.warpAffine(original_mask, matrix, (size, size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return ((aligned_mask > 0).astype(np.uint8) * 255, ok)

    @staticmethod
    def _fill_feature(mask: np.ndarray, points: np.ndarray, indices: list[int], dilate: int) -> None:
        valid = [index for index in indices if index < len(points)]
        if len(valid) < 3:
            return
        polygon = cv2.convexHull(np.rint(points[valid]).astype(np.int32))
        local = np.zeros_like(mask)
        cv2.fillConvexPoly(local, polygon, 255)
        if dilate > 1:
            local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
        cv2.bitwise_or(mask, local, dst=mask)

    @staticmethod
    def _forehead_geometry(shape: tuple[int, int], points: np.ndarray, base_face_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return expanded face geometry and the area added above Face Mesh.

        MediaPipe's face oval stops near landmark 10 and can cut through the
        visible forehead.  A deterministic trapezoid extends the oval toward
        the hairline; colour/segmentation evidence is deliberately not used to
        decide whether this geometric completion exists.
        """
        (height, width) = shape
        required = (10, 54, 103, 105, 284, 332, 334)
        if len(points) <= max(required):
            return (base_face_mask.copy(), np.zeros_like(base_face_mask))
        left_temple = points[54].astype(np.float32)
        right_temple = points[284].astype(np.float32)
        left_upper = points[103].astype(np.float32)
        right_upper = points[332].astype(np.float32)
        if left_temple[0] > right_temple[0]:
            (left_temple, right_temple) = (right_temple, left_temple)
            (left_upper, right_upper) = (right_upper, left_upper)
        top = points[10].astype(np.float32)
        brow_y = float(min(points[105, 1], points[334, 1]))
        forehead_height = max(brow_y - float(top[1]), 1.0)
        upper_y = float(np.clip(top[1] - Config.FOREHEAD_EXTENSION_RATIO * forehead_height, 0.0, height - 1.0))
        temple_span = max(float(right_temple[0] - left_temple[0]), 1.0)
        centre_x = float(top[0])
        half_width = Config.FOREHEAD_TOP_HALF_WIDTH_RATIO * temple_span
        upper_left = np.array([np.clip(centre_x - half_width, 0.0, width - 1.0), upper_y], dtype=np.float32)
        upper_right = np.array([np.clip(centre_x + half_width, 0.0, width - 1.0), upper_y], dtype=np.float32)
        forehead_polygon = np.rint(np.stack((left_temple, left_upper, upper_left, upper_right, right_upper, right_temple))).astype(np.int32)
        forehead_mask = np.zeros_like(base_face_mask)
        cv2.fillConvexPoly(forehead_mask, cv2.convexHull(forehead_polygon), 255)
        expanded_base = cv2.dilate(base_face_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.FACE_GEOMETRY_DILATE, Config.FACE_GEOMETRY_DILATE)))
        geometry_mask = cv2.bitwise_or(expanded_base, forehead_mask)
        completion = cv2.bitwise_and(forehead_mask, cv2.bitwise_not(base_face_mask))
        return (geometry_mask, completion)

    def _hair_mask(self, analysis_image: np.ndarray, face_mask: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Conservatively exclude dark components connected to the upper face edge."""
        lab = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        valid_l = l_channel[face_mask > 0]
        if valid_l.size < 100:
            return np.zeros_like(face_mask)
        adaptive_threshold = max(25.0, float(np.percentile(valid_l, 12)) - 10.0)
        dark = ((l_channel < adaptive_threshold) & (face_mask > 0)).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        boundary = cv2.morphologyEx(face_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        (count, labels, stats, _) = cv2.connectedComponentsWithStats(dark, connectivity=8)
        hair = np.zeros_like(face_mask)
        for label in range(1, count):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= 12 and np.any(component & (boundary > 0)):
                hair[component] = 255
        return cv2.dilate(hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    def _skin_mask(self, analysis_image: np.ndarray, points: np.ndarray, person_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size = Config.ANALYSIS_SIZE
        shape = (size, size)
        semantic_skin_mask = (person_mask > 0).astype(np.uint8) * 255
        base_face_mask = self._face_hull_mask(shape, points)
        (geometry_mask, forehead_completion) = self._forehead_geometry(shape, points, base_face_mask)
        semantic_face_skin = cv2.bitwise_or(semantic_skin_mask, forehead_completion)
        face_candidate = cv2.bitwise_and(semantic_face_skin, geometry_mask)
        feature_masks = build_feature_exclusion_masks(analysis_image, points)
        feature_exclusions = feature_masks['feature_exclusions']
        legacy_bulk_hair = self._hair_mask(analysis_image, face_candidate, points)
        hair_maps = detect_visible_hair_masks(analysis_image, face_candidate, points, feature_exclusions)
        bulk_hair = cv2.bitwise_or(hair_maps['bulk_hair_mask'], legacy_bulk_hair)
        hair = cv2.bitwise_or(hair_maps['hair_mask'], legacy_bulk_hair)
        exclusions = cv2.bitwise_or(feature_exclusions, hair)
        skin = cv2.bitwise_and(face_candidate, cv2.bitwise_not(exclusions))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        skin = cv2.bitwise_and(skin, cv2.bitwise_not(exclusions))
        skin = cv2.bitwise_and(skin, geometry_mask)
        display_face_mask = cv2.bitwise_and(face_candidate, cv2.bitwise_not(hair))
        self.last_debug_masks = {'semantic_skin_mask': semantic_skin_mask, 'face_geometry_mask': geometry_mask, 'forehead_completion': forehead_completion, 'feature_exclusions': feature_exclusions, 'nostril_mask': feature_masks['nostril_mask'], 'bulk_hair_mask': bulk_hair, 'color_hair_mask': hair_maps['color_hair_mask'], 'strand_hair_mask': hair_maps['strand_hair_mask'], 'facial_hair_mask': hair_maps['facial_hair_mask'], 'hair_mask': hair, 'hair_line_response': hair_maps['hair_line_response'], 'hair_scale_vote': hair_maps['hair_scale_vote'], 'display_face_mask': display_face_mask}
        return ((skin > 0).astype(np.uint8) * 255, display_face_mask)

    def preprocess_image(self, img: np.ndarray) -> PreprocessResultV2:
        """Build the complete V2 preprocessing result for one decoded photo."""
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3 or (img.size == 0):
            fallback = np.zeros((1, 1, 3), dtype=np.uint8)
            return self._fallback_result(fallback, ['INVALID_IMAGE'])
        detection = self.face_landmarker.detect(self._to_mp_image(img))
        faces = detection.face_landmarks or []
        if not faces:
            return self._fallback_result(img, ['NO_FACE'])
        (height, width) = img.shape[:2]
        all_points = [self._landmarks_to_pixels(face, width, height) for face in faces]
        primary_index = max(range(len(all_points)), key=lambda index: cv2.contourArea(cv2.convexHull(all_points[index].astype(np.float32))))
        primary_points = all_points[primary_index]
        primary_relative_z = np.asarray([landmark.z * width for landmark in faces[primary_index]], dtype=np.float32)
        (quality_score, quality_status, quality_flags) = self._quality_assessment(img, primary_points, len(faces))
        (original_person_mask, segmentation_ok) = self._person_mask_original(img)
        if not segmentation_ok:
            quality_flags.append('SEGMENTATION_FALLBACK')
            quality_score = max(0.0, quality_score - 5.0)
        matrix = self._similarity_transform(primary_points, img.shape[:2], original_person_mask)
        if matrix is None:
            return self._fallback_result(img, quality_flags + ['INVALID_FACE_GEOMETRY'])
        size = Config.ANALYSIS_SIZE
        analysis_image = cv2.warpAffine(img, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        inverse = cv2.invertAffineTransform(matrix).astype(np.float32)
        aligned_points = self._transform_points(primary_points, matrix)
        similarity_scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
        aligned_relative_z = primary_relative_z * similarity_scale
        person_mask = cv2.warpAffine(original_person_mask, matrix, (size, size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        person_mask = (person_mask > 0).astype(np.uint8) * 255
        (skin_mask, display_face_mask) = self._skin_mask(analysis_image, aligned_points, person_mask)
        skin_ratio = float(np.count_nonzero(skin_mask) / skin_mask.size)
        if skin_ratio < Config.MIN_SKIN_AREA_RATIO:
            quality_flags.append('INSUFFICIENT_SKIN_AREA')
            quality_score = max(0.0, quality_score - 15.0)
        (quality_score, quality_status) = self._status_from_score(quality_score, quality_flags)
        quality_score = round(float(np.clip(quality_score, 0.0, 100.0)), 2)
        capture_profile = getattr(self, 'capture_profile', CaptureProfile.CONSUMER)
        decision = evaluate_consumer_quality(analysis_image, skin_mask, self.last_debug_masks, quality_score, quality_status, quality_flags)
        quality_score = decision.score
        quality_status = decision.status
        quality_flags = list(decision.flags)
        quality_metrics = {'local_lighting_block_range': decision.local_lighting_block_range, 'local_lighting_gradient_p99': decision.local_lighting_gradient_p99, 'local_lighting_bright_ratio': decision.local_lighting_bright_ratio}
        if 'MULTIPLE_FACES' in quality_flags:
            skin_mask.fill(0)
        display_image = analysis_image.copy()
        result = PreprocessResultV2(analysis_image=analysis_image, display_image=display_image, skin_mask=skin_mask, landmarks=aligned_points, face_transform_matrix=matrix.astype(np.float32), inverse_transform_matrix=inverse, quality_score=quality_score, quality_status=quality_status, quality_flags=quality_flags, quality_metrics=quality_metrics, landmarks_relative_z=aligned_relative_z.astype(np.float32))
        self._attach_mask_bundle(result)
        self.last_result = result
        return result

    def remove_background(self, img: np.ndarray) -> np.ndarray:
        """Return the display product while caching the paired analysis result."""
        result = self.preprocess_image(img)
        self._compat_result = result
        self._compat_display_id = id(result.display_image)
        return result.display_image

    def color_transfer(self, source_img: np.ndarray) -> np.ndarray:
        """Deprecated V1 adapter; intentionally performs no colour transfer."""
        return source_img

    def crop_to_face_visia(self, img: np.ndarray) -> np.ndarray:
        """Return the V2 analysis image through the legacy method name."""
        if self._compat_result is not None and id(img) == self._compat_display_id:
            analysis = self._compat_result.analysis_image
            self._compat_result = None
            self._compat_display_id = None
            return analysis
        return self.preprocess_image(img).analysis_image

    def get_face_mask(self, img: np.ndarray) -> np.ndarray:
        """Compatibility helper returning the V2 valid-skin mask."""
        return self.preprocess_image(img).skin_mask

    def close(self) -> None:
        """Release MediaPipe task resources when the caller owns the lifecycle."""
        for model in (self.face_landmarker, self.segmenter):
            try:
                model.close()
            except Exception:
                pass
__all__ = ['AnalysisMaskBundle', 'ImagePreprocessor', 'PreprocessResultV2', 'build_analysis_mask_bundle', 'build_feature_exclusion_masks', 'detect_visible_hair_masks', 'evaluate_consumer_quality']
