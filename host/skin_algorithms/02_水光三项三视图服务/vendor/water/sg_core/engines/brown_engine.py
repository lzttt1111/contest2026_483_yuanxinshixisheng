"""VISIA-like Brown Spots / Brown Areas engine for ordinary RGB images.

This implementation detects *visible pigmentation differences* in the
standardized sRGB face image.  It cannot reproduce subsurface melanin imaging
or Canfield's proprietary RBX Brown algorithm, so every output is explicitly
labelled RBX-like rather than clinically equivalent to VISIA.
"""
from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass, replace
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from typing_extensions import assert_never
from sg_core.capture_profile import CaptureProfile
from sg_core.consumer_pigment.brown_contrast import enhance_brown_score_contrast
from sg_core.consumer_pigment.brown_detection_policy import BrownDetectionPolicy, policy_for_brown_profile, policy_for_consumer_brown_detection_recall
from sg_core.consumer_pigment.formal_markers import ConsumerBrownRecallPreset, ConsumerMarkerInput, ConsumerPigmentMarkerKind, policy_for_consumer_brown_recall, project_consumer_markers
from sg_core.consumer_pigment.markers import BROWN_STYLE_MARKER, render_compact_marker_mask
from sg_core.consumer_pigment.specular import build_specular_evidence
from sg_core.engines.rbx_engine import ErythemaAnalyzer
from sg_core.engines.visia_regions import BROW_LANDMARKS, MOUTH_LANDMARKS, NOSE_LANDMARKS, REGION_LABELS, REGION_ORDER, VISIA_BOUNDARY_COLOR, VisiaRegionSet, _smooth_binary_mask, assign_region, build_visia_regions, compact_region_counts, draw_region_boundaries, masked_gaussian, polygon_mask, regional_positive_z, robust_unit_map
from sg_core.preprocess.image_preprocessor import PreprocessResultV2
from sg_core.utils.io_utils import cv_imread, cv_imwrite
from sg_core.utils.gpu_backend import get_cuda_backend

class Config:
    """Brown V1 defaults for a 1024×1024 Preprocessor V2 analysis image."""
    LOCAL_SIGMAS = (8.0, 16.0, 32.0, 48.0)
    SCALE_Z_THRESHOLD = 1.65
    MIN_SCALE_VOTES = 3
    CANDIDATE_Z_THRESHOLD = 2.0
    DARK_WEIGHT = 0.44
    YELLOW_WEIGHT = 0.28
    RED_BROWN_WEIGHT = 0.16
    RGB_BROWN_WEIGHT = 0.12
    MIN_AREA_PX = 14
    MAX_AREA_RATIO = 0.012
    MIN_MEAN_SCORE = 0.28
    MIN_SOLIDITY = 0.18
    MAX_ASPECT_RATIO = 5.0
    PEAK_MIN_DISTANCE = 11
    LARGE_SPLIT_AREA = 160
    RENDER_FEATURE_EXCLUSION_MARGIN_PX = 25
    FEATURE_EXCLUSION_MARGIN_PX = 30
    NASOLABIAL_CORRIDOR_RADIUS_PX = 25
    NASOLABIAL_MIN_OVERLAP_RATIO = 0.1
    NASOLABIAL_HARD_OVERLAP_RATIO = 0.45
    NASOLABIAL_MIN_ASPECT_RATIO = 1.35
    NASOLABIAL_MAX_SOLIDITY = 0.7
    NASOLABIAL_LARGE_AREA_PX = 55
    COLOR_BASE = np.array([205, 220, 240], dtype=np.float32)
    COLOR_BROWN = np.array([10, 50, 150], dtype=np.float32)
    RENDER_PERCENTILE_LOW = 5.0
    RENDER_PERCENTILE_HIGH = 98.0
    RENDER_GAMMA = 0.65
    RENDER_COLOR_SMOOTH_SIGMA = 1.2
    RENDER_DENSITY_CONTRAST = 1.95
    RENDER_DENSITY_FLOOR = 0.22
    RENDER_TONE_WEIGHT = 0.12
    RENDER_TONE_WHITE_LEVEL = 0.92
    RENDER_TONE_BLACK_LEVEL = 0.18
    RENDER_TEXTURE_FINE_SIGMA = 0.75
    RENDER_TEXTURE_MEDIUM_SIGMA = 2.5
    RENDER_TEXTURE_FINE_WEIGHT = 0.75
    RENDER_TEXTURE_MEDIUM_WEIGHT = 0.25
    RENDER_TEXTURE_CLIP_LOW = -0.12
    RENDER_TEXTURE_CLIP_HIGH = 0.035
    RENDER_TEXTURE_WEIGHT = 1.35
    RENDER_DARK_WEIGHT = 0.56
    RENDER_YELLOW_WEIGHT = 0.22
    RENDER_RED_BROWN_WEIGHT = 0.14
    RENDER_RGB_BROWN_WEIGHT = 0.08
    INSTANCE_COLOR = (220, 235, 45)
    REGION_COLOR = VISIA_BOUNDARY_COLOR

@dataclass
class BrownResult:
    brown_spot_count: int
    brown_spot_area_ratio: float
    brown_spot_locations: list[dict]
    region_distribution: dict[str, dict]
    mean_brownness: float
    p90_brownness: float
    brownness_burden: float
    brown_score_map: np.ndarray
    instance_mask: np.ndarray
    brown_rbx_image: np.ndarray
    brown_overlay: np.ndarray
    partial_face: bool
    quality_score: float
    quality_status: str
    quality_flags: list[str]

    def metrics(self) -> dict:
        return {'algorithm': 'RBX-like Visible Brown Spots V1', 'brown_spot_count': self.brown_spot_count, 'brown_spot_area_ratio': self.brown_spot_area_ratio, 'brown_spot_locations': self.brown_spot_locations, 'region_distribution': self.region_distribution, 'mean_brownness': self.mean_brownness, 'p90_brownness': self.p90_brownness, 'brownness_burden': self.brownness_burden, 'partial_face': self.partial_face, 'quality_score': self.quality_score, 'quality_status': self.quality_status, 'quality_flags': self.quality_flags, 'disclaimer': '当前结果是普通RGB照片中的可见色素差异工程分析，不等价于VISIA官方RBX棕色或医学诊断。'}

def _component_locations(labels: np.ndarray, score: np.ndarray, regions: VisiaRegionSet, nasolabial_mask: np.ndarray, policy: BrownDetectionPolicy) -> tuple[list[dict], np.ndarray]:
    locations: list[dict] = []
    output_mask = np.zeros(labels.shape, dtype=np.uint8)
    skin_area = max(int(np.count_nonzero(regions.analysis_mask)), 1)
    max_area = int(Config.MAX_AREA_RATIO * skin_area)
    next_id = 1
    for label in range(1, int(labels.max()) + 1):
        component = labels == label
        area = int(np.count_nonzero(component))
        if area < policy.minimum_area_px or area > max_area:
            continue
        component_u8 = component.astype(np.uint8) * 255
        (contours, _) = cv2.findContours(component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        contour_area = max(float(cv2.contourArea(contour)), 1.0)
        hull_area = max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect = max(w, h) / max(min(w, h), 1)
        rotated_rect = cv2.minAreaRect(contour)
        (rotated_w, rotated_h) = rotated_rect[1]
        oriented_aspect = max(rotated_w, rotated_h) / max(min(rotated_w, rotated_h), 1.0)
        solidity = contour_area / hull_area
        mean_score = float(np.mean(score[component]))
        if mean_score < policy.minimum_mean_score or solidity < policy.minimum_solidity or aspect > policy.maximum_aspect_ratio:
            continue
        fold_overlap = float(np.count_nonzero(component & (nasolabial_mask > 0))) / max(area, 1)
        fold_like = fold_overlap >= Config.NASOLABIAL_HARD_OVERLAP_RATIO or (fold_overlap >= Config.NASOLABIAL_MIN_OVERLAP_RATIO and (oriented_aspect >= Config.NASOLABIAL_MIN_ASPECT_RATIO or solidity <= Config.NASOLABIAL_MAX_SOLIDITY or area >= Config.NASOLABIAL_LARGE_AREA_PX))
        if fold_like:
            continue
        component_scores = np.where(component, score, -1.0)
        (peak_y, peak_x) = np.unravel_index(int(np.argmax(component_scores)), component_scores.shape)
        compact_radius = int(np.clip(round(0.72 * np.sqrt(area / np.pi)), 3, 8))
        compact_disk = np.zeros_like(output_mask)
        cv2.circle(compact_disk, (int(peak_x), int(peak_y)), compact_radius, 255, -1)
        compact_component = component & (compact_disk > 0)
        if not np.any(compact_component):
            continue
        compact_u8 = compact_component.astype(np.uint8) * 255
        (compact_contours, _) = cv2.findContours(compact_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not compact_contours:
            continue
        contour = max(compact_contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour)
        area = int(np.count_nonzero(compact_component))
        mean_score = float(np.mean(score[compact_component]))
        moments = cv2.moments(contour)
        if abs(moments['m00']) < 1e-06:
            continue
        cx = float(moments['m10'] / moments['m00'])
        cy = float(moments['m01'] / moments['m00'])
        region = assign_region(cx, cy, regions.regions)
        if region == 'other':
            continue
        output_mask[compact_component] = 255
        locations.append({'id': next_id, 'centroid': [round(cx, 2), round(cy, 2)], 'bbox': [int(x), int(y), int(w), int(h)], 'area': area, 'region': region, 'mean_brownness': round(mean_score, 5), 'confidence': round(float(np.clip(0.35 + 0.65 * mean_score, 0, 1)), 4)})
        next_id += 1
    return (locations, output_mask)

class BrownAreaAnalyzer:
    """Detect visible brown/pigmented features from Preprocessor V2 output."""

    def __init__(self, capture_profile: CaptureProfile=CaptureProfile.CONSUMER, consumer_brown_recall_preset: ConsumerBrownRecallPreset=ConsumerBrownRecallPreset.RECALL_3) -> None:
        self.capture_profile = capture_profile
        self.consumer_brown_recall_preset = consumer_brown_recall_preset
        self.detection_policy = policy_for_brown_profile(capture_profile)
        self._face_mask_analyzer: ErythemaAnalyzer | None = None

    def close(self) -> None:
        if self._face_mask_analyzer is not None:
            self._face_mask_analyzer.close()
            self._face_mask_analyzer = None

    def _shared_render_masks(self, image: np.ndarray, landmarks: np.ndarray, skin_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return separate analysis-face and display-person masks.

        Brown colour/score calculations use the first mask indirectly through
        ``skin_mask``.  The second mask is intentionally broader: it keeps
        hair and other visible person pixels in the user-facing result while
        removing the background.
        """
        if self._face_mask_analyzer is None:
            self._face_mask_analyzer = ErythemaAnalyzer(color_transfer_enabled=False)
        face_mask = self._face_mask_analyzer._semantic_face_mask(image, landmarks, skin_mask)
        foreground_mask = self._face_mask_analyzer._person_foreground_mask(image, face_mask)
        return (face_mask, foreground_mask)

    def _profile_analysis_skin_mask(self, preprocess_result: PreprocessResultV2) -> np.ndarray:
        """Use the common consumer mask as the upper bound for Brown."""
        bundle = preprocess_result.mask_bundle
        if bundle is None:
            return np.asarray(preprocess_result.skin_mask).copy()
        return bundle.algorithm_mask()

    @staticmethod
    def _historical_analysis_regions(skin_mask: np.ndarray, landmarks: np.ndarray, current: VisiaRegionSet) -> VisiaRegionSet:
        """Restore the pre-filled-scope Brown analysis regions.

        The current VISIA object remains authoritative for display geometry
        and the narrow public-line guard. Candidate and quantification masks
        follow the historical safe-skin/landmark split instead of the filled
        public contour.
        """
        base_skin = (np.asarray(skin_mask) > 0).astype(np.uint8) * 255
        exclusions = (np.asarray(current.feature_exclusion_mask) > 0).astype(np.uint8) * 255
        safe_skin = cv2.bitwise_and(base_skin, cv2.bitwise_not(exclusions))
        (yy, xx) = np.indices(safe_skin.shape)
        (valid_y, _valid_x) = np.where(safe_skin > 0)
        if valid_y.size < 500:
            raise ValueError('valid skin region is too small')
        points = np.asarray(landmarks, dtype=np.float32)
        centre_x = int(np.clip(points[1, 0], 0, safe_skin.shape[1] - 1))
        brow_y = int(np.clip(min(points[BROW_LANDMARKS[0], 1], points[BROW_LANDMARKS[1], 1]), int(valid_y.min()), int(valid_y.max())))
        mouth_y = int(np.clip(np.mean(points[list(MOUTH_LANDMARKS), 1]), int(valid_y.min()), int(valid_y.max())))
        nose = polygon_mask(safe_skin.shape, points, list(NOSE_LANDMARKS))
        nose = cv2.dilate(nose, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        nose = cv2.bitwise_and(nose, safe_skin)
        skin = safe_skin > 0
        middle = (yy >= brow_y) & (yy <= mouth_y) & skin & (nose == 0)
        raw = {'forehead': ((yy < brow_y) & skin).astype(np.uint8) * 255, 'left_cheek': (middle & (xx < centre_x)).astype(np.uint8) * 255, 'right_cheek': (middle & (xx >= centre_x)).astype(np.uint8) * 255, 'nose': nose, 'chin': ((yy > mouth_y) & skin).astype(np.uint8) * 255}
        regions: dict[str, np.ndarray] = {}
        occupied = np.zeros_like(safe_skin)
        for name in ('nose', 'forehead', 'left_cheek', 'right_cheek', 'chin'):
            mask = _smooth_binary_mask(raw[name])
            mask = cv2.bitwise_and(mask, safe_skin)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(occupied))
            regions[name] = mask
            occupied = cv2.bitwise_or(occupied, mask)
        return replace(current, analysis_mask=occupied, regions=regions)

    @staticmethod
    def _marker_exclusion(exclusion: np.ndarray, regions: VisiaRegionSet) -> np.ndarray:
        boundary = regions.public_boundary_safety_mask
        if boundary is None:
            return (np.asarray(exclusion) > 0).astype(np.uint8) * 255
        return cv2.bitwise_or((np.asarray(exclusion) > 0).astype(np.uint8) * 255, (np.asarray(boundary) > 0).astype(np.uint8) * 255)

    def _regions_for_profile(self, skin_mask: np.ndarray, landmarks: np.ndarray, current: VisiaRegionSet) -> VisiaRegionSet:
        return current

    @staticmethod
    def _nasolabial_fold_mask(shape: tuple[int, int], landmarks: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
        """Build two nose-wing-to-mouth-corner wrinkle corridors."""
        mask = np.zeros(shape, dtype=np.uint8)
        for path_indices in ((209, 203, 206, 61), (429, 423, 426, 291)):
            if any((index >= len(landmarks) for index in path_indices)):
                continue
            path = np.rint(landmarks[list(path_indices), :2]).astype(np.int32)
            curve = path.astype(np.float32)
            for _ in range(3):
                first = 0.75 * curve[:-1] + 0.25 * curve[1:]
                second = 0.25 * curve[:-1] + 0.75 * curve[1:]
                smoothed = np.empty((2 * len(first) + 2, 2), dtype=np.float32)
                smoothed[0] = curve[0]
                smoothed[-1] = curve[-1]
                smoothed[1:-1:2] = first
                smoothed[2:-1:2] = second
                curve = smoothed
            cv2.polylines(mask, [np.rint(curve).astype(np.int32).reshape(-1, 1, 2)], False, 255, thickness=2 * Config.NASOLABIAL_CORRIDOR_RADIUS_PX + 1, lineType=cv2.LINE_AA)
        return cv2.bitwise_and(mask, analysis_mask)

    @staticmethod
    def _brown_maps(image: np.ndarray, regions: VisiaRegionSet, policy: BrownDetectionPolicy) -> tuple[np.ndarray, np.ndarray]:
        mask = regions.analysis_mask
        safe_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0] / 255.0
        a_channel = (lab[:, :, 1] - 128.0) / 127.0
        b_channel = (lab[:, :, 2] - 128.0) / 127.0
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_sum = np.maximum(np.sum(rgb, axis=2), 1e-05)
        red_chroma = rgb[:, :, 0] / rgb_sum
        blue_chroma = rgb[:, :, 2] / rgb_sum
        rgb_brown = red_chroma - blue_chroma
        scale_z_maps = []
        raw_maps = []
        backgrounds = get_cuda_backend().masked_gaussian_batch((l_channel, a_channel, b_channel, rgb_brown), mask, Config.LOCAL_SIGMAS)
        for sigma in Config.LOCAL_SIGMAS:
            (bg_l, bg_a, bg_b, bg_rgb) = backgrounds[float(sigma)]
            dark = np.maximum(bg_l - l_channel, 0.0)
            red_brown = np.maximum(a_channel - bg_a, 0.0)
            yellow = np.maximum(b_channel - bg_b, 0.0)
            chroma = np.maximum(rgb_brown - bg_rgb, 0.0)
            raw = Config.DARK_WEIGHT * dark / 0.035 + Config.YELLOW_WEIGHT * yellow / 0.025 + Config.RED_BROWN_WEIGHT * red_brown / 0.025 + Config.RGB_BROWN_WEIGHT * chroma / 0.018
            raw[mask == 0] = 0.0
            raw_maps.append(raw)
            scale_z_maps.append(regional_positive_z(raw, regions, sigma_floor=0.12))
        z_stack = np.stack(scale_z_maps, axis=0)
        raw_stack = np.stack(raw_maps, axis=0)
        scale_votes = np.sum(z_stack >= policy.scale_z_threshold, axis=0).astype(np.uint8)
        combined = 0.65 * np.max(z_stack, axis=0) + 0.35 * np.mean(z_stack, axis=0)
        score = robust_unit_map(combined, mask, 2.0, 99.0)
        candidate = ((scale_votes >= policy.minimum_scale_votes) & (combined >= policy.candidate_z_threshold) & (np.max(raw_stack, axis=0) > policy.minimum_raw_response) & (safe_mask > 0)).astype(np.uint8) * 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        return (score, candidate)

    @staticmethod
    def _split_instances(candidate: np.ndarray, score: np.ndarray, policy: BrownDetectionPolicy) -> np.ndarray:
        if not np.count_nonzero(candidate):
            return np.zeros(candidate.shape, dtype=np.int32)
        coordinates = peak_local_max(score, min_distance=policy.peak_min_distance, threshold_abs=policy.minimum_mean_score, labels=(candidate > 0).astype(np.uint8), exclude_border=False)
        markers = np.zeros(candidate.shape, dtype=np.int32)
        for (index, (y, x)) in enumerate(coordinates, 1):
            markers[y, x] = index
        if not np.count_nonzero(markers):
            (_, labels) = cv2.connectedComponents((candidate > 0).astype(np.uint8))
            return labels.astype(np.int32)
        return watershed(-score, markers, mask=candidate > 0).astype(np.int32)

    @staticmethod
    def _render_brown(image: np.ndarray, face_render_mask: np.ndarray, person_foreground_mask: np.ndarray, statistics_mask: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0] / 255.0
        a_channel = (lab[:, :, 1] - 128.0) / 127.0
        b_channel = (lab[:, :, 2] - 128.0) / 127.0
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_sum = np.maximum(np.sum(rgb, axis=2), 1e-05)
        rgb_brown = rgb[:, :, 0] / rgb_sum - rgb[:, :, 2] / rgb_sum
        raw_brown = Config.RENDER_DARK_WEIGHT * (1.0 - l_channel) + Config.RENDER_YELLOW_WEIGHT * np.maximum(b_channel, 0.0) + Config.RENDER_RED_BROWN_WEIGHT * np.maximum(a_channel, 0.0) + Config.RENDER_RGB_BROWN_WEIGHT * np.maximum(rgb_brown, 0.0)
        continuous_person = masked_gaussian(raw_brown, person_foreground_mask, Config.RENDER_COLOR_SMOOTH_SIGMA)
        continuous_face = masked_gaussian(raw_brown, face_render_mask, Config.RENDER_COLOR_SMOOTH_SIGMA)
        continuous_brown = continuous_person
        continuous_brown[face_render_mask > 0] = continuous_face[face_render_mask > 0]
        valid = continuous_brown[statistics_mask > 0]
        if valid.size:
            (low, high) = np.percentile(valid, [Config.RENDER_PERCENTILE_LOW, Config.RENDER_PERCENTILE_HIGH])
        else:
            (low, high) = (0.0, 1.0)
        low = float(low)
        high = float(high)
        if high - low < 0.0001:
            global_norm = np.zeros_like(continuous_brown, dtype=np.float32)
        else:
            global_norm = np.clip((continuous_brown - low) / (high - low), 0.0, 1.0)
        brown_norm = global_norm
        brown_norm = np.power(brown_norm, Config.RENDER_GAMMA)
        brown_density = 1.0 - np.power(np.clip(1.0 - brown_norm, 0.0, 1.0), Config.RENDER_DENSITY_CONTRAST)
        tone_density = np.clip((Config.RENDER_TONE_WHITE_LEVEL - l_channel) / max(Config.RENDER_TONE_WHITE_LEVEL - Config.RENDER_TONE_BLACK_LEVEL, 1e-06), 0.0, 1.0)
        density = np.clip(Config.RENDER_TONE_WEIGHT * tone_density + (1.0 - Config.RENDER_TONE_WEIGHT) * brown_density, 0.0, 1.0)
        density = Config.RENDER_DENSITY_FLOOR + (1.0 - Config.RENDER_DENSITY_FLOOR) * density
        canvas = Config.COLOR_BASE[None, None, :] * (1.0 - density[:, :, None]) + Config.COLOR_BROWN[None, None, :] * density[:, :, None]
        fine_person = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=Config.RENDER_TEXTURE_FINE_SIGMA, sigmaY=Config.RENDER_TEXTURE_FINE_SIGMA)
        medium_person = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=Config.RENDER_TEXTURE_MEDIUM_SIGMA, sigmaY=Config.RENDER_TEXTURE_MEDIUM_SIGMA)
        fine_face = masked_gaussian(l_channel, face_render_mask, Config.RENDER_TEXTURE_FINE_SIGMA)
        medium_face = masked_gaussian(l_channel, face_render_mask, Config.RENDER_TEXTURE_MEDIUM_SIGMA)
        fine_background = fine_person
        medium_background = medium_person
        face_pixels = face_render_mask > 0
        fine_background[face_pixels] = fine_face[face_pixels]
        medium_background[face_pixels] = medium_face[face_pixels]
        texture = Config.RENDER_TEXTURE_FINE_WEIGHT * (l_channel - fine_background) + Config.RENDER_TEXTURE_MEDIUM_WEIGHT * (l_channel - medium_background)
        texture = np.clip(texture, Config.RENDER_TEXTURE_CLIP_LOW, Config.RENDER_TEXTURE_CLIP_HIGH)
        canvas += texture[:, :, None] * 255.0 * Config.RENDER_TEXTURE_WEIGHT
        del person_foreground_mask
        return np.clip(canvas, 0, 255).astype(np.uint8)

    def detect_brown(self, preprocess_result: PreprocessResultV2) -> BrownResult:
        image = np.asarray(preprocess_result.analysis_image)
        analysis_skin_mask = self._profile_analysis_skin_mask(preprocess_result)
        current_render_regions = build_visia_regions(image, analysis_skin_mask, preprocess_result.landmarks, preprocess_result.quality_flags, include_chin=True, mode='full', feature_margin_px=Config.RENDER_FEATURE_EXCLUSION_MARGIN_PX)
        render_regions = self._regions_for_profile(analysis_skin_mask, preprocess_result.landmarks, current_render_regions)
        current_instance_regions = build_visia_regions(image, analysis_skin_mask, preprocess_result.landmarks, preprocess_result.quality_flags, include_chin=True, mode='full', feature_margin_px=Config.FEATURE_EXCLUSION_MARGIN_PX)
        instance_regions = self._regions_for_profile(analysis_skin_mask, preprocess_result.landmarks, current_instance_regions)
        active_detection_policy = policy_for_consumer_brown_detection_recall(self.consumer_brown_recall_preset) if self.capture_profile is CaptureProfile.CONSUMER else self.detection_policy
        (score, candidate) = self._brown_maps(image, render_regions, active_detection_policy)
        candidate = cv2.bitwise_and(candidate, instance_regions.analysis_mask)
        labels = self._split_instances(candidate, score, active_detection_policy)
        nasolabial_mask = self._nasolabial_fold_mask(image.shape[:2], preprocess_result.landmarks, instance_regions.analysis_mask)
        (locations, instance_mask) = _component_locations(labels, score, instance_regions, nasolabial_mask, active_detection_policy)
        exclusion = instance_regions.feature_exclusion_mask
        specular = build_specular_evidence(image, instance_regions.analysis_mask)
        exclusion = cv2.bitwise_or(exclusion, specular.mask)
        marker_policy = policy_for_consumer_brown_recall(self.consumer_brown_recall_preset) if self.capture_profile is CaptureProfile.CONSUMER else None
        projection = project_consumer_markers(ConsumerMarkerInput(instance_mask, score, instance_regions.analysis_mask, exclusion, instance_regions), ConsumerPigmentMarkerKind.BROWN, policy=marker_policy)
        instance_mask = projection.marker_mask
        locations = [{'id': index, 'centroid': list(item.centroid), 'bbox': list(item.bbox), 'area': item.marker_area_px, 'region': item.region, 'mean_brownness': round(item.peak_score, 5), 'confidence': item.confidence} for (index, item) in enumerate(projection.findings, 1)]
        skin_area = max(int(np.count_nonzero(instance_regions.analysis_mask)), 1)
        area_ratio = float(np.count_nonzero(instance_mask) / skin_area)
        valid_values = score[render_regions.analysis_mask > 0]
        mean_value = float(np.mean(valid_values)) if valid_values.size else 0.0
        p90 = float(np.percentile(valid_values, 90)) if valid_values.size else 0.0
        distribution = {}
        for name in REGION_ORDER:
            local = [item for item in locations if item['region'] == name]
            region_area = max(int(np.count_nonzero(instance_regions.regions[name])), 1)
            feature_area = sum((int(item['area']) for item in local))
            distribution[name] = {'label': REGION_LABELS[name], 'count': len(local), 'area_ratio': round(feature_area / region_area, 8)}
        (render_mask, foreground_mask) = self._shared_render_masks(image, preprocess_result.landmarks, preprocess_result.skin_mask)
        render = self._render_brown(image, render_mask, foreground_mask, render_regions.analysis_mask)
        render = enhance_brown_score_contrast(render, score, render_regions.analysis_mask)
        overlay = render_compact_marker_mask(render, instance_mask, BROWN_STYLE_MARKER)
        overlay = draw_region_boundaries(overlay, render_regions.display_regions, Config.REGION_COLOR, thickness=2, partial_face=render_regions.partial_face, closed_contour=render_regions.display_contour, separator_contour=render_regions.display_separator)
        result = BrownResult(brown_spot_count=len(locations), brown_spot_area_ratio=round(area_ratio, 8), brown_spot_locations=locations, region_distribution=distribution, mean_brownness=round(mean_value, 6), p90_brownness=round(p90, 6), brownness_burden=round(area_ratio * p90, 8), brown_score_map=score, instance_mask=instance_mask, brown_rbx_image=render, brown_overlay=overlay, partial_face=render_regions.partial_face, quality_score=float(preprocess_result.quality_score), quality_status=str(preprocess_result.quality_status), quality_flags=list(preprocess_result.quality_flags))
        result._medical_analysis_mask = instance_regions.analysis_mask.copy()
        result._medical_landmarks = np.asarray(preprocess_result.landmarks, dtype=np.float32).copy()
        result._filtered_hair_mask = np.zeros_like(result.instance_mask)
        return result

def detect_brown(preprocess_result: PreprocessResultV2) -> BrownResult:
    return BrownAreaAnalyzer().detect_brown(preprocess_result)
