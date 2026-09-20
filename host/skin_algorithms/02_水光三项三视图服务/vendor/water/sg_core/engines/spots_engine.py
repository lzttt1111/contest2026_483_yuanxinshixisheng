"""DermaVision Spots Engine V2.

This module detects visible, discrete colour/brightness anomalies in ordinary
RGB photographs. It does not detect RBX Brown Spots, Red Areas, UV Spots, and
it is not a medical diagnosis.

The algorithm consumes only Preprocessor V2's analysis image, skin mask, and
aligned landmarks. The display image is intentionally never read.
"""
from __future__ import annotations
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Any
import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.filters import frangi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sg_core.engines.visia_regions import VISIA_BOUNDARY_COLOR, build_nasolabial_shadow_mask, build_visia_regions, draw_region_boundaries
from sg_core.preprocess.image_preprocessor import LEFT_EYE, LEFT_EYEBROW, LIPS, RIGHT_EYE, RIGHT_EYEBROW, ImagePreprocessor, PreprocessResultV2, build_feature_exclusion_masks, detect_visible_hair_masks
from sg_core.utils.io_utils import cv_imread, cv_imwrite
from sg_core.utils.gpu_backend import get_cuda_backend
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Config:
    """Spots V2 唯一核心调参区（默认针对 1024×1024 分析图）。

    A. 快速调参指南：
    1. 想提高肉眼明显大斑召回：先小幅降低 LARGE/SALIENT 的 Z、MEAN_Z、
       MIN_CONFIDENCE，再逐项降低 brownness、deltaE、暗度或红度绝对证据。
    2. 想减少小碎点和毛孔：提高小斑 Z、MEAN_Z、MIN_AREA_PX、MIN_CONFIDENCE，
       或扩大 PORE_* 的过滤范围。
    3. 想减少头发、眉毛、睫毛和皱纹误检：降低对应 LINE_RESPONSE 阈值，
       缩小允许的遮挡重叠率，或增大与毛发/五官的安全距离。
    4. 想保留脸缘和发际线附近斑点：减小 SAFE_MASK_ERODE；代价是边界和
       头发误检可能上升。
    5. 权重参数控制的是“相对贡献”，同组权重建议总和约为 1。把同组权重
       整体同比例放大，对 robust Z-score 的影响通常很小。
    6. 每次只调整一组参数，并固定同一批图片比较数量、位置和 Mask。

    约定：带 KERNEL 的椭圆形态学核必须使用正奇数；阈值调低通常提高召回，
    但也会增加误检。除特别说明外，所有颜色数值均作用于算法分析而非医学诊断。
    """
    LOCAL_SIGMAS = (3.0, 6.0, 12.0, 24.0)
    SCALE_Z_THRESHOLD = 2.45
    SINGLE_SCALE_STRONG_Z = 4.0
    MIN_SCALE_VOTES = 2
    MIN_MEAN_Z = 1.7
    CANDIDATE_CLOSE_KERNEL = 3
    LARGE_LOCAL_SIGMAS = (18.0, 32.0, 48.0, 64.0)
    LARGE_SCALE_Z_THRESHOLD = 1.25
    LARGE_MIN_SCALE_VOTES = 2
    LARGE_MIN_MEAN_Z = 0.75
    LARGE_CANDIDATE_CLOSE_KERNEL = 25
    LARGE_CANDIDATE_OPEN_KERNEL = 5
    LARGE_TEXTURE_BLUR_SIGMA = 4.0
    LARGE_UNIFORMITY_Z_DIVISOR = 3.5
    LARGE_SCORE_DARK_WEIGHT = 0.32
    LARGE_SCORE_RED_WEIGHT = 0.28
    LARGE_SCORE_YELLOW_WEIGHT = 0.22
    LARGE_SCORE_DELTA_E_WEIGHT = 0.18
    LARGE_BROWN_DARK_WEIGHT = 0.36
    LARGE_BROWN_RED_WEIGHT = 0.28
    LARGE_BROWN_YELLOW_WEIGHT = 0.22
    LARGE_BROWN_DELTA_E_WEIGHT = 0.14
    LARGE_CANDIDATE_MIN_BROWNNESS = 0.8
    LARGE_CANDIDATE_MIN_REGIONAL_DELTA_E = 1.85
    SALIENT_CORE_SIGMAS = (4.0, 8.0, 16.0, 28.0)
    SALIENT_OUTER_RATIO = 2.8
    SALIENT_SCALE_Z_THRESHOLD = 1.1
    SALIENT_STRONG_Z_THRESHOLD = 2.75
    SALIENT_MIN_SCALE_VOTES = 2
    SALIENT_MIN_MEAN_Z = 0.7
    SALIENT_CANDIDATE_CLOSE_KERNEL = 9
    SALIENT_CANDIDATE_OPEN_KERNEL = 3
    SALIENT_SCORE_DARK_WEIGHT = 0.3
    SALIENT_SCORE_RED_WEIGHT = 0.3
    SALIENT_SCORE_YELLOW_WEIGHT = 0.16
    SALIENT_SCORE_DELTA_E_WEIGHT = 0.24
    SALIENT_ABSOLUTE_DELTA_E_MIN = 1.45
    SALIENT_ABSOLUTE_DARK_MIN = 0.85
    SALIENT_ABSOLUTE_RED_MIN = 0.65
    SALIENT_ABSOLUTE_YELLOW_MIN = 0.75
    SALIENT_STRONG_DELTA_E_MIN = 2.0
    COMPACT_VISIBLE_CORE_SIGMAS = (2.5, 4.5, 7.0)
    COMPACT_VISIBLE_OUTER_RATIO = 2.6
    COMPACT_VISIBLE_SCALE_Z_THRESHOLD = 0.85
    COMPACT_VISIBLE_STRONG_Z_THRESHOLD = 2.2
    COMPACT_VISIBLE_MIN_SCALE_VOTES = 2
    COMPACT_VISIBLE_DELTA_E_MIN = 1.15
    COMPACT_VISIBLE_DARK_MIN = 0.55
    COMPACT_VISIBLE_RED_MIN = 0.45
    COMPACT_VISIBLE_YELLOW_MIN = 0.55
    COMPACT_VISIBLE_CLOSE_KERNEL = 3
    COMPACT_VISIBLE_OPEN_KERNEL = 3
    COMPACT_VISIBLE_MIN_AREA_PX = 8.0
    COMPACT_VISIBLE_MAX_AREA_RATIO = 0.0015
    COMPACT_VISIBLE_MIN_CONFIDENCE = 0.24
    PROMINENT_RED_MIN_RED = 0.45
    PROMINENT_RGB_RED_MIN = 0.35
    PROMINENT_RED_LAB_WEIGHT = 0.52
    PROMINENT_RED_RGB_WEIGHT = 0.28
    PROMINENT_RED_DELTA_E_WEIGHT = 0.2
    PROMINENT_RED_MIN_DARK = 1.1
    PROMINENT_RED_MIN_DELTA_E = 1.0
    PROMINENT_RED_MIN_Z = 1.0
    PROMINENT_RED_PEAK_SCORE_MIN = 1.2
    PROMINENT_DARK_PEAK_SCORE_MIN = 2.4
    PROMINENT_RED_PEAK_MIN_DISTANCE = 17
    PROMINENT_RED_MAX_PEAKS = 100
    PROMINENT_RED_MAX_RADIUS = 7
    PROMINENT_RED_GROW_RELATIVE = 0.62
    PROMINENT_RED_GROW_ABSOLUTE = 0.9
    PROMINENT_RED_MIN_AREA_PX = 10.0
    PROMINENT_RED_SAFE_ERODE_KERNEL = 31
    PROMINENT_RED_FEATURE_SAFE_DISTANCE = 25.0
    PROMINENT_RED_DEDUP_DISTANCE = 10.0
    PROMINENT_BROAD_MIN_DELTA_E = 1.6
    PROMINENT_BROAD_MIN_RED = 0.3
    PROMINENT_BROAD_MIN_DARK = 1.0
    PROMINENT_BROAD_PEAK_SCORE_MIN = 1.25
    PROMINENT_BROAD_PEAK_MIN_DISTANCE = 20
    PROMINENT_BROAD_MAX_PEAKS = 60
    INFLAMED_RED_REGION_Z_MIN = 0.45
    INFLAMED_RED_PEAK_SCORE_MIN = 0.75
    INFLAMED_RED_PEAK_MIN_DISTANCE = 17
    INFLAMED_RED_MAX_PEAKS = 60
    INFLAMED_RED_MIN_DELTA_E = 0.5
    L_WEIGHT = 0.42
    CHROMA_WEIGHT = 0.24
    DELTA_E_WEIGHT = 0.24
    RGB_CHROMA_WEIGHT = 0.1
    FEATURE_DENOISE_SIGMA = 0.65
    HIGHLIGHT_VALUE = 0.92
    HIGHLIGHT_MAX_SATURATION = 0.25
    SHADOW_MAX_VALUE = 0.22
    SHADOW_MAX_SATURATION = 0.16
    SAFE_MASK_ERODE = 15
    MIN_AREA_PX = 10.0
    MAX_AREA_RATIO = 0.0007
    LARGE_MIN_AREA_PX = 90.0
    LARGE_MAX_AREA_RATIO = 0.008
    LARGE_CLASS_MIN_FRACTION = 0.22
    LARGE_MIN_SOLIDITY = 0.18
    LARGE_MIN_EXTENT = 0.08
    LARGE_MAX_ASPECT_RATIO = 3.8
    LARGE_MIN_CONFIDENCE = 0.34
    SALIENT_MIN_AREA_PX = 20.0
    SALIENT_MAX_AREA_RATIO = 0.006
    SALIENT_CLASS_MIN_FRACTION = 0.18
    SALIENT_MIN_SOLIDITY = 0.2
    SALIENT_MIN_EXTENT = 0.08
    SALIENT_MAX_ASPECT_RATIO = 3.8
    SALIENT_MIN_CONFIDENCE = 0.3
    MIN_EXTENT = 0.1
    MIN_SOLIDITY = 0.28
    MAX_ASPECT_RATIO = 3.2
    MIN_CONFIDENCE = 0.4
    SKIN_BOUNDARY_KERNEL = 11
    PORE_MAX_AREA = 36.0
    PORE_MIN_CIRCULARITY = 0.48
    PORE_MAX_DELTA_E = 2.8
    PORE_MAX_COLOR_DIFFERENCE = 1.6
    WRINKLE_MIN_ASPECT_RATIO = 2.7
    WRINKLE_MIN_LINE_RESPONSE = 0.3
    WRINKLE_FORCE_ASPECT_RATIO = 3.0
    HAIR_MIN_LINE_RESPONSE = 0.36
    HAIR_BOUNDARY_MIN_ASPECT_RATIO = 1.8
    HAIR_DARK_MIN_ASPECT_RATIO = 3.2
    HAIR_DARK_MAX_MEAN_L = 38.0
    SHADOW_MIN_AREA_RATIO = 0.0008
    SHADOW_MAX_DELTA_E = 4.0
    SHADOW_MAX_COLOR_DIFFERENCE = 1.8
    SHADOW_MAX_LOCAL_CONTRAST = 4.8
    LARGE_CONFIRM_MIN_BROWNNESS = 0.9
    LARGE_CONFIRM_MIN_REGIONAL_DELTA_E = 1.25
    SALIENT_SUPPORT_FRACTION_MIN = 0.12
    SALIENT_VISIBLE_DELTA_E_MIN = 1.55
    SALIENT_VISIBLE_DARK_MIN = 0.9
    SALIENT_VISIBLE_RED_MIN = 0.7
    SALIENT_VISIBLE_YELLOW_MIN = 0.8
    SMALL_WEAK_DELTA_E_MAX = 1.8
    SMALL_WEAK_COLOR_DIFFERENCE_MAX = 0.9
    SMALL_STRONG_LOCAL_CONTRAST_MIN = 4.8
    SMALL_STRONG_SCALE_VOTES_MIN = 3.0
    WEAK_CHROMA_COLOR_DIFFERENCE_MAX = 1.2
    WEAK_CHROMA_DELTA_E_MAX = 4.0
    COMPACT_DARK_MAX_AREA_PX = 100.0
    COMPACT_DARK_MAX_ASPECT_RATIO = 1.8
    COMPACT_DARK_MIN_CIRCULARITY = 0.15
    COMPACT_DARK_MIN_LOCAL_CONTRAST = 4.5
    COMPACT_DARK_MIN_SCALE_VOTES = 3.0
    SALIENT_COMPACT_DARK_MIN = 2.4
    SALIENT_COMPACT_DELTA_E_MIN = 2.8
    SALIENT_COMPACT_MAX_AREA_RATIO = 0.0025
    SALIENT_CHROMATIC_RED_MIN = 0.8
    SALIENT_CHROMATIC_YELLOW_MIN = 1.0
    LARGE_MERGE_DILATE_KERNEL = 9
    SMALL_SUPPRESS_OVERLAP = 0.6
    SMALL_SUPPRESS_CENTROID_OVERLAP = 0.35
    CANDIDATE_LINE_PERCENTILE = 90.0
    OCCLUSION_LINE_PERCENTILE = 90.0
    OCCLUSION_LINE_MIN_RESPONSE = 0.55
    OCCLUSION_LINE_MIN_ASPECT_RATIO = 2.0
    FOREHEAD_STRONG_LINE_MIN = 0.85
    FOREHEAD_STRONG_HAIR_VOTES_MIN = 1.5
    FOREHEAD_MID_LINE_MIN = 0.65
    FOREHEAD_MID_ASPECT_RATIO_MIN = 1.5
    FOREHEAD_NEAR_HAIR_DISTANCE = 12.0
    FOREHEAD_NEAR_ASPECT_RATIO_MIN = 1.5
    FOREHEAD_NEAR_CONFIDENCE_MAX = 0.55
    FOREHEAD_NEAR_LINE_MIN = 0.5
    FOREHEAD_FAR_HAIR_DISTANCE = 20.0
    FOREHEAD_FAR_LINE_MIN = 0.4
    FOREHEAD_FAR_CONFIDENCE_MAX = 0.5
    FOREHEAD_FAR_AREA_MAX = 80.0
    FEATURE_NEAR_DISTANCE = 5.0
    FEATURE_NEAR_LINE_MIN = 0.35
    OCCLUSION_OVERLAP_MAX = 0.12
    FEATURE_EXCLUSION_MARGIN_PX = 30
    NOSTRIL_OVERLAP_MAX = 0.05
    NOSTRIL_ENCLOSURE_MAX = 0.08
    NASOLABIAL_CORRIDOR_RADIUS_PX = 25
    NASOLABIAL_SHADOW_MARGIN_PX = 5
    NASOLABIAL_OVERLAP_MAX = 0.25
    NASOLABIAL_MIN_ASPECT_RATIO = 1.4
    NASOLABIAL_LINE_MIN_RESPONSE = 0.18
    NASOLABIAL_CENTROID_MIN_ASPECT_RATIO = 2.0
    CONFIDENCE_Z_OFFSET_LARGE = 0.75
    CONFIDENCE_Z_OFFSET_SALIENT = 0.65
    CONFIDENCE_Z_OFFSET_SMALL = 1.5
    CONFIDENCE_Z_RANGE = 3.5
    CONFIDENCE_DELTA_E_SCALE_LARGE = 6.5
    CONFIDENCE_DELTA_E_SCALE_OTHER = 8.0
    CONFIDENCE_CONTRAST_SCALE = 9.0
    CONFIDENCE_BROWNNESS_SCALE = 5.5
    CONFIDENCE_SALIENT_SCALE = 5.0
    CONFIDENCE_SALIENT_DELTA_E_WEIGHT = 0.45
    CONFIDENCE_SALIENT_RED_WEIGHT = 0.3
    CONFIDENCE_SALIENT_DARK_WEIGHT = 0.25
    CONFIDENCE_SHAPE_SOLIDITY_WEIGHT = 0.55
    CONFIDENCE_SHAPE_EXTENT_WEIGHT = 0.45
    CONFIDENCE_SHAPE_EXTENT_SCALE = 2.0
    CONFIDENCE_LINE_PENALTY_CAP = 0.7
    CONFIDENCE_LARGE_WEIGHTS = {'z': 0.18, 'vote': 0.18, 'delta_e': 0.2, 'contrast': 0.12, 'brown': 0.17, 'uniformity': 0.1, 'shape': 0.05, 'line': 0.1}
    CONFIDENCE_SALIENT_WEIGHTS = {'z': 0.18, 'vote': 0.18, 'delta_e': 0.24, 'contrast': 0.12, 'salient': 0.2, 'shape': 0.08, 'line': 0.1}
    CONFIDENCE_SMALL_WEIGHTS = {'z': 0.28, 'vote': 0.23, 'delta_e': 0.22, 'contrast': 0.15, 'shape': 0.12, 'line': 0.12}
    CONFIDENCE_SALIENT_BONUS_WEIGHT = 0.1
    QUALITY_CONFIDENCE_BASE = 0.75
    QUALITY_CONFIDENCE_WEIGHT = 0.25
    ROBUST_MIN_VALID_PIXELS = 100
    REGION_ROBUST_MIN_PIXELS = 200
    ROBUST_Z_CLIP_MIN = -5.0
    ROBUST_Z_CLIP_MAX = 12.0
    HEATMAP_HIGH_PERCENTILE = 99.5
    FRANGI_SIGMAS = (1, 2, 3)
    LINE_RESPONSE_HIGH_PERCENTILE = 99.5
    REGION_NOSE_DILATE_KERNEL = 17
    REGION_NOSE_LANDMARKS = (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 97, 98, 327, 326)
    REGION_LIP_LANDMARKS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
    COLOR_ZONE = VISIA_BOUNDARY_COLOR
    COLOR_SPOT = (237, 229, 188)
    ZONE_THICKNESS = 2
    SPOT_THICKNESS = 1
    SPOT_FILL_ALPHA = 0.68
    TINY_SPOT_FILL_AREA = 18
    SPLIT_MIN_AREA = 110
    SPLIT_PEAK_MIN_DISTANCE = 6
    SPLIT_PEAK_THRESHOLD_REL = 0.35
    SPLIT_MIN_CHILD_AREA = 8
    SPLIT_MAX_PEAKS = 20
    SPLIT_GROW_THRESHOLD_REL = 0.52
    SPLIT_GROW_THRESHOLD_ABS = 0.28
    SPLIT_MAX_RADIUS_SMALL = 10
    SPLIT_MAX_RADIUS_SALIENT = 13
    SPLIT_MAX_RADIUS_LARGE = 15
    SINGLE_PEAK_COMPACT_MIN_AREA = 420
    SINGLE_PEAK_GROW_THRESHOLD_REL = 0.58
    SINGLE_PEAK_GROW_THRESHOLD_ABS = 0.34
    SINGLE_PEAK_MAX_PARENT_FRACTION = 0.82

@dataclass
class SpotCandidateV2:
    spot_id: int
    region: str
    area: float
    area_ratio_skin: float
    bbox: list[int]
    centroid: list[float]
    circularity: float
    solidity: float
    extent: float
    aspect_ratio: float
    mean_deltaE: float
    color_difference: float
    scale_vote: float
    max_scale_vote: int
    local_contrast: float
    line_response: float
    score_z: float
    confidence: float
    spot_type: str = 'small'
    patch_contrast: float = 0.0
    regional_deltaE: float = 0.0
    brownness_score: float = 0.0
    uniformity_score: float = 0.0
    salient_deltaE: float = 0.0
    salient_dark_difference: float = 0.0
    salient_red_difference: float = 0.0
    salient_yellow_difference: float = 0.0
    salient_support_fraction: float = 0.0
    salient_supported: bool = False
    prominent_red_rescue: bool = False
    occlusion_overlap_ratio: float = 0.0
    hair_line_response_p90: float = 0.0
    hair_scale_vote: float = 0.0
    distance_to_feature_mask: float = 999.0
    distance_to_hair_mask: float = 999.0
    nostril_overlap_ratio: float = 0.0
    nostril_enclosure_ratio: float = 0.0

@dataclass
class SpotsResultV2:
    spot_count: int
    spot_area_ratio: float
    spot_locations: list[dict[str, Any]]
    spot_confidence: float
    region_distribution: dict[str, dict[str, float | int]]
    scale_vote_map: np.ndarray
    candidate_heatmap: np.ndarray
    candidate_mask: np.ndarray
    filtered_mask: np.ndarray
    spots_overlay: np.ndarray
    mean_deltaE: float
    definition: str
    hair_occlusion_mask: np.ndarray
    hair_rejected_mask: np.ndarray
    pre_occlusion_spot_count: int
    hair_filtered_count: int
    occlusion_filter_statistics: dict[str, int]
    small_spot_count: int
    large_spot_count: int
    merged_spot_count: int
    large_spot_area_ratio: float
    large_spot_locations: list[dict[str, Any]]
    large_spot_recall_notes: list[str]
    salient_spot_count: int
    nostril_filtered_count: int
    analysis_zone_area: int
    mean_spot_area: float
    median_spot_area: float
    pre_split_large_component_count: int
    post_split_instance_count: int

    def metrics(self) -> dict[str, Any]:
        """Return the JSON-safe portion of the result."""
        return {'definition': self.definition, 'spot_count': self.spot_count, 'spot_area_ratio': self.spot_area_ratio, 'spot_confidence': self.spot_confidence, 'mean_deltaE': self.mean_deltaE, 'pre_occlusion_spot_count': self.pre_occlusion_spot_count, 'hair_filtered_count': self.hair_filtered_count, 'small_spot_count': self.small_spot_count, 'large_spot_count': self.large_spot_count, 'merged_spot_count': self.merged_spot_count, 'large_spot_area_ratio': self.large_spot_area_ratio, 'large_spot_locations': self.large_spot_locations, 'large_spot_recall_notes': self.large_spot_recall_notes, 'salient_spot_count': self.salient_spot_count, 'nostril_filtered_count': self.nostril_filtered_count, 'analysis_zone_area': self.analysis_zone_area, 'mean_spot_area': self.mean_spot_area, 'median_spot_area': self.median_spot_area, 'pre_split_large_component_count': self.pre_split_large_component_count, 'post_split_instance_count': self.post_split_instance_count, 'occlusion_filter_statistics': self.occlusion_filter_statistics, 'spot_locations': self.spot_locations, 'region_distribution': self.region_distribution, 'parameters': {'local_sigmas': list(Config.LOCAL_SIGMAS), 'large_local_sigmas': list(Config.LARGE_LOCAL_SIGMAS), 'salient_core_sigmas': list(Config.SALIENT_CORE_SIGMAS), 'scale_z_threshold': Config.SCALE_Z_THRESHOLD, 'large_scale_z_threshold': Config.LARGE_SCALE_Z_THRESHOLD, 'minimum_scale_votes': Config.MIN_SCALE_VOTES, 'large_minimum_scale_votes': Config.LARGE_MIN_SCALE_VOTES, 'salient_minimum_scale_votes': Config.SALIENT_MIN_SCALE_VOTES, 'minimum_area_px': Config.MIN_AREA_PX, 'maximum_area_ratio': Config.MAX_AREA_RATIO, 'large_minimum_area_px': Config.LARGE_MIN_AREA_PX, 'large_maximum_area_ratio': Config.LARGE_MAX_AREA_RATIO, 'minimum_confidence': Config.MIN_CONFIDENCE, 'large_minimum_confidence': Config.LARGE_MIN_CONFIDENCE, 'salient_minimum_confidence': Config.SALIENT_MIN_CONFIDENCE}}

class MaskedGaussianBackgroundCache:
    """Reuse mask-only Gaussian terms across channels at the same scale.

    Mask-aware Gaussian filtering divides a blurred ``channel * mask`` by a
    blurred mask.  The denominator depends only on the mask and sigma, yet the
    Spots branches previously recomputed it for every LAB/RGB channel.  This
    cache keeps the exact OpenCV operation while eliminating that duplicate
    full-resolution work.
    """

    def __init__(self, mask: np.ndarray):
        self.mask_f = (mask > 0).astype(np.float32)
        self.mask_exterior = self.mask_f == 0
        self._denominators: dict[float, np.ndarray] = {}

    def denominator(self, sigma: float) -> np.ndarray:
        key = float(sigma)
        denominator = self._denominators.get(key)
        if denominator is None:
            denominator = cv2.GaussianBlur(self.mask_f, (0, 0), sigmaX=key, sigmaY=key, borderType=cv2.BORDER_REFLECT)
            self._denominators[key] = denominator
        return denominator

def masked_gaussian_background(channel: np.ndarray, mask: np.ndarray, sigma: float, cache: MaskedGaussianBackgroundCache | None=None) -> np.ndarray:
    """Estimate local background without allowing mask-exterior pixels in."""
    return get_cuda_backend().masked_gaussian(channel, mask, sigma)

def masked_gaussian_background_scale_space(channels: tuple[np.ndarray, ...], mask: np.ndarray, sigmas: tuple[float, ...]) -> dict[float, tuple[np.ndarray, ...]]:
    """Build broad masked backgrounds through a high-resolution scale space.

    Gaussian variances add.  Reusing the previous blurred numerator and
    denominator avoids repeatedly convolving the original 1024px image with
    every large kernel.  Pixel coordinates and all downstream thresholds stay
    at full resolution.
    """
    return get_cuda_backend().masked_gaussian_batch(channels, mask, sigmas)

def robust_z_score(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Median/MAD Z-score computed only over valid skin."""
    valid = values[mask > 0]
    if valid.size < Config.ROBUST_MIN_VALID_PIXELS:
        return (np.zeros_like(values, dtype=np.float32), 0.0, 1.0)
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    robust_sigma = max(1.4826 * mad, 0.0001)
    z_score = (values.astype(np.float32) - median) / robust_sigma
    z_score = np.clip(z_score, Config.ROBUST_Z_CLIP_MIN, Config.ROBUST_Z_CLIP_MAX)
    z_score[mask == 0] = 0.0
    return (z_score.astype(np.float32), median, robust_sigma)

def regional_robust_z_score(values: np.ndarray, mask: np.ndarray, region_masks: dict[str, np.ndarray]) -> np.ndarray:
    """Compute robust Z within each visible facial region.

    Side faces can have a strong left/right illumination difference. Region-
    local statistics prevent that global gradient from hiding a real compact
    colour anomaly on the darker side.
    """
    (global_z, _, _) = robust_z_score(values, mask)
    result = global_z.copy()
    covered = np.zeros_like(mask, dtype=bool)
    for name in ('forehead', 'left_cheek', 'right_cheek', 'nose', 'chin'):
        region = region_masks.get(name)
        if region is None:
            continue
        local_mask = cv2.bitwise_and(region, mask)
        if np.count_nonzero(local_mask) < Config.REGION_ROBUST_MIN_PIXELS:
            continue
        (local_z, _, _) = robust_z_score(values, local_mask)
        local_pixels = local_mask > 0
        result[local_pixels] = local_z[local_pixels]
        covered |= local_pixels
    result[mask == 0] = 0.0
    return result.astype(np.float32)

def _normalized_rgb(rgb: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.sum(rgb, axis=2, keepdims=True), 1e-05)
    return (rgb / denominator).astype(np.float32)

def _safe_heatmap(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = score[mask > 0]
    if valid.size == 0:
        return np.zeros((*score.shape, 3), dtype=np.uint8)
    high = max(float(np.percentile(valid, Config.HEATMAP_HIGH_PERCENTILE)), 1e-05)
    normalized = np.clip(score / high * 255.0, 0.0, 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heatmap[mask == 0] = 0
    return heatmap

def _line_response(l_channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Frangi response used only to reject line-like candidates."""
    valid = l_channel[mask > 0]
    if valid.size == 0:
        return np.zeros_like(l_channel, dtype=np.float32)
    filled = (l_channel / 100.0).astype(np.float32)
    filled[mask == 0] = float(np.median(valid) / 100.0)
    (height, width) = filled.shape
    reduced = cv2.resize(filled, (max(1, width // 2), max(1, height // 2)), interpolation=cv2.INTER_AREA)
    reduced_response = frangi(reduced, sigmas=tuple((max(0.5, sigma * 0.5) for sigma in Config.FRANGI_SIGMAS)), black_ridges=True).astype(np.float32)
    response = cv2.resize(reduced_response, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    valid_response = response[mask > 0]
    high = max(float(np.percentile(valid_response, Config.LINE_RESPONSE_HIGH_PERCENTILE)), 1e-08)
    response = np.clip(response / high, 0.0, 1.0)
    response[mask == 0] = 0.0
    return response.astype(np.float32)

def compute_multiscale_maps(analysis_image: np.ndarray, skin_mask: np.ndarray, region_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute LAB/RGB anomaly evidence and multi-scale robust votes."""
    bgr = analysis_image.astype(np.float32) / 255.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = rgb2lab(np.clip(rgb, 0.0, 1.0)).astype(np.float32)
    (l_channel, a_channel, b_channel) = cv2.split(lab)
    rgb_chroma = _normalized_rgb(rgb)
    rgb_red_chroma = (rgb_chroma[:, :, 0] - rgb_chroma[:, :, 1]) * 100.0
    hsv = cv2.cvtColor(np.clip(bgr, 0.0, 1.0), cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    highlight = (value > Config.HIGHLIGHT_VALUE) & (saturation < Config.HIGHLIGHT_MAX_SATURATION)
    deep_neutral_shadow = (value < Config.SHADOW_MAX_VALUE) & (saturation < Config.SHADOW_MAX_SATURATION)
    analysis_mask = ((skin_mask > 0) & ~highlight & ~deep_neutral_shadow).astype(np.uint8) * 255
    analysis_mask = cv2.erode(analysis_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.SAFE_MASK_ERODE, Config.SAFE_MASK_ERODE)))
    (l_feature, a_feature, b_feature) = get_cuda_backend().masked_gaussian_batch((l_channel, a_channel, b_channel), analysis_mask, (Config.FEATURE_DENOISE_SIGMA,))[float(Config.FEATURE_DENOISE_SIGMA)]
    small_backgrounds = get_cuda_backend().masked_gaussian_batch((l_feature, a_feature, b_feature, rgb_chroma[:, :, 0], rgb_chroma[:, :, 1], rgb_chroma[:, :, 2]), analysis_mask, Config.LOCAL_SIGMAS)
    shape = skin_mask.shape
    vote_map = np.zeros(shape, dtype=np.uint8)
    max_score_z = np.zeros(shape, dtype=np.float32)
    mean_score_z = np.zeros(shape, dtype=np.float32)
    max_delta_e = np.zeros(shape, dtype=np.float32)
    max_color_difference = np.zeros(shape, dtype=np.float32)
    max_local_contrast = np.zeros(shape, dtype=np.float32)
    single_scale_strong = np.zeros(shape, dtype=np.uint8)
    for sigma in Config.LOCAL_SIGMAS:
        (l_background, a_background, b_background, rgb_background_0, rgb_background_1, rgb_background_2) = small_backgrounds[float(sigma)]
        delta_l = l_feature - l_background
        delta_a = a_feature - a_background
        delta_b = b_feature - b_background
        local_l_difference = np.maximum(-delta_l, 0.0)
        color_difference = np.sqrt(delta_a ** 2 + delta_b ** 2)
        delta_e = np.sqrt(delta_l ** 2 + delta_a ** 2 + delta_b ** 2)
        score_delta_e = np.where(delta_l <= 0.0, delta_e, color_difference)
        rgb_background = np.stack((rgb_background_0, rgb_background_1, rgb_background_2), axis=2)
        rgb_difference = np.linalg.norm(rgb_chroma - rgb_background, axis=2) * 100.0
        raw_score = Config.L_WEIGHT * local_l_difference + Config.CHROMA_WEIGHT * color_difference + Config.DELTA_E_WEIGHT * score_delta_e + Config.RGB_CHROMA_WEIGHT * rgb_difference
        (z_score, _, _) = robust_z_score(raw_score, analysis_mask)
        scale_support = z_score >= Config.SCALE_Z_THRESHOLD
        vote_map += scale_support.astype(np.uint8)
        single_scale_strong |= (z_score >= Config.SINGLE_SCALE_STRONG_Z).astype(np.uint8)
        max_score_z = np.maximum(max_score_z, z_score)
        mean_score_z += np.maximum(z_score, 0.0)
        max_delta_e = np.maximum(max_delta_e, delta_e)
        max_color_difference = np.maximum(max_color_difference, color_difference)
        max_local_contrast = np.maximum(max_local_contrast, local_l_difference)
    mean_score_z /= float(len(Config.LOCAL_SIGMAS))
    candidate_mask = ((vote_map >= Config.MIN_SCALE_VOTES) & (mean_score_z >= Config.MIN_MEAN_Z) & (analysis_mask > 0)).astype(np.uint8) * 255
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.CANDIDATE_CLOSE_KERNEL, Config.CANDIDATE_CLOSE_KERNEL)))
    candidate_mask = cv2.bitwise_and(candidate_mask, analysis_mask)
    candidate_mask[vote_map < Config.MIN_SCALE_VOTES] = 0
    large_vote_map = np.zeros(shape, dtype=np.uint8)
    large_max_score_z = np.zeros(shape, dtype=np.float32)
    large_mean_score_z = np.zeros(shape, dtype=np.float32)
    large_patch_contrast = np.zeros(shape, dtype=np.float32)
    large_regional_delta_e = np.zeros(shape, dtype=np.float32)
    brownness_score = np.zeros(shape, dtype=np.float32)
    uniformity_score = np.zeros(shape, dtype=np.float32)
    salient_outer_sigmas = tuple((core_sigma * Config.SALIENT_OUTER_RATIO for core_sigma in Config.SALIENT_CORE_SIGMAS))
    compact_visible_outer_sigmas = tuple((core_sigma * Config.COMPACT_VISIBLE_OUTER_RATIO for core_sigma in Config.COMPACT_VISIBLE_CORE_SIGMAS))
    all_broad_sigmas = tuple(sorted(set(Config.LARGE_LOCAL_SIGMAS + salient_outer_sigmas + compact_visible_outer_sigmas + Config.SALIENT_CORE_SIGMAS + Config.COMPACT_VISIBLE_CORE_SIGMAS)))
    broad_backgrounds = masked_gaussian_background_scale_space((l_feature, a_feature, b_feature, rgb_red_chroma), analysis_mask, all_broad_sigmas)
    for sigma in Config.LARGE_LOCAL_SIGMAS:
        (l_background, a_background, b_background, _) = broad_backgrounds[float(sigma)]
        delta_l = l_feature - l_background
        delta_a = a_feature - a_background
        delta_b = b_feature - b_background
        dark_patch = np.maximum(-delta_l, 0.0)
        red_brown_shift = np.maximum(delta_a, 0.0)
        yellow_brown_shift = np.maximum(delta_b, 0.0)
        regional_delta_e = np.sqrt(delta_l ** 2 + delta_a ** 2 + delta_b ** 2)
        brownness = Config.LARGE_BROWN_DARK_WEIGHT * dark_patch + Config.LARGE_BROWN_RED_WEIGHT * red_brown_shift + Config.LARGE_BROWN_YELLOW_WEIGHT * yellow_brown_shift + Config.LARGE_BROWN_DELTA_E_WEIGHT * regional_delta_e
        raw_score = Config.LARGE_SCORE_DARK_WEIGHT * dark_patch + Config.LARGE_SCORE_RED_WEIGHT * red_brown_shift + Config.LARGE_SCORE_YELLOW_WEIGHT * yellow_brown_shift + Config.LARGE_SCORE_DELTA_E_WEIGHT * regional_delta_e
        (z_score, _, _) = robust_z_score(raw_score, analysis_mask)
        scale_support = z_score >= Config.LARGE_SCALE_Z_THRESHOLD
        large_vote_map += scale_support.astype(np.uint8)
        large_max_score_z = np.maximum(large_max_score_z, z_score)
        large_mean_score_z += np.maximum(z_score, 0.0)
        large_patch_contrast = np.maximum(large_patch_contrast, dark_patch)
        large_regional_delta_e = np.maximum(large_regional_delta_e, regional_delta_e)
        brownness_score = np.maximum(brownness_score, brownness)
    large_mean_score_z /= float(len(Config.LARGE_LOCAL_SIGMAS))
    texture_reference = cv2.GaussianBlur(large_max_score_z, (0, 0), sigmaX=Config.LARGE_TEXTURE_BLUR_SIGMA, sigmaY=Config.LARGE_TEXTURE_BLUR_SIGMA, borderType=cv2.BORDER_REFLECT)
    local_texture = np.abs(large_max_score_z - texture_reference)
    (texture_z, _, _) = robust_z_score(local_texture, analysis_mask)
    uniformity_score = np.clip(1.0 - texture_z / Config.LARGE_UNIFORMITY_Z_DIVISOR, 0.0, 1.0).astype(np.float32)
    large_candidate_mask = ((large_vote_map >= Config.LARGE_MIN_SCALE_VOTES) & (large_mean_score_z >= Config.LARGE_MIN_MEAN_Z) & ((brownness_score >= Config.LARGE_CANDIDATE_MIN_BROWNNESS) | (large_regional_delta_e >= Config.LARGE_CANDIDATE_MIN_REGIONAL_DELTA_E)) & (analysis_mask > 0)).astype(np.uint8) * 255
    large_candidate_mask = cv2.morphologyEx(large_candidate_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.LARGE_CANDIDATE_CLOSE_KERNEL, Config.LARGE_CANDIDATE_CLOSE_KERNEL)))
    large_candidate_mask = cv2.morphologyEx(large_candidate_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.LARGE_CANDIDATE_OPEN_KERNEL, Config.LARGE_CANDIDATE_OPEN_KERNEL)))
    large_candidate_mask = cv2.bitwise_and(large_candidate_mask, analysis_mask)
    large_candidate_mask[large_vote_map < Config.LARGE_MIN_SCALE_VOTES] = 0
    salient_vote_map = np.zeros(shape, dtype=np.uint8)
    salient_max_score_z = np.zeros(shape, dtype=np.float32)
    salient_mean_score_z = np.zeros(shape, dtype=np.float32)
    salient_delta_e = np.zeros(shape, dtype=np.float32)
    salient_dark_difference = np.zeros(shape, dtype=np.float32)
    salient_red_difference = np.zeros(shape, dtype=np.float32)
    salient_yellow_difference = np.zeros(shape, dtype=np.float32)
    salient_strong = np.zeros(shape, dtype=np.uint8)
    for core_sigma in Config.SALIENT_CORE_SIGMAS:
        outer_sigma = core_sigma * Config.SALIENT_OUTER_RATIO
        (core_l, core_a, core_b, _) = broad_backgrounds[float(core_sigma)]
        (outer_l, outer_a, outer_b, _) = broad_backgrounds[float(outer_sigma)]
        delta_l = core_l - outer_l
        delta_a = core_a - outer_a
        delta_b = core_b - outer_b
        dark_difference = np.maximum(-delta_l, 0.0)
        red_difference = np.maximum(delta_a, 0.0)
        yellow_difference = np.maximum(delta_b, 0.0)
        delta_e = np.sqrt(delta_l ** 2 + delta_a ** 2 + delta_b ** 2)
        raw_score = Config.SALIENT_SCORE_DARK_WEIGHT * dark_difference + Config.SALIENT_SCORE_RED_WEIGHT * red_difference + Config.SALIENT_SCORE_YELLOW_WEIGHT * yellow_difference + Config.SALIENT_SCORE_DELTA_E_WEIGHT * delta_e
        z_score = regional_robust_z_score(raw_score, analysis_mask, region_masks)
        absolute_evidence = (delta_e >= Config.SALIENT_ABSOLUTE_DELTA_E_MIN) & ((dark_difference >= Config.SALIENT_ABSOLUTE_DARK_MIN) | (red_difference >= Config.SALIENT_ABSOLUTE_RED_MIN) | (yellow_difference >= Config.SALIENT_ABSOLUTE_YELLOW_MIN))
        support = (z_score >= Config.SALIENT_SCALE_Z_THRESHOLD) & absolute_evidence
        strong = (z_score >= Config.SALIENT_STRONG_Z_THRESHOLD) & (delta_e >= Config.SALIENT_STRONG_DELTA_E_MIN) & absolute_evidence
        salient_vote_map += support.astype(np.uint8)
        salient_strong |= strong.astype(np.uint8)
        salient_max_score_z = np.maximum(salient_max_score_z, z_score)
        salient_mean_score_z += np.maximum(z_score, 0.0)
        salient_delta_e = np.maximum(salient_delta_e, delta_e)
        salient_dark_difference = np.maximum(salient_dark_difference, dark_difference)
        salient_red_difference = np.maximum(salient_red_difference, red_difference)
        salient_yellow_difference = np.maximum(salient_yellow_difference, yellow_difference)
    salient_mean_score_z /= float(len(Config.SALIENT_CORE_SIGMAS))
    salient_candidate_mask = (((salient_vote_map >= Config.SALIENT_MIN_SCALE_VOTES) | (salient_strong > 0)) & (salient_mean_score_z >= Config.SALIENT_MIN_MEAN_Z) & (salient_delta_e >= Config.SALIENT_ABSOLUTE_DELTA_E_MIN) & (analysis_mask > 0)).astype(np.uint8) * 255
    salient_candidate_mask = cv2.morphologyEx(salient_candidate_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.SALIENT_CANDIDATE_CLOSE_KERNEL, Config.SALIENT_CANDIDATE_CLOSE_KERNEL)))
    salient_candidate_mask = cv2.morphologyEx(salient_candidate_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.SALIENT_CANDIDATE_OPEN_KERNEL, Config.SALIENT_CANDIDATE_OPEN_KERNEL)))
    salient_candidate_mask = cv2.bitwise_and(salient_candidate_mask, analysis_mask)
    compact_visible_vote_map = np.zeros(shape, dtype=np.uint8)
    compact_visible_max_z = np.zeros(shape, dtype=np.float32)
    compact_visible_mean_z = np.zeros(shape, dtype=np.float32)
    compact_visible_delta_e = np.zeros(shape, dtype=np.float32)
    compact_visible_dark = np.zeros(shape, dtype=np.float32)
    compact_visible_red = np.zeros(shape, dtype=np.float32)
    compact_visible_rgb_red = np.zeros(shape, dtype=np.float32)
    compact_visible_yellow = np.zeros(shape, dtype=np.float32)
    compact_visible_strong = np.zeros(shape, dtype=np.uint8)
    for core_sigma in Config.COMPACT_VISIBLE_CORE_SIGMAS:
        outer_sigma = core_sigma * Config.COMPACT_VISIBLE_OUTER_RATIO
        (core_l, core_a, core_b, core_rgb_red) = broad_backgrounds[float(core_sigma)]
        (outer_l, outer_a, outer_b, outer_rgb_red) = broad_backgrounds[float(outer_sigma)]
        delta_l = core_l - outer_l
        delta_a = core_a - outer_a
        delta_b = core_b - outer_b
        dark = np.maximum(-delta_l, 0.0)
        red = np.maximum(delta_a, 0.0)
        rgb_red = np.maximum(core_rgb_red - outer_rgb_red, 0.0)
        yellow = np.maximum(delta_b, 0.0)
        delta_e = np.sqrt(delta_l ** 2 + delta_a ** 2 + delta_b ** 2)
        raw_score = 0.26 * dark + 0.24 * red + 0.14 * rgb_red + 0.14 * yellow + 0.22 * delta_e
        z_score = regional_robust_z_score(raw_score, analysis_mask, region_masks)
        absolute_evidence = (delta_e >= Config.COMPACT_VISIBLE_DELTA_E_MIN) & ((dark >= Config.COMPACT_VISIBLE_DARK_MIN) | (red >= Config.COMPACT_VISIBLE_RED_MIN) | (yellow >= Config.COMPACT_VISIBLE_YELLOW_MIN))
        support = (z_score >= Config.COMPACT_VISIBLE_SCALE_Z_THRESHOLD) & absolute_evidence
        strong = (z_score >= Config.COMPACT_VISIBLE_STRONG_Z_THRESHOLD) & absolute_evidence
        compact_visible_vote_map += support.astype(np.uint8)
        compact_visible_strong |= strong.astype(np.uint8)
        compact_visible_max_z = np.maximum(compact_visible_max_z, z_score)
        compact_visible_mean_z += np.maximum(z_score, 0.0)
        compact_visible_delta_e = np.maximum(compact_visible_delta_e, delta_e)
        compact_visible_dark = np.maximum(compact_visible_dark, dark)
        compact_visible_red = np.maximum(compact_visible_red, red)
        compact_visible_rgb_red = np.maximum(compact_visible_rgb_red, rgb_red)
        compact_visible_yellow = np.maximum(compact_visible_yellow, yellow)
    compact_visible_mean_z /= float(len(Config.COMPACT_VISIBLE_CORE_SIGMAS))
    compact_visible_candidate_mask = (((compact_visible_vote_map >= Config.COMPACT_VISIBLE_MIN_SCALE_VOTES) | (compact_visible_strong > 0)) & (compact_visible_delta_e >= Config.COMPACT_VISIBLE_DELTA_E_MIN) & (analysis_mask > 0)).astype(np.uint8) * 255
    compact_visible_candidate_mask = cv2.morphologyEx(compact_visible_candidate_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.COMPACT_VISIBLE_CLOSE_KERNEL, Config.COMPACT_VISIBLE_CLOSE_KERNEL)))
    compact_visible_candidate_mask = cv2.morphologyEx(compact_visible_candidate_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.COMPACT_VISIBLE_OPEN_KERNEL, Config.COMPACT_VISIBLE_OPEN_KERNEL)))
    compact_visible_candidate_mask = cv2.bitwise_and(compact_visible_candidate_mask, analysis_mask)
    prominent_red_score = cv2.GaussianBlur((Config.PROMINENT_RED_LAB_WEIGHT * compact_visible_red + Config.PROMINENT_RED_RGB_WEIGHT * compact_visible_rgb_red + Config.PROMINENT_RED_DELTA_E_WEIGHT * compact_visible_delta_e).astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT)
    prominent_dark_score = cv2.GaussianBlur((0.7 * compact_visible_dark + 0.3 * compact_visible_delta_e).astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT)
    prominent_broad_score = cv2.GaussianBlur((0.45 * salient_red_difference + 0.3 * salient_dark_difference + 0.25 * salient_delta_e).astype(np.float32), (0, 0), sigmaX=1.5, sigmaY=1.5, borderType=cv2.BORDER_REFLECT)
    prominent_feature_score = np.maximum.reduce((prominent_red_score, prominent_dark_score, prominent_broad_score))
    prominent_safe_mask = cv2.erode(analysis_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.PROMINENT_RED_SAFE_ERODE_KERNEL, Config.PROMINENT_RED_SAFE_ERODE_KERNEL)))
    prominent_common = (prominent_safe_mask > 0) & (compact_visible_delta_e >= Config.PROMINENT_RED_MIN_DELTA_E) & (compact_visible_mean_z >= Config.PROMINENT_RED_MIN_Z)
    prominent_red_valid = prominent_common & ((compact_visible_red >= Config.PROMINENT_RED_MIN_RED) | (compact_visible_rgb_red >= Config.PROMINENT_RGB_RED_MIN))
    prominent_dark_valid = prominent_common & (compact_visible_dark >= Config.PROMINENT_RED_MIN_DARK)
    prominent_broad_valid = (prominent_safe_mask > 0) & (salient_delta_e >= Config.PROMINENT_BROAD_MIN_DELTA_E) & ((salient_red_difference >= Config.PROMINENT_BROAD_MIN_RED) | (salient_dark_difference >= Config.PROMINENT_BROAD_MIN_DARK))
    regional_red_z = regional_robust_z_score(a_feature, analysis_mask, region_masks)
    prominent_inflamed_score = cv2.GaussianBlur((0.55 * np.maximum(regional_red_z, 0.0) + 0.2 * salient_red_difference + 0.15 * salient_dark_difference + 0.1 * salient_delta_e).astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2, borderType=cv2.BORDER_REFLECT)
    prominent_inflamed_valid = (prominent_safe_mask > 0) & (regional_red_z >= Config.INFLAMED_RED_REGION_Z_MIN) & (salient_delta_e >= Config.INFLAMED_RED_MIN_DELTA_E)
    red_peaks = peak_local_max(prominent_red_score, labels=prominent_red_valid.astype(np.uint8), min_distance=Config.PROMINENT_RED_PEAK_MIN_DISTANCE, threshold_abs=Config.PROMINENT_RED_PEAK_SCORE_MIN, num_peaks=Config.PROMINENT_RED_MAX_PEAKS, exclude_border=False)
    dark_peaks = peak_local_max(prominent_dark_score, labels=prominent_dark_valid.astype(np.uint8), min_distance=Config.PROMINENT_RED_PEAK_MIN_DISTANCE, threshold_abs=Config.PROMINENT_DARK_PEAK_SCORE_MIN, num_peaks=Config.PROMINENT_RED_MAX_PEAKS, exclude_border=False)
    broad_peaks = peak_local_max(prominent_broad_score, labels=prominent_broad_valid.astype(np.uint8), min_distance=Config.PROMINENT_BROAD_PEAK_MIN_DISTANCE, threshold_abs=Config.PROMINENT_BROAD_PEAK_SCORE_MIN, num_peaks=Config.PROMINENT_BROAD_MAX_PEAKS, exclude_border=False)
    inflamed_peaks = peak_local_max(prominent_inflamed_score, labels=prominent_inflamed_valid.astype(np.uint8), min_distance=Config.INFLAMED_RED_PEAK_MIN_DISTANCE, threshold_abs=Config.INFLAMED_RED_PEAK_SCORE_MIN, num_peaks=Config.INFLAMED_RED_MAX_PEAKS, exclude_border=False)
    prominent_feature_score = np.maximum(prominent_feature_score, prominent_inflamed_score)
    peak_candidates = sorted([(int(row), int(column)) for (row, column) in red_peaks] + [(int(row), int(column)) for (row, column) in dark_peaks] + [(int(row), int(column)) for (row, column) in broad_peaks] + [(int(row), int(column)) for (row, column) in inflamed_peaks], key=lambda point: float(prominent_feature_score[point]), reverse=True)
    peak_positions: list[tuple[int, int]] = []
    for point in peak_candidates:
        if any((math.hypot(point[0] - kept[0], point[1] - kept[1]) < Config.PROMINENT_RED_PEAK_MIN_DISTANCE for kept in peak_positions)):
            continue
        peak_positions.append(point)
        if len(peak_positions) >= Config.PROMINENT_RED_MAX_PEAKS:
            break
    prominent_peaks = np.asarray(peak_positions, dtype=np.int32).reshape(-1, 2)
    prominent_red_peak_mask = np.zeros(shape, dtype=np.uint8)
    for (peak_row, peak_column) in prominent_peaks:
        peak_score = float(prominent_feature_score[peak_row, peak_column])
        grow_threshold = max(Config.PROMINENT_RED_GROW_ABSOLUTE, Config.PROMINENT_RED_GROW_RELATIVE * peak_score)
        disk = np.zeros(shape, dtype=np.uint8)
        cv2.circle(disk, (int(peak_column), int(peak_row)), Config.PROMINENT_RED_MAX_RADIUS, 255, -1)
        local = ((prominent_feature_score >= grow_threshold) & (prominent_common | prominent_broad_valid | prominent_inflamed_valid) & (disk > 0)).astype(np.uint8)
        (labels_count, labels, _, _) = cv2.connectedComponentsWithStats(local, connectivity=8)
        peak_label = int(labels[peak_row, peak_column])
        if peak_label > 0 and labels_count > 1:
            prominent_red_peak_mask[labels == peak_label] = 255
        else:
            cv2.circle(prominent_red_peak_mask, (int(peak_column), int(peak_row)), 3, 255, -1)
    prominent_red_peak_mask = cv2.bitwise_and(prominent_red_peak_mask, prominent_safe_mask)
    salient_candidate_mask = cv2.bitwise_or(salient_candidate_mask, compact_visible_candidate_mask)
    merged_candidate_mask = cv2.bitwise_or(candidate_mask, large_candidate_mask)
    merged_candidate_mask = cv2.bitwise_or(merged_candidate_mask, salient_candidate_mask)
    raw_candidate_mask = merged_candidate_mask.copy()
    return {'analysis_mask': analysis_mask, 'candidate_mask': merged_candidate_mask, 'raw_candidate_mask': raw_candidate_mask, 'small_candidate_mask': candidate_mask, 'large_candidate_mask': large_candidate_mask, 'salient_candidate_mask': salient_candidate_mask, 'compact_visible_candidate_mask': compact_visible_candidate_mask, 'prominent_red_peak_mask': prominent_red_peak_mask, 'prominent_red_peak_positions': prominent_peaks, 'scale_vote_map': vote_map, 'large_scale_vote_map': large_vote_map, 'salient_scale_vote_map': salient_vote_map, 'compact_visible_vote_map': compact_visible_vote_map, 'single_scale_strong': single_scale_strong, 'max_score_z': max_score_z, 'mean_score_z': mean_score_z, 'large_max_score_z': large_max_score_z, 'large_mean_score_z': large_mean_score_z, 'salient_max_score_z': salient_max_score_z, 'salient_mean_score_z': salient_mean_score_z, 'compact_visible_mean_score_z': compact_visible_mean_z, 'prominent_red_score': prominent_feature_score, 'regional_red_z': regional_red_z, 'max_delta_e': max_delta_e, 'max_color_difference': max_color_difference, 'max_local_contrast': max_local_contrast, 'large_patch_contrast': large_patch_contrast, 'large_regional_delta_e': large_regional_delta_e, 'brownness_score': brownness_score, 'uniformity_score': uniformity_score, 'salient_delta_e': salient_delta_e, 'salient_dark_difference': salient_dark_difference, 'salient_red_difference': salient_red_difference, 'salient_yellow_difference': salient_yellow_difference, 'compact_visible_delta_e': compact_visible_delta_e, 'compact_visible_dark_difference': compact_visible_dark, 'compact_visible_red_difference': compact_visible_red, 'compact_visible_rgb_red_difference': compact_visible_rgb_red, 'compact_visible_yellow_difference': compact_visible_yellow, 'line_response': _line_response(l_feature, analysis_mask), 'l_channel': l_feature}

def _polygon_mask(shape: tuple[int, int], landmarks: np.ndarray, indices: list[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    valid = [index for index in indices if index < len(landmarks)]
    if len(valid) >= 3:
        points = np.rint(landmarks[valid]).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    return mask

def build_region_masks(landmarks: np.ndarray, skin_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Build five required regions; cheek names use image coordinates."""
    (height, width) = skin_mask.shape
    if len(landmarks) <= 334:
        empty = np.zeros_like(skin_mask)
        return {name: empty.copy() for name in ('forehead', 'left_cheek', 'right_cheek', 'nose', 'chin')}
    centre_x = int(np.clip(landmarks[1, 0], 0, width - 1))
    brow_y = int(np.clip(min(landmarks[105, 1], landmarks[334, 1]), 0, height - 1))
    mouth_y = int(np.clip(np.mean(landmarks[list(Config.REGION_LIP_LANDMARKS), 1]), 0, height - 1))
    (yy, xx) = np.indices((height, width))
    skin = skin_mask > 0
    nose = _polygon_mask((height, width), landmarks, list(Config.REGION_NOSE_LANDMARKS))
    nose = cv2.dilate(nose, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.REGION_NOSE_DILATE_KERNEL, Config.REGION_NOSE_DILATE_KERNEL)))
    nose = cv2.bitwise_and(nose, skin_mask)
    forehead = ((yy < brow_y) & skin).astype(np.uint8) * 255
    chin = ((yy > mouth_y) & skin).astype(np.uint8) * 255
    middle = (yy >= brow_y) & (yy <= mouth_y) & skin & (nose == 0)
    left_cheek = (middle & (xx < centre_x)).astype(np.uint8) * 255
    right_cheek = (middle & (xx >= centre_x)).astype(np.uint8) * 255
    return {'forehead': forehead, 'left_cheek': left_cheek, 'right_cheek': right_cheek, 'nose': nose, 'chin': chin}

def _assign_region(x: int, y: int, region_masks: dict[str, np.ndarray]) -> str:
    for name in ('nose', 'forehead', 'left_cheek', 'right_cheek', 'chin'):
        mask = region_masks[name]
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and (mask[y, x] > 0):
            return name
    return 'other'

def _component_values(array: np.ndarray, component: np.ndarray) -> float:
    values = array[component]
    return float(np.mean(values)) if values.size else 0.0

def recover_prominent_red_point_candidates(maps: dict[str, np.ndarray], skin_mask: np.ndarray, region_masks: dict[str, np.ndarray], existing_candidates: list[SpotCandidateV2], existing_candidate_mask: np.ndarray) -> tuple[list[SpotCandidateV2], list[np.ndarray], np.ndarray]:
    """Create compact candidates for obvious local red-brown point peaks.

    This intentionally runs *after* the regular candidate evaluation.  Its
    purpose is not to lower all general thresholds, which would turn facial
    texture into hundreds of spots, but to rescue a local peak only when it is
    clearly red/chromatic relative to the surrounding skin.  Five-feature and
    hair rejection still runs on these candidates in the common post-process.
    """
    peak_mask = maps['prominent_red_peak_mask']
    score = maps['prominent_red_score']
    peak_positions = maps['prominent_red_peak_positions']
    skin_area = max(1, int(np.count_nonzero(skin_mask)))
    output_candidates: list[SpotCandidateV2] = []
    output_contours: list[np.ndarray] = []
    output_mask = np.zeros_like(peak_mask)
    for (peak_row, peak_column) in peak_positions:
        local_disk = np.zeros_like(peak_mask)
        cv2.circle(local_disk, (int(peak_column), int(peak_row)), Config.PROMINENT_RED_MAX_RADIUS, 255, -1)
        local_mask = cv2.bitwise_and(peak_mask, local_disk)
        (labels_count, labels, stats, _) = cv2.connectedComponentsWithStats(local_mask, connectivity=8)
        label = int(labels[peak_row, peak_column])
        if label <= 0 or labels_count <= 1:
            continue
        component_mask = (labels == label).astype(np.uint8) * 255
        if stats[label, cv2.CC_STAT_AREA] < Config.PROMINENT_RED_MIN_AREA_PX:
            continue
        (contours, _) = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < Config.PROMINENT_RED_MIN_AREA_PX:
            continue
        (x, y, width, height) = cv2.boundingRect(contour)
        short_side = max(1, min(width, height))
        aspect_ratio = float(max(width, height) / short_side)
        if aspect_ratio > Config.SALIENT_MAX_ASPECT_RATIO:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-06:
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        circularity = float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-06))
        solidity = float(area / max(hull_area, 1e-06))
        extent = float(area / max(1, width * height))
        moments = cv2.moments(contour)
        if abs(moments['m00']) <= 1e-06:
            continue
        centroid_x = float(moments['m10'] / moments['m00'])
        centroid_y = float(moments['m01'] / moments['m00'])
        region = _assign_region(int(round(centroid_x)), int(round(centroid_y)), region_masks)
        if region == 'nose':
            continue
        if existing_candidate_mask[peak_row, peak_column] > 0:
            continue
        if any((math.hypot(centroid_x - existing.centroid[0], centroid_y - existing.centroid[1]) < Config.PROMINENT_RED_DEDUP_DISTANCE for existing in output_candidates)):
            continue
        component = component_mask > 0
        red = max(_component_values(maps['compact_visible_red_difference'], component), _component_values(maps['salient_red_difference'], component))
        dark = max(_component_values(maps['compact_visible_dark_difference'], component), _component_values(maps['salient_dark_difference'], component))
        yellow = _component_values(maps['compact_visible_yellow_difference'], component)
        delta_e = max(_component_values(maps['compact_visible_delta_e'], component), _component_values(maps['salient_delta_e'], component))
        regional_red_z = _component_values(maps['regional_red_z'], component)
        if red < Config.PROMINENT_RED_MIN_RED and dark < Config.PROMINENT_RED_MIN_DARK and (regional_red_z < Config.INFLAMED_RED_REGION_Z_MIN):
            continue
        score_z = max(_component_values(maps['compact_visible_mean_score_z'], component), _component_values(maps['salient_mean_score_z'], component))
        scale_vote = _component_values(maps['compact_visible_vote_map'], component)
        line_response = _component_values(maps['line_response'], component)
        peak_score = float(score[peak_row, peak_column])
        confidence = float(np.clip(0.26 + 0.16 * min(1.0, peak_score / 3.5) + 0.16 * min(1.0, delta_e / 3.0) + 0.12 * min(1.0, red / 2.0) + 0.1 * min(1.0, regional_red_z / 3.0) + 0.08 * min(1.0, solidity), 0.0, 1.0))
        output_candidates.append(SpotCandidateV2(spot_id=0, region=region, area=round(area, 3), area_ratio_skin=round(area / skin_area, 8), bbox=[int(x), int(y), int(width), int(height)], centroid=[round(centroid_x, 3), round(centroid_y, 3)], circularity=round(circularity, 4), solidity=round(solidity, 4), extent=round(extent, 4), aspect_ratio=round(aspect_ratio, 4), mean_deltaE=round(delta_e, 4), color_difference=round(float(np.hypot(red, yellow)), 4), scale_vote=round(scale_vote, 4), max_scale_vote=int(np.max(maps['compact_visible_vote_map'][component])), local_contrast=round(dark, 4), line_response=round(line_response, 4), score_z=round(score_z, 4), confidence=round(confidence, 4), spot_type='salient', salient_deltaE=round(delta_e, 4), salient_dark_difference=round(dark, 4), salient_red_difference=round(red, 4), salient_yellow_difference=round(yellow, 4), salient_support_fraction=1.0, salient_supported=True, prominent_red_rescue=True))
        output_contours.append(contour)
        output_mask[component] = 255
    return (output_candidates, output_contours, output_mask)

def evaluate_candidates(maps: dict[str, np.ndarray], skin_mask: np.ndarray, region_masks: dict[str, np.ndarray], quality_score: float) -> tuple[list[SpotCandidateV2], np.ndarray, list[np.ndarray]]:
    candidate_mask = maps['candidate_mask']
    (contours, _) = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    skin_area = max(1, int(np.count_nonzero(skin_mask)))
    maximum_area = max(Config.MIN_AREA_PX + 1.0, skin_area * Config.MAX_AREA_RATIO)
    maximum_large_area = max(Config.LARGE_MIN_AREA_PX + 1.0, skin_area * Config.LARGE_MAX_AREA_RATIO)
    maximum_salient_area = max(Config.SALIENT_MIN_AREA_PX + 1.0, skin_area * Config.SALIENT_MAX_AREA_RATIO)
    skin_boundary = cv2.morphologyEx(skin_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.SKIN_BOUNDARY_KERNEL, Config.SKIN_BOUNDARY_KERNEL)))
    accepted: list[SpotCandidateV2] = []
    accepted_contours: list[np.ndarray] = []
    filtered_mask = np.zeros_like(candidate_mask)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < Config.MIN_AREA_PX:
            continue
        component_mask = np.zeros_like(candidate_mask)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        component_mask = cv2.bitwise_and(component_mask, candidate_mask)
        component = component_mask > 0
        component_pixels = max(1, int(np.count_nonzero(component)))
        large_fraction = float(np.count_nonzero(component & (maps['large_candidate_mask'] > 0)) / component_pixels)
        small_fraction = float(np.count_nonzero(component & (maps['small_candidate_mask'] > 0)) / component_pixels)
        salient_fraction = float(np.count_nonzero(component & (maps['salient_candidate_mask'] > 0)) / component_pixels)
        compact_visible_fraction = float(np.count_nonzero(component & (maps['compact_visible_candidate_mask'] > 0)) / component_pixels)
        salient_min_area = Config.COMPACT_VISIBLE_MIN_AREA_PX if compact_visible_fraction >= Config.SALIENT_SUPPORT_FRACTION_MIN else Config.SALIENT_MIN_AREA_PX
        if area >= Config.LARGE_MIN_AREA_PX and large_fraction >= Config.LARGE_CLASS_MIN_FRACTION:
            spot_type = 'large'
        elif area >= salient_min_area and salient_fraction >= Config.SALIENT_CLASS_MIN_FRACTION:
            spot_type = 'salient'
        else:
            spot_type = 'small'
        if spot_type == 'large':
            if area > maximum_large_area:
                continue
        elif spot_type == 'salient':
            maximum_area_for_salient = max(Config.COMPACT_VISIBLE_MIN_AREA_PX + 1.0, skin_area * Config.COMPACT_VISIBLE_MAX_AREA_RATIO) if compact_visible_fraction >= Config.SALIENT_SUPPORT_FRACTION_MIN else maximum_salient_area
            if area > maximum_area_for_salient:
                continue
        elif area > maximum_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-06:
            continue
        (x, y, width, height) = cv2.boundingRect(contour)
        short_side = max(1, min(width, height))
        aspect_ratio = float(max(width, height) / short_side)
        max_aspect = {'large': Config.LARGE_MAX_ASPECT_RATIO, 'salient': Config.SALIENT_MAX_ASPECT_RATIO, 'small': Config.MAX_ASPECT_RATIO}[spot_type]
        if aspect_ratio > max_aspect:
            continue
        bbox_area = max(1, width * height)
        extent = float(area / bbox_area)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = float(area / max(hull_area, 1e-06))
        circularity = float(4.0 * math.pi * area / perimeter ** 2)
        min_extent = {'large': Config.LARGE_MIN_EXTENT, 'salient': Config.SALIENT_MIN_EXTENT, 'small': Config.MIN_EXTENT}[spot_type]
        min_solidity = {'large': Config.LARGE_MIN_SOLIDITY, 'salient': Config.SALIENT_MIN_SOLIDITY, 'small': Config.MIN_SOLIDITY}[spot_type]
        if extent < min_extent or solidity < min_solidity:
            continue
        mean_delta_e = _component_values(maps['max_delta_e'], component)
        color_difference = _component_values(maps['max_color_difference'], component)
        scale_vote = _component_values(maps['scale_vote_map'], component)
        max_scale_vote = int(np.max(maps['scale_vote_map'][component]))
        local_contrast = _component_values(maps['max_local_contrast'], component)
        patch_contrast = _component_values(maps['large_patch_contrast'], component)
        regional_delta_e = _component_values(maps['large_regional_delta_e'], component)
        brownness = _component_values(maps['brownness_score'], component)
        uniformity = _component_values(maps['uniformity_score'], component)
        salient_delta_e = _component_values(maps['salient_delta_e'], component)
        salient_dark = _component_values(maps['salient_dark_difference'], component)
        salient_red = _component_values(maps['salient_red_difference'], component)
        salient_yellow = _component_values(maps['salient_yellow_difference'], component)
        compact_visible_delta_e = _component_values(maps['compact_visible_delta_e'], component)
        compact_visible_dark = _component_values(maps['compact_visible_dark_difference'], component)
        compact_visible_red = _component_values(maps['compact_visible_red_difference'], component)
        compact_visible_yellow = _component_values(maps['compact_visible_yellow_difference'], component)
        salient_scale_vote = _component_values(maps['salient_scale_vote_map'], component)
        salient_score_z = _component_values(maps['salient_mean_score_z'], component)
        compact_visible_scale_vote = _component_values(maps['compact_visible_vote_map'], component)
        compact_visible_score_z = _component_values(maps['compact_visible_mean_score_z'], component)
        if spot_type == 'large':
            mean_delta_e = max(mean_delta_e, regional_delta_e)
            scale_vote = _component_values(maps['large_scale_vote_map'], component)
            max_scale_vote = int(np.max(maps['large_scale_vote_map'][component]))
            local_contrast = max(local_contrast, patch_contrast)
        elif spot_type == 'salient':
            mean_delta_e = max(mean_delta_e, salient_delta_e, compact_visible_delta_e)
            color_difference = max(color_difference, float(np.hypot(salient_red, salient_yellow)), float(np.hypot(compact_visible_red, compact_visible_yellow)))
            scale_vote = max(_component_values(maps['salient_scale_vote_map'], component), compact_visible_scale_vote)
            max_scale_vote = max(int(np.max(maps['salient_scale_vote_map'][component])), int(np.max(maps['compact_visible_vote_map'][component])))
            local_contrast = max(local_contrast, salient_dark, compact_visible_dark)
        line_response = _component_values(maps['line_response'], component)
        line_values = maps['line_response'][component]
        line_peak = float(np.percentile(line_values, Config.CANDIDATE_LINE_PERCENTILE)) if line_values.size else 0.0
        if spot_type == 'large':
            score_z = _component_values(maps['large_mean_score_z'], component)
        elif spot_type == 'salient':
            score_z = max(_component_values(maps['salient_mean_score_z'], component), compact_visible_score_z)
        else:
            score_z = _component_values(maps['mean_score_z'], component)
        mean_l = _component_values(maps['l_channel'], component)
        touches_boundary = bool(np.any(component & (skin_boundary > 0)))
        pore_like = area <= Config.PORE_MAX_AREA and circularity >= Config.PORE_MIN_CIRCULARITY and (mean_delta_e < Config.PORE_MAX_DELTA_E) and (color_difference < Config.PORE_MAX_COLOR_DIFFERENCE)
        wrinkle_like = aspect_ratio >= Config.WRINKLE_MIN_ASPECT_RATIO and line_peak >= Config.WRINKLE_MIN_LINE_RESPONSE or aspect_ratio >= Config.WRINKLE_FORCE_ASPECT_RATIO
        hair_like = line_peak >= Config.HAIR_MIN_LINE_RESPONSE and (touches_boundary and aspect_ratio >= Config.HAIR_BOUNDARY_MIN_ASPECT_RATIO or (aspect_ratio >= Config.HAIR_DARK_MIN_ASPECT_RATIO and mean_l < Config.HAIR_DARK_MAX_MEAN_L))
        shadow_like = area >= skin_area * Config.SHADOW_MIN_AREA_RATIO and mean_delta_e < Config.SHADOW_MAX_DELTA_E and (color_difference < Config.SHADOW_MAX_COLOR_DIFFERENCE) and (local_contrast < Config.SHADOW_MAX_LOCAL_CONTRAST)
        large_brown_patch = spot_type == 'large' and brownness >= Config.LARGE_CONFIRM_MIN_BROWNNESS and (regional_delta_e >= Config.LARGE_CONFIRM_MIN_REGIONAL_DELTA_E) and (scale_vote >= Config.LARGE_MIN_SCALE_VOTES)
        salient_visible_patch = salient_fraction >= Config.SALIENT_SUPPORT_FRACTION_MIN and salient_delta_e >= Config.SALIENT_VISIBLE_DELTA_E_MIN and (salient_dark >= Config.SALIENT_VISIBLE_DARK_MIN or salient_red >= Config.SALIENT_VISIBLE_RED_MIN or salient_yellow >= Config.SALIENT_VISIBLE_YELLOW_MIN) and (salient_scale_vote >= Config.SALIENT_MIN_SCALE_VOTES or salient_score_z >= Config.SALIENT_STRONG_Z_THRESHOLD)
        compact_visible_patch = compact_visible_fraction >= Config.SALIENT_SUPPORT_FRACTION_MIN and compact_visible_delta_e >= Config.COMPACT_VISIBLE_DELTA_E_MIN and (compact_visible_dark >= Config.COMPACT_VISIBLE_DARK_MIN or compact_visible_red >= Config.COMPACT_VISIBLE_RED_MIN or compact_visible_yellow >= Config.COMPACT_VISIBLE_YELLOW_MIN) and (compact_visible_scale_vote >= Config.COMPACT_VISIBLE_MIN_SCALE_VOTES or compact_visible_score_z >= Config.COMPACT_VISIBLE_STRONG_Z_THRESHOLD)
        salient_visible_patch = salient_visible_patch or compact_visible_patch
        if spot_type == 'small' and pore_like or wrinkle_like or hair_like or (shadow_like and (not (large_brown_patch or salient_visible_patch))):
            continue
        if spot_type == 'small' and mean_delta_e < Config.SMALL_WEAK_DELTA_E_MAX and (color_difference < Config.SMALL_WEAK_COLOR_DIFFERENCE_MAX) and (not salient_visible_patch) and (not (local_contrast >= Config.SMALL_STRONG_LOCAL_CONTRAST_MIN and scale_vote >= Config.SMALL_STRONG_SCALE_VOTES_MIN)):
            continue
        weak_chroma = color_difference < Config.WEAK_CHROMA_COLOR_DIFFERENCE_MAX and mean_delta_e < Config.WEAK_CHROMA_DELTA_E_MAX
        compact_strong_dark = area <= Config.COMPACT_DARK_MAX_AREA_PX and aspect_ratio <= Config.COMPACT_DARK_MAX_ASPECT_RATIO and (circularity >= Config.COMPACT_DARK_MIN_CIRCULARITY) and (local_contrast >= Config.COMPACT_DARK_MIN_LOCAL_CONTRAST) and (scale_vote >= Config.COMPACT_DARK_MIN_SCALE_VOTES)
        if spot_type == 'small' and weak_chroma and (not compact_strong_dark) and (not salient_visible_patch):
            continue
        if spot_type == 'large' and (not (large_brown_patch or salient_visible_patch)):
            continue
        if spot_type == 'salient':
            compact_dark = salient_dark >= Config.SALIENT_COMPACT_DARK_MIN and salient_delta_e >= Config.SALIENT_COMPACT_DELTA_E_MIN and (area <= skin_area * Config.SALIENT_COMPACT_MAX_AREA_RATIO)
            chromatic = salient_red >= Config.SALIENT_CHROMATIC_RED_MIN or salient_yellow >= Config.SALIENT_CHROMATIC_YELLOW_MIN
            if not salient_visible_patch or not (chromatic or compact_dark or compact_visible_patch):
                continue
        moments = cv2.moments(contour)
        if abs(moments['m00']) > 1e-06:
            centroid_x = float(moments['m10'] / moments['m00'])
            centroid_y = float(moments['m01'] / moments['m00'])
        else:
            centroid_x = float(x + width * 0.5)
            centroid_y = float(y + height * 0.5)
        region = _assign_region(int(round(centroid_x)), int(round(centroid_y)), region_masks)
        vote_denominator = {'large': len(Config.LARGE_LOCAL_SIGMAS), 'salient': len(Config.SALIENT_CORE_SIGMAS), 'small': len(Config.LOCAL_SIGMAS)}[spot_type]
        z_offset = {'large': Config.CONFIDENCE_Z_OFFSET_LARGE, 'salient': Config.CONFIDENCE_Z_OFFSET_SALIENT, 'small': Config.CONFIDENCE_Z_OFFSET_SMALL}[spot_type]
        z_term = float(np.clip((score_z - z_offset) / Config.CONFIDENCE_Z_RANGE, 0.0, 1.0))
        vote_term = float(np.clip(scale_vote / vote_denominator, 0.0, 1.0))
        delta_scale = Config.CONFIDENCE_DELTA_E_SCALE_LARGE if spot_type == 'large' else Config.CONFIDENCE_DELTA_E_SCALE_OTHER
        delta_term = float(np.clip(mean_delta_e / delta_scale, 0.0, 1.0))
        contrast_term = float(np.clip(local_contrast / Config.CONFIDENCE_CONTRAST_SCALE, 0.0, 1.0))
        brown_term = float(np.clip(brownness / Config.CONFIDENCE_BROWNNESS_SCALE, 0.0, 1.0))
        uniformity_term = float(np.clip(uniformity, 0.0, 1.0))
        salient_term = float(np.clip((Config.CONFIDENCE_SALIENT_DELTA_E_WEIGHT * salient_delta_e + Config.CONFIDENCE_SALIENT_RED_WEIGHT * salient_red + Config.CONFIDENCE_SALIENT_DARK_WEIGHT * salient_dark) / Config.CONFIDENCE_SALIENT_SCALE, 0.0, 1.0))
        if compact_visible_patch:
            compact_term = float(np.clip((0.45 * compact_visible_delta_e + 0.3 * compact_visible_red + 0.25 * compact_visible_dark) / Config.CONFIDENCE_SALIENT_SCALE, 0.0, 1.0))
            salient_term = max(salient_term, compact_term)
        shape_term = float(np.clip(Config.CONFIDENCE_SHAPE_SOLIDITY_WEIGHT * solidity + Config.CONFIDENCE_SHAPE_EXTENT_WEIGHT * min(1.0, extent * Config.CONFIDENCE_SHAPE_EXTENT_SCALE), 0.0, 1.0))
        line_penalty = float(np.clip(line_response, 0.0, Config.CONFIDENCE_LINE_PENALTY_CAP))
        if spot_type == 'large':
            weights = Config.CONFIDENCE_LARGE_WEIGHTS
            evidence_confidence = weights['z'] * z_term + weights['vote'] * vote_term + weights['delta_e'] * delta_term + weights['contrast'] * contrast_term + weights['brown'] * brown_term + weights['uniformity'] * uniformity_term + weights['shape'] * shape_term - weights['line'] * line_penalty
        elif spot_type == 'salient':
            weights = Config.CONFIDENCE_SALIENT_WEIGHTS
            evidence_confidence = weights['z'] * z_term + weights['vote'] * vote_term + weights['delta_e'] * delta_term + weights['contrast'] * contrast_term + weights['salient'] * salient_term + weights['shape'] * shape_term - weights['line'] * line_penalty
        else:
            weights = Config.CONFIDENCE_SMALL_WEIGHTS
            evidence_confidence = weights['z'] * z_term + weights['vote'] * vote_term + weights['delta_e'] * delta_term + weights['contrast'] * contrast_term + weights['shape'] * shape_term - weights['line'] * line_penalty
        if salient_visible_patch:
            evidence_confidence += Config.CONFIDENCE_SALIENT_BONUS_WEIGHT * salient_term
        evidence_confidence = float(np.clip(evidence_confidence, 0.0, 1.0))
        min_confidence = {'large': Config.LARGE_MIN_CONFIDENCE, 'salient': Config.SALIENT_MIN_CONFIDENCE, 'small': Config.MIN_CONFIDENCE}[spot_type]
        if salient_visible_patch:
            min_confidence = min(min_confidence, Config.SALIENT_MIN_CONFIDENCE)
        if compact_visible_patch:
            min_confidence = min(min_confidence, Config.COMPACT_VISIBLE_MIN_CONFIDENCE)
        if evidence_confidence < min_confidence:
            continue
        quality_factor = Config.QUALITY_CONFIDENCE_BASE + Config.QUALITY_CONFIDENCE_WEIGHT * float(np.clip(quality_score / 100.0, 0.0, 1.0))
        confidence = float(np.clip(evidence_confidence * quality_factor, 0.0, 1.0))
        candidate = SpotCandidateV2(spot_id=len(accepted) + 1, region=region, area=round(area, 3), area_ratio_skin=round(area / skin_area, 8), bbox=[int(x), int(y), int(width), int(height)], centroid=[round(centroid_x, 3), round(centroid_y, 3)], circularity=round(circularity, 4), solidity=round(solidity, 4), extent=round(extent, 4), aspect_ratio=round(aspect_ratio, 4), mean_deltaE=round(mean_delta_e, 4), color_difference=round(color_difference, 4), scale_vote=round(scale_vote, 4), max_scale_vote=max_scale_vote, local_contrast=round(local_contrast, 4), line_response=round(line_response, 4), score_z=round(score_z, 4), confidence=round(confidence, 4), spot_type=spot_type, patch_contrast=round(patch_contrast, 4), regional_deltaE=round(regional_delta_e, 4), brownness_score=round(brownness, 4), uniformity_score=round(uniformity, 4), salient_deltaE=round(salient_delta_e, 4), salient_dark_difference=round(salient_dark, 4), salient_red_difference=round(salient_red, 4), salient_yellow_difference=round(salient_yellow, 4), salient_support_fraction=round(salient_fraction, 4), salient_supported=bool(salient_visible_patch))
        accepted.append(candidate)
        accepted_contours.append(contour)
        filtered_mask[component] = 255
    return (accepted, filtered_mask, accepted_contours)

def merge_large_and_small_candidates(candidates: list[SpotCandidateV2], contours: list[np.ndarray], mask_shape: tuple[int, int]) -> tuple[list[SpotCandidateV2], np.ndarray, list[np.ndarray], int]:
    """Prefer one visible large patch over many embedded small fragments."""
    if not candidates:
        return (candidates, np.zeros(mask_shape, dtype=np.uint8), contours, 0)
    large_domain = np.zeros(mask_shape, dtype=np.uint8)
    for (candidate, contour) in zip(candidates, contours):
        if candidate.spot_type == 'large':
            cv2.drawContours(large_domain, [contour], -1, 255, -1)
    if np.count_nonzero(large_domain) > 0:
        large_domain = cv2.dilate(large_domain, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (Config.LARGE_MERGE_DILATE_KERNEL, Config.LARGE_MERGE_DILATE_KERNEL)))
    merged_candidates: list[SpotCandidateV2] = []
    merged_contours: list[np.ndarray] = []
    merged_mask = np.zeros(mask_shape, dtype=np.uint8)
    suppressed_small = 0
    for (candidate, contour) in zip(candidates, contours):
        component_mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        component = component_mask > 0
        pixel_count = max(1, int(np.count_nonzero(component)))
        overlap = float(np.count_nonzero(component & (large_domain > 0)) / pixel_count)
        cx = int(round(candidate.centroid[0]))
        cy = int(round(candidate.centroid[1]))
        centroid_inside_large = 0 <= cy < mask_shape[0] and 0 <= cx < mask_shape[1] and (large_domain[cy, cx] > 0)
        if candidate.spot_type == 'small' and (overlap >= Config.SMALL_SUPPRESS_OVERLAP or (centroid_inside_large and overlap >= Config.SMALL_SUPPRESS_CENTROID_OVERLAP)):
            suppressed_small += 1
            continue
        candidate.spot_id = len(merged_candidates) + 1
        merged_candidates.append(candidate)
        merged_contours.append(contour)
        merged_mask[component] = 255
    return (merged_candidates, merged_mask, merged_contours, suppressed_small)

def _spots_feature_masks(analysis_image: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shared = build_feature_exclusion_masks(analysis_image, landmarks, use_photometric_nostrils=False)
    radius = max(0, int(Config.FEATURE_EXCLUSION_MARGIN_PX))
    if radius <= 0:
        return (shared['feature_exclusions'], shared['nostril_mask'])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return (cv2.dilate(shared['feature_exclusions'], kernel), cv2.dilate(shared['nostril_mask'], kernel))

def _filled_face_domain(skin_mask: np.ndarray) -> np.ndarray:
    """Fill internal skin exclusions to recover the full face search domain."""
    points = cv2.findNonZero((skin_mask > 0).astype(np.uint8))
    domain = np.zeros_like(skin_mask)
    if points is not None and len(points) >= 3:
        cv2.fillConvexPoly(domain, cv2.convexHull(points), 255)
    return domain

def filter_occluded_candidates(analysis_image: np.ndarray, skin_mask: np.ndarray, landmarks: np.ndarray, candidates: list[SpotCandidateV2], contours: list[np.ndarray], pre_filtered_mask: np.ndarray) -> tuple[list[SpotCandidateV2], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Reject hair/feature instances and rebuild one consistent final mask."""
    (feature_mask, nostril_mask) = _spots_feature_masks(analysis_image, landmarks)
    hair_maps = detect_visible_hair_masks(analysis_image, skin_mask, landmarks, feature_mask)
    hair_mask = hair_maps['hair_mask']
    facial_hair_mask = hair_maps['facial_hair_mask']
    nasolabial_shadow = build_nasolabial_shadow_mask(analysis_image, landmarks, skin_mask, radius_px=Config.NASOLABIAL_CORRIDOR_RADIUS_PX, output_margin_px=Config.NASOLABIAL_SHADOW_MARGIN_PX, restrict_frangi_to_corridor=True)
    occlusion_mask = cv2.bitwise_or(hair_mask, feature_mask)
    distance_to_feature = cv2.distanceTransform((feature_mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_hair = cv2.distanceTransform((hair_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    accepted: list[SpotCandidateV2] = []
    accepted_contours: list[np.ndarray] = []
    final_mask = np.zeros_like(pre_filtered_mask)
    rejected_mask = np.zeros_like(pre_filtered_mask)
    statistics = {'hair': 0, 'eyebrow_eyelash_feature': 0, 'facial_hair': 0, 'nostril': 0, 'nasolabial_shadow': 0, 'line_like': 0, 'total': 0}
    for (candidate, contour) in zip(candidates, contours):
        contour_fill = np.zeros_like(pre_filtered_mask)
        cv2.drawContours(contour_fill, [contour], -1, 255, -1)
        component_mask = np.zeros_like(pre_filtered_mask)
        cv2.drawContours(component_mask, [contour], -1, 255, -1)
        component_mask = cv2.bitwise_and(component_mask, pre_filtered_mask)
        component = component_mask > 0
        pixel_count = max(1, int(np.count_nonzero(component)))
        hair_overlap = float(np.count_nonzero(component & (hair_mask > 0)) / pixel_count)
        facial_overlap = float(np.count_nonzero(component & (facial_hair_mask > 0)) / pixel_count)
        feature_overlap = float(np.count_nonzero(component & (feature_mask > 0)) / pixel_count)
        total_overlap = float(np.count_nonzero(component & (occlusion_mask > 0)) / pixel_count)
        nostril_overlap = float(np.count_nonzero(component & (nostril_mask > 0)) / pixel_count)
        nostril_pixels = max(1, int(np.count_nonzero(nostril_mask)))
        nostril_enclosure = float(np.count_nonzero((contour_fill > 0) & (nostril_mask > 0)) / nostril_pixels)
        nasolabial_overlap = float(np.count_nonzero(component & (nasolabial_shadow > 0)) / pixel_count)
        line_values = hair_maps['hair_line_response'][component]
        line_p90 = float(np.percentile(line_values, Config.OCCLUSION_LINE_PERCENTILE)) if line_values.size else 0.0
        vote_values = hair_maps['hair_scale_vote'][component]
        hair_scale_vote = float(np.mean(vote_values)) if vote_values.size else 0.0
        feature_distances = distance_to_feature[component]
        feature_distance = float(np.min(feature_distances)) if feature_distances.size else 999.0
        hair_distances = distance_to_hair[component]
        hair_distance = float(np.min(hair_distances)) if hair_distances.size else 999.0
        cx = int(np.clip(round(candidate.centroid[0]), 0, skin_mask.shape[1] - 1))
        cy = int(np.clip(round(candidate.centroid[1]), 0, skin_mask.shape[0] - 1))
        centroid_occluded = bool(occlusion_mask[cy, cx] > 0)
        centroid_in_nostril = bool(nostril_mask[cy, cx] > 0)
        centroid_in_nasolabial_shadow = bool(nasolabial_shadow[cy, cx] > 0)
        line_like = line_p90 >= Config.OCCLUSION_LINE_MIN_RESPONSE and candidate.aspect_ratio >= Config.OCCLUSION_LINE_MIN_ASPECT_RATIO
        forehead_strand = candidate.region == 'forehead' and (line_p90 >= Config.FOREHEAD_STRONG_LINE_MIN and hair_scale_vote >= Config.FOREHEAD_STRONG_HAIR_VOTES_MIN or (line_p90 >= Config.FOREHEAD_MID_LINE_MIN and candidate.aspect_ratio >= Config.FOREHEAD_MID_ASPECT_RATIO_MIN) or (hair_distance < Config.FOREHEAD_NEAR_HAIR_DISTANCE and candidate.aspect_ratio >= Config.FOREHEAD_NEAR_ASPECT_RATIO_MIN and (candidate.confidence < Config.FOREHEAD_NEAR_CONFIDENCE_MAX)) or (hair_distance < Config.FOREHEAD_NEAR_HAIR_DISTANCE and line_p90 >= Config.FOREHEAD_NEAR_LINE_MIN) or (hair_distance < Config.FOREHEAD_FAR_HAIR_DISTANCE and line_p90 >= Config.FOREHEAD_FAR_LINE_MIN and (candidate.confidence < Config.FOREHEAD_FAR_CONFIDENCE_MAX) and (candidate.area < Config.FOREHEAD_FAR_AREA_MAX)))
        near_feature_line = feature_distance < Config.FEATURE_NEAR_DISTANCE and line_p90 >= Config.FEATURE_NEAR_LINE_MIN
        near_feature_prominent = candidate.prominent_red_rescue and feature_distance < Config.PROMINENT_RED_FEATURE_SAFE_DISTANCE
        nasolabial_line = nasolabial_overlap >= Config.NASOLABIAL_OVERLAP_MAX and candidate.aspect_ratio >= Config.NASOLABIAL_MIN_ASPECT_RATIO and (line_p90 >= Config.NASOLABIAL_LINE_MIN_RESPONSE) or (centroid_in_nasolabial_shadow and candidate.aspect_ratio >= Config.NASOLABIAL_CENTROID_MIN_ASPECT_RATIO)
        reject = total_overlap >= Config.OCCLUSION_OVERLAP_MAX or centroid_occluded or centroid_in_nostril or (nostril_overlap >= Config.NOSTRIL_OVERLAP_MAX) or (nostril_enclosure >= Config.NOSTRIL_ENCLOSURE_MAX) or line_like or forehead_strand or near_feature_line or near_feature_prominent or nasolabial_line
        candidate.occlusion_overlap_ratio = round(total_overlap, 4)
        candidate.hair_line_response_p90 = round(line_p90, 4)
        candidate.hair_scale_vote = round(hair_scale_vote, 4)
        candidate.distance_to_feature_mask = round(feature_distance, 4)
        candidate.distance_to_hair_mask = round(hair_distance, 4)
        candidate.nostril_overlap_ratio = round(nostril_overlap, 4)
        candidate.nostril_enclosure_ratio = round(nostril_enclosure, 4)
        if reject:
            rejected_mask[component] = 255
            statistics['total'] += 1
            if centroid_in_nostril or nostril_overlap >= Config.NOSTRIL_OVERLAP_MAX or nostril_enclosure >= Config.NOSTRIL_ENCLOSURE_MAX:
                statistics['nostril'] += 1
            elif nasolabial_line:
                statistics['nasolabial_shadow'] += 1
            elif facial_overlap > 0.0:
                statistics['facial_hair'] += 1
            elif hair_overlap > 0.0 or centroid_occluded or hair_distance < Config.FOREHEAD_NEAR_HAIR_DISTANCE:
                statistics['hair'] += 1
            elif feature_overlap > 0.0 or near_feature_line or near_feature_prominent:
                statistics['eyebrow_eyelash_feature'] += 1
            else:
                statistics['line_like'] += 1
            continue
        candidate.spot_id = len(accepted) + 1
        accepted.append(candidate)
        accepted_contours.append(contour)
        final_mask[component] = 255
    return (accepted, accepted_contours, final_mask, occlusion_mask, rejected_mask, statistics)

def _split_score_map(candidate: SpotCandidateV2, maps: dict[str, np.ndarray]) -> np.ndarray:
    """Return the response surface that originally supported one instance."""
    if candidate.spot_type == 'large':
        return maps['large_max_score_z']
    if candidate.spot_type == 'salient':
        return maps['salient_max_score_z']
    return maps['max_score_z']

def _child_candidate(parent: SpotCandidateV2, child_mask: np.ndarray, contour: np.ndarray, maps: dict[str, np.ndarray], region_masks: dict[str, np.ndarray], analysis_area: int) -> SpotCandidateV2:
    """Recompute spatial and colour measurements after instance splitting."""
    component = child_mask > 0
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    (x, y, width, height) = cv2.boundingRect(contour)
    bbox_area = max(1, width * height)
    short_side = max(1, min(width, height))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    moments = cv2.moments(contour)
    if abs(moments['m00']) > 1e-06:
        centroid_x = float(moments['m10'] / moments['m00'])
        centroid_y = float(moments['m01'] / moments['m00'])
    else:
        centroid_x = float(x + width * 0.5)
        centroid_y = float(y + height * 0.5)
    if parent.spot_type == 'large':
        vote_map = maps['large_scale_vote_map']
        mean_score_map = maps['large_mean_score_z']
    elif parent.spot_type == 'salient':
        vote_map = maps['salient_scale_vote_map']
        mean_score_map = maps['salient_mean_score_z']
    else:
        vote_map = maps['scale_vote_map']
        mean_score_map = maps['mean_score_z']
    mean_delta_e = _component_values(maps['max_delta_e'], component)
    color_difference = _component_values(maps['max_color_difference'], component)
    local_contrast = _component_values(maps['max_local_contrast'], component)
    patch_contrast = _component_values(maps['large_patch_contrast'], component)
    regional_delta_e = _component_values(maps['large_regional_delta_e'], component)
    salient_delta_e = _component_values(maps['salient_delta_e'], component)
    if parent.spot_type == 'large':
        mean_delta_e = max(mean_delta_e, regional_delta_e)
        local_contrast = max(local_contrast, patch_contrast)
    elif parent.spot_type == 'salient':
        mean_delta_e = max(mean_delta_e, salient_delta_e)
        local_contrast = max(local_contrast, _component_values(maps['salient_dark_difference'], component))
    score_z = _component_values(mean_score_map, component)
    response_term = float(np.clip(score_z / 4.0, 0.0, 1.0))
    confidence = float(np.clip(0.85 * parent.confidence + 0.15 * response_term, 0.0, 1.0))
    region = _assign_region(int(round(centroid_x)), int(round(centroid_y)), region_masks)
    child_type = parent.spot_type
    if child_type == 'large' and area < Config.LARGE_MIN_AREA_PX:
        child_type = 'small'
    return replace(parent, region=region, area=round(area, 3), area_ratio_skin=round(area / max(analysis_area, 1), 8), bbox=[int(x), int(y), int(width), int(height)], centroid=[round(centroid_x, 3), round(centroid_y, 3)], circularity=round(4.0 * math.pi * area / max(perimeter * perimeter, 1e-06), 4), solidity=round(area / max(hull_area, 1e-06), 4), extent=round(area / bbox_area, 4), aspect_ratio=round(max(width, height) / short_side, 4), mean_deltaE=round(mean_delta_e, 4), color_difference=round(color_difference, 4), scale_vote=round(_component_values(vote_map, component), 4), max_scale_vote=int(np.max(vote_map[component])), local_contrast=round(local_contrast, 4), line_response=round(_component_values(maps['line_response'], component), 4), score_z=round(score_z, 4), confidence=round(confidence, 4), spot_type=child_type, patch_contrast=round(patch_contrast, 4), regional_deltaE=round(regional_delta_e, 4), brownness_score=round(_component_values(maps['brownness_score'], component), 4), uniformity_score=round(_component_values(maps['uniformity_score'], component), 4), salient_deltaE=round(salient_delta_e, 4), salient_dark_difference=round(_component_values(maps['salient_dark_difference'], component), 4), salient_red_difference=round(_component_values(maps['salient_red_difference'], component), 4), salient_yellow_difference=round(_component_values(maps['salient_yellow_difference'], component), 4))

def _compact_peak_component(component: np.ndarray, normalized_score: np.ndarray, peak: tuple[int, int], parent_type: str, *, grow_relative: float, grow_absolute: float) -> np.ndarray:
    """Grow one compact child around a supported local maximum."""
    (peak_row, peak_column) = peak
    peak_value = float(normalized_score[peak_row, peak_column])
    grow_threshold = max(grow_absolute, grow_relative * peak_value)
    radius = {'small': Config.SPLIT_MAX_RADIUS_SMALL, 'salient': Config.SPLIT_MAX_RADIUS_SALIENT, 'large': Config.SPLIT_MAX_RADIUS_LARGE}.get(parent_type, Config.SPLIT_MAX_RADIUS_SMALL)
    disk = np.zeros(component.shape, dtype=np.uint8)
    cv2.circle(disk, (peak_column, peak_row), radius, 1, -1)
    compact = (component & (normalized_score >= grow_threshold) & (disk > 0)).astype(np.uint8)
    compact = cv2.morphologyEx(compact, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    (count, labels, stats, _) = cv2.connectedComponentsWithStats(compact, connectivity=8)
    selected = int(labels[peak_row, peak_column])
    if selected <= 0 and count > 1:
        selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if selected <= 0:
        return np.zeros(component.shape, dtype=np.uint8)
    return (labels == selected).astype(np.uint8) * 255

def split_multipeak_candidates(candidates: list[SpotCandidateV2], contours: list[np.ndarray], final_mask: np.ndarray, maps: dict[str, np.ndarray], region_masks: dict[str, np.ndarray], analysis_area: int) -> tuple[list[SpotCandidateV2], list[np.ndarray], np.ndarray, int]:
    """Split multi-peak islands and compact oversized single-peak blobs."""
    output_candidates: list[SpotCandidateV2] = []
    output_contours: list[np.ndarray] = []
    output_mask = np.zeros_like(final_mask)
    split_parent_count = 0
    for (parent, parent_contour) in zip(candidates, contours):
        parent_mask = np.zeros_like(final_mask)
        cv2.drawContours(parent_mask, [parent_contour], -1, 255, -1)
        component = (parent_mask > 0) & (final_mask > 0)
        if parent.area < Config.SPLIT_MIN_AREA or np.count_nonzero(component) < 2 * Config.SPLIT_MIN_CHILD_AREA:
            parent.spot_id = len(output_candidates) + 1
            output_candidates.append(parent)
            output_contours.append(parent_contour)
            output_mask[component] = 255
            continue
        score = _split_score_map(parent, maps).astype(np.float32)
        values = score[component]
        value_min = float(np.min(values)) if values.size else 0.0
        value_max = float(np.max(values)) if values.size else 0.0
        if value_max - value_min <= 1e-06:
            parent.spot_id = len(output_candidates) + 1
            output_candidates.append(parent)
            output_contours.append(parent_contour)
            output_mask[component] = 255
            continue
        normalized = np.zeros_like(score, dtype=np.float32)
        normalized[component] = (score[component] - value_min) / (value_max - value_min)
        normalized = cv2.GaussianBlur(normalized, (0, 0), sigmaX=1.2)
        peaks_array = peak_local_max(normalized, labels=component.astype(np.uint8), min_distance=Config.SPLIT_PEAK_MIN_DISTANCE, threshold_rel=Config.SPLIT_PEAK_THRESHOLD_REL, num_peaks=Config.SPLIT_MAX_PEAKS, exclude_border=False)
        peaks = [(int(row), int(column)) for (row, column) in peaks_array]
        children_masks: list[np.ndarray] = []
        if len(peaks) >= 2:
            markers = np.zeros_like(final_mask, dtype=np.int32)
            for (marker_id, (row, column)) in enumerate(peaks, start=1):
                markers[row, column] = marker_id
            labels = watershed(-normalized, markers, mask=component, watershed_line=True)
            for (label_id, peak) in enumerate(peaks, start=1):
                child = _compact_peak_component(labels == label_id, normalized, peak, parent.spot_type, grow_relative=Config.SPLIT_GROW_THRESHOLD_REL, grow_absolute=Config.SPLIT_GROW_THRESHOLD_ABS)
                if np.count_nonzero(child) >= Config.SPLIT_MIN_CHILD_AREA:
                    children_masks.append(child)
        elif len(peaks) == 1 and parent.area >= Config.SINGLE_PEAK_COMPACT_MIN_AREA:
            child = _compact_peak_component(component, normalized, peaks[0], parent.spot_type, grow_relative=Config.SINGLE_PEAK_GROW_THRESHOLD_REL, grow_absolute=Config.SINGLE_PEAK_GROW_THRESHOLD_ABS)
            if np.count_nonzero(child) >= Config.SPLIT_MIN_CHILD_AREA and np.count_nonzero(child) <= parent.area * Config.SINGLE_PEAK_MAX_PARENT_FRACTION:
                children_masks.append(child)
        child_items: list[tuple[SpotCandidateV2, np.ndarray, np.ndarray]] = []
        for child_mask in children_masks:
            (child_contours, _) = cv2.findContours(child_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not child_contours:
                continue
            child_contour = max(child_contours, key=cv2.contourArea)
            if cv2.contourArea(child_contour) < Config.SPLIT_MIN_CHILD_AREA:
                continue
            child_items.append((_child_candidate(parent, child_mask, child_contour, maps, region_masks, analysis_area), child_contour, child_mask))
        if len(child_items) >= 2:
            split_parent_count += 1
        if child_items:
            for (child, child_contour, child_mask) in child_items:
                child.spot_id = len(output_candidates) + 1
                output_candidates.append(child)
                output_contours.append(child_contour)
                output_mask[child_mask > 0] = 255
        else:
            parent.spot_id = len(output_candidates) + 1
            output_candidates.append(parent)
            output_contours.append(parent_contour)
            output_mask[component] = 255
    return (output_candidates, output_contours, output_mask, split_parent_count)

def _region_distribution(candidates: list[SpotCandidateV2], skin_area: int) -> dict[str, dict[str, float | int]]:
    names = ('forehead', 'left_cheek', 'right_cheek', 'nose', 'chin', 'other')
    result: dict[str, dict[str, float | int]] = {name: {'count': 0, 'area_ratio': 0.0} for name in names}
    for candidate in candidates:
        local = result[candidate.region]
        local['count'] = int(local['count']) + 1
        local['area_ratio'] = float(local['area_ratio']) + candidate.area / max(skin_area, 1)
    for local in result.values():
        local['area_ratio'] = round(float(local['area_ratio']), 8)
    return result

def _draw_overlay(analysis_image: np.ndarray, display_regions: dict[str, np.ndarray], partial_face: bool, filtered_mask: np.ndarray, display_contour: np.ndarray | None=None, display_separator: np.ndarray | None=None) -> np.ndarray:
    overlay = analysis_image.copy()
    (spot_contours, _) = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_pixels = filtered_mask > 0
    if np.any(spot_pixels):
        tint = np.empty_like(overlay)
        tint[:] = Config.COLOR_SPOT
        blended = cv2.addWeighted(overlay, 1.0 - Config.SPOT_FILL_ALPHA, tint, Config.SPOT_FILL_ALPHA, 0.0)
        overlay[spot_pixels] = blended[spot_pixels]
    overlay = draw_region_boundaries(overlay, display_regions, color=Config.COLOR_ZONE, thickness=Config.ZONE_THICKNESS, partial_face=partial_face, closed_contour=display_contour, separator_contour=display_separator)
    for contour in spot_contours:
        if cv2.contourArea(contour) <= Config.TINY_SPOT_FILL_AREA:
            cv2.drawContours(overlay, [contour], -1, Config.COLOR_SPOT, -1, cv2.LINE_AA)
        else:
            cv2.drawContours(overlay, [contour], -1, Config.COLOR_SPOT, Config.SPOT_THICKNESS, cv2.LINE_AA)
    return overlay

def _empty_result(preprocess_result: PreprocessResultV2) -> SpotsResultV2:
    (height, width) = preprocess_result.analysis_image.shape[:2]
    empty = np.zeros((height, width), dtype=np.uint8)
    regions = _region_distribution([], max(1, int(np.count_nonzero(preprocess_result.skin_mask))))
    result = SpotsResultV2(spot_count=0, spot_area_ratio=0.0, spot_locations=[], spot_confidence=0.0, region_distribution=regions, scale_vote_map=empty.copy(), candidate_heatmap=np.zeros((height, width, 3), dtype=np.uint8), candidate_mask=empty.copy(), filtered_mask=empty.copy(), spots_overlay=preprocess_result.analysis_image.copy(), mean_deltaE=0.0, definition='Visible discrete spots in ordinary RGB; not Brown Spots, Red Areas, UV Spots, or medical diagnosis', hair_occlusion_mask=empty.copy(), hair_rejected_mask=empty.copy(), pre_occlusion_spot_count=0, hair_filtered_count=0, occlusion_filter_statistics={'hair': 0, 'eyebrow_eyelash_feature': 0, 'facial_hair': 0, 'nostril': 0, 'line_like': 0, 'total': 0, 'small_suppressed_by_large': 0}, small_spot_count=0, large_spot_count=0, merged_spot_count=0, large_spot_area_ratio=0.0, large_spot_locations=[], large_spot_recall_notes=[], salient_spot_count=0, nostril_filtered_count=0, analysis_zone_area=0, mean_spot_area=0.0, median_spot_area=0.0, pre_split_large_component_count=0, post_split_instance_count=0)
    result._medical_analysis_mask = (np.asarray(preprocess_result.skin_mask) > 0).astype(np.uint8) * 255
    result._medical_landmarks = np.asarray(preprocess_result.landmarks, dtype=np.float32).copy()
    return result

class SpotsEngine:

    def __init__(self):
        self.last_result: SpotsResultV2 | None = None
        self._compat_preprocessor: ImagePreprocessor | None = None

    def detect_spots(self, preprocess_result: PreprocessResultV2) -> SpotsResultV2:
        """Detect visible spots from the three algorithm-safe V2 fields."""
        analysis_image = preprocess_result.analysis_image
        skin_mask = preprocess_result.skin_mask
        landmarks = preprocess_result.landmarks
        hard_failure_flags = {'NO_FACE', 'MULTIPLE_FACES', 'INVALID_IMAGE', 'INVALID_FACE_GEOMETRY'}
        if hard_failure_flags.intersection(preprocess_result.quality_flags) or np.count_nonzero(skin_mask) == 0:
            result = _empty_result(preprocess_result)
            self.last_result = result
            return result
        if analysis_image.ndim != 3 or analysis_image.shape[:2] != skin_mask.shape:
            raise ValueError('analysis_image and skin_mask shapes do not match')
        if landmarks.ndim != 2 or landmarks.shape[1] != 2:
            raise ValueError('landmarks must be aligned Nx2 coordinates')
        region_image = analysis_image.copy()
        region_image[skin_mask == 0] = 0
        visia_regions = build_visia_regions(region_image, skin_mask, landmarks, preprocess_result.quality_flags, include_chin=True, mode='full', feature_margin_px=2)
        analysis_mask = visia_regions.analysis_mask
        region_masks = visia_regions.regions
        analysis_area = int(np.count_nonzero(analysis_mask))
        if analysis_area == 0:
            result = _empty_result(preprocess_result)
            self.last_result = result
            return result
        maps = compute_multiscale_maps(analysis_image, analysis_mask, region_masks)
        (candidates, pre_filtered_mask, candidate_contours) = evaluate_candidates(maps, analysis_mask, region_masks, preprocess_result.quality_score)
        (candidates, pre_filtered_mask, candidate_contours, suppressed_small_by_large) = merge_large_and_small_candidates(candidates, candidate_contours, skin_mask.shape)
        (prominent_red_candidates, prominent_red_contours, prominent_red_mask) = recover_prominent_red_point_candidates(maps, analysis_mask, region_masks, candidates, pre_filtered_mask)
        candidates.extend(prominent_red_candidates)
        candidate_contours.extend(prominent_red_contours)
        pre_filtered_mask = cv2.bitwise_or(pre_filtered_mask, prominent_red_mask)
        pre_occlusion_spot_count = len(candidates)
        (candidates, candidate_contours, filtered_mask, hair_occlusion_mask, hair_rejected_mask, occlusion_statistics) = filter_occluded_candidates(analysis_image, analysis_mask, landmarks, candidates, candidate_contours, pre_filtered_mask)
        occlusion_statistics['small_suppressed_by_large'] = suppressed_small_by_large
        post_occlusion_spot_count = len(candidates)
        pre_split_large_component_count = sum((candidate.area >= Config.SPLIT_MIN_AREA for candidate in candidates))
        (candidates, candidate_contours, filtered_mask, split_parent_count) = split_multipeak_candidates(candidates, candidate_contours, filtered_mask, maps, region_masks, analysis_area)
        occlusion_statistics['multipeak_parents_split'] = split_parent_count
        skin_area = max(1, analysis_area)
        spot_area = float(sum((candidate.area for candidate in candidates)))
        spot_areas = [candidate.area for candidate in candidates]
        small_candidates = [candidate for candidate in candidates if candidate.spot_type == 'small']
        large_candidates = [candidate for candidate in candidates if candidate.spot_type == 'large']
        salient_candidates = [candidate for candidate in candidates if candidate.salient_supported]
        large_area = float(sum((candidate.area for candidate in large_candidates)))
        mean_delta_e = float(np.mean([candidate.mean_deltaE for candidate in candidates])) if candidates else 0.0
        mean_confidence = float(np.mean([candidate.confidence for candidate in candidates])) if candidates else 0.0
        heatmap = _safe_heatmap(np.maximum.reduce((maps['max_score_z'], maps['large_max_score_z'], maps['salient_max_score_z'], maps['compact_visible_mean_score_z'])), analysis_mask)
        overlay = _draw_overlay(analysis_image, visia_regions.display_regions, visia_regions.partial_face, filtered_mask, visia_regions.display_contour, visia_regions.display_separator)
        result = SpotsResultV2(spot_count=len(candidates), spot_area_ratio=round(spot_area / skin_area, 8), spot_locations=[asdict(candidate) for candidate in candidates], spot_confidence=round(mean_confidence, 4), region_distribution=_region_distribution(candidates, skin_area), scale_vote_map=maps['scale_vote_map'], candidate_heatmap=heatmap, candidate_mask=cv2.bitwise_or(maps['candidate_mask'], prominent_red_mask), filtered_mask=filtered_mask, spots_overlay=overlay, mean_deltaE=round(mean_delta_e, 4), definition='Visible discrete spots in ordinary RGB; not Brown Spots, Red Areas, UV Spots, or medical diagnosis', hair_occlusion_mask=hair_occlusion_mask, hair_rejected_mask=hair_rejected_mask, pre_occlusion_spot_count=pre_occlusion_spot_count, hair_filtered_count=pre_occlusion_spot_count - post_occlusion_spot_count, occlusion_filter_statistics=occlusion_statistics, small_spot_count=len(small_candidates), large_spot_count=len(large_candidates), merged_spot_count=len(candidates), large_spot_area_ratio=round(large_area / skin_area, 8), large_spot_locations=[asdict(candidate) for candidate in large_candidates], large_spot_recall_notes=['large_visible_spot_branch_enabled', 'large branch targets soft low-contrast brown/dark patches in ordinary RGB', f'small_candidates_suppressed_by_large={suppressed_small_by_large}'], salient_spot_count=len(salient_candidates), nostril_filtered_count=int(occlusion_statistics.get('nostril', 0)), analysis_zone_area=analysis_area, mean_spot_area=round(float(np.mean(spot_areas)) if spot_areas else 0.0, 4), median_spot_area=round(float(np.median(spot_areas)) if spot_areas else 0.0, 4), pre_split_large_component_count=pre_split_large_component_count, post_split_instance_count=len(candidates))
        result._medical_analysis_mask = analysis_mask.copy()
        result._medical_landmarks = np.asarray(landmarks, dtype=np.float32).copy()
        self.last_result = result
        return result

    def _get_compat_preprocessor(self) -> ImagePreprocessor:
        if self._compat_preprocessor is None:
            self._compat_preprocessor = ImagePreprocessor(None, None, None)
        return self._compat_preprocessor

    @staticmethod
    def _compact_metrics(result: SpotsResultV2, quality_flags: list[str]) -> dict[str, int]:
        """生成云端 CSV/JSON 共用的精简斑点量化内容。"""
        region_names = {'forehead': '额头', 'left_cheek': '左脸颊', 'right_cheek': '右脸颊', 'nose': '鼻部', 'chin': '下巴'}
        headers = ['总计']
        values: list[object] = [int(result.spot_count)]
        is_partial_face = 'PARTIAL_FACE' in set(quality_flags or [])
        if not is_partial_face:
            headers.extend(region_names.values())
            values.extend((int(result.region_distribution.get(region_key, {}).get('count', 0)) for region_key in region_names))
        return {header: int(value) for (header, value) in zip(headers, values)}

    def close(self) -> None:
        if self._compat_preprocessor is not None:
            self._compat_preprocessor.close()
            self._compat_preprocessor = None

def detect_spots(preprocess_result: PreprocessResultV2) -> SpotsResultV2:
    """Module-level V2 convenience entry point."""
    return SpotsEngine().detect_spots(preprocess_result)
