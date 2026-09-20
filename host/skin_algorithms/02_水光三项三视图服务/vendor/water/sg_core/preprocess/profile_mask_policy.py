from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import cv2
import numpy as np
from typing_extensions import assert_never
from sg_core.capture_profile import CaptureProfile
from sg_core.preprocess.analysis_mask_bundle import AnalysisMaskBundle, build_analysis_mask_bundle, expand_common_hair_masks
from sg_core.preprocess.image_preprocessor import detect_visible_hair_masks

@dataclass(frozen=True, slots=True)
class AppliedProfileMasks:
    skin_mask: np.ndarray
    forbidden_mask: np.ndarray
    bundle: AnalysisMaskBundle

class PreprocessMaskConsumer(Protocol):
    skin_mask: np.ndarray
    mask_bundle: AnalysisMaskBundle | None

@dataclass(frozen=True, slots=True)
class ProfileMaskCoverageError(RuntimeError):
    valid_pixels: int
    source_pixels: int

    def __str__(self) -> str:
        return f'consumer common mask removed implausibly large skin area: valid={self.valid_pixels}, source={self.source_pixels}'

def apply_profile_mask_policy(profile: CaptureProfile, analysis_image: np.ndarray, skin_mask: np.ndarray, forbidden_mask: np.ndarray, landmarks: np.ndarray, coordinate_space: str) -> AppliedProfileMasks:
    empty = np.zeros_like(skin_mask, dtype=np.uint8)
    hair = detect_visible_hair_masks(analysis_image, skin_mask, landmarks, forbidden_mask)
    (hair_mask, facial_hair_mask) = expand_common_hair_masks(hair['hair_mask'], hair['facial_hair_mask'])
    bundle = build_analysis_mask_bundle(valid_skin_mask=skin_mask, hair_mask=hair_mask, facial_hair_mask=facial_hair_mask, feature_exclusion_mask=forbidden_mask, nostril_mask=empty, display_face_mask=skin_mask, coordinate_space=coordinate_space)
    source_pixels = int(np.count_nonzero(skin_mask))
    valid_pixels = int(np.count_nonzero(bundle.valid_skin_mask))
    if source_pixels >= 100 and valid_pixels < 0.15 * source_pixels:
        required_fallback_keys = {'bulk_hair_mask', 'color_hair_mask', 'strand_hair_mask', 'facial_hair_mask'}
        bulk_pixels = int(np.count_nonzero(hair.get('bulk_hair_mask')))
        if required_fallback_keys.issubset(hair) and bulk_pixels <= 0.5 * source_pixels:
            conservative_hair = cv2.bitwise_or(hair['color_hair_mask'], hair['strand_hair_mask'])
            conservative_hair = cv2.bitwise_or(conservative_hair, hair['facial_hair_mask'])
            (hair_mask, facial_hair_mask) = expand_common_hair_masks(conservative_hair, hair['facial_hair_mask'])
            bundle = build_analysis_mask_bundle(valid_skin_mask=skin_mask, hair_mask=hair_mask, facial_hair_mask=facial_hair_mask, feature_exclusion_mask=forbidden_mask, nostril_mask=empty, display_face_mask=skin_mask, coordinate_space=coordinate_space)
            valid_pixels = int(np.count_nonzero(bundle.valid_skin_mask))
        if valid_pixels < 0.15 * source_pixels:
            raise ProfileMaskCoverageError(valid_pixels, source_pixels)
    removed = cv2.bitwise_and((np.asarray(skin_mask) > 0).astype(np.uint8) * 255, cv2.bitwise_not(np.asarray(bundle.valid_skin_mask)))
    return AppliedProfileMasks(skin_mask=np.asarray(bundle.valid_skin_mask).copy(), forbidden_mask=cv2.bitwise_or(forbidden_mask, removed), bundle=bundle)

def apply_preprocess_mask_bundle(profile: CaptureProfile, preprocess_result: PreprocessMaskConsumer) -> None:
    """Publish the common consumer algorithm mask to every DermaVision engine."""
    if preprocess_result.mask_bundle is not None:
        preprocess_result.skin_mask = preprocess_result.mask_bundle.algorithm_mask()
__all__ = ['AppliedProfileMasks', 'PreprocessMaskConsumer', 'ProfileMaskCoverageError', 'apply_preprocess_mask_bundle', 'apply_profile_mask_policy']
