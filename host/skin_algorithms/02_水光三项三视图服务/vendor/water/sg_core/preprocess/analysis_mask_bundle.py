from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Final, TypedDict
import cv2
import numpy as np
MASK_CONTRACT_VERSION: Final = 'analysis_mask_bundle_v1'
COMMON_HAIR_SAFETY_MARGIN_PX: Final = 7

class MaskEvidence(TypedDict):
    mask_contract_version: str
    coordinate_space: str
    sha256: dict[str, str]
    pixels: dict[str, int]

@dataclass(frozen=True, slots=True)
class MaskShapeError(ValueError):
    label: str
    expected: tuple[int, ...]
    actual: tuple[int, ...]

    def __str__(self) -> str:
        return f'{self.label} mask shape mismatch: expected={self.expected}, actual={self.actual}'

def _binary_readonly(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if np.asarray(mask).shape != shape:
        raise MaskShapeError('analysis', shape, np.asarray(mask).shape)
    result = ((np.asarray(mask) > 0).astype(np.uint8) * 255).copy()
    result.flags.writeable = False
    return result

def _mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask).tobytes(order='C')).hexdigest()

def expand_common_hair_masks(hair_mask: np.ndarray, facial_hair_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the common seven-pixel safety margin around hair evidence."""
    kernel_size = 2 * COMMON_HAIR_SAFETY_MARGIN_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    hair = cv2.dilate((np.asarray(hair_mask) > 0).astype(np.uint8) * 255, kernel)
    facial = cv2.dilate((np.asarray(facial_hair_mask) > 0).astype(np.uint8) * 255, kernel)
    return (hair, facial)

@dataclass(frozen=True, slots=True)
class AnalysisMaskBundle:
    """Immutable upper bound shared by every detector in one profile."""
    valid_skin_mask: np.ndarray
    hair_mask: np.ndarray
    facial_hair_mask: np.ndarray
    feature_exclusion_mask: np.ndarray
    nostril_mask: np.ndarray
    display_face_mask: np.ndarray
    coordinate_space: str

    def algorithm_mask(self, extra_exclusion: np.ndarray | None=None) -> np.ndarray:
        """Return a writable detector mask that can only remove pixels."""
        result = np.asarray(self.valid_skin_mask).copy()
        if extra_exclusion is not None:
            if np.asarray(extra_exclusion).shape != result.shape:
                raise MaskShapeError('extra_exclusion', result.shape, np.asarray(extra_exclusion).shape)
            result[np.asarray(extra_exclusion) > 0] = 0
        result[np.asarray(self.hair_mask) > 0] = 0
        result[np.asarray(self.facial_hair_mask) > 0] = 0
        result[np.asarray(self.feature_exclusion_mask) > 0] = 0
        result[np.asarray(self.nostril_mask) > 0] = 0
        return result

    def evidence(self) -> MaskEvidence:
        masks = {'valid_skin': self.valid_skin_mask, 'hair': self.hair_mask, 'facial_hair': self.facial_hair_mask, 'feature_exclusion': self.feature_exclusion_mask, 'nostril': self.nostril_mask, 'display_face': self.display_face_mask}
        return {'mask_contract_version': MASK_CONTRACT_VERSION, 'coordinate_space': self.coordinate_space, 'sha256': {name: _mask_sha256(mask) for (name, mask) in masks.items()}, 'pixels': {name: int(cv2.countNonZero(np.asarray(mask))) for (name, mask) in masks.items()}}

def build_analysis_mask_bundle(*, valid_skin_mask: np.ndarray, hair_mask: np.ndarray, facial_hair_mask: np.ndarray, feature_exclusion_mask: np.ndarray, nostril_mask: np.ndarray, display_face_mask: np.ndarray, coordinate_space: str) -> AnalysisMaskBundle:
    shape = np.asarray(valid_skin_mask).shape
    if len(shape) != 2:
        raise MaskShapeError('valid_skin', (-1, -1), shape)
    hair = _binary_readonly(hair_mask, shape)
    facial_hair = _binary_readonly(facial_hair_mask, shape)
    features = _binary_readonly(feature_exclusion_mask, shape)
    nostrils = _binary_readonly(nostril_mask, shape)
    excluded = cv2.bitwise_or(hair, facial_hair)
    excluded = cv2.bitwise_or(excluded, features)
    excluded = cv2.bitwise_or(excluded, nostrils)
    valid = cv2.bitwise_and(_binary_readonly(valid_skin_mask, shape), cv2.bitwise_not(excluded))
    valid.flags.writeable = False
    return AnalysisMaskBundle(valid_skin_mask=valid, hair_mask=hair, facial_hair_mask=facial_hair, feature_exclusion_mask=features, nostril_mask=nostrils, display_face_mask=_binary_readonly(display_face_mask, shape), coordinate_space=coordinate_space)
__all__ = ['AnalysisMaskBundle', 'COMMON_HAIR_SAFETY_MARGIN_PX', 'MASK_CONTRACT_VERSION', 'MaskEvidence', 'MaskShapeError', 'build_analysis_mask_bundle', 'expand_common_hair_masks']
