"""Display-only local contrast for consumer Brown results."""
from __future__ import annotations
from typing import Final
import numpy as np
_DEEP_BROWN_BGR: Final = np.asarray((8, 42, 125), dtype=np.float32)
_SCORE_START: Final = 0.35
_MAX_ALPHA: Final = 0.62

def enhance_brown_score_contrast(base_image: np.ndarray, score_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Darken high-score local evidence without drawing or changing findings."""
    base = np.asarray(base_image)
    score = np.clip(np.asarray(score_map, dtype=np.float32), 0.0, 1.0)
    supported = np.asarray(valid_mask) > 0
    normalized = np.clip((score - _SCORE_START) / (1.0 - _SCORE_START), 0.0, 1.0)
    smooth = normalized * normalized * (3.0 - 2.0 * normalized)
    alpha = (_MAX_ALPHA * smooth)[:, :, None]
    output = base.astype(np.float32)
    blended = output * (1.0 - alpha) + _DEEP_BROWN_BGR * alpha
    output[supported] = blended[supported]
    return np.clip(output, 0.0, 255.0).astype(np.uint8)
__all__ = ['enhance_brown_score_contrast']
