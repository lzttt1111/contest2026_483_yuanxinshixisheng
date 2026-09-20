"""Small display variants derived from the existing consumer Brown image."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from typing_extensions import assert_never

from src.consumer_pigment.specular import (
    build_specular_evidence,
    compress_specular_highlights,
)


class ConsumerBrownPalette(str, Enum):
    SOFT = "soft"
    REFERENCE = "reference_repaired"
    ENHANCED = "enhanced"


@dataclass(frozen=True, slots=True)
class BrownRenderInput:
    source_image: np.ndarray
    mother_base: np.ndarray
    valid_mask: np.ndarray


def _soften(image: np.ndarray) -> np.ndarray:
    filtered = cv2.bilateralFilter(image, 7, 16.0, 16.0)
    return cv2.addWeighted(image, 0.78, filtered, 0.22, 3.0)


def _enhance(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.4, sigmaY=1.4)
    sharpened = image.astype(np.float32) + 0.16 * (
        image.astype(np.float32) - blurred.astype(np.float32)
    )
    return np.clip(1.035 * sharpened - 4.0, 0.0, 255.0).astype(np.uint8)


def render_consumer_brown_base(
    render_input: BrownRenderInput,
    palette: ConsumerBrownPalette,
) -> np.ndarray:
    """Repair glare, then make only a small display adjustment to the mother."""
    source = np.asarray(render_input.source_image)
    mother = np.asarray(render_input.mother_base)
    evidence = build_specular_evidence(source, render_input.valid_mask)
    repaired = compress_specular_highlights(mother, evidence)
    match palette:
        case ConsumerBrownPalette.SOFT:
            return _soften(repaired)
        case ConsumerBrownPalette.REFERENCE:
            return repaired
        case ConsumerBrownPalette.ENHANCED:
            return _enhance(repaired)
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "BrownRenderInput",
    "ConsumerBrownPalette",
    "render_consumer_brown_base",
]
