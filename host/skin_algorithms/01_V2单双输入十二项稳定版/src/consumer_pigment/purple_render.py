"""Small display variants derived from existing UV and porphyrin bases."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from typing_extensions import assert_never


class ConsumerPurplePalette(str, Enum):
    SOFT = "soft"
    REFERENCE = "reference"
    ENHANCED = "enhanced"


@dataclass(frozen=True, slots=True)
class PurpleRenderInput:
    uv_mother_base: np.ndarray
    porphyrin_mother_base: np.ndarray


@dataclass(frozen=True, slots=True)
class PurplePaletteResult:
    palette: ConsumerPurplePalette
    uv_base: np.ndarray
    porphyrin_base: np.ndarray


def _soften(image: np.ndarray) -> np.ndarray:
    filtered = cv2.bilateralFilter(image, 7, 14.0, 14.0)
    mixed = cv2.addWeighted(image, 0.82, filtered, 0.18, 0.0)
    return np.clip(mixed.astype(np.float32) * 0.96 + 5.0, 0, 255).astype(np.uint8)


def _enhance(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)
    detail = image.astype(np.float32) - blurred.astype(np.float32)
    return np.clip(
        image.astype(np.float32) * 1.035 + 0.14 * detail - 4.0,
        0,
        255,
    ).astype(np.uint8)


def render_consumer_purple_palette(
    render_input: PurpleRenderInput,
    palette: ConsumerPurplePalette,
) -> PurplePaletteResult:
    """Keep current visual identity and vary only softness/detail strength."""
    uv = np.asarray(render_input.uv_mother_base)
    porphyrin = np.asarray(render_input.porphyrin_mother_base)
    match palette:
        case ConsumerPurplePalette.SOFT:
            return PurplePaletteResult(palette, _soften(uv), _soften(porphyrin))
        case ConsumerPurplePalette.REFERENCE:
            return PurplePaletteResult(palette, uv.copy(), porphyrin.copy())
        case ConsumerPurplePalette.ENHANCED:
            return PurplePaletteResult(palette, _enhance(uv), _enhance(porphyrin))
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "ConsumerPurplePalette",
    "PurplePaletteResult",
    "PurpleRenderInput",
    "render_consumer_purple_palette",
]
