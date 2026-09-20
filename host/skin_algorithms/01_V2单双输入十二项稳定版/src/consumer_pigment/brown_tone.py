"""Highlight-protected display tone variants for consumer Brown images."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from typing_extensions import assert_never


class ConsumerBrownTonePreset(str, Enum):
    LIGHT = "light"
    NATURAL = "natural_clear"
    BRIGHT = "bright_contrast"


@dataclass(frozen=True, slots=True)
class BrownTonePolicy:
    midtone_lift: float
    local_contrast_gain: float


@dataclass(frozen=True, slots=True)
class BrownToneInput:
    base_image: np.ndarray
    foreground_mask: np.ndarray
    protected_highlight_alpha: np.ndarray


def policy_for_brown_tone(
    preset: ConsumerBrownTonePreset,
) -> BrownTonePolicy:
    match preset:
        case ConsumerBrownTonePreset.LIGHT:
            return BrownTonePolicy(6.0, 0.05)
        case ConsumerBrownTonePreset.NATURAL:
            return BrownTonePolicy(10.0, 0.10)
        case ConsumerBrownTonePreset.BRIGHT:
            return BrownTonePolicy(14.0, 0.18)
        case unreachable:
            assert_never(unreachable)


def render_brown_tone(
    render_input: BrownToneInput,
    preset: ConsumerBrownTonePreset,
) -> np.ndarray:
    """Lift non-highlight midtones while keeping hue and background stable."""
    policy = policy_for_brown_tone(preset)
    base = np.asarray(render_input.base_image)
    foreground = np.asarray(render_input.foreground_mask) > 0
    protected = np.clip(
        np.asarray(render_input.protected_highlight_alpha, dtype=np.float32),
        0.0,
        1.0,
    )
    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    normalized = lightness / 255.0
    midtone_weight = 4.0 * normalized * (1.0 - normalized)
    local_background = cv2.GaussianBlur(
        lightness,
        (0, 0),
        sigmaX=8.0,
        sigmaY=8.0,
    )
    local_detail = np.clip(lightness - local_background, -12.0, 12.0)
    available = 1.0 - protected
    adjusted = (
        lightness
        + policy.midtone_lift * midtone_weight * available
        + policy.local_contrast_gain * local_detail * available
    )
    output_lab = lab.copy()
    output_lab[:, :, 0] = np.where(
        foreground,
        np.clip(adjusted, 0.0, 255.0),
        lightness,
    ).astype(np.uint8)
    output = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output[~foreground] = base[~foreground]
    return output


__all__ = [
    "BrownToneInput",
    "BrownTonePolicy",
    "ConsumerBrownTonePreset",
    "policy_for_brown_tone",
    "render_brown_tone",
]
