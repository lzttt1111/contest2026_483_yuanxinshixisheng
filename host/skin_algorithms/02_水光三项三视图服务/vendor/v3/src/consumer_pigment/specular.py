"""Shared consumer specular evidence and display compression."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SpecularEvidence:
    mask: np.ndarray
    alpha: np.ndarray
    value_threshold: float
    saturation_threshold: float


def _masked_gaussian(
    image: np.ndarray,
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    weights = (np.asarray(mask) > 0).astype(np.float32)
    values = np.asarray(image, dtype=np.float32)
    if values.ndim == 3:
        weights = weights[:, :, None]
    numerator = cv2.GaussianBlur(
        values * weights,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    denominator = cv2.GaussianBlur(
        weights,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )
    if values.ndim == 3 and denominator.ndim == 2:
        denominator = denominator[:, :, None]
    return numerator / np.maximum(denominator, 1e-6)


def build_specular_evidence(
    image: np.ndarray,
    valid_mask: np.ndarray,
) -> SpecularEvidence:
    """Find bright, low-saturation local outliers on valid skin."""
    valid = np.asarray(valid_mask) > 0
    hsv = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    samples_v = value[valid]
    samples_s = saturation[valid]
    value_threshold = max(
        0.90,
        float(np.percentile(samples_v, 97.0)) if samples_v.size else 0.90,
    )
    saturation_threshold = min(
        0.30,
        float(np.percentile(samples_s, 35.0)) if samples_s.size else 0.30,
    )
    local_background = _masked_gaussian(value, valid_mask, 12.0)
    residual = value - local_background
    candidate = (
        valid
        & (value >= value_threshold)
        & (saturation <= saturation_threshold)
        & (residual >= 0.04)
    ).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    candidate = cv2.dilate(
        candidate,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    candidate[~valid] = 0
    alpha = cv2.GaussianBlur(
        (candidate > 0).astype(np.float32),
        (0, 0),
        sigmaX=4.0,
        sigmaY=4.0,
    )
    return SpecularEvidence(
        mask=candidate,
        alpha=np.clip(alpha * 0.78, 0.0, 0.78),
        value_threshold=round(value_threshold, 6),
        saturation_threshold=round(saturation_threshold, 6),
    )


def compress_specular_highlights(
    image: np.ndarray,
    evidence: SpecularEvidence,
) -> np.ndarray:
    """Roll highlights toward surrounding tone without painting flat patches."""
    context_mask = np.where(evidence.mask > 0, 0, 255).astype(np.uint8)
    local_tone = _masked_gaussian(image, context_mask, 12.0)
    alpha = evidence.alpha[:, :, None]
    output = (
        np.asarray(image, dtype=np.float32) * (1.0 - alpha)
        + local_tone * alpha
    )
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def build_source_and_render_specular_evidence(
    source_image: np.ndarray,
    rendered_image: np.ndarray,
    valid_mask: np.ndarray,
) -> SpecularEvidence:
    """Union camera glare with clipping introduced by the Brown display."""
    source = build_specular_evidence(source_image, valid_mask)
    rendered = build_specular_evidence(rendered_image, valid_mask)
    return SpecularEvidence(
        mask=cv2.bitwise_or(source.mask, rendered.mask),
        alpha=np.maximum(source.alpha, rendered.alpha),
        value_threshold=min(source.value_threshold, rendered.value_threshold),
        saturation_threshold=max(
            source.saturation_threshold,
            rendered.saturation_threshold,
        ),
    )


__all__ = [
    "SpecularEvidence",
    "build_specular_evidence",
    "build_source_and_render_specular_evidence",
    "compress_specular_highlights",
]
