# -*- coding: utf-8 -*-
"""Shared user-review masks for all five skin-analysis outputs.

These files explain which pixels were analyzable and which visible hair was
excluded.  They never feed back into detection or quantification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io_utils import cv_imwrite


DEBUG_VALID_SKIN_NAME = "03_有效皮肤区域.png"
DEBUG_HAIR_OCCLUSION_NAME = "04_毛发遮挡区域.png"
DEBUG_HAIR_REJECTED_NAME = "05_已过滤毛发误检.png"


def capture_preprocess_debug(preprocess_result: Any) -> dict[str, np.ndarray]:
    """Copy stable uint8 masks without changing PreprocessResultV2 fields."""
    valid = (np.asarray(preprocess_result.skin_mask) > 0).astype(np.uint8) * 255
    raw = getattr(preprocess_result, "_debug_masks", {}) or {}
    hair = raw.get("hair_mask")
    if hair is None or np.asarray(hair).shape != valid.shape:
        hair = np.zeros_like(valid)
    else:
        hair = (np.asarray(hair) > 0).astype(np.uint8) * 255
    return {
        "valid_skin_mask": valid.copy(),
        "hair_occlusion_mask": hair.copy(),
    }


def write_analysis_debug_outputs(
    sample_dir: str | Path,
    debug_masks: dict[str, np.ndarray] | None,
    filtered_hair_mask: np.ndarray | None = None,
) -> dict[str, str]:
    """Write the same three diagnostic masks for every detection project."""
    directory = Path(sample_dir)
    directory.mkdir(parents=True, exist_ok=True)
    debug = debug_masks or {}
    valid = np.asarray(debug.get("valid_skin_mask", np.zeros((1, 1), np.uint8)))
    valid = (valid > 0).astype(np.uint8) * 255
    hair = np.asarray(debug.get("hair_occlusion_mask", np.zeros_like(valid)))
    if hair.shape != valid.shape:
        hair = np.zeros_like(valid)
    hair = (hair > 0).astype(np.uint8) * 255
    rejected = (
        np.zeros_like(valid)
        if filtered_hair_mask is None
        else (np.asarray(filtered_hair_mask) > 0).astype(np.uint8) * 255
    )
    if rejected.shape != valid.shape:
        rejected = np.zeros_like(valid)

    paths = {
        "valid_skin": str(directory / DEBUG_VALID_SKIN_NAME),
        "hair_occlusion": str(directory / DEBUG_HAIR_OCCLUSION_NAME),
        "hair_rejected": str(directory / DEBUG_HAIR_REJECTED_NAME),
    }
    cv_imwrite(paths["valid_skin"], valid)
    cv_imwrite(paths["hair_occlusion"], hair)
    cv_imwrite(paths["hair_rejected"], rejected)
    return paths
