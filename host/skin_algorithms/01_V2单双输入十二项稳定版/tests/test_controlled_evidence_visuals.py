from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.aisia_medical_report.controlled_evidence import (
    _red_brown_overlay,
    _surface_irregularity,
)
from src.engines.visia_regions import VisiaRegionSet


def _regions(shape: tuple[int, int]) -> VisiaRegionSet:
    empty = np.zeros(shape, dtype=np.uint8)
    contour = np.asarray([[4, 4], [27, 4], [27, 27], [4, 27]], np.int32)
    return VisiaRegionSet(
        analysis_mask=np.full(shape, 255, np.uint8),
        feature_exclusion_mask=empty,
        regions={},
        display_regions={},
        partial_face=False,
        display_contour=contour,
        scope_mask=np.full(shape, 255, np.uint8),
    )


def test_red_brown_derived_image_adds_public_visia_contour(
    tmp_path: Path,
) -> None:
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[14:18, 14:18] = 255
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    output = _red_brown_overlay(
        image,
        mask_path,
        tmp_path / "result.jpg",
        regions=_regions(mask.shape),
    )
    rendered = cv2.imread(str(output))

    assert rendered is not None
    assert int(rendered[4, 16, 0]) > 140
    assert int(rendered[4, 16, 1]) > 140


def test_surface_irregularity_uses_sparse_texture_mask_not_dense_score(
    tmp_path: Path,
) -> None:
    image = np.full((32, 32, 3), (70, 90, 120), dtype=np.uint8)
    marker = np.zeros((32, 32, 3), dtype=np.uint8)
    marker[14:17, 14:17] = (255, 255, 0)
    marker_path = tmp_path / "texture-mask.png"
    cv2.imwrite(str(marker_path), marker)

    output = _surface_irregularity(
        image,
        marker_path,
        tmp_path / "surface.png",
        regions=_regions(image.shape[:2]),
    )
    rendered = cv2.imread(str(output))
    changed = np.any(np.abs(rendered.astype(int) - image.astype(int)) > 30, axis=2)

    assert rendered is not None
    assert 8 <= np.count_nonzero(changed[10:22, 10:22]) <= 30
    assert np.count_nonzero(changed) < 600
