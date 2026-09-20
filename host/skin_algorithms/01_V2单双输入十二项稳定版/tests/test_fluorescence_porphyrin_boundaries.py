from __future__ import annotations

import cv2
import numpy as np

from src.engines.fluorescence_porphyrin_engine import (
    FluorescencePorphyrinAnalyzer,
)
from src.engines.visia_regions import (
    REGION_ORDER,
    VisiaRegionSet,
    draw_region_boundaries,
)


class ZeroScorePorphyrinAnalyzer(FluorescencePorphyrinAnalyzer):
    def _score_map(
        self,
        image: np.ndarray,
        regions: VisiaRegionSet,
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = image.shape[:2]
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.uint8)


def test_porphyrin_overlay_uses_unified_visia_contour_without_changing_detection() -> None:
    shape = (96, 96)
    base = np.full((*shape, 3), 30, dtype=np.uint8)
    analysis_mask = np.full(shape, 255, dtype=np.uint8)
    empty_region = np.zeros(shape, dtype=np.uint8)
    regions = VisiaRegionSet(
        analysis_mask=analysis_mask,
        feature_exclusion_mask=empty_region.copy(),
        regions={name: analysis_mask.copy() for name in REGION_ORDER},
        display_regions={name: empty_region.copy() for name in REGION_ORDER},
        partial_face=False,
        display_contour=np.asarray(
            [[[10, 10]], [[85, 10]], [[85, 85]], [[10, 85]]],
            dtype=np.int32,
        ),
        display_separator=np.asarray(
            [[[12, 48]], [[83, 48]]],
            dtype=np.int32,
        ),
    )
    landmarks = np.zeros((478, 2), dtype=np.float32)

    analyzer = ZeroScorePorphyrinAnalyzer()
    result = analyzer.detect(base, regions, landmarks)
    expected = draw_region_boundaries(
        base,
        regions.display_regions,
        color=analyzer.config.region_color_bgr,
        thickness=analyzer.config.region_thickness,
        partial_face=False,
        nose_style=analyzer.config.nose_style,
        landmarks=landmarks,
        closed_contour=regions.display_contour,
        separator_contour=regions.display_separator,
    )

    assert np.array_equal(result.overlay, expected)
    assert cv2.countNonZero(result.instance_mask) == 0
    assert result.locations == []
    assert all(item["count"] == 0 for item in result.region_distribution.values())
