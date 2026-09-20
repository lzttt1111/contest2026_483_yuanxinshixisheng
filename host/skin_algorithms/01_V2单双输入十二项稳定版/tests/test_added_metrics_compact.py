from __future__ import annotations

import pytest

from src.nine_analysis.metrics import extract_item


@pytest.mark.parametrize(
    ("item", "metric"),
    (
        ("surface_gloss", "gloss_area_ratio_total"),
        ("vascular", "vascular_count_total"),
        ("contour_firmness", "jaw_continuity_ratio"),
    ),
)
def test_added_item_metrics_are_valid_scoring_inputs(item: str, metric: str) -> None:
    raw = {metric: 0.25, "ignored_text": "not-a-number"}

    summary, scoring = extract_item(item, raw)

    assert summary == {metric: 0.25}
    assert scoring == {"工程量化指标": {metric: 0.25}}
