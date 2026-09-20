from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest

from src.engines.visia_regions import (
    VISIA_BOUNDARY_COLOR,
    build_visia_regions,
    draw_region_boundaries,
)
from src.engines.rbx_engine import ErythemaAnalyzer
from src.engines.brown_engine import BrownAreaAnalyzer
from src.capture_profile import CaptureProfile
from src.engines.visia_unified_contour import _front_template


def _evaluator():
    module = importlib.import_module("src.preprocess.image_preprocessor")
    evaluator = getattr(module, "evaluate_consumer_quality", None)
    assert callable(evaluator), "consumer独立质量门禁尚未实现"
    return evaluator


def test_consumer_forehead_occlusion_becomes_warning() -> None:
    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    skin = np.full((256, 256), 255, dtype=np.uint8)
    forehead = np.zeros_like(skin)
    forehead[:80] = 255
    hair = np.zeros_like(skin)
    hair[:80, :220] = 255

    decision = _evaluator()(
        image,
        skin,
        {"forehead_completion": forehead, "hair_mask": hair},
        90.0,
        "PASS",
        [],
    )

    assert decision.status == "WARNING"
    assert "FOREHEAD_OCCLUDED" in decision.flags
    assert decision.forehead_occluded is True


def test_consumer_severe_local_lighting_is_rejected() -> None:
    image = np.full((256, 256, 3), 30, dtype=np.uint8)
    image[:, 128:] = 230
    skin = np.full((256, 256), 255, dtype=np.uint8)

    decision = _evaluator()(image, skin, {}, 90.0, "PASS", [])

    assert decision.status == "REJECT"
    assert decision.score < 60.0
    assert "SEVERE_LOCAL_LIGHTING" in decision.flags


def test_consumer_smooth_brightness_gradient_is_warning_not_reject() -> None:
    gradient = np.linspace(40, 220, 256, dtype=np.uint8)
    gray = np.repeat(gradient[np.newaxis, :], 256, axis=0)
    image = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    skin = np.full((256, 256), 255, dtype=np.uint8)

    decision = _evaluator()(image, skin, {}, 90.0, "PASS", [])

    assert decision.status == "WARNING"
    assert "LOCAL_LIGHTING_VARIATION" in decision.flags
    assert "SEVERE_LOCAL_LIGHTING" not in decision.flags
    assert decision.local_lighting_block_range > 125.0


def test_consumer_uniform_capture_keeps_existing_quality() -> None:
    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    skin = np.full((256, 256), 255, dtype=np.uint8)

    decision = _evaluator()(image, skin, {}, 88.0, "PASS", ["BLUR"])

    assert decision.status == "PASS"
    assert decision.score == 88.0
    assert decision.flags == ("BLUR",)


def test_forehead_occlusion_keeps_visia_display_but_removes_metrics() -> None:
    landmarks, _ = _front_template()
    image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    skin = np.full((1024, 1024), 255, dtype=np.uint8)
    hair = np.zeros_like(skin)
    hair[:380, :560] = 255
    skin[hair > 0] = 0

    regions = build_visia_regions(
        image,
        skin,
        landmarks,
        ["FOREHEAD_OCCLUDED"],
    )

    assert np.count_nonzero(regions.regions["forehead"]) == 0
    assert np.count_nonzero(regions.display_regions["forehead"]) == 0
    assert np.count_nonzero(regions.regions["left_cheek"]) > 0
    assert np.count_nonzero(regions.regions["right_cheek"]) > 0
    assert regions.display_contour is not None
    assert regions.display_separator is not None
    assert np.count_nonzero(regions.scope_mask) > 0
    rendered = draw_region_boundaries(
        image,
        regions.display_regions,
        closed_contour=regions.display_contour,
        separator_contour=regions.display_separator,
    )
    boundary = np.all(
        rendered == np.asarray(VISIA_BOUNDARY_COLOR, dtype=np.uint8),
        axis=2,
    )
    hair_interior = hair.copy()
    hair_interior[:8] = 0
    hair_interior[:, :8] = 0
    assert np.count_nonzero(boundary & (hair_interior > 0)) > 0


def test_analysis_scope_stays_seven_pixels_away_from_public_visia_lines() -> None:
    landmarks, _ = _front_template()
    image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    skin = np.full((1024, 1024), 255, dtype=np.uint8)

    regions = build_visia_regions(image, skin, landmarks, [])

    public_lines = np.zeros_like(skin)
    cv2.polylines(
        public_lines,
        [np.asarray(regions.display_contour, dtype=np.int32)],
        True,
        255,
        1,
    )
    cv2.polylines(
        public_lines,
        [np.asarray(regions.display_separator, dtype=np.int32)],
        False,
        255,
        1,
    )
    safety_band = cv2.dilate(
        public_lines,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    assert np.count_nonzero(regions.analysis_mask & safety_band) == 0
    assert np.count_nonzero(regions.scope_mask & safety_band) == 0


def test_redness_forehead_domain_ignores_filled_scope_but_avoids_cyan_line() -> None:
    landmarks, _ = _front_template()
    image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    stats = np.full((1024, 1024), 255, dtype=np.uint8)
    regions = build_visia_regions(image, stats, landmarks, [])
    forehead = np.zeros_like(stats)
    forehead[80:360, 280:744] = 255
    # Reproduce the clinic28-09 failure mode: the filled public scope has no
    # usable forehead, while the historical red region remains valid.
    regions.scope_mask[:420] = 0

    feature_regions = ErythemaAnalyzer._red_feature_region_masks(
        {"forehead": forehead},
        regions,
    )

    restored = feature_regions["forehead"]
    assert np.count_nonzero(restored) > 100_000
    assert np.count_nonzero(restored & regions.scope_mask) == 0
    assert np.count_nonzero(
        restored & regions.public_boundary_safety_mask
    ) == 0


def test_consumer_redness_keeps_accepted_filled_scope_domain() -> None:
    landmarks, _ = _front_template()
    image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    stats = np.full((1024, 1024), 255, dtype=np.uint8)
    regions = build_visia_regions(image, stats, landmarks, [])
    regions.scope_mask[:420] = 0

    quantification = ErythemaAnalyzer._public_quantification_mask(stats, regions)

    assert np.array_equal(
        quantification,
        cv2.bitwise_and(stats, regions.scope_mask),
    )
    assert np.count_nonzero(quantification[:420]) == 0


def test_redness_profile_is_explicit_without_loading_real_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.engines.rbx_engine as module

    sentinel = object()
    monkeypatch.setattr(module, "load_face_landmarker", lambda: sentinel)
    monkeypatch.setattr(module, "load_selfie_multiclass_segmenter", lambda: sentinel)
    monkeypatch.setattr(module, "load_selfie_segmenter", lambda: sentinel)

    consumer = ErythemaAnalyzer(capture_profile=CaptureProfile.CONSUMER)
    institution = ErythemaAnalyzer(capture_profile=CaptureProfile.INSTITUTION)

    assert consumer.capture_profile is CaptureProfile.CONSUMER
    assert institution.capture_profile is CaptureProfile.INSTITUTION


def test_consumer_brown_keeps_current_scope_while_institution_restores_forehead() -> None:
    landmarks, _ = _front_template()
    image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    skin = np.full((1024, 1024), 255, dtype=np.uint8)
    current = build_visia_regions(image, skin, landmarks, [])
    current.scope_mask[:420] = 0

    consumer = BrownAreaAnalyzer(capture_profile=CaptureProfile.CONSUMER)
    institution = BrownAreaAnalyzer(capture_profile=CaptureProfile.INSTITUTION)
    consumer_regions = consumer._regions_for_profile(skin, landmarks, current)
    institution_regions = institution._regions_for_profile(skin, landmarks, current)

    assert consumer_regions is current
    assert np.count_nonzero(consumer_regions.scope_mask[:420]) == 0
    assert institution_regions is not current
    assert np.count_nonzero(institution_regions.regions["forehead"]) > 100_000


def test_brown_forehead_domain_ignores_filled_scope_but_markers_avoid_line() -> None:
    landmarks, _ = _front_template()
    image = np.full((1024, 1024, 3), 128, dtype=np.uint8)
    skin = np.full((1024, 1024), 255, dtype=np.uint8)
    current = build_visia_regions(image, skin, landmarks, [])
    current.scope_mask[:420] = 0

    restored = BrownAreaAnalyzer._historical_analysis_regions(
        skin,
        landmarks,
        current,
    )
    marker_exclusion = BrownAreaAnalyzer._marker_exclusion(
        restored.feature_exclusion_mask,
        restored,
    )

    assert np.count_nonzero(restored.regions["forehead"]) > 100_000
    assert np.count_nonzero(
        restored.regions["forehead"] & current.scope_mask
    ) == 0
    assert np.count_nonzero(
        marker_exclusion & restored.public_boundary_safety_mask
    ) == np.count_nonzero(restored.public_boundary_safety_mask)
