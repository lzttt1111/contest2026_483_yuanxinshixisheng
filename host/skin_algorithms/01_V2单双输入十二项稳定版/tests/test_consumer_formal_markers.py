from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np

from src.capture_profile import CaptureProfile
from src.consumer_pigment.formal_markers import ConsumerBrownRecallPreset
from src.engines.brown_engine import BrownAreaAnalyzer
from src.engines.purple_analysis_engine import PurpleAnalysisEngine
from src.engines.visia_regions import REGION_ORDER, VisiaRegionSet
from src.preprocess.analysis_mask_bundle import (
    build_analysis_mask_bundle,
    expand_common_hair_masks,
)


def _regions(shape: tuple[int, int]) -> VisiaRegionSet:
    analysis = np.full(shape, 255, dtype=np.uint8)
    empty = np.zeros(shape, dtype=np.uint8)
    regions = {name: empty.copy() for name in REGION_ORDER}
    regions["left_cheek"][:, : shape[1] // 2] = 255
    regions["right_cheek"][:, shape[1] // 2 :] = 255
    return VisiaRegionSet(
        analysis_mask=analysis,
        feature_exclusion_mask=empty,
        regions=regions,
        display_regions={name: mask.copy() for name, mask in regions.items()},
        partial_face=False,
    )


def test_consumer_formal_marker_policies_are_kind_specific() -> None:
    module = importlib.import_module("src.consumer_pigment.formal_markers")
    kind_type = getattr(module, "ConsumerPigmentMarkerKind", None)
    resolver = getattr(module, "policy_for_consumer_marker", None)
    assert kind_type is not None and callable(resolver)

    policies = [resolver(kind) for kind in kind_type]

    assert [item.minimum_peak_score for item in policies] == [0.50, 0.90]
    assert [item.minimum_distance_px for item in policies] == [12, 24]
    assert [item.minimum_local_prominence_z for item in policies] == [8.0, 0.0]


def test_consumer_brown_recall_presets_only_relax_prominence() -> None:
    module = importlib.import_module("src.consumer_pigment.formal_markers")
    preset_type = getattr(module, "ConsumerBrownRecallPreset")
    resolver = getattr(module, "policy_for_consumer_brown_recall")

    policies = [resolver(preset) for preset in preset_type]

    assert [item.minimum_local_prominence_z for item in policies] == [
        8.0, 6.5, 5.0, 3.5, 2.0,
    ]
    assert {item.minimum_peak_score for item in policies} == {0.50}
    assert [item.minimum_distance_px for item in policies] == [12, 15, 15, 15, 15]
    assert [item.minimum_marker_edge_gap_px for item in policies] == [0, 0, 0, 0, 0]
    assert [item.large_component_prominence_relief for item in policies] == [
        0.0, 0.35, 0.35, 0.35, 0.35,
    ]
    assert [item.minimum_marker_radius_px for item in policies] == [2, 3, 3, 3, 3]
    assert [item.maximum_marker_radius_px for item in policies] == [7, 5, 5, 5, 5]


def test_consumer_brown_large_component_gets_area_support() -> None:
    module = importlib.import_module("src.consumer_pigment.formal_markers")
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[32:47, 32:47] = 255
    score = np.full(mask.shape, 0.66, dtype=np.float32)
    score[32:47, 32:47] = 0.75
    valid = np.full(mask.shape, 255, dtype=np.uint8)
    marker_input = module.ConsumerMarkerInput(
        mask,
        score,
        valid,
        np.zeros_like(mask),
        _regions(mask.shape),
    )

    current = module.project_consumer_markers(
        marker_input,
        module.ConsumerPigmentMarkerKind.BROWN,
        policy=module.policy_for_consumer_brown_recall(
            module.ConsumerBrownRecallPreset.CURRENT
        ),
    )
    recall = module.project_consumer_markers(
        marker_input,
        module.ConsumerPigmentMarkerKind.BROWN,
        policy=module.policy_for_consumer_brown_recall(
            module.ConsumerBrownRecallPreset.RECALL_1
        ),
    )

    assert not current.findings
    assert len(recall.findings) == 1
    assert recall.findings[0].marker_radius_px == 5


def test_consumer_brown_15px_centres_keep_visible_edge_gap() -> None:
    module = importlib.import_module("src.consumer_pigment.formal_markers")
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[24:37, 24:36] = 255
    mask[24:37, 39:51] = 255
    score = np.zeros(mask.shape, dtype=np.float32)
    score[30, 30] = 0.99
    score[30, 45] = 0.98
    valid = np.full(mask.shape, 255, dtype=np.uint8)

    result = module.project_consumer_markers(
        module.ConsumerMarkerInput(
            mask,
            score,
            valid,
            np.zeros_like(mask),
            _regions(mask.shape),
        ),
        module.ConsumerPigmentMarkerKind.BROWN,
        policy=module.policy_for_consumer_brown_recall(
            module.ConsumerBrownRecallPreset.RECALL_4
        ),
    )

    assert len(result.findings) == 2
    assert result.rejected_nms == 0
    first, second = result.findings
    assert first.marker_radius_px == second.marker_radius_px == 5
    assert abs(first.centroid[0] - second.centroid[0]) == 15


def test_brown_projection_rejects_weak_local_contrast_without_widening_spacing() -> None:
    module = importlib.import_module("src.consumer_pigment.formal_markers")
    kind_type = getattr(module, "ConsumerPigmentMarkerKind")
    input_type = getattr(module, "ConsumerMarkerInput")
    project = getattr(module, "project_consumer_markers")
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[20:27, 20:27] = 255
    mask[60:67, 60:67] = 255
    score = np.full(mask.shape, 0.10, dtype=np.float32)
    score[3:38, 3:38] = 0.88
    score[20:27, 20:27] = 0.89
    score[60:67, 60:67] = 0.90
    valid = np.full(mask.shape, 255, dtype=np.uint8)

    result = project(
        input_type(mask, score, valid, np.zeros_like(mask), _regions(mask.shape)),
        kind_type.BROWN,
    )

    assert [finding.centroid for finding in result.findings] == [(60, 60)]


def test_brown_projection_preserves_distinct_peaks_at_red_spacing() -> None:
    module = importlib.import_module("src.consumer_pigment.formal_markers")
    kind_type = getattr(module, "ConsumerPigmentMarkerKind")
    input_type = getattr(module, "ConsumerMarkerInput")
    project = getattr(module, "project_consumer_markers")
    mask = np.zeros((128, 128), dtype=np.uint8)
    score = np.zeros(mask.shape, dtype=np.float32)
    peaks = (
        (20, 20, 0.99),
        (36, 20, 0.51),
        (68, 20, 0.90),
        (84, 20, 0.60),
        (20, 68, 0.80),
        (36, 68, 0.70),
    )
    for x, y, value in peaks:
        mask[y - 3:y + 4, x - 3:x + 4] = 255
        score[y, x] = value
    valid = np.full(mask.shape, 255, dtype=np.uint8)

    result = project(
        input_type(mask, score, valid, np.zeros_like(mask), _regions(mask.shape)),
        kind_type.BROWN,
    )

    assert len(result.findings) == 6
    assert all(2 <= finding.marker_radius_px <= 7 for finding in result.findings)


def test_uv_spot_marker_uses_high_contrast_gold() -> None:
    from src.consumer_pigment.markers import UV_STYLE_MARKER

    assert UV_STYLE_MARKER.fill_bgr == (0, 199, 255)
    assert UV_STYLE_MARKER.outline_bgr == (0, 199, 255)


def test_brown_analyzer_profile_defaults_to_institution() -> None:
    institution = BrownAreaAnalyzer()
    consumer = BrownAreaAnalyzer(capture_profile=CaptureProfile.CONSUMER)

    assert institution.capture_profile is CaptureProfile.INSTITUTION
    assert consumer.capture_profile is CaptureProfile.CONSUMER
    assert consumer.consumer_brown_recall_preset is ConsumerBrownRecallPreset.RECALL_3


def test_brown_profiles_require_four_scale_support() -> None:
    module = importlib.import_module(
        "src.consumer_pigment.brown_detection_policy"
    )
    resolver = getattr(module, "policy_for_brown_profile", None)
    assert callable(resolver)

    institution = resolver(CaptureProfile.INSTITUTION)
    consumer = resolver(CaptureProfile.CONSUMER)

    assert institution.minimum_scale_votes == 4
    assert institution.candidate_z_threshold == 2.0
    assert institution.minimum_area_px == 14
    assert institution.minimum_mean_score == 0.50
    assert consumer.minimum_scale_votes == 4
    assert consumer.candidate_z_threshold == 1.40
    assert consumer.minimum_raw_response == 0.10
    assert consumer.minimum_area_px == 4
    assert consumer.minimum_mean_score == 0.16
    assert consumer.peak_min_distance == 6


def test_consumer_brown_front_recall_presets_do_not_change_institution() -> None:
    module = importlib.import_module(
        "src.consumer_pigment.brown_detection_policy"
    )
    resolver = getattr(module, "policy_for_consumer_brown_detection_recall")
    institution = module.policy_for_brown_profile(CaptureProfile.INSTITUTION)

    policies = [resolver(preset) for preset in ConsumerBrownRecallPreset]

    assert [item.minimum_scale_votes for item in policies] == [4, 4, 3, 3, 3]
    assert [item.candidate_z_threshold for item in policies] == [
        1.40, 1.22, 1.10, 0.95, 0.80,
    ]
    assert [item.minimum_raw_response for item in policies] == [
        0.10, 0.085, 0.075, 0.060, 0.045,
    ]
    assert institution.minimum_scale_votes == 4
    assert institution.candidate_z_threshold == 2.0
    assert institution.minimum_raw_response == 0.18


def test_consumer_brown_projection_keeps_moderate_supported_peak() -> None:
    module = importlib.import_module("src.consumer_pigment.formal_markers")
    kind_type = getattr(module, "ConsumerPigmentMarkerKind")
    input_type = getattr(module, "ConsumerMarkerInput")
    project = getattr(module, "project_consumer_markers")
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:31, 24:31] = 255
    score = np.zeros(mask.shape, dtype=np.float32)
    score[27, 27] = 0.55
    valid = np.full(mask.shape, 255, dtype=np.uint8)

    result = project(
        input_type(mask, score, valid, np.zeros_like(mask), _regions(mask.shape)),
        kind_type.BROWN,
    )

    assert len(result.findings) == 1


def test_purple_analyzer_profile_defaults_to_institution() -> None:
    institution = PurpleAnalysisEngine()
    consumer = PurpleAnalysisEngine(capture_profile=CaptureProfile.CONSUMER)

    assert institution.capture_profile is CaptureProfile.INSTITUTION
    assert consumer.capture_profile is CaptureProfile.CONSUMER


def test_consumer_porphyrin_keeps_original_detection_with_smaller_display() -> None:
    institution = PurpleAnalysisEngine()
    consumer = PurpleAnalysisEngine(capture_profile=CaptureProfile.CONSUMER)

    assert institution.fluorescence_porphyrin_detector.config.display_min_radius_px == 2
    assert institution.fluorescence_porphyrin_detector.config.display_max_radius_px == 4
    assert consumer.fluorescence_porphyrin_detector.config.display_min_radius_px == 1
    assert consumer.fluorescence_porphyrin_detector.config.display_max_radius_px == 1
    assert consumer.fluorescence_porphyrin_detector.config.peak_threshold == (
        institution.fluorescence_porphyrin_detector.config.peak_threshold
    )


def test_consumer_brown_projection_accepts_balanced_recall_and_rejects_noise() -> None:
    module = importlib.import_module("src.consumer_pigment.formal_markers")
    kind_type = getattr(module, "ConsumerPigmentMarkerKind", None)
    input_type = getattr(module, "ConsumerMarkerInput", None)
    project = getattr(module, "project_consumer_markers", None)
    assert kind_type is not None and input_type is not None and callable(project)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[20:27, 20:27] = 255
    mask[29:36, 29:36] = 255
    mask[62:69, 62:69] = 255
    mask[76:83, 18:25] = 255
    score = np.zeros(mask.shape, dtype=np.float32)
    score[23, 23] = 0.96
    score[32, 32] = 0.91
    score[65, 65] = 0.95
    score[79, 21] = 0.80
    valid = np.full(mask.shape, 255, dtype=np.uint8)
    exclusion = np.zeros_like(mask)
    exclusion[58:73, 58:73] = 255

    result = project(
        input_type(mask, score, valid, exclusion, _regions(mask.shape)),
        kind_type.BROWN,
    )

    assert len(result.findings) == 3
    assert result.findings[0].centroid == (23, 23)
    assert result.findings[0].region == "left_cheek"
    assert result.findings[0].marker_radius_px == 4
    assert result.rejected_low_score == 0
    assert result.rejected_exclusion == 1
    assert result.rejected_nms == 0
    assert np.count_nonzero(result.marker_mask & exclusion) == 0


def test_consumer_brown_analysis_mask_consumes_common_hair_bundle() -> None:
    skin = np.full((64, 64), 255, dtype=np.uint8)
    hair = np.zeros_like(skin)
    hair[8:56, 30:34] = 255
    empty = np.zeros_like(skin)
    common_hair, common_facial = expand_common_hair_masks(hair, empty)
    bundle = build_analysis_mask_bundle(
        valid_skin_mask=skin,
        hair_mask=common_hair,
        facial_hair_mask=common_facial,
        feature_exclusion_mask=empty,
        nostril_mask=empty,
        display_face_mask=skin,
        coordinate_space="test",
    )
    analyzer = object.__new__(BrownAreaAnalyzer)
    analyzer.capture_profile = CaptureProfile.CONSUMER
    preprocess = SimpleNamespace(skin_mask=skin, mask_bundle=bundle)

    result = analyzer._profile_analysis_skin_mask(preprocess)

    assert np.count_nonzero(result[common_hair > 0]) == 0
    assert np.count_nonzero(result[common_hair == 0]) > 0


def test_consumer_marker_projection_uses_circular_marker_as_formal_area() -> None:
    module = importlib.import_module("src.consumer_pigment.formal_markers")
    kind_type = getattr(module, "ConsumerPigmentMarkerKind", None)
    input_type = getattr(module, "ConsumerMarkerInput", None)
    project = getattr(module, "project_consumer_markers", None)
    assert kind_type is not None and input_type is not None and callable(project)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[18:39, 18:39] = 255
    score = np.zeros(mask.shape, dtype=np.float32)
    score[28, 28] = 0.98
    valid = np.full(mask.shape, 255, dtype=np.uint8)

    result = project(
        input_type(
            mask,
            score,
            valid,
            np.zeros_like(mask),
            _regions(mask.shape),
        ),
        kind_type.UV_SPOTS,
    )

    assert len(result.findings) == 1
    assert result.findings[0].marker_area_px == np.count_nonzero(result.marker_mask)
    assert 8 <= result.findings[0].marker_radius_px <= 12
    assert result.findings[0].bbox[2] <= 25
    assert result.findings[0].bbox[3] <= 25
