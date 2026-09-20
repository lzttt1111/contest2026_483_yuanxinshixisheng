from __future__ import annotations

import importlib
from dataclasses import replace

import numpy as np

from src.engines.visia_regions import REGION_ORDER, VisiaRegionSet


def _regions(shape: tuple[int, int]) -> VisiaRegionSet:
    mask = np.full(shape, 255, dtype=np.uint8)
    empty = np.zeros(shape, dtype=np.uint8)
    regions = {name: empty.copy() for name in REGION_ORDER}
    regions["left_cheek"][:, : shape[1] // 2] = 255
    regions["right_cheek"][:, shape[1] // 2 :] = 255
    return VisiaRegionSet(
        analysis_mask=mask,
        feature_exclusion_mask=empty,
        regions=regions,
        display_regions={name: value.copy() for name, value in regions.items()},
        partial_face=False,
    )


def test_purple_presets_have_ordered_precision_contract() -> None:
    module = importlib.import_module("src.consumer_pigment.purple_presets")
    resolver = getattr(module, "policy_for_purple_preset", None)
    preset_type = getattr(module, "ConsumerFindingPreset", None)
    assert callable(resolver) and preset_type is not None

    policies = [resolver(preset) for preset in preset_type]

    assert [policy.minimum_scale_votes for policy in policies] == [4, 3, 2]
    assert [policy.score_percentile for policy in policies] == [90.0, 75.0, 60.0]
    assert [policy.nms_distance for policy in policies] == [18, 12, 8]


def test_uv_selector_requires_support_and_rejects_suppression() -> None:
    module = importlib.import_module("src.consumer_pigment.purple_presets")
    selector = getattr(module, "select_purple_findings", None)
    evidence_type = getattr(module, "PurpleFindingEvidence", None)
    kind_type = getattr(module, "PurpleFindingKind", None)
    preset_type = getattr(module, "ConsumerFindingPreset", None)
    assert callable(selector)
    assert evidence_type is not None and kind_type is not None and preset_type is not None
    score = np.zeros((96, 96), dtype=np.float32)
    score[22:29, 22:29] = 0.95
    score[46:53, 46:53] = 0.94
    score[70:77, 70:77] = 0.96
    votes = np.full(score.shape, 4, dtype=np.uint8)
    valid = np.full(score.shape, 255, dtype=np.uint8)
    support = np.zeros_like(valid)
    support[18:33, 18:33] = 255
    support[66:81, 66:81] = 255
    suppression = np.zeros_like(valid)
    suppression[66:81, 66:81] = 255
    evidence = evidence_type(
        kind_type.UV_SPOTS,
        score,
        votes,
        valid,
        support,
        suppression,
    )

    result = selector(evidence, _regions(score.shape), preset_type.SOFT)

    assert len(result.findings) == 1
    assert result.findings[0].region == "left_cheek"
    assert result.rejected_support >= 1
    assert result.rejected_suppression >= 1


def test_porphyrin_selector_uses_multiscale_evidence_without_uv_support() -> None:
    module = importlib.import_module("src.consumer_pigment.purple_presets")
    selector = getattr(module, "select_purple_findings", None)
    evidence_type = getattr(module, "PurpleFindingEvidence", None)
    kind_type = getattr(module, "PurpleFindingKind", None)
    preset_type = getattr(module, "ConsumerFindingPreset", None)
    assert callable(selector)
    assert evidence_type is not None and kind_type is not None and preset_type is not None
    score = np.zeros((96, 96), dtype=np.float32)
    score[42:49, 42:49] = 0.93
    votes = np.full(score.shape, 4, dtype=np.uint8)
    valid = np.full(score.shape, 255, dtype=np.uint8)
    empty = np.zeros_like(valid)
    evidence = evidence_type(
        kind_type.PORPHYRIN,
        score,
        votes,
        valid,
        empty,
        empty,
    )

    result = selector(evidence, _regions(score.shape), preset_type.SOFT)

    assert len(result.findings) == 1
    assert 2 <= result.findings[0].marker_radius_px <= 7


def test_purple_palettes_are_small_variants_of_existing_mother_bases() -> None:
    module = importlib.import_module("src.consumer_pigment.purple_render")
    renderer = getattr(module, "render_consumer_purple_palette", None)
    input_type = getattr(module, "PurpleRenderInput", None)
    palette_type = getattr(module, "ConsumerPurplePalette", None)
    assert callable(renderer) and input_type is not None and palette_type is not None
    uv_mother = np.full((96, 96, 3), (145, 150, 155), dtype=np.uint8)
    porphyrin_mother = np.full((96, 96, 3), (118, 48, 18), dtype=np.uint8)
    uv_original = uv_mother.copy()
    porphyrin_original = porphyrin_mother.copy()
    render_input = input_type(uv_mother, porphyrin_mother)

    results = [renderer(render_input, palette) for palette in palette_type]

    assert len(results) == 3
    assert len({result.porphyrin_base.tobytes() for result in results}) == 3
    assert len({result.uv_base.tobytes() for result in results}) == 3
    reference = renderer(render_input, palette_type.REFERENCE)
    np.testing.assert_array_equal(reference.uv_base, uv_mother)
    np.testing.assert_array_equal(reference.porphyrin_base, porphyrin_mother)
    np.testing.assert_array_equal(uv_mother, uv_original)
    np.testing.assert_array_equal(porphyrin_mother, porphyrin_original)


def test_display_palette_never_changes_frozen_detection_bases() -> None:
    from src.engines.purple_base_renderer import (
        PurpleBaseStyleConfig,
        PurpleBaseStyleRenderer,
    )

    image = np.full((96, 96, 3), (80, 110, 145), dtype=np.uint8)
    yy, xx = np.indices((96, 96))
    image[:, :, 0] = np.clip(image[:, :, 0] + xx // 3, 0, 255)
    foreground = np.full((96, 96), 255, dtype=np.uint8)
    skin = np.zeros((96, 96), dtype=np.uint8)
    skin[16:82, 18:78] = 255
    base = PurpleBaseStyleConfig()
    alternative = replace(
        base,
        uv_black_level=0.20,
        uv_white_level=0.60,
        porphyrin_mid_bgr=(120, 70, 30),
    )

    reference = PurpleBaseStyleRenderer(base).render(image, foreground, skin)
    changed = PurpleBaseStyleRenderer(alternative).render(image, foreground, skin)

    assert not np.array_equal(reference.uv_spots_base, changed.uv_spots_base)
    assert not np.array_equal(reference.porphyrin_base, changed.porphyrin_base)
    np.testing.assert_array_equal(
        reference.uv_spots_detection_base,
        changed.uv_spots_detection_base,
    )
    np.testing.assert_array_equal(
        reference.porphyrin_detection_base,
        changed.porphyrin_detection_base,
    )
