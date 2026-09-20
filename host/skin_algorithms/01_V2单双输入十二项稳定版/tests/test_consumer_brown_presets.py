from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest

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


def test_brown_presets_have_ordered_precision_contract() -> None:
    module = importlib.import_module("src.consumer_pigment.brown_presets")
    resolver = getattr(module, "policy_for_brown_preset", None)
    preset = getattr(module, "ConsumerFindingPreset", None)
    assert callable(resolver)
    assert preset is not None

    soft = resolver(preset.SOFT)
    balanced = resolver(preset.BALANCED)
    enhanced = resolver(preset.ENHANCED)

    assert (soft.minimum_scale_votes, soft.score_percentile, soft.nms_distance) == (
        4,
        90.0,
        18,
    )
    assert (
        balanced.minimum_scale_votes,
        balanced.score_percentile,
        balanced.nms_distance,
    ) == (3, 75.0, 12)
    assert (
        enhanced.minimum_scale_votes,
        enhanced.score_percentile,
        enhanced.nms_distance,
    ) == (2, 60.0, 8)


def test_brown_selector_rejects_specular_peak_and_uses_compact_marker() -> None:
    module = importlib.import_module("src.consumer_pigment.brown_presets")
    selector = getattr(module, "select_brown_findings", None)
    evidence_type = getattr(module, "BrownFindingEvidence", None)
    preset = getattr(module, "ConsumerFindingPreset", None)
    assert callable(selector)
    assert evidence_type is not None and preset is not None

    score = np.zeros((96, 96), dtype=np.float32)
    score[28:35, 28:35] = 0.93
    score[61:70, 61:70] = 0.96
    votes = np.full((96, 96), 4, dtype=np.uint8)
    valid = np.full((96, 96), 255, dtype=np.uint8)
    specular = np.zeros((96, 96), dtype=np.uint8)
    specular[58:73, 58:73] = 255
    evidence = evidence_type(score, votes, valid, specular)

    result = selector(evidence, _regions(score.shape), preset.SOFT)

    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.region == "left_cheek"
    assert 2 <= finding.marker_radius_px <= 7
    assert result.rejected_specular >= 1
    assert np.count_nonzero(result.marker_mask & specular) == 0


def test_brown_selector_warns_on_density_without_truncating_findings() -> None:
    module = importlib.import_module("src.consumer_pigment.brown_presets")
    selector = getattr(module, "select_brown_findings", None)
    evidence_type = getattr(module, "BrownFindingEvidence", None)
    preset = getattr(module, "ConsumerFindingPreset", None)
    assert callable(selector)
    assert evidence_type is not None and preset is not None

    score = np.zeros((128, 128), dtype=np.float32)
    for y in range(8, 121, 12):
        for x in range(8, 121, 12):
            score[y, x] = 0.95
    votes = np.full(score.shape, 4, dtype=np.uint8)
    valid = np.full(score.shape, 255, dtype=np.uint8)
    evidence = evidence_type(score, votes, valid, np.zeros_like(valid))

    result = selector(evidence, _regions(score.shape), preset.ENHANCED)

    assert len(result.findings) > 20
    assert result.density_status == "WARNING"


def test_shared_specular_mask_detects_low_saturation_highlight() -> None:
    module = importlib.import_module("src.consumer_pigment.specular")
    builder = getattr(module, "build_specular_evidence", None)
    assert callable(builder)
    image = np.full((96, 96, 3), (90, 115, 145), dtype=np.uint8)
    image[36:60, 36:60] = 250
    valid = np.full((96, 96), 255, dtype=np.uint8)

    evidence = builder(image, valid)

    assert np.count_nonzero(evidence.mask[40:56, 40:56]) > 0
    assert np.count_nonzero(evidence.mask[:20, :20]) == 0


def test_highlight_compression_reduces_glare_without_flat_fill() -> None:
    module = importlib.import_module("src.consumer_pigment.specular")
    builder = getattr(module, "build_specular_evidence", None)
    compressor = getattr(module, "compress_specular_highlights", None)
    assert callable(builder) and callable(compressor)
    image = np.full((96, 96, 3), (70, 90, 130), dtype=np.uint8)
    image[36:60, 36:60] = (248, 248, 248)
    valid = np.full((96, 96), 255, dtype=np.uint8)
    evidence = builder(image, valid)

    compressed = compressor(image, evidence)

    before = float(np.mean(image[42:54]))
    after = float(np.mean(compressed[42:54]))
    surround = float(np.mean(compressed[24:32, 24:32]))
    assert after < before - 20.0
    assert after > surround
    assert np.std(compressed[36:60, 36:60]) > 0.0


def test_compact_marker_renderer_uses_one_uniform_style() -> None:
    module = importlib.import_module("src.consumer_pigment.markers")
    renderer = getattr(module, "render_compact_marker_mask", None)
    style_type = getattr(module, "CompactMarkerStyle", None)
    assert callable(renderer) and style_type is not None
    base = np.full((64, 64, 3), 100, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:25, 20:25] = 255
    mask[40:45, 40:45] = 255
    style = style_type((190, 220, 55), (110, 150, 25), 0.90)

    rendered = renderer(base, mask, style)

    assert np.array_equal(rendered[22, 22], np.asarray((181, 208, 59)))
    assert np.array_equal(rendered[42, 42], rendered[22, 22])


def test_compact_marker_renderer_accepts_saved_three_channel_mask() -> None:
    module = importlib.import_module("src.consumer_pigment.markers")
    renderer = getattr(module, "render_compact_marker_mask")
    style = getattr(module, "BROWN_STYLE_MARKER")
    base = np.full((32, 32, 3), 100, dtype=np.uint8)
    mask = np.zeros((32, 32, 3), dtype=np.uint8)
    mask[12:17, 12:17, :] = 255

    rendered = renderer(base, mask, style)

    assert np.array_equal(rendered[14, 14], np.asarray(style.fill_bgr))


def test_consumer_marker_styles_are_module_specific_and_high_contrast() -> None:
    marker_module = importlib.import_module("src.consumer_pigment.markers")
    redness_module = importlib.import_module("src.engines.rbx_engine")
    brown = getattr(marker_module, "BROWN_STYLE_MARKER", None)
    uv = getattr(marker_module, "UV_STYLE_MARKER", None)
    porphyrin = getattr(marker_module, "PORPHYRIN_STYLE_MARKER", None)

    assert brown is not None and uv is not None and porphyrin is not None
    expected = redness_module.Config.RED_FEATURE_MARKER_COLOR
    assert brown.fill_bgr == (255, 220, 35)
    assert uv.fill_bgr == (0, 199, 255)
    assert uv.outline_bgr == uv.fill_bgr
    assert porphyrin.fill_bgr == expected
    assert len({brown.fill_bgr, uv.fill_bgr, porphyrin.fill_bgr}) == 3
    assert all(style.fill_alpha == 1.0 for style in (brown, uv, porphyrin))


def test_existing_candidate_consolidation_uses_strongest_compact_circle() -> None:
    module = importlib.import_module("src.consumer_pigment.markers")
    consolidate = getattr(module, "consolidate_existing_marker_mask", None)
    assert callable(consolidate)
    candidate = np.zeros((96, 96), dtype=np.uint8)
    candidate[28:37, 28:37] = 255
    candidate[38:48, 38:48] = 255
    candidate[68:77, 68:77] = 255
    score = np.zeros(candidate.shape, dtype=np.float32)
    score[32, 32] = 0.94
    score[42, 42] = 0.88
    score[72, 72] = 0.97
    valid = np.full(candidate.shape, 255, dtype=np.uint8)
    exclusion = np.zeros_like(candidate)
    exclusion[64:82, 64:82] = 255

    result = consolidate(
        candidate,
        score,
        valid,
        exclusion,
        minimum_peak_score=0.85,
        minimum_distance_px=18,
    )

    assert len(result.findings) == 1
    assert result.findings[0].centroid == (32, 32)
    assert 2 <= result.findings[0].marker_radius_px <= 7
    assert result.rejected_nms == 1
    assert result.rejected_exclusion == 1
    assert np.count_nonzero(result.marker_mask & exclusion) == 0


def test_brown_evidence_keeps_dark_brown_peak_and_masks_highlight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("src.consumer_pigment.brown_evidence")
    builder = getattr(module, "build_brown_finding_evidence", None)
    assert callable(builder)

    class CpuGaussianBackend:
        def masked_gaussian_batch(
            self,
            channels: tuple[np.ndarray, ...],
            mask: np.ndarray,
            sigmas: tuple[float, ...],
        ) -> dict[float, tuple[np.ndarray, ...]]:
            del mask
            return {
                float(sigma): tuple(
                    cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma)
                    for channel in channels
                )
                for sigma in sigmas
            }

    monkeypatch.setattr(
        module,
        "get_cuda_backend",
        lambda: CpuGaussianBackend(),
    )
    image = np.full((128, 128, 3), (105, 125, 155), dtype=np.uint8)
    image[44:55, 44:55] = (40, 65, 100)
    image[76:92, 76:92] = 250
    regions = _regions(image.shape[:2])

    evidence = builder(image, regions)

    assert evidence.score_map[49, 49] > evidence.score_map[20, 20]
    assert evidence.scale_votes[49, 49] >= 2
    assert evidence.specular_mask[82, 82] > 0


def test_brown_palettes_preserve_mother_image_and_compress_highlight() -> None:
    module = importlib.import_module("src.consumer_pigment.brown_render")
    renderer = getattr(module, "render_consumer_brown_base", None)
    palette_type = getattr(module, "ConsumerBrownPalette", None)
    input_type = getattr(module, "BrownRenderInput", None)
    assert callable(renderer) and palette_type is not None and input_type is not None
    source = np.full((96, 96, 3), (85, 110, 150), dtype=np.uint8)
    source[38:58, 38:58] = 250
    mother = np.full((96, 96, 3), (82, 118, 188), dtype=np.uint8)
    mother[38:58, 38:58] = 245
    mask = np.full((96, 96), 255, dtype=np.uint8)
    render_input = input_type(source, mother, mask)

    outputs = [
        renderer(render_input, palette)
        for palette in palette_type
    ]

    assert len(outputs) == 3
    assert all(output.shape == source.shape for output in outputs)
    assert len({output.tobytes() for output in outputs}) == 3
    assert all(float(np.mean(output[42:54, 42:54])) < 235.0 for output in outputs)
    reference = renderer(render_input, palette_type.REFERENCE)
    np.testing.assert_array_equal(reference[10:20, 10:20], mother[10:20, 10:20])
    np.testing.assert_array_equal(source, render_input.source_image)
    np.testing.assert_array_equal(mother, render_input.mother_base)
