from __future__ import annotations

import importlib

import numpy as np


def test_common_mask_bundle_excludes_hair_features_and_extra_masks() -> None:
    module = importlib.import_module("src.preprocess.image_preprocessor")
    builder = getattr(module, "build_analysis_mask_bundle", None)
    assert callable(builder), "公共AnalysisMaskBundle构建器尚未实现"
    base = np.full((12, 12), 255, dtype=np.uint8)
    hair = np.zeros_like(base)
    hair[:, :3] = 255
    features = np.zeros_like(base)
    features[4:8, 4:8] = 255
    extra = np.zeros_like(base)
    extra[9:, 9:] = 255

    bundle = builder(
        valid_skin_mask=base,
        hair_mask=hair,
        facial_hair_mask=np.zeros_like(base),
        feature_exclusion_mask=features,
        nostril_mask=np.zeros_like(base),
        display_face_mask=base,
        coordinate_space="test_12",
    )
    final_mask = bundle.algorithm_mask(extra)

    assert np.count_nonzero(final_mask[:, :3]) == 0
    assert np.count_nonzero(final_mask[4:8, 4:8]) == 0
    assert np.count_nonzero(final_mask[9:, 9:]) == 0
    assert np.count_nonzero(final_mask & hair) == 0
    assert np.all(final_mask[bundle.valid_skin_mask == 0] == 0)


def test_common_mask_bundle_has_stable_private_evidence_and_readonly_arrays() -> None:
    module = importlib.import_module("src.preprocess.image_preprocessor")
    builder = getattr(module, "build_analysis_mask_bundle", None)
    assert callable(builder), "公共AnalysisMaskBundle构建器尚未实现"
    base = np.full((8, 8), 255, dtype=np.uint8)
    empty = np.zeros_like(base)

    first = builder(
        valid_skin_mask=base,
        hair_mask=empty,
        facial_hair_mask=empty,
        feature_exclusion_mask=empty,
        nostril_mask=empty,
        display_face_mask=base,
        coordinate_space="aligned_8",
    )
    second = builder(
        valid_skin_mask=base,
        hair_mask=empty,
        facial_hair_mask=empty,
        feature_exclusion_mask=empty,
        nostril_mask=empty,
        display_face_mask=base,
        coordinate_space="aligned_8",
    )

    assert first.evidence() == second.evidence()
    assert first.evidence()["mask_contract_version"] == "analysis_mask_bundle_v1"
    assert first.valid_skin_mask.flags.writeable is False


def test_common_hair_safety_margin_expands_thin_strands_by_seven_pixels() -> None:
    module = importlib.import_module("src.preprocess.analysis_mask_bundle")
    expand = getattr(module, "expand_common_hair_masks", None)
    assert callable(expand)
    hair = np.zeros((64, 64), dtype=np.uint8)
    hair[8:56, 31:33] = 255
    empty = np.zeros_like(hair)

    expanded_hair, expanded_facial = expand(hair, empty)

    assert np.count_nonzero(expanded_hair[:, 24:40]) > np.count_nonzero(hair)
    assert np.count_nonzero(expanded_hair[:, :20]) == 0
    assert np.count_nonzero(expanded_facial) == 0
