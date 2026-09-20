from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from src.capture_profile import CaptureProfile
from src.preprocess.image_preprocessor import _plausible_strand_component
from src.preprocess.profile_mask_policy import ProfileMaskCoverageError


def test_consumer_profile_applies_shared_hair_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocess_package = importlib.import_module("src.preprocess")
    policy = getattr(preprocess_package, "apply_profile_mask_policy", None)
    assert callable(policy), "跨resident公共Mask策略尚未实现"
    policy_module = importlib.import_module("src.preprocess.profile_mask_policy")
    base = np.full((64, 64), 255, dtype=np.uint8)
    hair = np.zeros_like(base)
    hair[:, :4] = 255
    empty = np.zeros_like(base)
    monkeypatch.setattr(
        policy_module,
        "detect_visible_hair_masks",
        lambda *_args: {
            "hair_mask": hair,
            "facial_hair_mask": empty,
        },
    )

    applied = policy(
        CaptureProfile.CONSUMER,
        np.full((64, 64, 3), 120, dtype=np.uint8),
        base,
        empty,
        np.zeros((478, 2), dtype=np.float32),
        "consumer_64",
    )

    assert np.count_nonzero(applied.skin_mask[:, :4]) == 0
    assert np.count_nonzero(applied.skin_mask[:, 4:11]) == 0
    assert np.count_nonzero(applied.forbidden_mask[:, :11]) == 64 * 11
    assert np.count_nonzero(applied.skin_mask[:, 14:]) > 0
    assert applied.bundle.evidence()["coordinate_space"] == "consumer_64"


def test_institution_profile_keeps_existing_masks_without_hair_redetection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocess_package = importlib.import_module("src.preprocess")
    policy = getattr(preprocess_package, "apply_profile_mask_policy", None)
    assert callable(policy), "跨resident公共Mask策略尚未实现"
    policy_module = importlib.import_module("src.preprocess.profile_mask_policy")
    monkeypatch.setattr(
        policy_module,
        "detect_visible_hair_masks",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("institution must not change historical masks")
        ),
    )
    base = np.full((8, 8), 255, dtype=np.uint8)
    forbidden = np.zeros_like(base)
    forbidden[0, 0] = 255

    applied = policy(
        CaptureProfile.INSTITUTION,
        np.full((8, 8, 3), 120, dtype=np.uint8),
        base,
        forbidden,
        np.zeros((478, 2), dtype=np.float32),
        "institution_8",
    )

    assert np.array_equal(applied.skin_mask, base)
    assert np.array_equal(applied.forbidden_mask, forbidden)


def test_consumer_profile_rejects_catastrophic_hair_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_module = importlib.import_module("src.preprocess.profile_mask_policy")
    base = np.full((32, 32), 255, dtype=np.uint8)
    empty = np.zeros_like(base)
    monkeypatch.setattr(
        policy_module,
        "detect_visible_hair_masks",
        lambda *_args: {
            "hair_mask": base,
            "facial_hair_mask": empty,
        },
    )

    with pytest.raises(ProfileMaskCoverageError):
        policy_module.apply_profile_mask_policy(
            CaptureProfile.CONSUMER,
            np.full((32, 32, 3), 120, dtype=np.uint8),
            base,
            empty,
            np.zeros((478, 2), dtype=np.float32),
            "consumer_catastrophic",
        )


def test_consumer_profile_discards_unreliable_bulk_hair_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_module = importlib.import_module("src.preprocess.profile_mask_policy")
    base = np.full((64, 64), 255, dtype=np.uint8)
    empty = np.zeros_like(base)
    bulk = np.zeros_like(base)
    bulk[:, :26] = 255
    catastrophic = base.copy()
    strand = np.zeros_like(base)
    strand[:, 30:32] = 255
    monkeypatch.setattr(
        policy_module,
        "detect_visible_hair_masks",
        lambda *_args: {
            "hair_mask": catastrophic,
            "bulk_hair_mask": bulk,
            "color_hair_mask": empty,
            "strand_hair_mask": strand,
            "facial_hair_mask": empty,
        },
    )

    applied = policy_module.apply_profile_mask_policy(
        CaptureProfile.CONSUMER,
        np.full((64, 64, 3), 120, dtype=np.uint8),
        base,
        empty,
        np.zeros((478, 2), dtype=np.float32),
        "consumer_bulk_false_positive",
    )

    assert np.count_nonzero(applied.skin_mask) > 0.70 * np.count_nonzero(base)
    assert np.count_nonzero(applied.skin_mask[:, 27:35]) == 0


def test_broad_texture_field_is_not_a_hair_strand_component() -> None:
    assert not _plausible_strand_component(
        area=90_000.0,
        aspect_ratio=1.2,
        touches_boundary=True,
        in_forehead=True,
        domain_pixels=100_000,
    )
    assert _plausible_strand_component(
        area=10_000.0,
        aspect_ratio=3.2,
        touches_boundary=True,
        in_forehead=True,
        domain_pixels=100_000,
    )


def test_consumer_shared_preprocess_applies_bundle_before_dermavision_dispatch() -> None:
    policy_module = importlib.import_module("src.preprocess.profile_mask_policy")
    apply_bundle = getattr(policy_module, "apply_preprocess_mask_bundle", None)
    assert callable(apply_bundle)
    base = np.full((16, 16), 255, dtype=np.uint8)
    hair = np.zeros_like(base)
    hair[:, 7:9] = 255
    empty = np.zeros_like(base)
    bundle_module = importlib.import_module("src.preprocess.analysis_mask_bundle")
    bundle = bundle_module.build_analysis_mask_bundle(
        valid_skin_mask=base,
        hair_mask=hair,
        facial_hair_mask=empty,
        feature_exclusion_mask=empty,
        nostril_mask=empty,
        display_face_mask=base,
        coordinate_space="test",
    )
    preprocess = SimpleNamespace(skin_mask=base.copy(), mask_bundle=bundle)

    apply_bundle(CaptureProfile.CONSUMER, preprocess)

    assert np.array_equal(preprocess.skin_mask, bundle.algorithm_mask())
