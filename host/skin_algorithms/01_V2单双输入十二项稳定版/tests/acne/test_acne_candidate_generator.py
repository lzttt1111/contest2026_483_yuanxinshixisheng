from __future__ import annotations

import cv2
import numpy as np

from src.acne.acne_candidate_generator import (
    AcneCandidateGenerator,
    MaxRecallConfig,
    _robust_positive_score,
    _tier,
    combine_candidates,
)


def test_mask_aware_local_score_does_not_create_skin_edge_response() -> None:
    channel = np.zeros((96, 96), dtype=np.float32)
    valid = np.zeros((96, 96), dtype=np.uint8)
    valid[16:80, 16:80] = 1
    channel[valid > 0] = 120.0

    score = _robust_positive_score(channel, valid, 21)

    assert float(score[valid > 0].max()) < 0.01
    assert float(score[valid == 0].max()) == 0.0


def test_high_tier_requires_two_independent_evidence_groups() -> None:
    config = MaxRecallConfig()

    assert _tier(0.82, 1, config) == "medium"
    assert _tier(0.82, 2, config) == "high"
    assert _tier(0.50, 4, config) == "medium"
    assert _tier(0.30, 4, config) == "low"


def test_diffuse_erythema_is_separate_from_focal_count() -> None:
    generator = AcneCandidateGenerator(
        MaxRecallConfig(diffuse_min_area_ratio_of_skin=0.01, diffuse_close_kernel=11)
    )
    valid = np.ones((128, 128), dtype=np.uint8)
    redness = np.zeros((128, 128), dtype=np.float32)
    redness[28:100, 24:104] = 0.80
    weak_structure = np.full((128, 128), 0.05, dtype=np.float32)
    scores = {
        "redness_only": redness,
        "blob_score": weak_structure,
        "local_contrast": weak_structure,
        "clahe_texture": weak_structure,
    }

    regions, diffuse_mask, diffuse_heatmap = generator._detect_diffuse_regions(scores, valid)

    assert len(regions) == 1
    assert regions[0]["countable"] is False
    assert int(diffuse_mask.sum()) > 0
    assert float(diffuse_heatmap.max()) > 0.0


def test_combined_candidates_report_source_provenance() -> None:
    yolo = [{"bbox_xyxy": [10.0, 10.0, 24.0, 24.0], "confidence": 0.7}]
    unsupervised = [
        {
            "id": 1,
            "bbox_xyxy": [11.0, 11.0, 25.0, 25.0],
            "center_xy": [18.0, 18.0],
            "area": 196,
            "final_candidate_score": 0.8,
            "candidate_score": 0.8,
            "evidence_channels": ["lab_a_local_redness", "blob_score"],
            "candidate_branch": "inflammatory_branch",
            "source": "unsupervised_max_recall",
            "label": "acne_candidate",
            "label_zh": "疑似异常皮肤区域",
        },
        {
            "id": 2,
            "bbox_xyxy": [70.0, 70.0, 82.0, 82.0],
            "center_xy": [76.0, 76.0],
            "area": 144,
            "final_candidate_score": 0.6,
            "candidate_score": 0.6,
            "source": "unsupervised_max_recall",
            "label": "acne_candidate",
            "label_zh": "疑似异常皮肤区域",
        },
    ]

    combined, counts = combine_candidates(yolo, unsupervised, MaxRecallConfig())

    assert len(combined) == 2
    assert counts == {"yolo_only": 0, "unsupervised_only": 1, "yolo_and_unsupervised": 1}
    assert {item["combined_source"] for item in combined} == {
        "unsupervised_only",
        "yolo_and_unsupervised",
    }


def test_generator_schema_and_invalid_regions_are_excluded() -> None:
    image = np.full((256, 256, 3), (150, 170, 190), dtype=np.uint8)
    cv2.circle(image, (128, 128), 7, (70, 80, 185), -1)
    skin = np.full((256, 256), 255, dtype=np.uint8)
    forbidden = np.zeros((256, 256), dtype=np.uint8)
    forbidden[20:70, 20:70] = 255

    result = AcneCandidateGenerator().generate(image, skin, forbidden)

    assert result["status"] == "ok"
    assert set(result["tier_counts"]) == {"high", "medium", "low"}
    assert "diffuse_erythema_regions" in result
    assert "unsupervised_focal_candidates" in result
    assert "candidate_explosion_warning" in result["statistics"]
    for candidate in result["focal_acne_candidates"]:
        assert candidate["tier"] in {"high", "medium", "low"}
        assert candidate["label_zh"] == "疑似异常皮肤区域"
        assert candidate["final_candidate_score"] >= candidate["raw_candidate_score"] * 0.70
        cx, cy = [int(round(value)) for value in candidate["center_xy"]]
        assert forbidden[cy, cx] == 0
