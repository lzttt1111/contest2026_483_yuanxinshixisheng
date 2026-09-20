from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import src.acne.algorithms.acne_analysis as analysis_module
from src.acne.algorithms.acne_analysis import AcneAnalysisRunner, DetectorProfile
from src.acne.artifact_policy import (
    AcneArtifactPolicy,
    selected_acne_algorithms,
)
from src.acne.face_preprocess import PreprocessResult
from src.acne.pipeline import Pipeline
from src.acne.quantification_result import build_quantification_result


def _preprocess_result() -> PreprocessResult:
    image = np.full((32, 32, 3), 64, dtype=np.uint8)
    return PreprocessResult(
        original_bgr=image.copy(),
        standardized_bgr=image.copy(),
        masked_face_bgr=image.copy(),
        skin_mask=np.full((32, 32), 255, dtype=np.uint8),
        forbidden_mask=np.zeros((32, 32), dtype=np.uint8),
        landmarks_xy=np.asarray(
            [[0.0, 0.0], [16.0, 16.0], [31.0, 31.0]],
            dtype=np.float32,
        ),
        crop_box_xyxy=(0, 0, 32, 32),
        square_offset_xy=(0, 0),
        resize_scale=1.0,
        metadata={
            "input_mode": "full_face",
            "quality": {"warnings": []},
            "landmark_quality": {
                "is_reliable": True,
                "full_face_score": 1.0,
                "partial_face_score": 0.0,
                "inside_ratio": 1.0,
                "face_bbox_ratio": 1.0,
                "border_touch_ratio": 0.0,
                "geometric_checks": {},
                "visible_regions": {},
                "reasons": [],
            },
            "visible_regions": {},
            "grading_allowed": True,
            "global_region_analysis_allowed": True,
            "local_detection_allowed": True,
            "face_count": 1,
            "skin_pixels": 1024,
            "forbidden_pixels": 0,
            "_debug_original_landmarks_xy": None,
            "_debug_original_forbidden_mask": None,
            "_debug_input_mode": "full_face",
        },
    )


def _candidate_result() -> dict[str, Any]:
    candidate = {
        "id": 1,
        "bbox_xyxy": [4.0, 4.0, 10.0, 10.0],
        "center_xy": [7.0, 7.0],
        "confidence": 0.8,
        "label": "acne_candidate",
        "source": "unsupervised",
    }
    return {
        "status": "ok",
        "config": {},
        "scores": {},
        "focal_acne_candidates": [candidate],
        "diffuse_erythema_regions": [],
        "tier_counts": {"high": 0, "medium": 0, "low": 1},
        "channel_statistics": {},
        "statistics": {"focal_candidate_count": 1},
    }


def _runner(tmp_path: Path, monkeypatch) -> AcneAnalysisRunner:
    runner = object.__new__(AcneAnalysisRunner)
    runner.profile = DetectorProfile(
        name="fixture",
        weights=Path("fixture.pt"),
        imgsz=32,
        conf=0.1,
        device="cpu",
    )
    checkpoint = tmp_path / "grading.pth"
    checkpoint.write_bytes(b"fixture")
    runner.enable_grading = True
    runner.grading_checkpoint = checkpoint
    runner.grading_python = Path("python")
    runner.grading_timeout = 1
    runner._preprocess_config = object()
    runner._get_mediapipe_models = lambda: (None, None, None)

    class Detector:
        def predict(self, _image: np.ndarray) -> dict[str, Any]:
            return {
                "status": "ok",
                "count": 1,
                "elapsed_seconds": 0.01,
                "detections": [{
                    "bbox_xyxy": [4.0, 4.0, 10.0, 10.0],
                    "confidence": 0.9,
                    "label": "acne_candidate",
                }],
            }

    runner._get_detector = lambda: Detector()
    monkeypatch.setattr(
        analysis_module,
        "preprocess_image",
        lambda *_args: _preprocess_result(),
    )
    return runner


def _formal_values(summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "count": summary["detection"]["count"],
        "region_counts": summary["detection"]["region_counts"],
        "grading": summary["grading"],
        "metrics": build_quantification_result(summary),
        "standardized_image": (output_dir / "17_acne_candidate_circles.jpg").read_bytes(),
        "original_image": (output_dir / "24_original_yolo_circles.jpg").read_bytes(),
    }


def test_formal_fast_selection_token_is_explicit_and_not_an_algorithm() -> None:
    # Given: a local acne request carrying the fast policy token.
    algorithms = ["acne", "formal-fast"]

    # When: policy selection and algorithm selection parse the request.
    policy = AcneArtifactPolicy.from_algorithms(algorithms)
    selected = selected_acne_algorithms(algorithms)

    # Then: formal-fast is explicit but never reaches algorithm validation.
    assert policy is AcneArtifactPolicy.FORMAL_FAST
    assert selected == ["acne"]
    assert (
        AcneArtifactPolicy.from_algorithms([*algorithms, "debug-artifacts"])
        is AcneArtifactPolicy.DEBUG_ARTIFACTS
    )
    assert selected_acne_algorithms([*algorithms, "debug-artifacts"]) == ["acne"]


def test_formal_fast_skips_max_recall_with_formal_value_parity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given: counted Acne-LDS and MaxRecall fakes around deterministic YOLO.
    grading_calls: list[Path] = []
    generator_calls = {"init": 0, "generate": 0}

    class CountingGenerator:
        config = object()

        def __init__(self) -> None:
            generator_calls["init"] += 1

        def generate(self, *_args) -> dict[str, Any]:
            generator_calls["generate"] += 1
            return _candidate_result()

    monkeypatch.setattr(analysis_module, "AcneCandidateGenerator", CountingGenerator)
    monkeypatch.setattr(
        analysis_module,
        "combine_candidates",
        lambda yolo, _unsupervised, _config: (list(yolo), {"yolo": len(yolo)}),
    )
    monkeypatch.setattr(
        analysis_module,
        "run_acne_lds_grading",
        lambda **kwargs: (
            grading_calls.append(Path(kwargs["image_path"]))
            or {"status": "ok", "severity_level": 2}
        ),
    )
    runner = _runner(tmp_path, monkeypatch)
    formal_dir = tmp_path / "formal"
    fast_dir = tmp_path / "formal-fast"
    formal = runner.run(formal_dir / "ignored.jpg", formal_dir)
    calls_after_formal = dict(generator_calls)

    # When: the same image takes the formal-fast path.
    fast = runner.run(
        fast_dir / "ignored.jpg",
        fast_dir,
        artifact_policy=AcneArtifactPolicy.FORMAL_FAST,
    )

    # Then: only MaxRecall disappears; formal clinical values remain identical.
    assert calls_after_formal == {"init": 1, "generate": 1}
    assert generator_calls == calls_after_formal
    assert len(grading_calls) == 2
    assert _formal_values(fast, fast_dir) == _formal_values(formal, formal_dir)
    assert sum(fast["detection"]["region_counts"].values()) == 1
    assert fast["timing_seconds"]["candidate_generation"] == 0.0
    assert fast["max_recall"] == {
        "status": "skipped",
        "reason": "formal_fast_policy",
        "yolo_count": 1,
        "unsupervised_focal_count": 0,
        "diffuse_erythema_region_count": 0,
        "combined_count": 1,
    }
    assert fast["max_recall_outputs"] == {}
    assert not (fast_dir / "22_candidate.json").exists()
    assert (fast_dir / "18_detections.csv").is_file()


def test_pipeline_formal_fast_keeps_compact_and_medical_v2_reports(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given: a fast runner whose MaxRecall constructor must never execute.
    class ForbiddenGenerator:
        def __init__(self) -> None:
            raise AssertionError("formal-fast constructed MaxRecall")

    monkeypatch.setattr(analysis_module, "AcneCandidateGenerator", ForbiddenGenerator)
    monkeypatch.setattr(
        analysis_module,
        "run_acne_lds_grading",
        lambda **_kwargs: {"status": "ok", "severity_level": 2},
    )
    pipeline = object.__new__(Pipeline)
    pipeline.output_root = tmp_path / "output"
    pipeline.runner = _runner(tmp_path, monkeypatch)

    # When: Pipeline completes formal-fast.
    result = pipeline.process_single(str(tmp_path / "fixture.jpg"), ["acne", "formal-fast"])

    # Then: official YOLO and clinical compact/V2 outputs remain available.
    output_dir = Path(result["output_dir"])
    assert result["status"] == "success"
    assert result["metadata"]["artifact_policy"] == "formal-fast"
    assert result["metadata"]["timing_seconds"]["candidate_generation"] == 0.0
    assert Path(result["results"]["acne_circles"]).is_file()
    assert Path(result["results"]["acne_detections"]).is_file()
    assert Path(result["results"]["original_yolo_circles"]).is_file()
    assert (output_dir / "痤疮量化指标.csv").is_file()
    assert (output_dir / "痤疮量化指标.json").is_file()
    assert (output_dir / "痤疮医学量化指标_V2.csv").is_file()
    assert "max_recall_json" not in result["results"]
