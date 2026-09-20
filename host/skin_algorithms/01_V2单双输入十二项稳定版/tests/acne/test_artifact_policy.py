from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import src.acne.algorithms.acne_analysis as analysis_module
import src.acne.artifact_policy as artifact_policy_module
import src.acne.pipeline as pipeline_module
from src.acne.algorithms.acne_analysis import AcneAnalysisRunner, DetectorProfile
from src.acne.artifact_policy import AcneArtifactPolicy, AcnePhaseTimings
from src.acne.detector_postprocess import save_detection_outputs
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
        landmarks_xy=np.empty((0, 2), dtype=np.float32),
        crop_box_xyxy=(0, 0, 32, 32),
        square_offset_xy=(0, 0),
        resize_scale=1.0,
        metadata={
            "input_mode": "skin_patch",
            "quality": {"warnings": []},
            "landmark_quality": {
                "is_reliable": False,
                "full_face_score": 0.0,
                "partial_face_score": 0.0,
                "inside_ratio": 0.0,
                "face_bbox_ratio": 0.0,
                "border_touch_ratio": 0.0,
                "geometric_checks": {},
                "visible_regions": {},
                "reasons": ["fixture"],
            },
            "visible_regions": {},
            "grading_allowed": False,
            "global_region_analysis_allowed": False,
            "local_detection_allowed": True,
            "face_count": 0,
            "skin_pixels": 1024,
            "forbidden_pixels": 0,
            "_debug_original_landmarks_xy": None,
            "_debug_original_forbidden_mask": None,
            "_debug_input_mode": "skin_patch",
        },
    )


def _candidate_result() -> dict[str, Any]:
    score = np.zeros((32, 32), dtype=np.float32)
    candidate = {
        "id": 1,
        "bbox_xyxy": [4.0, 4.0, 10.0, 10.0],
        "center_xy": [7.0, 7.0],
        "confidence": 0.8,
        "label": "acne_candidate",
        "source": "unsupervised",
    }
    score_names = {
        "acne_candidate_heatmap",
        "diffuse_erythema_heatmap",
        "focal_candidate_heatmap",
        "redness_only",
        "texture_only",
        "blob_score",
        "blackhat_after_line_suppression",
        "linear_dark_structure",
        "repetitive_normal_texture",
        "tophat_before_suppression",
        "tophat_after_suppression",
        "specular_highlight_mask",
        "normality_suppression",
    }
    return {
        "status": "ok",
        "config": {},
        "scores": {name: score.copy() for name in score_names},
        "focal_acne_candidates": [candidate],
        "diffuse_erythema_regions": [],
        "tier_counts": {"high": 0, "medium": 0, "low": 1},
        "channel_statistics": {},
        "statistics": {"focal_candidate_count": 1},
    }


def _runner(monkeypatch) -> AcneAnalysisRunner:
    runner = object.__new__(AcneAnalysisRunner)
    runner.profile = DetectorProfile(
        name="fixture",
        weights=Path("fixture.pt"),
        imgsz=32,
        conf=0.1,
        device="cpu",
    )
    runner.enable_grading = False
    runner.grading_checkpoint = None
    runner.grading_python = Path("python")
    runner.grading_timeout = 1
    runner._preprocess_config = object()
    runner._get_mediapipe_models = lambda: (None, None, None)

    class Detector:
        def predict(self, image: np.ndarray) -> dict[str, Any]:
            return {
                "status": "ok",
                "count": 1,
                "elapsed_seconds": 0.01,
                "detections": [
                    {
                        "bbox_xyxy": [4.0, 4.0, 10.0, 10.0],
                        "confidence": 0.9,
                        "label": "acne_candidate",
                    }
                ],
            }

    runner._get_detector = lambda: Detector()

    class Generator:
        config = object()

        def generate(self, *_args) -> dict[str, Any]:
            return _candidate_result()

    monkeypatch.setattr(analysis_module, "preprocess_image", lambda *_args: _preprocess_result())
    monkeypatch.setattr(analysis_module, "AcneCandidateGenerator", Generator)
    monkeypatch.setattr(
        analysis_module,
        "combine_candidates",
        lambda yolo, unsupervised, config: (list(yolo), {"yolo": len(yolo), "unsupervised": 0}),
    )
    return runner


def _formal_observables(summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "count": summary["detection"]["count"],
        "grading": summary["grading"],
        "metrics": build_quantification_result(summary),
        "public_standardized_image": (output_dir / "17_acne_candidate_circles.jpg").read_bytes(),
        "public_original_image": (output_dir / "24_original_yolo_circles.jpg").read_bytes(),
    }


def test_formal_policy_skips_debug_images_without_changing_observables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner(monkeypatch)
    formal_dir = tmp_path / "formal"
    debug_dir = tmp_path / "debug"

    formal = runner.run(formal_dir / "ignored.jpg", formal_dir)
    debug = runner.run(
        debug_dir / "ignored.jpg",
        debug_dir,
        artifact_policy=AcneArtifactPolicy.DEBUG_ARTIFACTS,
    )

    formal_observables = _formal_observables(formal, formal_dir)
    assert formal_observables == _formal_observables(debug, debug_dir)
    assert formal_observables["count"] == 1
    assert formal_observables["grading"]["status"] == "skipped"
    assert formal_observables["metrics"]["疑似痤疮圈选数量"] == 1
    assert formal_observables["metrics"]["痤疮严重程度等级"]["等级"] is None
    assert (formal_dir / "22_candidate.json").is_file()
    formal_candidate = json.loads((formal_dir / "22_candidate.json").read_text(encoding="utf-8"))
    debug_candidate = json.loads((debug_dir / "22_candidate.json").read_text(encoding="utf-8"))
    assert formal_candidate["debug_outputs"] == {}
    assert len(debug_candidate["debug_outputs"]) == 13
    assert {
        key: value for key, value in formal_candidate.items() if key != "debug_outputs"
    } == {
        key: value for key, value in debug_candidate.items() if key != "debug_outputs"
    }
    debug_image_names = {
        "03_masked_face.jpg",
        "06_region_debug.jpg",
        "09_landmark_debug.jpg",
        "10_visible_regions_debug.jpg",
        "16_raw_boxes.jpg",
        "19_candidate_heatmap.jpg",
        "19a_diffuse_erythema_heatmap.jpg",
        "19b_focal_candidate_heatmap.jpg",
        "20_candidate_circle_overlay.jpg",
        "20a_combined_candidate_overlay.jpg",
        "21_candidate_score_debug.jpg",
        "25_original_unsupervised_candidate_circles.jpg",
        "26_original_combined_candidate_circles.jpg",
    }
    assert all(not (formal_dir / name).exists() for name in debug_image_names)
    assert all((debug_dir / name).exists() for name in debug_image_names)
    assert not (formal_dir / "23_debug_channels").exists()
    assert (debug_dir / "23_debug_channels" / "final_heatmap.jpg").exists()

    required_timings = {
        "preprocess",
        "lds",
        "yolo",
        "candidate_generation",
        "image_encoding",
    }
    assert required_timings <= set(formal["timing_seconds"])
    assert all(formal["timing_seconds"][key] >= 0 for key in required_timings)


def test_debug_artifact_selection_token_is_explicit() -> None:
    assert AcneArtifactPolicy.from_algorithms(None) is AcneArtifactPolicy.FORMAL
    assert AcneArtifactPolicy.from_algorithms(["acne"]) is AcneArtifactPolicy.FORMAL
    assert (
        AcneArtifactPolicy.from_algorithms(["acne", "debug-artifacts"])
        is AcneArtifactPolicy.DEBUG_ARTIFACTS
    )


def test_image_encoding_excludes_structured_artifact_persistence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ticks = iter((0.0, 2.0, 2.0, 7.0, 7.0, 18.0))
    monkeypatch.setattr(
        artifact_policy_module,
        "perf_counter",
        lambda: next(ticks),
        raising=False,
    )
    timings = AcnePhaseTimings()
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    save_detection_outputs(
        tmp_path,
        image,
        {"status": "ok", "detections": [], "count": 0},
        [],
        timings=timings,
    )

    assert timings.image_encoding == 5.0
    assert timings.quantification == 13.0


def test_pipeline_adds_quantification_timing_at_result_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    received_policies: list[AcneArtifactPolicy] = []

    class Runner:
        def run(
            self,
            _source: Path,
            output_dir: Path,
            *,
            artifact_policy: AcneArtifactPolicy,
        ) -> dict[str, Any]:
            received_policies.append(artifact_policy)
            output_dir.mkdir(parents=True)
            summary = {
                "status": "ok",
                "detection": {"status": "ok", "count": 1},
                "grading": {"status": "ok", "severity_level": 2},
                "timing_seconds": {
                    "preprocess": 0.1,
                    "lds": 0.2,
                    "yolo": 0.3,
                    "candidate_generation": 0.4,
                    "image_encoding": 0.5,
                    "quantification": 0.6,
                },
            }
            (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return summary

    pipeline = object.__new__(Pipeline)
    pipeline.output_root = tmp_path / "output"
    pipeline.runner = Runner()
    monkeypatch.setattr(pipeline_module, "write_clinical_quantification", lambda *_args: {})
    monkeypatch.setattr(pipeline_module, "try_merge_medical_v2_report", lambda *_args: None)

    result = pipeline.process_single(
        str(tmp_path / "fixture.jpg"),
        ["acne", "debug-artifacts"],
    )

    assert received_policies == [AcneArtifactPolicy.DEBUG_ARTIFACTS]
    assert result["metadata"]["artifact_policy"] == "debug-artifacts"
    assert set(result["metadata"]["timing_seconds"]) == {
        "preprocess",
        "lds",
        "yolo",
        "candidate_generation",
        "image_encoding",
        "quantification",
    }
    assert result["metadata"]["timing_seconds"]["quantification"] >= 0.6
    saved_summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
    assert saved_summary["timing_seconds"] == result["metadata"]["timing_seconds"]


def test_pipeline_formal_result_maps_public_candidate_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.output_root = tmp_path / "output"
    pipeline.runner = _runner(monkeypatch)
    monkeypatch.setattr(pipeline_module, "write_clinical_quantification", lambda *_args: {})
    monkeypatch.setattr(pipeline_module, "try_merge_medical_v2_report", lambda *_args: None)

    result = pipeline.process_single(str(tmp_path / "fixture.jpg"), ["acne"])

    candidate_json = Path(result["results"]["max_recall_json"])
    assert candidate_json.name == "22_candidate.json"
    assert candidate_json.is_file()
