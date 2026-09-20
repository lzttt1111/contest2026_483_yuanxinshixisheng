from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.added_algorithms.public_metrics import PublicMetricFiles
from src.added_algorithms.runner import AddedAlgorithmArtifacts
from src.capture_profile import CaptureProfile
from src.nine_analysis.orchestrator import NineAnalysisOrchestrator
from src.preprocess.image_preprocessor import PreprocessResultV2
from src import pipeline as pipeline_module


def _preprocess() -> PreprocessResultV2:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    return PreprocessResultV2(
        analysis_image=image,
        display_image=image.copy(),
        skin_mask=np.full((64, 64), 255, dtype=np.uint8),
        landmarks=np.zeros((478, 2), dtype=np.float32),
        face_transform_matrix=np.eye(2, 3, dtype=np.float32),
        inverse_transform_matrix=np.eye(2, 3, dtype=np.float32),
        quality_score=90.0,
        quality_status="PASS",
        quality_flags=[],
    )


def test_contour_failure_preserves_completed_dermavision_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "phone.jpg"
    assert cv2.imwrite(str(input_path), _preprocess().analysis_image)

    class FakePreprocessor:
        def preprocess_image(self, _image: np.ndarray) -> PreprocessResultV2:
            return _preprocess()

    class FakeCuda:
        def begin_task_profile(self):
            return None

        def finish_task_profile(self, _baseline):
            return {
                "device": "cpu",
                "task_gpu_kernel_seconds": 0.0,
                "task_gpu_kernel_calls": 0,
                "task_peak_memory_mib": 0.0,
            }

    def fake_gloss(_preprocess, output_dir, source_name):
        root = Path(output_dir) / Path(source_name).stem
        root.mkdir(parents=True)
        overlay = root / "01_表面油光检测结果图.jpg"
        assert cv2.imwrite(str(overlay), np.zeros((64, 64, 3), dtype=np.uint8))
        files = PublicMetricFiles(
            root / "metrics.json",
            root / "compact.csv",
            root / "medical.csv",
            root / "full.json",
        )
        for path in (
            files.metrics_json,
            files.compact_csv,
            files.medical_csv_v2,
            files.full_metrics_json,
        ):
            path.write_text("{}\n" if path.suffix == ".json" else "x\n")
        return AddedAlgorithmArtifacts(overlay, files)

    monkeypatch.setattr(pipeline_module, "get_cuda_backend", lambda: FakeCuda())
    monkeypatch.setattr(
        "src.added_algorithms.runner.run_surface_gloss",
        fake_gloss,
    )
    monkeypatch.setattr(
        "src.added_algorithms.runner.run_contour_firmness",
        lambda *_args: None,
    )
    pipeline = pipeline_module.DermaVisionPipeline(
        capture_profile=CaptureProfile.CONSUMER
    )
    pipeline.preprocessor = FakePreprocessor()
    pipeline.preprocessed_dir = str(tmp_path / "preprocessed")
    pipeline.surface_gloss_dir = str(tmp_path / "gloss")
    pipeline.contour_firmness_dir = str(tmp_path / "contour")
    for directory in (
        pipeline.preprocessed_dir,
        pipeline.surface_gloss_dir,
        pipeline.contour_firmness_dir,
    ):
        Path(directory).mkdir(parents=True)

    response = pipeline.process_single(
        str(input_path),
        ["surface_gloss", "contour_firmness"],
    )

    assert response["status"] == "partial_success"
    assert response["failed_algorithms"] == {
        "contour_firmness": "轮廓紧致度几何QC失败或相对Z不可用"
    }
    assert Path(response["results"]["surface_gloss"]).is_file()

    orchestrator = NineAnalysisOrchestrator(
        tmp_path / "orchestrator",
        algorithms=("surface_gloss", "contour_firmness"),
        capture_profile=CaptureProfile.CONSUMER,
    )
    items = orchestrator._canonical_items({
        "dermavision": {
            "status": "partial_success",
            "result": response,
        }
    })
    assert items["surface_gloss"]["状态"] == "success"
    assert items["contour_firmness"]["状态"] == "failed"
