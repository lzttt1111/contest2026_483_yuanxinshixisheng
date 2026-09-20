"""worker 调用点：门禁由"下载原图 bytes"真实计算并持久化（零模型推理）。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from src.scoring_input import gate as gate_module
from src.scoring_input.gate import compute_input_quality_gate

import src.acne.worker as acne_worker
import src.wrinkle.worker as wrinkle_worker
from src import worker as root_worker

from input_gate_test_utils import StubPreprocessor, center_collage, clean_synthetic, encode_png

EXPECTED_COLLAGE = {"status": "REJECT", "reason_codes": ["collage"]}


def _stub_preprocessor_factory(flags: tuple[str, ...] = ()):
    def factory():
        return StubPreprocessor(flags=flags)

    return factory


def test_root_worker_gate_comes_from_downloaded_bytes(tmp_path, monkeypatch) -> None:
    from test_worker_contract import _FakePipeline

    collage_bytes = encode_png(center_collage())
    output_dir = tmp_path / "root-out"
    pipeline = _FakePipeline(output_dir)

    monkeypatch.setattr(gate_module, "_get_default_preprocessor", _stub_preprocessor_factory())
    monkeypatch.setattr(root_worker, "_pipeline", pipeline)
    monkeypatch.setattr(
        root_worker, "download_via_internal_api", lambda oss_key: collage_bytes
    )
    monkeypatch.setattr(
        root_worker,
        "upload_via_internal_api",
        lambda local_path, report_id, algo_name, file_name: (
            f"report/{report_id}/{algo_name}/{file_name}"
        ),
    )

    result = root_worker.analyze_image.run("gate-root", "input/test.jpg", None, ["redness"])

    expected = compute_input_quality_gate(collage_bytes)
    assert expected == EXPECTED_COLLAGE
    assert result["raw_result"]["scoring_input"]["quality"]["input_quality_gate"] == expected


def test_root_worker_gate_follows_downloaded_bytes_not_pipeline(tmp_path, monkeypatch) -> None:
    from test_worker_contract import _FakePipeline

    clean_bytes = encode_png(clean_synthetic())
    output_dir = tmp_path / "root-out-clean"
    pipeline = _FakePipeline(output_dir)

    monkeypatch.setattr(
        gate_module, "_get_default_preprocessor", _stub_preprocessor_factory()
    )
    monkeypatch.setattr(root_worker, "_pipeline", pipeline)
    monkeypatch.setattr(
        root_worker, "download_via_internal_api", lambda oss_key: clean_bytes
    )
    monkeypatch.setattr(
        root_worker,
        "upload_via_internal_api",
        lambda local_path, report_id, algo_name, file_name: (
            f"report/{report_id}/{algo_name}/{file_name}"
        ),
    )

    result = root_worker.analyze_image.run("gate-root-clean", "input/test.jpg", None, ["pores"])

    expected = compute_input_quality_gate(clean_bytes)
    assert expected == {"status": "PASS", "reason_codes": []}
    assert result["raw_result"]["scoring_input"]["quality"]["input_quality_gate"] == expected


def test_acne_worker_gate_comes_from_downloaded_bytes(tmp_path, monkeypatch) -> None:
    collage_bytes = encode_png(center_collage())
    raw_dir = tmp_path / "acne-raw"
    raw_dir.mkdir()
    summary_path = raw_dir / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    class FakePipeline:
        def process_single(self, input_path, algorithms):
            return {
                "status": "success",
                "output_dir": str(raw_dir),
                "summary_path": str(summary_path),
                "results": {},
            }

    monkeypatch.setattr(gate_module, "_get_default_preprocessor", _stub_preprocessor_factory())
    monkeypatch.setattr(acne_worker, "download_via_internal_api", lambda oss_key: collage_bytes)
    monkeypatch.setattr(acne_worker, "_get_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(
        acne_worker,
        "build_display_outputs",
        lambda **kwargs: ({"summary_json": str(summary_path)}, {"量化结果": {}}, {"device": "cpu"}),
    )
    monkeypatch.setattr(acne_worker, "_upload_result_group", lambda **kwargs: {})

    result = acne_worker.analyze_image.run(
        "gate-acne", "inputs/face.jpg", None, ["acne", "formal-fast"]
    )

    expected = compute_input_quality_gate(collage_bytes)
    assert expected == EXPECTED_COLLAGE
    assert result["_scoring_input"]["quality"]["input_quality_gate"] == expected


def test_wrinkle_worker_gate_comes_from_downloaded_bytes(tmp_path, monkeypatch) -> None:
    collage_bytes = encode_png(center_collage())
    raw_dir = tmp_path / "wrinkle-raw"
    raw_dir.mkdir()
    artifact = raw_dir / "artifact.jpg"
    artifact.write_bytes(b"image")
    result_keys = (
        "analysis_face", "preprocessed_face", "stage1_candidates",
        "vote_heatmap", "stage2_overlay", "stage2_centerline",
        "face_filter_debug", "texture_reference", "comparison",
        "region_overlay", "region_tiles", "region_metrics_csv", "summary_json",
    )

    class FakePipeline:
        def process_single(self, _path, _algorithms):
            return {
                "status": "success",
                "output_dir": str(raw_dir),
                "results": {key: str(artifact) for key in result_keys},
                "display_results": {},
                "display_manifest": {},
                "upload_relative_paths": {},
                "metadata": {
                    "run_preset": "balanced",
                    "device": "0",
                    "successful_runs": 3,
                    "failed_runs": 0,
                    "region_analysis_status": "ok",
                    "elapsed_seconds": 1.2,
                    "summary": {"region_metrics": []},
                },
            }

    monkeypatch.setattr(gate_module, "_get_default_preprocessor", _stub_preprocessor_factory())
    monkeypatch.setattr(wrinkle_worker, "_pipeline", FakePipeline())
    monkeypatch.setattr(
        wrinkle_worker, "download_via_internal_api", lambda oss_key: collage_bytes
    )
    monkeypatch.setattr(
        wrinkle_worker,
        "upload_via_internal_api",
        lambda *args, **kwargs: "report/gate-wrinkle/wrinkle/artifact.jpg",
    )
    monkeypatch.setattr(wrinkle_worker, "_cleanup_success_output", lambda *args, **kwargs: None)

    result = wrinkle_worker.analyze_image.run("gate-wrinkle", "input/test.jpg")

    expected = compute_input_quality_gate(collage_bytes)
    assert expected == EXPECTED_COLLAGE
    assert result["raw_result"]["scoring_input"]["quality"]["input_quality_gate"] == expected
