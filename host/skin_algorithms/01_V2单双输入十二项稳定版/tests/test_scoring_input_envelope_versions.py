"""信封升版：worker schema_version、scoring_input 落位、真实契约校验与历史兼容。"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import pytest
from aisia_contracts import REGISTRY, get_envelope
from aisia_contracts.algorithms.redness.v2 import RednessV2RawResult
from aisia_contracts.algorithms.redness.v3 import RednessV3RawResult
from aisia_contracts.scoring_input.v1 import ScoringInputV1

from src import worker
from test_worker_contract import TEST_IMAGE, _FakePipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERMAVISION_ALGORITHMS = (
    "redness", "spots", "brown", "texture", "pores",
    "purple", "surface_gloss", "vascular", "contour_firmness",
)
EXPECTED_VERSIONS = {
    "redness": "3",
    "spots": "3",
    "brown": "3",
    "texture": "3",
    "pores": "3",
    "purple": "2",
    "surface_gloss": "1",
    "vascular": "1",
    "contour_firmness": "1",
}


def _run_root_worker(output_dir: Path, algorithm: str) -> dict:
    pipeline = _FakePipeline(output_dir)

    def upload(local_path: str, report_id: str, algo_name: str, file_name: str) -> str:
        return f"report/{report_id}/{algo_name}/{file_name}"

    with (
        mock.patch.object(worker, "_pipeline", pipeline),
        mock.patch.object(
            worker, "download_via_internal_api", return_value=TEST_IMAGE.read_bytes()
        ),
        mock.patch.object(worker, "upload_via_internal_api", side_effect=upload),
        # 门禁模型接线由 test_input_quality_gate_worker.py 覆盖，这里避免模型推理。
        mock.patch.object(
            worker,
            "compute_input_quality_gate",
            return_value={"status": "PASS", "reason_codes": []},
        ),
    ):
        return worker.analyze_image.run("envelope-test", "input/test.jpg", None, [algorithm])


@pytest.mark.parametrize("algorithm", DERMAVISION_ALGORITHMS)
def test_envelope_version_scoring_input_and_real_contract(algorithm: str) -> None:
    output_dir = PROJECT_ROOT / "output" / "scoring_input_envelope_test"
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        result = _run_root_worker(output_dir, algorithm)
        version = EXPECTED_VERSIONS[algorithm]
        assert worker._SCHEMA_VERSIONS[algorithm] == version
        assert result["schema_version"] == version
        assert "scoring_input" in result["raw_result"]
        ScoringInputV1.model_validate(result["raw_result"]["scoring_input"])

        envelope_cls = get_envelope(algorithm, version)
        envelope_cls.model_validate(result)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_worker_versions_are_registered_in_contract() -> None:
    for algorithm, version in EXPECTED_VERSIONS.items():
        assert (algorithm, version) in REGISTRY


def test_old_version_envelope_without_scoring_input_still_validates() -> None:
    old_cls = get_envelope("purple", "1")
    response = {
        "record_id": "old-purple",
        "status": "success",
        "schema_version": "1",
        "meta_data": {"name": "purple", "version": "1"},
        "raw_result": {
            "uv_base": "purple/uv.png",
            "uv_spots_overlay": "purple/uv_spots.jpg",
            "fluorescence_base": "purple/fluorescence.png",
            "porphyrin_overlay": "purple/porphyrin.jpg",
            "metrics": {
                "uv_spots_total": 1,
                "porphyrin_total": 2,
            },
            "quality_score": None,
            "quality_status": None,
            "quality_flags": [],
        },
        "debug_info": {"report_csv": "purple/report.csv"},
    }
    old_cls.model_validate(response)
    assert "scoring_input" not in RednessV2RawResult.model_fields
    assert "scoring_input" in RednessV3RawResult.model_fields
