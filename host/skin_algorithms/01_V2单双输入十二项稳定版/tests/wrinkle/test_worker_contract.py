from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class WorkerContractTest(unittest.TestCase):
    def test_success_uses_fixed_six_field_envelope(self) -> None:
        with mock.patch.dict("os.environ", {"SERVICE_NAME": "wrinkle"}):
            from src.wrinkle import worker

        raw_dir = Path(tempfile.mkdtemp(prefix="wrinkle_contract_"))
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

        with (
            mock.patch.object(worker, "_pipeline", FakePipeline()),
            mock.patch.object(worker, "download_via_internal_api", return_value=b"jpg"),
            mock.patch.object(
                worker,
                "upload_via_internal_api",
                return_value="report/record-1/wrinkle/artifact.jpg",
            ),
            mock.patch.object(worker, "_cleanup_success_output"),
        ):
            result = worker.analyze_image.run("record-1", "input/test.jpg")

        self.assertEqual(
            set(result),
            {"record_id", "status", "schema_version", "meta_data", "raw_result", "debug_info"},
        )
        self.assertEqual(result["schema_version"], "2")
        self.assertEqual(result["meta_data"], {"name": "wrinkle", "version": "1"})
        expected_raw = set(result_keys) | {
            "region_metrics", "run_preset", "device", "successful_runs",
            "failed_runs", "region_analysis_status", "scoring_input",
        }
        self.assertEqual(set(result["raw_result"]), expected_raw)
        self.assertEqual(result["raw_result"]["run_preset"], "balanced")
        self.assertEqual(
            set(result["debug_info"]),
            {"display_result", "algorithm_version", "elapsed_seconds", "output_dir"},
        )

    def test_medical_v2_additions_are_optional_and_fail_soft(self) -> None:
        with mock.patch.dict("os.environ", {"SERVICE_NAME": "wrinkle"}):
            from src.wrinkle import worker
            from src.wrinkle.clinical_quantification import (
                build_medical_v2_report,
                build_report,
                write_medical_v2_report,
            )
            from src.wrinkle.medical_v2_schema import to_english_document

        output_dir = Path(tempfile.mkdtemp(prefix="wrinkle_medical_v2_"))
        csv_path = output_dir / "皱纹医学量化指标_V2.csv"
        json_path = output_dir / "皱纹量化指标.json"
        report = build_report({})
        json_path.write_text(
            json.dumps(
                {
                    "旧字段": "保持",
                    "medical_metrics_v2": to_english_document(
                        build_medical_v2_report(report)
                    ),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        write_medical_v2_report(report, csv_path, None)
        with (
            mock.patch.object(worker.settings, "enable_medical_metrics_v2", True),
            mock.patch.object(
                worker,
                "upload_via_internal_api",
                return_value="report/record-v2/wrinkle/皱纹医学量化指标_V2.csv",
            ) as upload,
        ):
            additions = worker._build_medical_v2_additions(
                record_id="record-v2",
                output_dir=output_dir,
            )
            self.assertEqual(
                set(additions),
                {"medical_metrics_v2", "medical_report_csv_v2"},
            )
            self.assertEqual(
                additions["medical_metrics_v2"]["scoring_status"],
                "uncalibrated",
            )
            upload.assert_called_once()

            json_path.unlink()
            self.assertEqual(
                worker._build_medical_v2_additions(
                    record_id="record-v2",
                    output_dir=output_dir,
                ),
                {},
            )


if __name__ == "__main__":
    unittest.main()
