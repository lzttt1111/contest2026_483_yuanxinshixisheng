from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.nine_analysis.clinical_reports import (
    build_acne_report,
    build_medical_v2_report,
    write_medical_v2_report,
)


class IntegratedMedicalMetricsV2Test(unittest.TestCase):
    def test_acne_v2_keeps_scoring_uncalibrated(self) -> None:
        raw = {
            "detection": {
                "status": "ok",
                "input_mode": "full_face",
                "detection_scope": "global",
                "count": 1,
                "region_counts": {"forehead": 1},
                "detections": [{
                    "region": "forehead", "box_area": 120.0,
                    "confidence": 0.8,
                }],
            },
            "preprocess": {"skin_pixels": 100000, "face_count": 1},
            "grading": {"status": "ok", "severity_level": 1},
        }
        report = build_acne_report(raw)
        document = build_medical_v2_report(report, "acne")
        self.assertEqual(document["评分状态"], "uncalibrated")
        self.assertNotIn("目标明细", document)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "痤疮医学量化指标_V2.csv"
            json_path = Path(directory) / "痤疮医学量化指标_V2.json"
            write_medical_v2_report(report, "acne", csv_path, json_path)
            self.assertEqual(json.loads(json_path.read_text()), document)
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["核心-候选框面积占比"], "0.0012")


if __name__ == "__main__":
    unittest.main()
