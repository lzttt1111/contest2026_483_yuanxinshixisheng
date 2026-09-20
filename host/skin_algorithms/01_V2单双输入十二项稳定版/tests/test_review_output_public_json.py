from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.nine_analysis.review_output import ReviewOutputExporter


PUBLIC_KEYS = {
    "project",
    "metrics_version",
    "scoring_status",
    "imaging_and_units",
    "overall_metrics",
    "region_metrics",
    "left_right_comparison",
    "quality_control",
    "medical_limitations",
}


class ReviewOutputPublicJsonTest(unittest.TestCase):
    def test_acne_and_wrinkle_public_json_excludes_runtime_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            acne_image = source / "acne.jpg"
            wrinkle_image = source / "wrinkle.jpg"
            acne_image.write_bytes(b"acne-image")
            wrinkle_image.write_bytes(b"wrinkle-image")

            acne_raw = {
                "profile": {"weights": "/home/example/acne.pt"},
                "elapsed_seconds": 7.0,
                "outputs": {"original": "/home/example/.runtime/input.jpg"},
                "detection": {
                    "status": "ok",
                    "input_mode": "full_face",
                    "detection_scope": "global",
                    "count": 1,
                    "region_counts": {"forehead": 1},
                    "detections": [{
                        "region": "forehead",
                        "box_area": 120.0,
                        "confidence": 0.8,
                    }],
                },
                "preprocess": {"skin_pixels": 100000, "face_count": 1},
                "grading": {"status": "ok", "severity_level": 1},
            }
            wrinkle_raw = {
                "weights": "/home/example/wrinkle.pt",
                "source": "/home/example/.runtime/input.jpg",
                "output_dir": "/home/example/.runtime/output",
                "elapsed_seconds": 8.0,
                "region_analysis_status": "ok",
                "run_preset": "balanced",
                "successful_runs": 84,
                "failed_runs": 0,
                "face_filter_status": "ok",
                "semantic_skin_status": "ok",
                "stage2_recommended_pixels": 20,
                "region_metrics": [{
                    "region_name": "额头纹",
                    "area_px": 1000,
                    "segment_count": 2,
                    "wrinkle_pixels": 40,
                    "mean_segment_length": 20,
                    "max_segment_length": 30,
                    "relative_score": 25,
                }],
            }
            acne_json = source / "acne.json"
            wrinkle_json = source / "wrinkle.json"
            acne_json.write_text(json.dumps(acne_raw), encoding="utf-8")
            wrinkle_json.write_text(json.dumps(wrinkle_raw), encoding="utf-8")

            target = root / "review" / "sample"
            exporter = ReviewOutputExporter(root / "review")
            exporter.export_wrinkle_acne(
                {
                    "acne": {"主结果图": str(acne_image), "量化JSON": str(acne_json)},
                    "wrinkle": {
                        "主结果图": str(wrinkle_image),
                        "量化JSON": str(wrinkle_json),
                    },
                },
                target,
            )

            for relative in ("痤疮/痤疮量化指标.json", "皱纹/皱纹量化指标.json"):
                path = target / relative
                text = path.read_text(encoding="utf-8")
                document = json.loads(text)
                self.assertEqual(set(document), PUBLIC_KEYS)
                self.assertNotIn("/home/", text)
                self.assertNotIn(".runtime", text)
                self.assertNotIn("output_dir", text)
                self.assertNotIn("elapsed_seconds", text)
                self.assertNotIn("weights", text)

            for relative in (
                "痤疮/痤疮量化指标.csv",
                "痤疮/痤疮医学量化指标_V2.csv",
                "皱纹/皱纹量化指标.csv",
                "皱纹/皱纹医学量化指标_V2.csv",
            ):
                with (target / relative).open(
                    "r", encoding="utf-8-sig", newline=""
                ) as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(rows, relative)
                for row in rows:
                    self.assertNotIn("", row.values(), f"{relative}: {row}")

            with (target / "痤疮/痤疮医学量化指标_V2.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                acne_rows = list(csv.DictReader(handle))
            self.assertEqual(acne_rows[1]["有效皮肤面积（像素）"], "仅全面部统计")
            self.assertEqual(
                acne_rows[1]["核心-单位面积密度（个/10万有效皮肤像素）"],
                "仅全面部统计",
            )
            self.assertEqual(
                acne_rows[1]["核心-Acne-LDS评估状态"], "仅全面部适用"
            )

            with (target / "皱纹/皱纹医学量化指标_V2.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                wrinkle_rows = list(csv.DictReader(handle))
            self.assertEqual(
                wrinkle_rows[1]["核心-P50线段长度（像素，分区均值代理）"],
                "仅全面部统计",
            )
            self.assertEqual(
                wrinkle_rows[1]["核心-平均可见宽度（像素，面积/中心线代理）"],
                "仅全面部统计",
            )
            self.assertEqual(
                wrinkle_rows[1]["核心-总纹路长度（中心线像素）"], "40"
            )
            self.assertEqual(
                wrinkle_rows[1]["核心-P90视觉对比度（0～1，响应代理）"],
                "0.25",
            )


if __name__ == "__main__":
    unittest.main()
