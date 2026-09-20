from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from docx import Document

from src.aisia_medical_report import generate_report_from_review_result


class MedicalReportIntegrationTests(unittest.TestCase):
    def test_optional_report_accepts_successful_nine_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sample"
            root.mkdir()
            index = {
                "状态": "success",
                "成功项目数": 9,
                "项目总数": 9,
                "九项结果": {name: {"状态": "success"} for name in (
                    "redness", "spots", "brown", "texture", "pores",
                    "uv_spots", "porphyrin", "wrinkle", "acne",
                )},
            }
            (root / "九项检测结果索引.json").write_text(
                json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            output = generate_report_from_review_result(root, report_id="TEST-REPORT")
            json_path = Path(output["json"])
            docx_path = Path(output["docx"])
            self.assertTrue(json_path.is_file())
            self.assertTrue(docx_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["模块总数"], 11)
            self.assertIsNone(payload["报告信息"]["综合得分"])
            self.assertGreater(len(Document(docx_path).tables), 0)
            updated = json.loads((root / "九项检测结果索引.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["医学报告"]["状态"], "success")

    def test_optional_report_rejects_incomplete_nine_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "九项检测结果索引.json").write_text(
                json.dumps({"状态": "failed", "成功项目数": 8}, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                generate_report_from_review_result(root)


if __name__ == "__main__":
    unittest.main()
