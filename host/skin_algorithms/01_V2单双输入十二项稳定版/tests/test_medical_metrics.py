from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.utils.detailed_metrics import (
    MEDICAL_REGION_ORDER,
    _boundary_gradient,
    build_medical_payload,
    build_medical_report_regions,
    write_medical_metrics,
    write_medical_metrics_v2,
)
from src.engines.visia_regions import build_moustache_feature_exclusion_mask


class MedicalMetricsTest(unittest.TestCase):
    @staticmethod
    def _fixture() -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        cv2.ellipse(mask, (512, 520), (330, 430), 0, 0, 360, 255, -1)
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[:] = (512, 512)
        landmarks[1] = (512, 500)
        landmarks[105] = (410, 340)
        landmarks[334] = (614, 340)
        landmarks[168] = (512, 390)
        landmarks[2] = (512, 610)
        landmarks[0] = (512, 690)
        landmarks[17] = (512, 750)
        return mask, landmarks

    def test_medical_regions_are_mutually_exclusive_and_complete(self) -> None:
        mask, landmarks = self._fixture()
        regions = build_medical_report_regions(mask, landmarks)
        stack = np.stack([regions.regions[name] > 0 for name in MEDICAL_REGION_ORDER])
        self.assertTrue(np.array_equal(np.sum(stack, axis=0) > 0, mask > 0))
        self.assertLessEqual(int(np.max(np.sum(stack, axis=0))), 1)

    def test_chin_remains_assessable_when_valid_skin_ends_near_lower_lip(self) -> None:
        mask, landmarks = self._fixture()
        mask[820:, :] = 0

        regions = build_medical_report_regions(mask, landmarks)

        self.assertTrue(regions.available["chin"])
        self.assertGreaterEqual(int(np.count_nonzero(regions.regions["chin"])), 100)
        stack = np.stack([regions.regions[name] > 0 for name in MEDICAL_REGION_ORDER])
        self.assertTrue(np.array_equal(np.sum(stack, axis=0) > 0, mask > 0))
        self.assertLessEqual(int(np.max(np.sum(stack, axis=0))), 1)

    def test_boundary_gradient_uses_boolean_masks(self) -> None:
        score = np.full((64, 64), 0.2, dtype=np.float32)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        score[mask > 0] = 0.8

        value = _boundary_gradient(score, mask)

        self.assertAlmostEqual(value, 0.6, places=5)

    def test_medical_json_and_csv_share_summary_rows(self) -> None:
        mask, landmarks = self._fixture()
        score = np.zeros(mask.shape, dtype=np.float32)
        cv2.circle(score, (420, 550), 20, 0.8, -1)
        instance_mask = np.zeros_like(mask)
        cv2.circle(instance_mask, (420, 550), 5, 255, -1)
        payload = build_medical_payload(
            project="spots",
            project_label="普通RGB可见斑点",
            analysis_mask=mask,
            landmarks=landmarks,
            instances=[{
                "spot_id": 1,
                "centroid": [420, 550],
                "area": 81,
                "mean_deltaE": 5.2,
                "confidence": .8,
                "local_contrast": 4.0,
                "uniformity_score": .7,
                "spot_type": "small",
                "intensity": .6,
            }],
            score_map=score,
            instance_mask=instance_mask,
            quality_control={"图像质量状态": "PASS"},
            limitations=["测试局限"],
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "metrics.json"
            csv_path = Path(directory) / "metrics.csv"
            write_medical_metrics(json_path, csv_path, payload, "spots")
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            json_line_count = json_path.read_text(encoding="utf-8").count("\n")
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(loaded["指标版本"], "medical_metrics_v1")
        self.assertEqual(
            set(loaded),
            {"检测项目", "指标版本", "总体指标", "分区指标"},
        )
        self.assertNotIn("医学解释", rows[0])
        self.assertNotIn("目标明细", loaded)
        self.assertGreater(json_line_count, 10)
        self.assertLessEqual(len(rows) + 1, 20)
        self.assertEqual(rows[0]["检测范围"], "全面部")
        self.assertEqual(
            rows[0]["特征数量（个）"],
            str(loaded["总体指标"]["特征数量（个）"]),
        )
        self.assertEqual(
            sum(int(row["特征数量（个）"]) for row in rows[1:] if row["评估状态"] == "可评估"),
            loaded["总体指标"]["特征数量（个）"],
        )
        json.dumps(loaded, ensure_ascii=False, allow_nan=False)

    def test_moustache_exclusion_is_local_and_binary(self) -> None:
        mask = np.full((1024, 1024), 255, dtype=np.uint8)
        landmarks = np.zeros((478, 2), dtype=np.float32)
        landmarks[98] = (430, 520)
        landmarks[97] = (470, 530)
        landmarks[2] = (512, 540)
        landmarks[326] = (550, 530)
        landmarks[327] = (590, 520)
        landmarks[291] = (620, 650)
        landmarks[267] = (560, 625)
        landmarks[0] = (512, 615)
        landmarks[37] = (464, 625)
        landmarks[61] = (404, 650)
        exclusion = build_moustache_feature_exclusion_mask(
            mask.shape,
            landmarks,
            mask,
            margin_px=12,
        )
        self.assertEqual(exclusion.dtype, np.uint8)
        self.assertTrue(set(np.unique(exclusion)).issubset({0, 255}))
        self.assertEqual(int(exclusion[580, 512]), 255)
        self.assertEqual(int(exclusion[720, 512]), 0)

    def test_partial_missing_side_is_unavailable_not_zero(self) -> None:
        mask, landmarks = self._fixture()
        mask[:, 512:] = 0
        payload = build_medical_payload(
            project="pores",
            project_label="可见毛孔负担",
            analysis_mask=mask,
            landmarks=landmarks,
            instances=[],
            score_map=np.zeros(mask.shape, np.float32),
            instance_mask=np.zeros_like(mask),
            quality_control={"图像质量状态": "WARNING"},
            limitations=["测试局限"],
        )
        right_rows = [
            row for row in payload["分区指标"]
            if row["检测范围"] in {"画面右鼻旁", "画面右颧部", "画面右面颊", "画面右下颌"}
        ]
        self.assertTrue(right_rows)
        self.assertTrue(any(row["评估状态"] == "不可评估" for row in right_rows))
        for row in right_rows:
            if row["评估状态"] == "不可评估":
                self.assertEqual(row["有效皮肤面积（像素）"], "不可评估")

    def test_pore_v2_area_units_are_square_pixels(self) -> None:
        mask, landmarks = self._fixture()
        payload = build_medical_payload(
            project="pores",
            project_label="可见毛孔负担",
            analysis_mask=mask,
            landmarks=landmarks,
            instances=[],
            score_map=np.zeros(mask.shape, np.float32),
            instance_mask=np.zeros_like(mask),
            quality_control={"图像质量状态": "PASS"},
            limitations=["测试局限"],
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "pores.json"
            csv_path = Path(directory) / "pores.csv"
            document = write_medical_metrics_v2(
                json_path,
                csv_path,
                payload,
                "pores",
            )
            header = csv_path.read_text(encoding="utf-8-sig").splitlines()[0]

        self.assertEqual(
            document["成像与单位说明"]["面积单位"],
            "标准化1024图像平方像素（px²）",
        )
        self.assertIn("P50单体面积（像素²）", header)
        self.assertIn("特征总面积（像素²）", header)


if __name__ == "__main__":
    unittest.main()
