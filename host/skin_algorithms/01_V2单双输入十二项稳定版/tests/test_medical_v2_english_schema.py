from __future__ import annotations

import unittest
from pathlib import Path

from src.nine_analysis.medical_v2_schema import (
    INCREMENTAL_OVERALL_METRICS,
    LEGACY_CANONICAL_METRICS,
    to_chinese_document,
    to_english_document,
    to_incremental_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _contains_chinese_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            any("\u4e00" <= character <= "\u9fff" for character in str(key))
            or _contains_chinese_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_chinese_key(child) for child in value)
    return False


class MedicalV2EnglishSchemaTest(unittest.TestCase):
    def test_public_document_uses_only_english_keys_and_round_trips(self) -> None:
        chinese = {
            "检测项目": "红区",
            "指标版本": "medical_metrics_v2_20260728",
            "评分状态": "uncalibrated",
            "成像与单位说明": {
                "强度说明": "0～1 工程归一化值",
            },
            "总体指标": {
                "核心指标": {
                    "数量与密度": {"特征数量（个）": 12},
                },
                "辅助指标": {
                    "分析范围": {
                        "检测范围": "全面部",
                        "评估状态": "可评估",
                        "有效皮肤面积（像素）": 1000,
                    }
                },
            },
            "分区指标": [],
            "左右比较": {},
            "质量控制": {},
            "医学局限性": [],
        }
        public = to_english_document(chinese)
        self.assertFalse(_contains_chinese_key(public))
        self.assertEqual(public["scoring_status"], "uncalibrated")
        self.assertEqual(
            public["overall_metrics"]["core_metrics"]["count_and_density"][
                "feature_count"
            ],
            12,
        )
        self.assertEqual(to_chinese_document(public), chinese)

    def test_all_service_schemas_keep_the_shared_translation_contract(self) -> None:
        paths = [
            PROJECT_ROOT / "src/nine_analysis/medical_v2_schema.py",
            PROJECT_ROOT / "src/medical_v2_schema.py",
            PROJECT_ROOT / "src/acne/medical_v2_schema.py",
            PROJECT_ROOT / "src/wrinkle/medical_v2_schema.py",
        ]
        for path in paths:
            with self.subTest(path=path):
                content = path.read_text(encoding="utf-8")
                self.assertIn("KEYS_ZH_TO_EN", content)
                self.assertIn("def to_english_document", content)
                self.assertIn("def to_incremental_document", content)
                self.assertIn("_assert_no_chinese_keys(result)", content)

    def test_incremental_registry_does_not_repeat_legacy_canonical_ids(self) -> None:
        for project, incremental in INCREMENTAL_OVERALL_METRICS.items():
            with self.subTest(project=project):
                self.assertFalse(
                    incremental & LEGACY_CANONICAL_METRICS.get(project, frozenset())
                )

    def test_incremental_document_keeps_only_new_english_metrics(self) -> None:
        full_document = {
            "检测项目": "可见斑点",
            "指标版本": "medical_metrics_v2_20260728",
            "评分状态": "uncalibrated",
            "成像与单位说明": {"强度说明": "0～1 工程归一化值"},
            "总体指标": {
                "核心指标": {
                    "数量与密度": {
                        "特征数量（个）": 12,
                        "单位面积密度（个/10万有效皮肤像素）": 34.5,
                    }
                },
                "辅助指标": {
                    "分析范围": {
                        "检测范围": "全面部",
                        "评估状态": "可评估",
                        "有效皮肤面积（像素）": 1000,
                    }
                },
            },
            "分区指标": [],
            "左右比较": {},
            "质量控制": {},
            "医学局限性": [],
        }
        incremental = to_incremental_document(full_document, "spots")
        self.assertFalse(_contains_chinese_key(incremental))
        count_and_density = incremental["overall_metrics"]["core_metrics"][
            "count_and_density"
        ]
        self.assertNotIn("feature_count", count_and_density)
        self.assertEqual(
            count_and_density["feature_density_per_100k_skin_px"], 34.5
        )

    def test_dermavision_project_labels_resolve_without_explicit_override(self) -> None:
        labels = {
            "可见红区": "redness",
            "普通RGB可见斑点": "spots",
            "棕区（普通RGB综合色素代理）": "brown",
            "二维可见表面纹理代理": "texture",
            "可见毛孔负担": "pores",
        }
        for label, project in labels.items():
            metric = next(iter(INCREMENTAL_OVERALL_METRICS[project]))
            document = {
                "project": label,
                "metrics_version": "medical_metrics_v2_20260728",
                "scoring_status": "uncalibrated",
                "imaging_and_units": {},
                "overall_metrics": {
                    "core_metrics": {"distribution": {metric: 1}}
                },
                "region_metrics": [],
                "left_right_comparison": {},
                "quality_control": {},
                "medical_limitations": [],
            }
            with self.subTest(label=label):
                incremental = to_incremental_document(document)
                self.assertEqual(
                    incremental["overall_metrics"]["core_metrics"][
                        "distribution"
                    ][metric],
                    1,
                )


if __name__ == "__main__":
    unittest.main()
