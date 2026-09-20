from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from pydantic import ValidationError

from cloud_contracts import (
    contract_catalog,
    public_metric_documentation_rows,
    validate_failure_envelope,
    validate_worker_envelope,
    worker_field_documentation_rows,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSIONS = {
    "redness": "3",
    "spots": "3",
    "brown": "3",
    "texture": "3",
    "pores": "3",
    "purple": "2",
    "acne": "1",
    "wrinkle": "2",
    "surface_gloss": "1",
    "vascular": "1",
    "contour_firmness": "1",
}
# 这三个定义直接来自 aisia-contracts 的 scoring_input SSOT，不是本地镜像，
# 中文字段说明由契约仓维护；本地 schema 完整性门不重复要求。
EXTERNAL_CONTRACT_DEFINITIONS = {
    "ScoringInputV1",
    "ScoringInputQuality",
    "InputQualityGate",
}


def _scoring_input(algorithm: str) -> dict:
    return {
        "schema_version": "scoring_input_v1",
        "algorithm_name": algorithm,
        "detection_schema_version": SCHEMA_VERSIONS[algorithm],
        "detection_impl_version": "1",
        "report_id": f"contract-{algorithm}",
        "detection_attempt_id": f"attempt-{algorithm}",
        "source_image_sha256": "a" * 64,
        "capture_profile": "consumer",
        "input_route": "consumer_queue",
        "evidence_status": "missing",
        "evidence_field_list_sha256": "b" * 64,
        "quality": {},
        "evidence": {},
        "missing_fields": [],
        "missing_reason": "算法证据缺失",
    }
MANIFEST = json.loads(
    (ROOT / "cloud" / "contracts" / "internal_dev_contracts.json").read_text(
        encoding="utf-8"
    )
)["source_repositories"]


def _dermavision_payload(algorithm: str) -> dict:
    if algorithm in {"brown", "pores"}:
        metrics = {"总计": 3, "额头": 1, "左脸颊": 1, "右脸颊": 1, "鼻部": 0, "下巴": 0}
    elif algorithm == "texture":
        metrics = {
            "总计": 3, "凸起样（黄）": 2, "凹陷样（蓝）": 1,
            "额头": 1, "左脸颊": 1, "右脸颊": 1, "鼻部": 0, "下巴": 0,
        }
    elif algorithm == "surface_gloss":
        metrics = {
            "油光面积占比": {
                name: 0.1 for name in ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")
            },
            "油光区域数量": {
                name: 1 for name in ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")
            },
        }
    elif algorithm == "vascular":
        metrics = {
            "血管样结构数量": {
                name: 1 for name in ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")
            },
            "血管样结构总长度": {
                name: 10.0 for name in ("总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴")
            },
        }
    elif algorithm == "contour_firmness":
        metrics = {
            "中面部曲面连续性": 0.8,
            "下颌缘连续性": 0.7,
            "左右轮廓差异": 0.1,
        }
    elif algorithm == "purple":
        metrics = {
            f"{project}_{region}": 1
            for project in ("uv_spots", "porphyrin")
            for region in (
                "total",
                "forehead",
                "left_cheek",
                "right_cheek",
                "nose",
                "chin",
            )
        }
    elif algorithm == "redness":
        metrics = {
            "algorithm": "RBX-like Visible Redness",
            "metric_version": "v1",
            "source_name": "fixture.jpg",
            "reference_image": None,
            "color_transfer_enabled": False,
            "color_transfer_strength": 1.0,
            "red_area_threshold": 0.45,
            "high_red_area_threshold": 0.7,
            "skin_foreground_area": 100,
            "face_stats_area": 90,
            "red_area_ratio": 0.1,
            "high_red_area_ratio": 0.05,
            "mean_redness": 0.2,
            "p50_redness": 0.2,
            "p90_redness": 0.4,
            "p95_redness": 0.5,
            "max_redness": 0.7,
            "redness_burden": 0.02,
            "region_statistics": {},
            "red_feature_count": 1,
            "red_feature_area": 4,
            "red_feature_area_ratio": 0.01,
            "red_feature_locations": [],
            "red_feature_region_distribution": {},
            "quality_score": 80.0,
            "quality_status": "PASS",
            "quality_flags": [],
            "disclaimer": "fixture",
            "redness_base_mean": 0.1,
            "redness_detail_mean": 0.1,
            "raw_redness_mean": 0.1,
            "render_default": "natural",
            "render_presets": {},
            "color_transfer_scope": "face",
            "eye_rendering": "none",
            "face_mask_source": "fixture",
            "red_feature_overlay_base": "natural",
            "red_feature_marker_count": 1,
            "red_feature_pre_filter_count": 1,
            "red_feature_filtered_count": 0,
            "red_feature_filter_reasons": {
                "eyes": 0, "eyebrows": 0, "nostrils": 0, "lips": 0,
                "boundary": 0, "nasolabial": 0,
            },
        }
    else:
        metrics = {
            "definition": "fixture",
            "spot_count": 1,
            "spot_area_ratio": 0.01,
            "spot_confidence": 0.8,
            "mean_deltaE": 2.0,
            "pre_occlusion_spot_count": 1,
            "hair_filtered_count": 0,
            "small_spot_count": 1,
            "large_spot_count": 0,
            "merged_spot_count": 1,
            "large_spot_area_ratio": 0.0,
            "large_spot_locations": [],
            "large_spot_recall_notes": [],
            "salient_spot_count": 0,
            "nostril_filtered_count": 0,
            "analysis_zone_area": 100,
            "mean_spot_area": 1.0,
            "median_spot_area": 1.0,
            "pre_split_large_component_count": 0,
            "post_split_instance_count": 1,
            "occlusion_filter_statistics": {
                "hair": 0, "eyebrow_eyelash_feature": 0, "facial_hair": 0,
                "nostril": 0, "nasolabial_shadow": 0, "line_like": 0,
                "total": 0, "small_suppressed_by_large": 0,
                "multipeak_parents_split": 0,
            },
            "spot_locations": [],
            "region_distribution": {},
            "parameters": {
                "local_sigmas": [3.0], "large_local_sigmas": [18.0],
                "salient_core_sigmas": [4.0], "scale_z_threshold": 2.45,
                "large_scale_z_threshold": 1.25, "minimum_scale_votes": 2,
                "large_minimum_scale_votes": 2, "salient_minimum_scale_votes": 2,
                "minimum_area_px": 10.0, "maximum_area_ratio": 0.0007,
                "large_minimum_area_px": 90.0, "large_maximum_area_ratio": 0.008,
                "minimum_confidence": 0.4, "large_minimum_confidence": 0.34,
                "salient_minimum_confidence": 0.3,
            },
        }
    raw = {
        "metrics": metrics,
        "quality_score": 80.0,
        "quality_status": "PASS",
        "quality_flags": [],
    }
    if algorithm == "purple":
        raw.update(
            {
                "uv_base": "purple/uv.png",
                "uv_spots_overlay": "purple/uv_spots.jpg",
                "fluorescence_base": "purple/fluorescence.png",
                "porphyrin_overlay": "purple/porphyrin.jpg",
            }
        )
    else:
        raw["overlay"] = f"{algorithm}/result.jpg"
    if algorithm in {"surface_gloss", "vascular", "contour_firmness"}:
        raw["medical_report_csv_v2"] = f"{algorithm}/medical_v2.csv"
    if algorithm == "redness":
        raw["red_areas_overlay"] = "redness/instances.jpg"
    if algorithm == "brown":
        raw["brown_spots_overlay"] = "brown/instances.jpg"
    raw["scoring_input"] = _scoring_input(algorithm)
    return {
        "record_id": f"contract-{algorithm}",
        "status": "success",
        "schema_version": SCHEMA_VERSIONS[algorithm],
        "meta_data": {"name": algorithm, "version": "1"},
        "raw_result": raw,
        "debug_info": {
            "report_csv": f"{algorithm}/report.csv",
            "timing_seconds": {"pipeline": 1.2},
            "execution_mode": "single_algorithm",
        },
    }


def _medical_v2(project: str) -> dict:
    return {
        "metrics_version": "medical_metrics_v2_20260728",
        "scoring_status": "uncalibrated",
        "units": {"project": project},
        "overall_metrics": {},
        "region_metrics": [],
        "left_right_comparison": {},
        "medical_limitations": [],
    }


def _acne_payload() -> dict:
    spec = MANIFEST["acne"]
    raw = {key: f"acne/{key}.json" for key in spec["raw_result_oss_keys"]}
    raw.update(
        {
            "acne_presence": {"status": "detected", "message": "已检测到疑似痤疮"},
            "acne_count": 4,
            "region_counts": [{"region": "额头", "count": 2}],
            "grading_status": "ok",
            "grading_reason": None,
            "detector_status": "ok",
            "detector_reason": None,
            "input_mode": "full_face",
            "detection_scope": "full_face",
            "量化结果": {
                "疑似痤疮圈选数量": 4,
                "痤疮严重程度等级": {"等级": 1, "注释": "1级"},
            },
        }
    )
    return {
        "record_id": "contract-acne",
        "status": "success",
        "schema_version": SCHEMA_VERSIONS["acne"],
        "meta_data": {"name": "acne", "version": "1"},
        "raw_result": raw,
        "debug_info": {
            "display_result": {},
            "algorithm_version": "1",
            "elapsed_seconds": 2.5,
        },
    }


def _wrinkle_payload() -> dict:
    spec = MANIFEST["wrinkle"]
    raw = {key: f"wrinkle/{key}.jpg" for key in spec["raw_result_oss_keys"]}
    raw.update(
        {
            "region_metrics": [{
                "region_key": "forehead", "region_name": "额头纹", "short_name": "FH",
                "relative_score": 50.0, "segment_count": 2, "wrinkle_pixels": 20,
                "mean_segment_length": 10.0, "max_segment_length": 12,
                "density_per_10k": 5.0, "share_pct": 10.0, "area_px": 40000,
            }],
            "run_preset": "balanced",
            "device": "cuda:0",
            "successful_runs": 84,
            "failed_runs": 0,
            "region_analysis_status": "completed",
            "scoring_input": _scoring_input("wrinkle"),
        }
    )
    return {
        "record_id": "contract-wrinkle",
        "status": "success",
        "schema_version": SCHEMA_VERSIONS["wrinkle"],
        "meta_data": {"name": "wrinkle", "version": "1"},
        "raw_result": raw,
        "debug_info": {
            "display_result": {},
            "algorithm_version": "1",
            "elapsed_seconds": 5.5,
            "output_dir": "/tmp/wrinkle-contract",
        },
    }


class PydanticCloudContractTest(unittest.TestCase):
    def test_all_eleven_worker_contracts_validate_without_mutation(self) -> None:
        payloads = {
            **{
                name: _dermavision_payload(name)
                for name in ("redness", "spots", "brown", "texture", "pores", "purple")
            },
            **{
                name: _dermavision_payload(name)
                for name in ("surface_gloss", "vascular", "contour_firmness")
            },
            "acne": _acne_payload(),
            "wrinkle": _wrinkle_payload(),
        }
        for algorithm, payload in payloads.items():
            with self.subTest(algorithm=algorithm):
                before = deepcopy(payload)
                returned = validate_worker_envelope(algorithm, payload)
                self.assertIs(returned, payload)
                self.assertEqual(payload, before)

    def test_raw_result_keys_match_three_agents_contracts(self) -> None:
        for algorithm in (
            "redness", "spots", "brown", "texture", "pores", "purple",
            "surface_gloss", "vascular", "contour_firmness",
        ):
            actual = set(_dermavision_payload(algorithm)["raw_result"])
            expected = set(MANIFEST["dermavision"]["single_algorithm_raw_result"][algorithm])
            self.assertEqual(actual, expected)
        for service, payload in (("acne", _acne_payload()), ("wrinkle", _wrinkle_payload())):
            spec = MANIFEST[service]
            expected = set(spec["raw_result_oss_keys"]) | set(spec["raw_result_structured_keys"])
            self.assertEqual(set(payload["raw_result"]), expected)

    def test_unknown_fields_and_wrong_version_are_rejected(self) -> None:
        payload = _dermavision_payload("pores")
        payload["raw_result"]["unexpected"] = 1
        with self.assertRaises(ValidationError):
            validate_worker_envelope("pores", payload)

    def test_schema_version_is_required_and_matches_algorithm_contract(self) -> None:
        payload = _dermavision_payload("redness")
        del payload["schema_version"]
        with self.assertRaises(ValidationError):
            validate_worker_envelope("redness", payload)

        payload = _dermavision_payload("redness")
        payload["schema_version"] = "1"
        with self.assertRaises((ValidationError, ValueError)):
            validate_worker_envelope("redness", payload)

        payload = _dermavision_payload("pores")
        payload["meta_data"]["version"] = "2"
        with self.assertRaises(ValidationError):
            validate_worker_envelope("pores", payload)

    def test_purple_rejects_chinese_wide_metrics_and_medical_extensions(self) -> None:
        payload = _dermavision_payload("purple")
        payload["raw_result"]["metrics"] = {
            "检测项目": "标准化UV紫外线色斑与紫质工程分析",
            "总体指标": {"特征数量（个）": 1},
        }
        with self.assertRaises(ValidationError):
            validate_worker_envelope("purple", payload)

        payload = _dermavision_payload("purple")
        payload["raw_result"]["medical_metrics_v2"] = _medical_v2("purple")
        with self.assertRaises(ValidationError):
            validate_worker_envelope("purple", payload)

    def test_purple_accepts_partial_face_two_total_metrics(self) -> None:
        payload = _dermavision_payload("purple")
        payload["raw_result"]["metrics"] = {
            "uv_spots_total": 8,
            "porphyrin_total": 5,
        }
        before = deepcopy(payload)
        self.assertIs(validate_worker_envelope("purple", payload), payload)
        self.assertEqual(payload, before)

    def test_failure_contract_supports_existing_service_extensions(self) -> None:
        for payload in (
            {"record_id": "1", "status": "failed", "error_message": "failed"},
            {
                "record_id": "2",
                "status": "failed",
                "error_message": "failed",
                "error_code": "detector_deployment_error",
            },
            {
                "record_id": "3",
                "status": "failed",
                "error_message": "failed",
                "error_details": {"stage": "pipeline"},
            },
        ):
            self.assertIs(validate_failure_envelope(payload), payload)

    def test_every_declared_schema_property_has_chinese_description(self) -> None:
        schemas = contract_catalog()
        self.assertEqual(set(schemas), {
            "redness", "spots", "brown", "texture", "pores", "purple",
            "acne", "acne_v2", "wrinkle", "surface_gloss", "vascular",
            "contour_firmness",
        })
        for algorithm, entry in schemas.items():
            schema = entry["schema"]
            for definition_name, definition in schema.get("$defs", {}).items():
                if definition_name in EXTERNAL_CONTRACT_DEFINITIONS:
                    continue
                for field_name, field_schema in definition.get("properties", {}).items():
                    with self.subTest(
                        algorithm=algorithm,
                        definition=definition_name,
                        field=field_name,
                    ):
                        self.assertTrue(field_schema.get("description"))

    def test_purple_public_metrics_have_twelve_leaf_descriptions(self) -> None:
        payload = _dermavision_payload("purple")
        rows = public_metric_documentation_rows(
            "purple", payload["raw_result"]
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            {row["json_path"] for row in rows},
            {
                f"raw_result.metrics.{project}_{region}"
                for project in ("uv_spots", "porphyrin")
                for region in (
                    "total", "forehead", "left_cheek", "right_cheek", "nose", "chin"
                )
            },
        )
        for row in rows:
            self.assertTrue(any("\u4e00" <= char <= "\u9fff" for char in row["title"]))
            self.assertTrue(row["description"])
            self.assertEqual(row["unit"], "个")
            self.assertEqual(row["value_type"], "integer")

    def test_worker_field_rows_use_json_paths_not_internal_model_names(self) -> None:
        payload = _dermavision_payload("purple")
        rows = worker_field_documentation_rows("purple", payload)
        paths = {row["json_path"] for row in rows}
        self.assertIn("raw_result.metrics", paths)
        self.assertIn("meta_data.version", paths)
        self.assertNotIn("FullMedicalMetricsV2", paths)
        self.assertNotIn("MedicalMetricsV2", paths)


if __name__ == "__main__":
    unittest.main()
