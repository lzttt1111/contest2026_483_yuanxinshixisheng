from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import worker
from src.medical_v2_delivery import (
    document_rows,
    validate_csv_matches_document,
)
from src.medical_v2_schema import to_english_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE = PROJECT_ROOT / "data" / "reference.jpg"


class _FakePipeline:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.calls: list[list[str]] = []
        self.with_medical_v2 = False

    def _write_medical_v2(self, algorithm: str, label: str) -> dict[str, str]:
        document = {
            "检测项目": label,
            "指标版本": "medical_metrics_v2_20260728",
            "评分状态": "uncalibrated",
            "成像与单位说明": {},
            "总体指标": {
                "核心指标": {"数量与密度": {"特征数量（个）": 1}},
                "辅助指标": {"分析范围": {
                    "检测范围": "全面部",
                    "评估状态": "可评估",
                    "有效皮肤面积（像素）": 100,
                }},
            },
            "分区指标": [],
            "左右比较": {},
            "质量控制": {},
            "医学局限性": [],
        }
        json_path = self.output_dir / f"{algorithm}_量化指标.json"
        csv_path = self.output_dir / f"{algorithm}_医学量化指标_V2.csv"
        public_document = to_english_document(document)
        json_path.write_text(
            json.dumps(
                {"旧字段": 1, "medical_metrics_v2": public_document},
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        rows = document_rows(document)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return {"json": str(json_path), "csv": str(csv_path)}

    def process_single(self, image_path: str, algorithms: list[str]) -> dict:
        self.calls.append(list(algorithms))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        names = {
            "redness": "03_RBX红区结果图.jpg",
            "red_areas_overlay": "06_VISIA红色区实例图.jpg",
            "redness_report": "红区量化指标.csv",
            "redness_metrics": "红区量化指标.json",
            "spots": "01_Spots斑点结果图.jpg",
            "spots_report": "02_Spots量化指标.csv",
            "spots_metrics": "02_Spots量化指标.json",
            "brown": "01_RBX棕区结果图.jpg",
            "brown_spots_overlay": "02_VISIA棕色斑实例图.jpg",
            "brown_report": "棕色斑量化指标.csv",
            "brown_metrics": "棕色斑量化指标.json",
            "texture": "01_纹理检测结果图.jpg",
            "texture_report": "纹理量化指标.csv",
            "texture_metrics": "纹理量化指标.json",
            "pores": "01_毛孔检测结果图.jpg",
            "pores_report": "毛孔量化指标.csv",
            "pores_metrics": "毛孔量化指标.json",
            "purple_uv_base": "01_紫外线色斑底图.png",
            "purple_uv_spots_overlay": "02_紫外线色斑检测结果.jpg",
            "purple_fluorescence_base": "03_紫质荧光底图.png",
            "purple_porphyrin_overlay": "04_紫质检测结果.jpg",
            "purple_report": "紫区量化指标.csv",
            "purple_metrics": "紫区量化指标.json",
            "surface_gloss": "01_表面油光检测结果图.jpg",
            "surface_gloss_report": "表面油光量化指标.csv",
            "surface_gloss_metrics": "表面油光量化指标.json",
            "surface_gloss_medical_report_csv_v2": "表面油光医学量化指标_V2.csv",
            "vascular": "01_血管样结构检测结果图.jpg",
            "vascular_report": "血管样结构量化指标.csv",
            "vascular_metrics": "血管样结构量化指标.json",
            "vascular_medical_report_csv_v2": "血管样结构医学量化指标_V2.csv",
            "contour_firmness": "01_轮廓紧致度检测结果图.jpg",
            "contour_firmness_report": "轮廓紧致度量化指标.csv",
            "contour_firmness_metrics": "轮廓紧致度量化指标.json",
            "contour_firmness_medical_report_csv_v2": "轮廓紧致度医学量化指标_V2.csv",
        }
        results = {}
        for key, filename in names.items():
            path = self.output_dir / filename
            if path.suffix == ".csv":
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    if key == "purple_report":
                        csv.writer(handle).writerows([
                            ["检测项目", "总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"],
                            ["紫外线色斑", 0, 0, 0, 0, 0, 0],
                            ["紫质", 0, 0, 0, 0, 0, 0],
                        ])
                    else:
                        csv.writer(handle).writerows([["总计"], [0]])
            elif path.suffix == ".json":
                if key == "redness_metrics":
                    payload = {
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
                        "red_area_ratio": 0.125,
                        "high_red_area_ratio": 0.05,
                        "mean_redness": 0.2,
                        "p50_redness": 0.2,
                        "p90_redness": 0.4,
                        "p95_redness": 0.5,
                        "max_redness": 0.7,
                        "redness_burden": 0.02,
                        "region_statistics": {},
                        "red_feature_count": 0,
                        "red_feature_area": 0,
                        "red_feature_area_ratio": 0.0,
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
                        "red_feature_marker_count": 0,
                        "red_feature_pre_filter_count": 0,
                        "red_feature_filtered_count": 0,
                        "red_feature_filter_reasons": {
                            "eyes": 0, "eyebrows": 0, "nostrils": 0,
                            "lips": 0, "boundary": 0, "nasolabial": 0,
                        },
                    }
                elif key == "spots_metrics":
                    payload = {
                        "definition": "VISIA-like visible spots",
                        "spot_count": 0,
                        "spot_area_ratio": 0.0,
                        "spot_confidence": 0.0,
                        "mean_deltaE": 0.0,
                        "pre_occlusion_spot_count": 0,
                        "hair_filtered_count": 0,
                        "small_spot_count": 0,
                        "large_spot_count": 0,
                        "merged_spot_count": 0,
                        "large_spot_area_ratio": 0.0,
                        "spot_locations": [],
                        "large_spot_locations": [],
                        "large_spot_recall_notes": [],
                        "salient_spot_count": 0,
                        "nostril_filtered_count": 0,
                        "analysis_zone_area": 100,
                        "mean_spot_area": 0.0,
                        "median_spot_area": 0.0,
                        "pre_split_large_component_count": 0,
                        "post_split_instance_count": 0,
                        "occlusion_filter_statistics": {
                            "hair": 0, "eyebrow_eyelash_feature": 0,
                            "facial_hair": 0, "nostril": 0,
                            "nasolabial_shadow": 0, "line_like": 0,
                            "total": 0, "small_suppressed_by_large": 0,
                            "multipeak_parents_split": 0,
                        },
                        "region_distribution": {},
                        "parameters": {
                            "local_sigmas": [3.0], "large_local_sigmas": [18.0],
                            "salient_core_sigmas": [4.0], "scale_z_threshold": 2.45,
                            "large_scale_z_threshold": 1.25, "minimum_scale_votes": 2,
                            "large_minimum_scale_votes": 2,
                            "salient_minimum_scale_votes": 2,
                            "minimum_area_px": 10.0, "maximum_area_ratio": 0.0007,
                            "large_minimum_area_px": 90.0,
                            "large_maximum_area_ratio": 0.008,
                            "minimum_confidence": 0.4,
                            "large_minimum_confidence": 0.34,
                            "salient_minimum_confidence": 0.3,
                        },
                    }
                elif key == "purple_metrics":
                    payload = {
                        "uv_spots_total": 0,
                        "uv_spots_forehead": 0,
                        "uv_spots_left_cheek": 0,
                        "uv_spots_right_cheek": 0,
                        "uv_spots_nose": 0,
                        "uv_spots_chin": 0,
                        "porphyrin_total": 0,
                        "porphyrin_forehead": 0,
                        "porphyrin_left_cheek": 0,
                        "porphyrin_right_cheek": 0,
                        "porphyrin_nose": 0,
                        "porphyrin_chin": 0,
                    }
                elif key == "surface_gloss_metrics":
                    payload = {
                        "油光面积占比": {
                            name: 0.1 for name in (
                                "总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"
                            )
                        },
                        "油光区域数量": {
                            name: 1 for name in (
                                "总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"
                            )
                        },
                    }
                elif key == "vascular_metrics":
                    payload = {
                        "血管样结构数量": {
                            name: 1 for name in (
                                "总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"
                            )
                        },
                        "血管样结构总长度": {
                            name: 10.0 for name in (
                                "总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"
                            )
                        },
                    }
                elif key == "contour_firmness_metrics":
                    payload = {
                        "中面部曲面连续性": 0.8,
                        "下颌缘连续性": 0.7,
                        "左右轮廓差异": 0.1,
                    }
                else:
                    payload = {"总计": 0}
                path.write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                path.write_bytes(b"contract-image")
            results[key] = str(path)
        medical_v2_results = {}
        if self.with_medical_v2:
            for algorithm in algorithms:
                if algorithm == "purple":
                    spec = self._write_medical_v2(
                        "purple", "紫区（紫外线色斑与紫质）"
                    )
                    payload = json.loads(Path(spec["json"]).read_text(encoding="utf-8"))
                    payload["medical_metrics_v2"]["overall_metrics"] = {
                        "uv_spots": {},
                        "porphyrin": {},
                    }
                    Path(spec["json"]).write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    medical_v2_results[algorithm] = spec
                else:
                    medical_v2_results[algorithm] = self._write_medical_v2(
                        algorithm, algorithm
                    )
        return {
            "status": "success",
            "results": results,
            "medical_v2_results": medical_v2_results,
            "cleanup_paths": [str(self.output_dir)],
            "metadata": {"quality_status": "PASS"},
        }


class _SelectiveFailurePipeline(_FakePipeline):
    def process_single(self, image_path: str, algorithms: list[str]) -> dict:
        if algorithms == ["brown"]:
            self.calls.append(list(algorithms))
            return {
                "status": "failed",
                "message": "模拟棕区算法失败",
                "cleanup_paths": [],
            }
        return super().process_single(image_path, algorithms)


class WorkerContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output_dir = PROJECT_ROOT / "output" / "worker_contract_test"
        shutil.rmtree(self.output_dir, ignore_errors=True)
        self.uploads: list[tuple[str, str]] = []
        self.pipeline = _FakePipeline(self.output_dir)
        # 本合同测试只关心信封结构，避免在单测中运行 MediaPipe 原图门禁模型；
        # 门禁由"下载原图 bytes 计算"的接线由 test_input_quality_gate_worker.py 覆盖。
        self._gate_patch = mock.patch.object(
            worker,
            "compute_input_quality_gate",
            return_value={"status": "PASS", "reason_codes": []},
        )
        self._gate_patch.start()

    def tearDown(self) -> None:
        self._gate_patch.stop()
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _run_worker(self, algorithms: list[str] | None) -> dict:
        image_bytes = TEST_IMAGE.read_bytes()

        def upload(local_path: str, report_id: str, algo_name: str, file_name: str) -> str:
            object_key = f"report/{report_id}/{algo_name}/{file_name}"
            self.uploads.append((local_path, object_key))
            return object_key

        with (
            mock.patch.object(worker, "_pipeline", self.pipeline),
            mock.patch.object(worker, "download_via_internal_api", return_value=image_bytes),
            mock.patch.object(worker, "upload_via_internal_api", side_effect=upload),
        ):
            return worker.analyze_image.run(
                "contract-test",
                "input/test.jpg",
                "dermavision/v1/",
                algorithms,
            )

    def _assert_single_envelope(self, result: dict, algorithm: str) -> None:
        self.assertEqual(
            set(result),
            {
                "record_id", "status", "schema_version",
                "meta_data", "raw_result", "debug_info",
            },
        )
        self.assertEqual(result["record_id"], "contract-test")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["schema_version"], worker._SCHEMA_VERSIONS[algorithm])
        self.assertEqual(
            result["meta_data"],
            {"name": algorithm, "version": "1"},
        )
        expected_raw = {
            "overlay", "metrics", "quality_score", "quality_status", "quality_flags",
            "scoring_input",
        }
        if algorithm in {"surface_gloss", "vascular", "contour_firmness"}:
            expected_raw.add("medical_report_csv_v2")
        if algorithm == "purple":
            expected_raw = {
                "uv_base",
                "uv_spots_overlay",
                "fluorescence_base",
                "porphyrin_overlay",
                "metrics",
                "quality_score",
                "quality_status",
                "quality_flags",
                "scoring_input",
            }
        if algorithm == "redness":
            expected_raw.add("red_areas_overlay")
        if algorithm == "brown":
            expected_raw.add("brown_spots_overlay")
        self.assertEqual(set(result["raw_result"]), expected_raw)
        metrics = result["raw_result"]["metrics"]
        if algorithm == "redness":
            self.assertEqual(metrics["algorithm"], "RBX-like Visible Redness")
            self.assertIn("red_area_ratio", metrics)
            self.assertEqual(metrics["red_feature_locations"], [])
        elif algorithm == "spots":
            self.assertIn("spot_count", metrics)
            self.assertEqual(metrics["spot_locations"], [])
            self.assertEqual(metrics["large_spot_locations"], [])
        elif algorithm == "purple":
            self.assertEqual(
                metrics,
                {
                    "uv_spots_total": 0,
                    "uv_spots_forehead": 0,
                    "uv_spots_left_cheek": 0,
                    "uv_spots_right_cheek": 0,
                    "uv_spots_nose": 0,
                    "uv_spots_chin": 0,
                    "porphyrin_total": 0,
                    "porphyrin_forehead": 0,
                    "porphyrin_left_cheek": 0,
                    "porphyrin_right_cheek": 0,
                    "porphyrin_nose": 0,
                    "porphyrin_chin": 0,
                },
            )
        elif algorithm == "surface_gloss":
            self.assertEqual(
                set(metrics),
                {"油光面积占比", "油光区域数量"},
            )
        elif algorithm == "vascular":
            self.assertEqual(
                set(metrics),
                {"血管样结构数量", "血管样结构总长度"},
            )
        elif algorithm == "contour_firmness":
            self.assertEqual(
                set(metrics),
                {"中面部曲面连续性", "下颌缘连续性", "左右轮廓差异"},
            )
        else:
            self.assertEqual(metrics, {"总计": 0})
        self.assertEqual(
            set(result["debug_info"]),
            {"report_csv", "timing_seconds", "execution_mode"},
        )
        self.assertEqual(
            result["debug_info"]["execution_mode"],
            "single_algorithm_consumer",
        )

    def test_redness_and_spots_return_exact_five_files(self) -> None:
        result = self._run_worker(["redness", "spots"])

        expected_keys = {
            "redness",
            "red_areas_overlay",
            "redness_report",
            "redness_metrics",
            "spots",
            "spots_report",
            "spots_metrics",
        }
        self.assertEqual(result["status"], "success")
        self.assertEqual(set(result["raw_result"]), expected_keys)
        self.assertEqual(len(self.uploads), 5)
        self.assertTrue(all("contract-test/" in key for _, key in self.uploads))
        self.assertIn("03_RBX红区结果图.jpg", result["raw_result"]["redness"])
        self.assertIn(
            "06_VISIA红色区实例图.jpg",
            result["raw_result"]["red_areas_overlay"],
        )
        self.assertIn("红区量化指标.csv", result["raw_result"]["redness_report"])
        redness_metrics = result["raw_result"]["redness_metrics"]
        self.assertEqual(redness_metrics["algorithm"], "RBX-like Visible Redness")
        self.assertEqual(redness_metrics["red_area_ratio"], 0.125)
        self.assertEqual(redness_metrics["region_statistics"], {})
        self.assertEqual(redness_metrics["red_feature_locations"], [])
        self.assertIn("01_Spots斑点结果图.jpg", result["raw_result"]["spots"])
        self.assertIn("02_Spots量化指标.csv", result["raw_result"]["spots_report"])
        spots_metrics = result["raw_result"]["spots_metrics"]
        self.assertEqual(spots_metrics["definition"], "VISIA-like visible spots")
        self.assertEqual(spots_metrics["spot_count"], 0)
        self.assertEqual(spots_metrics["spot_locations"], [])
        self.assertEqual(spots_metrics["large_spot_locations"], [])
        self.assertEqual(spots_metrics["region_distribution"], {})
        self.assertEqual(self.pipeline.calls, [["redness", "spots"]])
        self.assertFalse(self.output_dir.exists())

    def test_redness_only_returns_three_files(self) -> None:
        result = self._run_worker(["redness"])
        self._assert_single_envelope(result, "redness")
        self.assertEqual(len(self.uploads), 3)
        self.assertIn("03_RBX红区结果图.jpg", result["raw_result"]["overlay"])
        self.assertIn("红区量化指标.csv", result["debug_info"]["report_csv"])
        self.assertEqual(self.pipeline.calls, [["redness"]])

    def test_spots_only_returns_two_files(self) -> None:
        result = self._run_worker(["spots"])
        self._assert_single_envelope(result, "spots")
        self.assertEqual(len(self.uploads), 2)
        self.assertEqual(self.pipeline.calls, [["spots"]])

    def test_brown_only_returns_three_uploads_and_inline_metrics(self) -> None:
        result = self._run_worker(["brown"])
        self._assert_single_envelope(result, "brown")
        self.assertEqual(len(self.uploads), 3)
        self.assertEqual(self.pipeline.calls, [["brown"]])

    def test_texture_only_returns_two_uploads_and_inline_metrics(self) -> None:
        result = self._run_worker(["texture"])
        self._assert_single_envelope(result, "texture")
        self.assertEqual(len(self.uploads), 2)
        self.assertEqual(self.pipeline.calls, [["texture"]])

    def test_pores_only_returns_two_uploads_and_inline_metrics(self) -> None:
        result = self._run_worker(["pores"])
        self._assert_single_envelope(result, "pores")
        self.assertEqual(len(self.uploads), 2)
        self.assertEqual(self.pipeline.calls, [["pores"]])

    def test_purple_returns_four_images_csv_and_inline_metrics(self) -> None:
        result = self._run_worker(["purple"])
        self._assert_single_envelope(result, "purple")
        self.assertEqual(len(self.uploads), 5)
        raw = result["raw_result"]
        self.assertIn("01_紫外线色斑底图.png", raw["uv_base"])
        self.assertIn("02_紫外线色斑检测结果.jpg", raw["uv_spots_overlay"])
        self.assertIn("03_紫质荧光底图.png", raw["fluorescence_base"])
        self.assertIn("04_紫质检测结果.jpg", raw["porphyrin_overlay"])
        self.assertEqual(
            set(raw["metrics"]),
            {
                "uv_spots_total", "uv_spots_forehead", "uv_spots_left_cheek",
                "uv_spots_right_cheek", "uv_spots_nose", "uv_spots_chin",
                "porphyrin_total", "porphyrin_forehead",
                "porphyrin_left_cheek", "porphyrin_right_cheek",
                "porphyrin_nose", "porphyrin_chin",
            },
        )
        self.assertEqual(raw["metrics"]["uv_spots_total"], 0)
        self.assertEqual(raw["metrics"]["porphyrin_total"], 0)
        self.assertIn("紫区量化指标.csv", result["debug_info"]["report_csv"])
        self.assertEqual(self.pipeline.calls, [["purple"]])

    def test_three_additive_algorithms_keep_six_field_envelopes(self) -> None:
        for algorithm in ("surface_gloss", "vascular", "contour_firmness"):
            with self.subTest(algorithm=algorithm):
                self.uploads.clear()
                result = self._run_worker([algorithm])
                self._assert_single_envelope(result, algorithm)
                self.assertEqual(len(self.uploads), 3)
                self.assertIn("medical_report_csv_v2", result["raw_result"])
                self.assertIn("医学量化指标_V2.csv", result["raw_result"]["medical_report_csv_v2"])

    def test_medical_v2_enabled_adds_only_two_optional_fields(self) -> None:
        self.pipeline.with_medical_v2 = True
        with mock.patch.object(worker.settings, "enable_medical_metrics_v2", True):
            result = self._run_worker(["redness"])
        expected_old = {
            "overlay", "metrics", "quality_score", "quality_status",
            "quality_flags", "red_areas_overlay", "scoring_input",
        }
        self.assertEqual(
            set(result["raw_result"]),
            expected_old | {"medical_metrics_v2", "medical_report_csv_v2"},
        )
        self.assertEqual(
            result["raw_result"]["medical_metrics_v2"]["scoring_status"],
            "uncalibrated",
        )
        self.assertIn(
            "redness_医学量化指标_V2.csv",
            result["raw_result"]["medical_report_csv_v2"],
        )

    def test_medical_v2_failure_does_not_break_old_success(self) -> None:
        with mock.patch.object(worker.settings, "enable_medical_metrics_v2", True):
            result = self._run_worker(["spots"])
        self.assertEqual(result["status"], "success")
        self.assertNotIn("medical_metrics_v2", result["raw_result"])
        self.assertNotIn("medical_report_csv_v2", result["raw_result"])

    def test_purple_does_not_expose_internal_medical_wide_table(self) -> None:
        self.pipeline.with_medical_v2 = True
        with mock.patch.object(worker.settings, "enable_medical_metrics_v2", True):
            result = self._run_worker(["purple"])
        self.assertNotIn("medical_metrics_v2", result["raw_result"])
        self.assertNotIn("medical_report_csv_v2", result["raw_result"])

    def test_purple_combined_medical_csv_matches_by_project_and_region(self) -> None:
        def metrics(project: str, count: int) -> dict:
            return {
                "核心指标": {"数量与密度": {"特征数量（个）": count}},
                "辅助指标": {"分析范围": {
                    "检测范围": "全面部",
                    "评估状态": "可评估",
                    "有效皮肤面积（像素）": 100,
                }},
            }

        document = {
            "总体指标": {
                "标准化UV紫外线色斑工程代理": metrics("uv_spots", 2),
                "标准化荧光UV紫质工程代理": metrics("porphyrin", 3),
            },
            "分区指标": [
                {"subproject": "uv_spots", "regions": []},
                {"subproject": "porphyrin", "regions": []},
            ],
        }
        rows = document_rows(document)
        self.assertEqual(
            {(row["检测项目"], row["检测范围"]) for row in rows},
            {
                ("标准化UV紫外线色斑工程代理", "全面部"),
                ("标准化荧光UV紫质工程代理", "全面部"),
            },
        )
        csv_path = self.output_dir / "紫区医学量化指标_V2.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        columns: list[str] = []
        for row in rows:
            columns.extend(name for name in row if name not in columns)
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        validate_csv_matches_document(csv_path, document)

    def test_all_five_algorithms_return_exact_seventeen_fields(self) -> None:
        algorithms = ["redness", "spots", "brown", "texture", "pores"]
        result = self._run_worker(algorithms)
        expected_keys = {
            "redness",
            "red_areas_overlay",
            "redness_report",
            "redness_metrics",
            "spots",
            "spots_report",
            "spots_metrics",
            "brown",
            "brown_spots_overlay",
            "brown_report",
            "brown_metrics",
            "texture",
            "texture_report",
            "texture_metrics",
            "pores",
            "pores_report",
            "pores_metrics",
        }
        self.assertEqual(result["status"], "success")
        self.assertEqual(set(result["raw_result"]), expected_keys)
        self.assertEqual(len(self.uploads), 12)
        self.assertIn("red_area_ratio", result["raw_result"]["redness_metrics"])
        self.assertIn("spot_locations", result["raw_result"]["spots_metrics"])
        for key in ("brown_metrics", "texture_metrics", "pores_metrics"):
            self.assertEqual(result["raw_result"][key], {"总计": 0})
            self.assertTrue(
                all(
                    any("\u4e00" <= character <= "\u9fff" for character in name)
                    for name in result["raw_result"][key]
                ),
                msg=f"{key} 的量化指标名称必须使用中文",
            )
        self.assertEqual(self.pipeline.calls, [algorithms])

    def test_invalid_algorithm_request_fails_before_download(self) -> None:
        with mock.patch.object(worker, "download_via_internal_api") as download:
            result = worker.analyze_image.run(
                "invalid-request",
                "input/test.jpg",
                "dermavision/v1/",
                ["redness", "redness"],
            )
        self.assertEqual(result["status"], "failed")
        self.assertIn("重复项目", result["error_message"])
        download.assert_not_called()

    def test_metrics_contract_rejects_english_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_path = root / "指标.json"
            report_path = root / "指标.csv"
            metrics_path.write_text('{"total": 3}', encoding="utf-8")
            with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
                csv.writer(handle).writerows([["total"], [3]])
            with self.assertRaisesRegex(ValueError, "必须使用中文"):
                worker._load_and_validate_metrics(
                    str(metrics_path),
                    str(report_path),
                )

    def test_metrics_contract_rejects_json_csv_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_path = root / "指标.json"
            report_path = root / "指标.csv"
            metrics_path.write_text('{"总计": 3}', encoding="utf-8")
            with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
                csv.writer(handle).writerows([["总计"], [4]])
            with self.assertRaisesRegex(ValueError, "不一致"):
                worker._load_and_validate_metrics(
                    str(metrics_path),
                    str(report_path),
                )

    def test_purple_metrics_contract_supports_partial_face_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_path = root / "紫区量化指标.json"
            report_path = root / "紫区量化指标.csv"
            metrics_path.write_text(
                '{"uv_spots_total": 12, "porphyrin_total": 9}',
                encoding="utf-8",
            )
            with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
                csv.writer(handle).writerows([
                    ["检测项目", "总计"],
                    ["紫外线色斑", 12],
                    ["紫质", 9],
                ])
            self.assertEqual(
                worker._load_and_validate_purple_metrics(
                    str(metrics_path), str(report_path)
                ),
                {"uv_spots_total": 12, "porphyrin_total": 9},
            )

    def test_purple_metrics_contract_rejects_json_csv_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_path = root / "紫区量化指标.json"
            report_path = root / "紫区量化指标.csv"
            metrics_path.write_text(
                '{"uv_spots_total": 12, "porphyrin_total": 9}',
                encoding="utf-8",
            )
            with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
                csv.writer(handle).writerows([
                    ["检测项目", "总计"],
                    ["紫外线色斑", 13],
                    ["紫质", 9],
                ])
            with self.assertRaisesRegex(ValueError, "不一致"):
                worker._load_and_validate_purple_metrics(
                    str(metrics_path), str(report_path)
                )

    def test_independent_task_failure_does_not_block_other_algorithms(self) -> None:
        pipeline = _SelectiveFailurePipeline(self.output_dir)
        image_bytes = TEST_IMAGE.read_bytes()
        uploads: list[tuple[str, str]] = []

        def upload(local_path: str, report_id: str, algo_name: str, file_name: str) -> str:
            object_key = f"report/{report_id}/{algo_name}/{file_name}"
            uploads.append((local_path, object_key))
            return object_key

        results = {}
        with (
            mock.patch.object(worker, "_pipeline", pipeline),
            mock.patch.object(
                worker,
                "download_via_internal_api",
                return_value=image_bytes,
            ),
            mock.patch.object(
                worker,
                "upload_via_internal_api",
                side_effect=upload,
            ),
        ):
            for algorithm in ("redness", "spots", "brown", "texture", "pores"):
                record_id = f"independent-{algorithm}"
                results[algorithm] = worker.analyze_image.run(
                    record_id,
                    "input/shared.jpg",
                    "dermavision/v1/",
                    [algorithm],
                )

        self.assertEqual(results["brown"]["status"], "failed")
        self.assertIn("模拟棕区算法失败", results["brown"]["error_message"])
        for algorithm in ("redness", "spots", "texture", "pores"):
            self.assertEqual(results[algorithm]["status"], "success")
            self.assertEqual(
                results[algorithm]["meta_data"],
                {"name": algorithm, "version": "1"},
            )
            self.assertEqual(
                results[algorithm]["debug_info"]["execution_mode"],
                "single_algorithm_consumer",
            )

        uploaded_keys = [oss_key for _, oss_key in uploads]
        for algorithm in ("redness", "spots", "texture", "pores"):
            self.assertTrue(
                any(f"independent-{algorithm}/" in key for key in uploaded_keys)
            )

    def test_default_request_runs_all_five_algorithms(self) -> None:
        result = self._run_worker(None)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["raw_result"]), 17)
        self.assertEqual(len(self.uploads), 12)
        self.assertEqual(
            self.pipeline.calls,
            [[
                "redness",
                "spots",
                "brown",
                "texture",
                "pores",
            ]],
        )


if __name__ == "__main__":
    unittest.main()
