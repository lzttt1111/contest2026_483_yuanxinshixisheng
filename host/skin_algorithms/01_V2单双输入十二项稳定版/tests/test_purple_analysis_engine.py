# -*- coding: utf-8 -*-
"""Purple Analysis 的轻量接口与确定性测试。

测试使用合成数据和注入式假 Analyzer，不加载 MediaPipe 模型，也不修改
现有五项引擎。可由 pytest 或标准 unittest 执行。
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.capture_profile import CaptureProfile
from src.engines.purple_analysis_engine import PurpleAnalysisEngine


class FakePorphyrinAnalyzer:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def detect_porphyrins(self, _preprocess_result):
        self.calls += 1
        return self.result

    def close(self):
        pass


class FakeSpotsAnalyzer:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def detect_spots(self, _preprocess_result):
        self.calls += 1
        return self.result

    def close(self):
        pass


class FakeBrownAnalyzer:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def detect_brown(self, _preprocess_result):
        self.calls += 1
        return self.result

    def close(self):
        pass


class NoDisplayReadPreprocess:
    def __init__(self, image, skin_mask, landmarks, status="PASS"):
        self.analysis_image = image
        self.skin_mask = skin_mask
        self.landmarks = landmarks
        self.quality_score = 90.0
        self.quality_status = status
        self.quality_flags = [] if status != "REJECT" else ["INVALID_IMAGE"]

    @property
    def display_image(self):
        raise AssertionError("Purple Analysis 不得读取 display_image")


class PurpleAnalysisEngineTest(unittest.TestCase):
    def setUp(self):
        height = width = 96
        yy, xx = np.indices((height, width))
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = np.clip(90 + 0.4 * xx, 0, 255)
        image[:, :, 1] = np.clip(105 + 0.2 * yy, 0, 255)
        image[:, :, 2] = np.clip(135 + 0.2 * xx, 0, 255)
        skin = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(skin, (48, 50), (38, 43), 0, 0, 360, 255, -1)
        feature = np.zeros_like(skin)
        cv2.ellipse(feature, (35, 40), (8, 4), 0, 0, 360, 255, -1)
        cv2.ellipse(feature, (61, 40), (8, 4), 0, 0, 360, 255, -1)

        regions = {
            "forehead": np.where((skin > 0) & (yy < 34), 255, 0).astype(np.uint8),
            "left_cheek": np.where((skin > 0) & (xx < 48) & (yy >= 34) & (yy <= 70), 255, 0).astype(np.uint8),
            "right_cheek": np.where((skin > 0) & (xx >= 48) & (yy >= 34) & (yy <= 70), 255, 0).astype(np.uint8),
            "nose": np.where((skin > 0) & (np.abs(xx - 48) < 7) & (yy >= 34) & (yy <= 68), 255, 0).astype(np.uint8),
            "chin": np.where((skin > 0) & (yy > 70), 255, 0).astype(np.uint8),
        }
        # 鼻区优先，模拟互斥区域。
        for name in ("forehead", "left_cheek", "right_cheek", "chin"):
            regions[name][regions["nose"] > 0] = 0
        analysis = np.zeros_like(skin)
        for mask in regions.values():
            analysis = cv2.bitwise_or(analysis, mask)
        display = {name: mask.copy() for name, mask in regions.items()}
        self.regions = SimpleNamespace(
            analysis_mask=analysis,
            feature_exclusion_mask=feature,
            regions=regions,
            display_regions=display,
            partial_face=False,
            display_contour=None,
            display_separator=None,
        )
        self.image = image
        self.skin = skin
        self.landmarks = np.tile(np.array([[48.0, 50.0]], dtype=np.float32), (478, 1))
        self.preprocess = NoDisplayReadPreprocess(image, skin, self.landmarks)

        porphyrin_mask = np.zeros_like(skin)
        cv2.circle(porphyrin_mask, (35, 55), 3, 255, -1)
        porphyrin_score = np.zeros_like(skin, dtype=np.float32)
        porphyrin_score[porphyrin_mask > 0] = 0.9
        self.porphyrin_result = SimpleNamespace(
            porphyrin_mask=porphyrin_mask,
            porphyrin_score_map=porphyrin_score,
            porphyrin_locations=[
                {
                    "id": 1,
                    "centroid": [35, 55],
                    "region": "left_cheek",
                    "confidence": 0.9,
                }
            ],
            region_distribution={
                "left_cheek": {"label": "左脸颊", "count": 1},
                "t_zone": {"label": "T区（工程代理）", "count": 0},
                "non_t_zone": {"label": "非T区（工程代理）", "count": 1},
            },
            analysis_mask=analysis,
            feature_exclusion_mask=feature,
            partial_face=False,
        )

        spots_mask = np.zeros_like(skin)
        cv2.ellipse(spots_mask, (64, 59), (6, 4), 0, 0, 360, 255, -1)
        self.spots_result = SimpleNamespace(
            filtered_mask=spots_mask,
            spot_locations=[
                {
                    "spot_id": 1,
                    "centroid": [64.0, 59.0],
                    "bbox": [58, 55, 13, 9],
                    "area": 70.0,
                    "confidence": 0.8,
                    "spot_type": "salient",
                    "brownness_score": 1.0,
                    "regional_deltaE": 2.0,
                    "local_contrast": 1.1,
                    "salient_dark_difference": 1.0,
                    "salient_red_difference": 0.3,
                    "salient_yellow_difference": 0.9,
                    "prominent_red_rescue": False,
                }
            ],
        )

        brown_mask = np.zeros_like(skin)
        cv2.ellipse(brown_mask, (64, 59), (5, 3), 0, 0, 360, 255, -1)
        brown_score = np.zeros_like(skin, dtype=np.float32)
        brown_score[brown_mask > 0] = 0.92
        self.brown_result = SimpleNamespace(
            brown_score_map=brown_score,
            instance_mask=brown_mask,
            brown_spot_locations=[{"centroid": [64, 59], "area": 45}],
        )

    def _engine(self):
        return PurpleAnalysisEngine(
            porphyrin_analyzer=FakePorphyrinAnalyzer(self.porphyrin_result),
            spots_analyzer=FakeSpotsAnalyzer(self.spots_result),
            brown_analyzer=FakeBrownAnalyzer(self.brown_result),
        )

    def test_result_contract_and_json(self):
        engine = self._engine()
        with patch(
            "src.engines.purple_analysis_engine.build_visia_regions",
            return_value=self.regions,
        ), patch(
            "src.engines.purple_analysis_engine.build_nasolabial_shadow_mask",
            return_value=np.zeros_like(self.skin),
        ):
            result = engine.analyze(self.preprocess)
        for mask in (result.porphyrin_mask, result.uv_spots_mask):
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)).issubset({0, 255}))
        for score in (result.porphyrin_score_map, result.uv_spots_score_map):
            self.assertEqual(score.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(score)))
            self.assertGreaterEqual(float(score.min()), 0.0)
            self.assertLessEqual(float(score.max()), 1.0)
        json.dumps(result.metrics(), ensure_ascii=False)

    def test_centres_inside_analysis_mask(self):
        engine = self._engine()
        with patch(
            "src.engines.purple_analysis_engine.build_visia_regions",
            return_value=self.regions,
        ), patch(
            "src.engines.purple_analysis_engine.build_nasolabial_shadow_mask",
            return_value=np.zeros_like(self.skin),
        ):
            result = engine.analyze(self.preprocess)
        for collection in (result.porphyrin_locations, result.uv_spots_instances):
            for item in collection:
                x, y = item["centroid"]
                self.assertGreater(result.analysis_mask[int(round(y)), int(round(x))], 0)

    def test_determinism_and_mask_exterior_invariance(self):
        engine = self._engine()
        changed = self.image.copy()
        changed[self.skin == 0] = (255, 0, 255)
        changed_preprocess = NoDisplayReadPreprocess(changed, self.skin, self.landmarks)
        with patch(
            "src.engines.purple_analysis_engine.build_visia_regions",
            return_value=self.regions,
        ), patch(
            "src.engines.purple_analysis_engine.build_nasolabial_shadow_mask",
            return_value=np.zeros_like(self.skin),
        ):
            first = engine.analyze(self.preprocess)
            second = engine.analyze(self.preprocess)
            exterior_changed = engine.analyze(changed_preprocess)
        self.assertTrue(np.array_equal(first.porphyrin_mask, second.porphyrin_mask))
        self.assertTrue(np.array_equal(first.uv_spots_mask, second.uv_spots_mask))
        self.assertEqual(first.porphyrin_locations, second.porphyrin_locations)
        self.assertEqual(first.uv_spots_instances, second.uv_spots_instances)
        self.assertTrue(np.array_equal(first.porphyrin_mask, exterior_changed.porphyrin_mask))
        self.assertTrue(np.array_equal(first.uv_spots_mask, exterior_changed.uv_spots_mask))
        self.assertTrue(np.allclose(first.uv_spots_score_map, exterior_changed.uv_spots_score_map))

    def test_feature_holes_never_change_formal_base_pixels(self):
        """五官排除只能删实例，不能在正式底图留下颜色接缝。"""
        engine = self._engine()
        skin_with_holes = self.skin.copy()
        cv2.ellipse(skin_with_holes, (35, 40), (10, 6), 0, 0, 360, 0, -1)
        cv2.ellipse(skin_with_holes, (61, 40), (10, 6), 0, 0, 360, 0, -1)
        preprocess_with_holes = NoDisplayReadPreprocess(
            self.image,
            skin_with_holes,
            self.landmarks,
        )
        with patch(
            "src.engines.purple_analysis_engine.build_visia_regions",
            return_value=self.regions,
        ), patch(
            "src.engines.purple_analysis_engine.build_nasolabial_shadow_mask",
            return_value=np.zeros_like(self.skin),
        ):
            complete = engine.analyze(self.preprocess)
            holes = engine.analyze(preprocess_with_holes)
        self.assertTrue(np.array_equal(complete.uv_like_base, holes.uv_like_base))
        self.assertTrue(np.array_equal(complete.porphyrin_base, holes.porphyrin_base))

    def test_moustache_exclusion_removes_instance_only(self):
        engine = self._engine()
        no_exclusion = np.zeros_like(self.skin)
        exclude_candidate = np.zeros_like(self.skin)
        cv2.circle(exclude_candidate, (35, 55), 6, 255, -1)
        common_patches = (
            patch(
                "src.engines.purple_analysis_engine.build_visia_regions",
                return_value=self.regions,
            ),
            patch(
                "src.engines.purple_analysis_engine.build_nasolabial_shadow_mask",
                return_value=np.zeros_like(self.skin),
            ),
        )
        with common_patches[0], common_patches[1], patch(
            "src.engines.purple_analysis_engine.build_moustache_feature_exclusion_mask",
            return_value=no_exclusion,
        ):
            baseline = engine.analyze(self.preprocess)
        with patch(
            "src.engines.purple_analysis_engine.build_visia_regions",
            return_value=self.regions,
        ), patch(
            "src.engines.purple_analysis_engine.build_nasolabial_shadow_mask",
            return_value=np.zeros_like(self.skin),
        ), patch(
            "src.engines.purple_analysis_engine.build_moustache_feature_exclusion_mask",
            return_value=exclude_candidate,
        ):
            filtered = engine.analyze(self.preprocess)
        self.assertEqual(baseline.porphyrin_count, 1)
        self.assertEqual(filtered.porphyrin_count, 0)
        self.assertTrue(np.array_equal(baseline.uv_like_base, filtered.uv_like_base))
        self.assertTrue(np.array_equal(baseline.porphyrin_base, filtered.porphyrin_base))

    def test_consumer_restores_original_porphyrin_findings(self):
        low_score = self.porphyrin_result.porphyrin_score_map.copy()
        low_score[low_score > 0] = 0.80
        source = SimpleNamespace(
            **{
                **self.porphyrin_result.__dict__,
                "porphyrin_score_map": low_score,
            }
        )
        engine = PurpleAnalysisEngine(
            capture_profile=CaptureProfile.CONSUMER,
            porphyrin_analyzer=FakePorphyrinAnalyzer(source),
        )
        with patch(
            "src.engines.purple_analysis_engine.build_visia_regions",
            return_value=self.regions,
        ), patch(
            "src.engines.purple_analysis_engine.build_nasolabial_shadow_mask",
            return_value=np.zeros_like(self.skin),
        ):
            result = engine.analyze(self.preprocess)

        self.assertEqual(result.porphyrin_count, 1)
        self.assertEqual(result.porphyrin_locations[0]["centroid"], [35, 55])
        self.assertGreater(np.count_nonzero(result.porphyrin_mask), 0)

    def test_save_result_writes_four_formal_images_and_matching_metrics(self):
        engine = self._engine()
        with patch(
            "src.engines.purple_analysis_engine.build_visia_regions",
            return_value=self.regions,
        ), patch(
            "src.engines.purple_analysis_engine.build_nasolabial_shadow_mask",
            return_value=np.zeros_like(self.skin),
        ):
            result = engine.analyze(self.preprocess)
        with tempfile.TemporaryDirectory() as directory:
            paths = engine.save_result(result, directory, "sample")
            sample_dir = os.path.join(directory, "sample")
            formal_images = sorted(
                name
                for name in os.listdir(sample_dir)
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            self.assertEqual(
                formal_images,
                [
                    "01_紫外线色斑底图.png",
                    "02_紫外线色斑检测结果.jpg",
                    "03_紫质荧光底图.png",
                    "04_紫质检测结果.jpg",
                ],
            )
            self.assertEqual(len(paths), 13)
            for path in paths.values():
                self.assertTrue(os.path.isfile(path), path)
            self.assertEqual(
                sorted(
                    name
                    for name in os.listdir(sample_dir)
                    if name.lower().endswith(".json")
                ),
                ["紫区完整指标.json", "紫区量化指标.json"],
            )
            self.assertEqual(
                sorted(
                    name
                    for name in os.listdir(sample_dir)
                    if name.lower().endswith(".csv")
                ),
                ["紫区医学量化指标_V2.csv", "紫区量化指标.csv"],
            )
            with open(paths["metrics_json"], encoding="utf-8") as handle:
                metrics_json = json.load(handle)
            with open(paths["metrics_csv"], encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(metrics_json), 12)
            self.assertEqual(len(rows), 2)
            with open(paths["full_metrics_json"], encoding="utf-8") as handle:
                full_metrics = json.load(handle)
            self.assertEqual(set(full_metrics), {"uv_spots", "porphyrin"})
            self.assertTrue(full_metrics["uv_spots"])
            self.assertTrue(full_metrics["porphyrin"])
            self.assertEqual(
                list(rows[0]),
                ["检测项目", "总计", "额头", "左脸颊", "右脸颊", "鼻部", "下巴"],
            )
            self.assertEqual(rows[0]["检测项目"], "紫外线色斑")
            self.assertEqual(rows[1]["检测项目"], "紫质")
            self.assertEqual(
                rows[0]["总计"],
                str(result.uv_spots_count),
            )
            self.assertEqual(
                rows[1]["总计"],
                str(result.porphyrin_count),
            )
            self.assertEqual(
                {
                    key.removeprefix("uv_spots_"): value
                    for key, value in metrics_json.items()
                    if key.startswith("uv_spots_")
                },
                {
                    "total": result.uv_spots_count,
                    "forehead": result.uv_spots_region_distribution.get(
                        "forehead", {}
                    ).get("count", 0),
                    "left_cheek": result.uv_spots_region_distribution.get(
                        "left_cheek", {}
                    ).get("count", 0),
                    "right_cheek": result.uv_spots_region_distribution.get(
                        "right_cheek", {}
                    ).get("count", 0),
                    "nose": result.uv_spots_region_distribution.get(
                        "nose", {}
                    ).get("count", 0),
                    "chin": result.uv_spots_region_distribution.get(
                        "chin", {}
                    ).get("count", 0),
                },
            )
            self.assertEqual(
                {
                    key.removeprefix("porphyrin_"): value
                    for key, value in metrics_json.items()
                    if key.startswith("porphyrin_")
                },
                {
                    "total": result.porphyrin_count,
                    "forehead": result.porphyrin_region_distribution.get(
                        "forehead", {}
                    ).get("count", 0),
                    "left_cheek": result.porphyrin_region_distribution.get(
                        "left_cheek", {}
                    ).get("count", 0),
                    "right_cheek": result.porphyrin_region_distribution.get(
                        "right_cheek", {}
                    ).get("count", 0),
                    "nose": result.porphyrin_region_distribution.get(
                        "nose", {}
                    ).get("count", 0),
                    "chin": result.porphyrin_region_distribution.get(
                        "chin", {}
                    ).get("count", 0),
                },
            )
            self.assertNotIn("目标明细", metrics_json)
            self.assertLess(
                len(json.dumps(metrics_json, ensure_ascii=False)),
                500,
            )
            self.assertTrue(all(key.isascii() for key in metrics_json))
            self.assertTrue(all(type(value) is int for value in metrics_json.values()))

    def test_uv_spots_is_independent_from_spots_and_brown(self):
        porphyrin = FakePorphyrinAnalyzer(self.porphyrin_result)
        spots = FakeSpotsAnalyzer(self.spots_result)
        brown = FakeBrownAnalyzer(self.brown_result)
        engine = PurpleAnalysisEngine(
            porphyrin_analyzer=porphyrin,
            spots_analyzer=spots,
            brown_analyzer=brown,
        )
        with patch(
            "src.engines.purple_analysis_engine.build_visia_regions",
            return_value=self.regions,
        ), patch(
            "src.engines.purple_analysis_engine.build_nasolabial_shadow_mask",
            return_value=np.zeros_like(self.skin),
        ):
            engine.analyze(self.preprocess)
        self.assertEqual(porphyrin.calls, 1)
        self.assertEqual(spots.calls, 0)
        self.assertEqual(brown.calls, 0)

    def test_reject_short_circuits_without_analyzers(self):
        porphyrin = FakePorphyrinAnalyzer(self.porphyrin_result)
        spots = FakeSpotsAnalyzer(self.spots_result)
        brown = FakeBrownAnalyzer(self.brown_result)
        engine = PurpleAnalysisEngine(
            porphyrin_analyzer=porphyrin,
            spots_analyzer=spots,
            brown_analyzer=brown,
        )
        rejected = NoDisplayReadPreprocess(
            self.image,
            self.skin,
            self.landmarks,
            status="REJECT",
        )
        self.assertIsNone(engine.process_preprocess_result(rejected, "/tmp", "reject.jpg"))
        self.assertEqual(porphyrin.calls, 0)
        self.assertEqual(spots.calls, 0)
        self.assertEqual(brown.calls, 0)


if __name__ == "__main__":
    unittest.main()
