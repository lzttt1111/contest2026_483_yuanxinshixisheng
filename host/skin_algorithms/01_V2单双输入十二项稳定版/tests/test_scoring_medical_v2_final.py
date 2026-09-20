from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.nine_analysis.calibration import (
    database_progress,
    ingest_feature_jsonl,
    ingest_feature_paths,
)
from src.nine_analysis.metrics import extract_item
from src.nine_analysis.scoring import _percentile


class MedicalScoringV2FinalTest(unittest.TestCase):
    def test_constant_metric_does_not_produce_a_fake_percentile(self) -> None:
        profile = {
            "transform": "identity",
            "usable": False,
            "q005": 1.0,
            "q995": 1.0,
        }
        self.assertIsNone(_percentile(1.0, profile))
        profile.pop("usable")
        profile.update({
            "q10": 1.0,
            "q25": 1.0,
            "q50": 1.0,
            "q75": 1.0,
            "q90": 1.0,
            "q95": 1.0,
        })
        self.assertIsNone(_percentile(1.0, profile))

    def test_registry_available_weights_are_valid(self) -> None:
        registry = json.loads(Path("calibration/metric_registry.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["activation_status"], "uncalibrated")
        for item, rule in registry["rules"].items():
            weights = [group["weight"] for group in rule["groups"] if group["available"]]
            self.assertGreater(sum(weights), 0, item)
            self.assertLessEqual(sum(weights), 1.0001, item)
        composite = registry["composites"]["pigmentation"]["components"]
        self.assertAlmostEqual(sum(item["weight"] for item in composite), 1.0)

    def test_public_acne_and_wrinkle_v2_produce_scoring_inputs(self) -> None:
        acne = {
            "metrics_version": "medical_metrics_v2_20260728",
            "overall_metrics": {"core_metrics": {
                "scope_and_morphology": {
                    "feature_density_per_100k_skin_px": 12.0,
                    "candidate_box_area_ratio": 0.03,
                    "p90_candidate_box_area_px": 44.0,
                },
                "signal_intensity": {"p90_confidence": 0.8},
            }},
        }
        wrinkle = {
            "metrics_version": "medical_metrics_v2_20260728",
            "overall_metrics": {"core_metrics": {
                "scope_and_morphology": {
                    "total_wrinkle_length_px": 900,
                    "p90_segment_length_px_proxy": 100,
                    "mean_visible_width_px_proxy": 4.0,
                    "wrinkle_continuity": 0.3,
                },
                "signal_intensity": {"p90_visual_contrast_proxy": 0.6},
            }},
        }
        self.assertTrue(any(extract_item("acne", acne)[1].values()))
        self.assertTrue(any(extract_item("wrinkle", wrinkle)[1].values()))

    def test_incremental_ingest_is_idempotent(self) -> None:
        document = {"九项评分输入": {"redness": {"泛红覆盖负担": {"area_ratio": 0.2}}}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "scoring_features.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            database = root / "metrics.sqlite3"
            first = ingest_feature_paths([source], database)
            second = ingest_feature_paths([source], database)
            self.assertEqual(first["imported"], 1)
            self.assertEqual(second["duplicates"], 1)
            self.assertEqual(database_progress(database)["samples"], 1)

    def test_streaming_jsonl_ingest_is_idempotent_and_skips_invalid_rows(self) -> None:
        observation = {
            "signature": {
                "relative_path": "image.jpg",
                "size": 10,
                "mtime_ns": 20,
            },
            "quality": {"status": "PASS"},
            "scoring_features": {
                "redness": {"泛红覆盖负担": {"area_ratio": 0.2}},
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "scoring.jsonl"
            source.write_text(
                json.dumps(observation, ensure_ascii=False) + "\n{broken\n",
                encoding="utf-8",
            )
            database = root / "metrics.sqlite3"
            first = ingest_feature_jsonl([source], database)
            second = ingest_feature_jsonl([source], database)
            self.assertEqual(first["imported"], 1)
            self.assertEqual(first["invalid"], 1)
            self.assertEqual(second["duplicates"], 1)
            self.assertEqual(database_progress(database)["samples"], 1)
            self.assertEqual(database_progress(database)["by_quality"], {"PASS": 1})


if __name__ == "__main__":
    unittest.main()
