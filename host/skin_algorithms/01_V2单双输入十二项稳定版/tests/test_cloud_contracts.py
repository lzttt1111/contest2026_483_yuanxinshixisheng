from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

import run_cloud_worker
from scripts.validation import verify_cloud_contracts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CloudContractSnapshotTest(unittest.TestCase):
    def test_source_files_match_snapshot_or_declared_additive_extension(self) -> None:
        self.assertEqual(verify_cloud_contracts.verify(), [])

    def test_all_worker_targets_resolve_to_root_project_modules(self) -> None:
        targets = [*run_cloud_worker.DERMAVISION_TARGETS, "acne", "wrinkle"]
        for target in targets:
            with self.subTest(target=target):
                launch = run_cloud_worker.resolve_launch(target)
                run_cloud_worker.validate_layout(launch)
                module = launch.app.split(":", 1)[0]
                module_path = PROJECT_ROOT.joinpath(*module.split(".")).with_suffix(".py")
                self.assertTrue(module_path.is_file())

    def test_dermavision_production_targets_use_one_queue_and_one_slot(self) -> None:
        for queue in run_cloud_worker.DERMAVISION_TARGETS:
            launch = run_cloud_worker.resolve_launch(queue)
            self.assertEqual(launch.queues, (queue,))
            self.assertEqual(launch.concurrency, 1)

    def test_all_eleven_targets_keep_public_task_and_queue(self) -> None:
        expected = {
            "redness": ("src.worker:celery_app", "redness", "dermavision.analyze_image"),
            "spots": ("src.worker:celery_app", "spots", "dermavision.analyze_image"),
            "brown": ("src.worker:celery_app", "brown", "dermavision.analyze_image"),
            "texture": ("src.worker:celery_app", "texture", "dermavision.analyze_image"),
            "pores": ("src.worker:celery_app", "pores", "dermavision.analyze_image"),
            "purple": ("src.worker:celery_app", "purple", "dermavision.analyze_image"),
            "acne": ("src.acne.worker:celery_app", "acne", "acne.analyze_image"),
            "wrinkle": ("src.wrinkle.worker:celery_app", "wrinkle", "wrinkle.analyze_image"),
            "surface_gloss": ("src.worker:celery_app", "surface_gloss", "dermavision.analyze_image"),
            "vascular": ("src.worker:celery_app", "vascular", "dermavision.analyze_image"),
            "contour_firmness": ("src.worker:celery_app", "contour_firmness", "dermavision.analyze_image"),
        }
        for target, (app, queue, task) in expected.items():
            launch = run_cloud_worker.resolve_launch(target)
            self.assertEqual((launch.app, launch.queues[0], launch.task), (app, queue, task))

    def test_acne_and_wrinkle_preserve_service_name_queue(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SERVICE_NAME", None)
            self.assertEqual(
                run_cloud_worker.resolve_launch("acne").queues,
                ("acne",),
            )
            self.assertEqual(
                run_cloud_worker.resolve_launch("wrinkle").queues,
                ("wrinkle",),
            )

    def test_manifest_keeps_exact_envelope_and_field_counts(self) -> None:
        document = json.loads(
            run_cloud_worker.CONTRACT_MANIFEST.read_text(encoding="utf-8")
        )
        manifest = document["source_repositories"]
        expected_envelope = [
            "record_id", "status", "schema_version", "meta_data", "raw_result",
            "debug_info",
        ]
        for spec in manifest.values():
            self.assertEqual(spec["success_envelope"], expected_envelope)
            self.assertEqual(
                spec["medical_v2_optional_raw_result_keys"],
                ["medical_metrics_v2", "medical_report_csv_v2"],
            )
        self.assertEqual(len(manifest["acne"]["raw_result_oss_keys"]), 17)
        self.assertEqual(len(manifest["acne"]["raw_result_structured_keys"]), 10)
        self.assertEqual(len(manifest["wrinkle"]["raw_result_oss_keys"]), 13)
        # scoring_input 为本次升版新增的可选结构化键。
        self.assertEqual(len(manifest["wrinkle"]["raw_result_structured_keys"]), 7)
        self.assertEqual(
            set(manifest["dermavision"]["single_algorithm_raw_result"]),
            {
                "redness", "spots", "brown", "texture", "pores", "purple",
                "surface_gloss", "vascular", "contour_firmness",
            },
        )
        profiles = document["capture_profiles"]
        self.assertEqual(
            profiles["routing_policy"],
            "explicit_queue_only_never_pixel_inference",
        )
        self.assertEqual(len(profiles["institution"]["queues"]), 11)
        self.assertEqual(len(profiles["consumer"]["queues"]), 11)
        self.assertTrue(
            all(
                queue.startswith("consumer_")
                for queue in profiles["consumer"]["queues"]
            )
        )
        self.assertEqual(profiles["consumer"]["arguments"], [
            "task_id", "oss_key", "oss_result_prefix", "algorithms",
        ])
        self.assertEqual(
            profiles["consumer"]["success_envelope"],
            expected_envelope,
        )


if __name__ == "__main__":
    unittest.main()
