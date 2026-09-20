from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import run


class NineBatchRunTest(unittest.TestCase):
    def test_report_subject_id_uses_alias_or_extension_free_stem(self) -> None:
        self.assertEqual(
            run.report_subject_id_for_image(Path("02_clinic28-09_RGB_M.jpg")),
            "clinic28-09",
        )
        self.assertEqual(
            run.report_subject_id_for_image(Path("1000037_1.jpg")),
            "1000037_1",
        )
        self.assertEqual(
            run.report_subject_id_for_image(Path("clinic28-25_CP_M.png")),
            "clinic28-25",
        )
        self.assertEqual(
            run.report_subject_id_for_image(Path("张三 正脸照片.jpeg")),
            "张三 正脸照片",
        )

    def test_input_and_output_paths_are_explicit_and_output_alias_is_supported(self) -> None:
        parser = run.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
        args = parser.parse_args(["--input-dir", "/input", "--output-dir", "/output"])
        self.assertEqual(args.input_dir, Path("/input"))
        self.assertEqual(args.output_dir, Path("/output"))
        alias_args = parser.parse_args(["--input-dir", "/input", "--output", "/legacy"])
        self.assertEqual(alias_args.output_dir, Path("/legacy"))
        self.assertFalse(args.precount)
        self.assertEqual(args.output_profile, "review")
        self.assertIsNone(args.runtime_dir)
        self.assertFalse(args.keep_runtime)
        debug_args = parser.parse_args([
            "--input-dir", "/input",
            "--output-dir", "/output",
            "--save-debug-artifacts",
        ])
        self.assertTrue(debug_args.keep_runtime)
        self.assertEqual(run.resolve_algorithms(args.algorithms), run.ALL_TWELVE_ALGORITHMS)
        self.assertFalse(run.GENERATE_MEDICAL_REPORT_DEFAULT)

    def test_local_word_default_depends_on_output_profile_and_can_be_disabled(self) -> None:
        parser = run.build_parser()
        review = parser.parse_args([
            "--input-dir", "/input", "--output-dir", "/output",
        ])
        scoring = parser.parse_args([
            "--input-dir", "/input", "--output-dir", "/output",
            "--output-profile", "scoring",
        ])
        disabled = parser.parse_args([
            "--input-dir", "/input", "--output-dir", "/output",
            "--no-generate-medical-report",
        ])
        enabled = parser.parse_args([
            "--input-dir", "/input", "--output-dir", "/output",
            "--generate-medical-report",
        ])

        self.assertFalse(run.resolve_medical_report_default(
            review.output_profile,
            review.generate_medical_report,
        ))
        self.assertFalse(run.resolve_medical_report_default(
            scoring.output_profile,
            scoring.generate_medical_report,
        ))
        self.assertFalse(run.resolve_medical_report_default(
            disabled.output_profile,
            disabled.generate_medical_report,
        ))
        self.assertTrue(run.resolve_medical_report_default(
            enabled.output_profile,
            enabled.generate_medical_report,
        ))

    def test_algorithm_selector_supports_all_legacy_and_subset(self) -> None:
        self.assertEqual(run.resolve_algorithms(["all"]), run.ALL_TWELVE_ALGORITHMS)
        self.assertEqual(
            run.resolve_algorithms(["legacy-nine"]), run.LEGACY_NINE_ALGORITHMS
        )
        self.assertEqual(
            run.resolve_algorithms(["vascular", "surface_gloss"]),
            ("vascular", "surface_gloss"),
        )

    def test_medical_report_is_rejected_in_scoring_only_mode(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run.main([
                    "--input-dir", "/input",
                    "--output-dir", "/output",
                    "--output-profile", "scoring",
                    "--generate-medical-report",
                ])

    def test_runtime_cleanup_removes_only_its_empty_job_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            runtime_root = temporary_root / "runtime" / "job" / "worker"
            run_root = runtime_root / "run-id"
            run_root.mkdir(parents=True)
            (run_root / "debug.txt").write_text("temporary", encoding="utf-8")
            run.remove_runtime_tree(run_root, runtime_root)
            self.assertFalse(run_root.exists())
            self.assertFalse(runtime_root.exists())
            self.assertFalse((temporary_root / "runtime" / "job").exists())
            # The shared runtime root is intentionally not removed.
            self.assertTrue((temporary_root / "runtime").exists())

    def test_batch_statistics_keeps_only_aggregates(self) -> None:
        statistics = run.BatchStatistics()
        statistics.add({"status": "success", "seconds": 2.0})
        statistics.add({"status": "claimed_elsewhere", "seconds": 0.1})
        statistics.add({
            "status": "partial_success",
            "seconds": 4.0,
            "failure_category": "input_quality_reject",
        })
        self.assertEqual(statistics.processed, 3)
        self.assertEqual(statistics.duration_count, 2)
        self.assertEqual(statistics.duration_total, 6.0)
        self.assertEqual(statistics.statuses["partial_success"], 1)
        self.assertEqual(statistics.failure_categories["input_quality_reject"], 1)
        self.assertFalse(statistics.all_successful)

    def test_recursive_collection_patterns_and_output_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "output" / "fake").mkdir(parents=True)
            (root / "a.jpg").write_bytes(b"a")
            (root / "nested" / "b.png").write_bytes(b"b")
            (root / "nested" / "skip.txt").write_text("x")
            (root / "output" / "fake" / "old.jpg").write_bytes(b"old")
            images = sorted(
                run.iter_images(
                    root,
                    recursive=True,
                    patterns=("*.jpg", "nested/*.png"),
                    excluded_root=root / "output",
                ),
                key=lambda p: p.relative_to(root).as_posix(),
            )
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in images],
                ["a.jpg", "nested/b.png"],
            )

    def test_duplicate_stems_receive_stable_unique_ids(self) -> None:
        from collections import Counter

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").mkdir()
            (root / "b").mkdir()
            first = root / "a" / "same.jpg"
            second = root / "b" / "same.png"
            first.write_bytes(b"1")
            second.write_bytes(b"2")
            stem_counts: Counter[str] = Counter({"same": 2})
            id_first = run.compute_sample_id(first, root, stem_counts)
            id_second = run.compute_sample_id(second, root, stem_counts)
            self.assertNotEqual(id_first, id_second)
            self.assertTrue(id_first.startswith("same__"))

    def test_streaming_ids_do_not_require_a_full_directory_precount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            flat = root / "same.jpg"
            nested = root / "nested" / "same.jpg"
            flat.write_bytes(b"1")
            nested.write_bytes(b"2")
            self.assertEqual(run.compute_sample_id(flat, root, None), "same.jpg")
            nested_id = run.compute_sample_id(nested, root, None)
            self.assertTrue(nested_id.startswith("same.jpg__"))
            self.assertEqual(nested_id, run.compute_sample_id(nested, root, None))

    def test_failure_diagnostics_separates_quality_reject_from_retryable_error(self) -> None:
        quality = {
            "九项结果": {"redness": {"状态": "failed"}},
            "服务原始响应": {
                "dermavision": {
                    "status": "failed",
                    "result": {"message": "图像质量门禁拒绝: NO_FACE"},
                }
            },
        }
        quality_result = run.failure_diagnostics(quality)
        self.assertEqual(quality_result["failure_category"], "input_quality_reject")
        self.assertFalse(quality_result["retry_recommended"])

        wrinkle = {
            "九项结果": {"wrinkle": {"状态": "failed"}},
            "服务原始响应": {
                "wrinkle": {
                    "status": "failed",
                    "result": {"message": "IndexError: index is out of bounds"},
                }
            },
        }
        wrinkle_result = run.failure_diagnostics(wrinkle)
        self.assertEqual(
            wrinkle_result["failure_category"], "recoverable_algorithm_error"
        )
        self.assertTrue(wrinkle_result["retry_recommended"])

    def test_failure_archive_does_not_copy_input_or_debug_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime"
            source.mkdir()
            for name in ("manifest.json", "timing.json", "nine_metrics.json"):
                (source / name).write_text("{}", encoding="utf-8")
            (source / "00_input.jpg").write_bytes(b"input")
            (source / "debug.jpg").write_bytes(b"debug")
            archived = run.preserve_failure_diagnostics(
                source, root / "failures", "sample"
            )
            self.assertIsNotNone(archived)
            self.assertEqual(
                sorted(path.name for path in archived.iterdir()),
                ["manifest.json", "nine_metrics.json", "timing.json"],
            )

    def test_partial_archive_preserves_completed_algorithm_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime"
            result = source / "dermavision" / "surface_gloss" / "result.jpg"
            result.parent.mkdir(parents=True)
            result.write_bytes(b"completed-result")
            (source / "manifest.json").write_text("{}", encoding="utf-8")
            (source / "00_input.jpg").write_bytes(b"input")

            archived = run.preserve_failure_diagnostics(
                source,
                root / "failures",
                "sample",
                include_artifacts=True,
            )

            self.assertEqual(
                (
                    archived
                    / "partial_artifacts/dermavision/surface_gloss/result.jpg"
                ).read_bytes(),
                b"completed-result",
            )
            self.assertFalse((archived / "partial_artifacts/00_input.jpg").exists())

    def test_resume_uses_path_size_and_mtime_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "image.jpg"
            image.write_bytes(b"data")
            signature = run.file_signature(image, root)
            state = root / "state.jsonl"
            state.write_text(
                json.dumps({"status": "success", "signature": signature}) + "\n"
                + "{interrupted",
                encoding="utf-8",
            )
            self.assertEqual(run.load_completed(state), {run.signature_key(signature)})

    def test_resume_merges_state_files_from_multiple_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = {"relative_path": "a.jpg", "size": 1, "mtime_ns": 10}
            second = {"relative_path": "b.jpg", "size": 2, "mtime_ns": 20}
            (root / "state.worker-a.jsonl").write_text(
                json.dumps({"status": "success", "signature": first}) + "\n",
                encoding="utf-8",
            )
            (root / "state.worker-b.jsonl").write_text(
                json.dumps({"status": "success", "signature": second}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                run.load_completed_states(root),
                {run.signature_key(first), run.signature_key(second)},
            )

    def test_resume_merges_same_job_across_shard_topologies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            batch_root = Path(temporary)
            old = batch_root / "skin-380k-shard-0002-of-0003"
            split_a = batch_root / "skin-380k-shard-0002-of-0006"
            split_b = batch_root / "skin-380k-shard-0005-of-0006"
            unrelated = batch_root / "skin-380k-test-shard-0002-of-0003"
            for directory in (old, split_a, split_b, unrelated):
                directory.mkdir()
            signatures = [
                {"relative_path": f"{name}.jpg", "size": index, "mtime_ns": index}
                for index, name in enumerate(("old", "a", "b", "other"), 1)
            ]
            for directory, signature in zip(
                (old, split_a, split_b, unrelated), signatures
            ):
                (directory / "state.worker.jsonl").write_text(
                    json.dumps({"status": "success", "signature": signature}) + "\n",
                    encoding="utf-8",
                )
            completed = run.load_completed_job_states(batch_root, "skin-380k")
            self.assertEqual(
                completed,
                {run.signature_key(signature) for signature in signatures[:3]},
            )

    def test_sharding_is_stable_and_covers_every_image_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = []
            for index in range(20):
                path = root / f"image-{index}.jpg"
                path.write_bytes(str(index).encode("ascii"))
                images.append(path)
            assignments = [run.shard_number(path, root, 4) for path in images]
            self.assertTrue(all(0 <= shard < 4 for shard in assignments))
            self.assertEqual(assignments, [run.shard_number(path, root, 4) for path in images])
            self.assertEqual(sum(assignments.count(shard) for shard in range(4)), 20)

    def test_output_groups_limit_each_directory_to_two_thousand_images(self) -> None:
        counts: dict[str, int] = {}
        for index in range(4001):
            group = run.compute_output_group(index, 2000)
            counts[group] = counts.get(group, 0) + 1
        self.assertEqual(
            counts,
            {"batch_000001": 2000, "batch_000002": 2000, "batch_000003": 1},
        )

    def test_grouped_result_directory_is_used_for_resume_check(self) -> None:
        root = Path("/output")
        output_group = run.compute_output_group(12000, 2000)  # batch_000007
        sample_id = "image"
        target = root / output_group / sample_id
        self.assertEqual(target, Path("/output/batch_000007/image"))

    def test_shared_claim_prevents_duplicate_processing_and_can_be_released(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            claim_root = Path(temporary) / "claims"
            signature = {"relative_path": "a/image.jpg", "size": 10, "mtime_ns": 20}
            first = run.acquire_work_claim(
                claim_root, signature, worker_id="worker-a"
            )
            self.assertIsNotNone(first)
            second = run.acquire_work_claim(
                claim_root, signature, worker_id="worker-b"
            )
            self.assertIsNone(second)
            self.assertTrue(run.release_work_claim(first))
            third = run.acquire_work_claim(
                claim_root, signature, worker_id="worker-b"
            )
            self.assertIsNotNone(third)
            self.assertTrue(run.release_work_claim(third))

    def test_stale_shared_claim_can_be_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            claim_root = Path(temporary) / "claims"
            signature = {"relative_path": "image.jpg", "size": 10, "mtime_ns": 20}
            first = run.acquire_work_claim(
                claim_root, signature, worker_id="offline-worker"
            )
            self.assertIsNotNone(first)
            stale_time = 1
            os.utime(first.path, (stale_time, stale_time))
            recovered = run.acquire_work_claim(
                claim_root,
                signature,
                worker_id="replacement-worker",
                stale_after_seconds=1,
            )
            self.assertIsNotNone(recovered)
            self.assertNotEqual(first.token, recovered.token)
            self.assertTrue(run.release_work_claim(recovered))

    def test_batch_metadata_allows_cross_worker_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary) / "batch_000001" / "image"
            result_root.mkdir(parents=True)
            (result_root / "九项检测结果索引.json").write_text("{}", encoding="utf-8")
            signature = {"relative_path": "image.jpg", "size": 10, "mtime_ns": 20}
            run.write_batch_metadata(
                result_root,
                signature=signature,
                output_group="batch_000001",
            )
            self.assertTrue(run.result_matches_signature(result_root, signature))
            changed = dict(signature, size=11)
            self.assertFalse(run.result_matches_signature(result_root, changed))

    def test_final_twelve_receipt_allows_cross_worker_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary) / "batch_000001" / "image"
            result_root.mkdir(parents=True)
            (result_root / "十二项检测结果索引.json").write_text("{}", encoding="utf-8")
            (result_root / "运行回执.json").write_text(
                '{"timing": {}}', encoding="utf-8"
            )
            signature = {"relative_path": "image.jpg", "size": 10, "mtime_ns": 20}

            run.write_batch_metadata(
                result_root,
                signature=signature,
                output_group="batch_000001",
            )

            self.assertTrue(run.result_matches_signature(result_root, signature))
            self.assertFalse((result_root / "批处理元数据.json").exists())


if __name__ == "__main__":
    unittest.main()
