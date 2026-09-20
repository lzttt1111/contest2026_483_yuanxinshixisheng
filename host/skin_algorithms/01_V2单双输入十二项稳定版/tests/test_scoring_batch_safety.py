from __future__ import annotations

import io
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

import run
from src.nine_analysis.batch_runtime import state_store


def _manifest_row(path: Path, root: Path, *, subject_id: str) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "subject_id": subject_id,
        "legacy_record_sha256": "a" * 64,
        "split": "train",
    }


def test_manifest_reads_only_explicit_paths_in_file_order() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "1000001_1.jpg"
        second = root / "nested" / "1000002_1.png"
        unlisted = root / "must-not-be-seen.jpg"
        second.parent.mkdir()
        first.write_bytes(b"one")
        second.write_bytes(b"two")
        unlisted.write_bytes(b"hidden")
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    _manifest_row(second, root, subject_id="1000002"),
                    _manifest_row(first, root, subject_id="1000001"),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        with mock.patch.object(run, "iter_images", side_effect=AssertionError("scan")):
            images = list(run.iter_manifest_images(root, manifest))

        assert images == [second, first]


@pytest.mark.parametrize("relative_path", ["../escape.jpg", "/absolute.jpg"])
def test_manifest_rejects_paths_outside_input_root(relative_path: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "relative_path": relative_path,
                    "size": 1,
                    "mtime_ns": 1,
                    "subject_id": "subject",
                    "legacy_record_sha256": "b" * 64,
                    "split": "train",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(run.InputManifestError):
            list(run.iter_manifest_images(root, manifest))


def test_manifest_rejects_duplicate_relative_paths_and_changed_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        image = root / "sample.jpg"
        image.write_bytes(b"current")
        row = _manifest_row(image, root, subject_id="sample")
        row["size"] = 1
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            json.dumps(row) + "\n" + json.dumps(row) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(run.InputManifestError):
            list(run.iter_manifest_images(root, manifest))


def test_parser_exposes_long_run_safety_switches() -> None:
    args = run.build_parser().parse_args(
        [
            "--input-dir",
            "/input",
            "--input-manifest",
            "/manifest.jsonl",
            "--output-dir",
            "/output",
            "--allow-shared-gpu",
            "--skip-recorded-failures",
            "--qc-review-dir",
            "/review",
        ]
    )

    assert args.input_manifest == Path("/manifest.jsonl")
    assert args.allow_shared_gpu is True
    assert args.skip_recorded_failures is True
    assert args.qc_review_dir == Path("/review")
    assert args.recycle_every == 8
    assert args.image_timeout_seconds == 1800
    assert args.max_swap_gib == 6.0
    assert args.retain_scoring_artifacts is False
    assert args.slots == 1


def test_recorded_failures_are_optional_resume_terminals() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        state_dir = Path(temporary)
        rows = [
            {"status": "success", "signature": {"relative_path": "a", "size": 1, "mtime_ns": 1}},
            {"status": "partial_success", "signature": {"relative_path": "b", "size": 2, "mtime_ns": 2}},
            {"status": "failed", "signature": {"relative_path": "c", "size": 3, "mtime_ns": 3}},
            {"status": "claimed_elsewhere", "signature": {"relative_path": "d", "size": 4, "mtime_ns": 4}},
        ]
        (state_dir / "state.worker.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        successes = run.load_terminal_states(state_dir, include_failures=False)
        all_terminals = run.load_terminal_states(state_dir, include_failures=True)

        assert successes == {run.signature_key(rows[0]["signature"])}
        assert all_terminals == {
            run.signature_key(row["signature"]) for row in rows[:3]
        }


def test_recorded_failure_skip_does_not_require_review_output_directory() -> None:
    signature = ("failed.jpg", 10, 20, "consumer")
    assert run.should_skip_terminal(
        signature,
        {signature},
        output_profile="review",
        skip_recorded_failures=True,
        result_index_exists=False,
    )
    assert not run.should_skip_terminal(
        signature,
        {signature},
        output_profile="review",
        skip_recorded_failures=False,
        result_index_exists=False,
    )


def test_terminal_state_is_fsynced_before_claim_release() -> None:
    events: list[str] = []
    claim = run.WorkClaim(Path("/claim"), "token")

    with (
        mock.patch.object(
            state_store,
            "append_state",
            side_effect=lambda handle, record: events.append("state"),
        ),
        mock.patch.object(
            state_store,
            "release_work_claim",
            side_effect=lambda value: events.append("release") or True,
        ),
    ):
        run.commit_terminal_record(io.StringIO(), {"status": "success"}, claim)

    assert events == ["state", "release"]


def test_preserved_terminal_claim_blocks_peer_after_failure(tmp_path) -> None:
    signature = {
        "relative_path": "failed.jpg",
        "size": 10,
        "mtime_ns": 20,
        "capture_profile": "consumer",
    }
    claim_root = tmp_path / "claims"
    claim = run.acquire_work_claim(claim_root, signature, worker_id="worker-a")
    assert claim is not None
    state = tmp_path / "state.jsonl"
    with state.open("a", encoding="utf-8") as handle:
        run.commit_terminal_record(
            handle,
            {"status": "partial_success", "signature": signature},
            claim,
            preserve_claim=True,
        )

    assert not claim.path.exists()
    assert claim.path.with_suffix(".done").is_dir()
    assert run.acquire_work_claim(
        claim_root,
        signature,
        worker_id="worker-b",
    ) is None


def test_preserved_failed_claim_can_be_retried_explicitly(tmp_path) -> None:
    signature = {
        "relative_path": "failed.jpg",
        "size": 10,
        "mtime_ns": 20,
        "capture_profile": "consumer",
    }
    claim_root = tmp_path / "claims"
    claim = run.acquire_work_claim(claim_root, signature, worker_id="worker-a")
    assert claim is not None
    state = tmp_path / "state.jsonl"
    with state.open("a", encoding="utf-8") as handle:
        run.commit_terminal_record(
            handle,
            {"status": "partial_success", "signature": signature},
            claim,
            preserve_claim=True,
        )

    retry = run.acquire_work_claim(
        claim_root,
        signature,
        worker_id="worker-b",
        retry_failed_terminal=True,
    )

    assert retry is not None
    assert run.release_work_claim(retry)
    assert list(claim_root.glob(".*.retry-*"))


def test_retry_cutoff_blocks_failure_created_during_same_run(tmp_path) -> None:
    signature = {
        "relative_path": "failed.jpg",
        "size": 10,
        "mtime_ns": 20,
        "capture_profile": "consumer",
    }
    claim_root = tmp_path / "claims"
    claim = run.acquire_work_claim(claim_root, signature, worker_id="worker-a")
    assert claim is not None
    state = tmp_path / "state.jsonl"
    with state.open("a", encoding="utf-8") as handle:
        run.commit_terminal_record(
            handle,
            {
                "status": "partial_success",
                "signature": signature,
                "finished_at": "2026-08-27T13:00:01",
            },
            claim,
            preserve_claim=True,
        )

    assert run.acquire_work_claim(
        claim_root,
        signature,
        worker_id="worker-b",
        retry_failed_terminal=True,
        retry_failed_terminal_before=datetime.fromisoformat(
            "2026-08-27T13:00:00"
        ),
    ) is None


def test_shared_gpu_lease_allows_two_workers_and_rejects_third() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = run.acquire_gpu_run_lease(root, "0", allow_shared=True)
        second = run.acquire_gpu_run_lease(root, "0", allow_shared=True)
        try:
            with pytest.raises(run.GpuRunLeaseUnavailable):
                run.acquire_gpu_run_lease(root, "0", allow_shared=True)
        finally:
            first.close()
            second.close()


def test_resource_limits_request_recycle_before_exhaustion() -> None:
    snapshot = run.MemorySnapshot(
        available_gib=12.0,
        swap_used_gib=1.0,
        rss_gib=3.0,
    )
    assert run.resource_recycle_reason(
        snapshot,
        min_available_gib=16.0,
        max_swap_gib=6.0,
    ) == "available_memory_below_limit"


def test_quality_rejection_is_copied_into_review_category(tmp_path) -> None:
    input_root = tmp_path / "input"
    image = input_root / "batch_000001" / "sample" / "face.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpeg-bytes")
    record = {
        "sample_id": "sample",
        "signature": {
            "relative_path": "batch_000001/sample/face.jpg",
            "size": len(b"jpeg-bytes"),
            "mtime_ns": image.stat().st_mtime_ns,
            "capture_profile": "consumer",
        },
        "failure_category": "input_quality_reject",
        "service_errors": {
            "dermavision": "图像质量门禁拒绝: BLUR,UNEVEN_LIGHTING,SEVERE_LOCAL_LIGHTING"
        },
        "finished_at": "2026-08-27T15:00:00",
    }

    archived = run.archive_quality_rejection(
        tmp_path / "review",
        image,
        input_root,
        record,
    )

    assert archived is not None
    assert archived.parent.name == "03_光照与曝光"
    assert archived.read_bytes() == b"jpeg-bytes"
    metadata = json.loads(
        archived.with_name(archived.name + ".json").read_text(encoding="utf-8")
    )
    assert metadata["quality_flags"] == [
        "BLUR", "UNEVEN_LIGHTING", "SEVERE_LOCAL_LIGHTING"
    ]
    assert metadata["source_relative_path"] == "batch_000001/sample/face.jpg"


def test_quality_review_category_prioritizes_face_and_pose_failures() -> None:
    assert run.quality_review_category(["NO_FACE", "BLUR"]) == "01_人脸异常"
    assert run.quality_review_category(["POSE_YAW", "BLUR"]) == "02_姿态与裁切"
    assert run.quality_review_category(["FOREHEAD_OCCLUDED"]) == "05_遮挡与有效域"
