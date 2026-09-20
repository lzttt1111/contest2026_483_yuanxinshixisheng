from __future__ import annotations

import json
from pathlib import Path

import run
from scripts.acceptance.surface_commands import build_surface_command


def test_batch_summary_excludes_absolute_runtime_and_model_paths(tmp_path: Path) -> None:
    summary = tmp_path / "summary"
    statistics = run.BatchStatistics()
    statistics.add({"status": "success", "seconds": 1.0})
    run.write_summary(
        summary,
        job_name="job",
        input_root=Path("/private/input"),
        output_root=Path("/private/output"),
        images_per_group=2000,
        worker_id="worker",
        selected=1,
        skipped=0,
        statistics=statistics,
        cold_start={
            "python_executable": "/private/.venv/bin/python",
            "weights": "/private/models/model.pt",
            "gpu": {"gpu_name": "private", "cuda_runtime": "12.8"},
            "cuda": {"device": "cuda:0", "compute_capability": [12, 0]},
            "seconds": 1.2,
        },
    )
    payload = json.loads((summary / "job.worker.summary.json").read_text(encoding="utf-8"))
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "/private/" not in encoded
    assert payload["input_root"] == "input"
    assert payload["output_root"] == "output"
    assert payload["cold_start"] == {"seconds": 1.2}


def test_batch_csv_reduces_absolute_output_to_public_relative_label() -> None:
    row = run.summary_csv_row(
        {
            "index": 1,
            "signature": {"relative_path": "face.jpg"},
            "output_group": "batch_000001",
            "output_dir": "/private/results/batch_000001/face",
            "status": "success",
            "seconds": 1.0,
        }
    )
    assert row["输出目录"] == "batch_000001/face"


def test_formal_report_subject_defaults_to_each_sample_alias(tmp_path: Path) -> None:
    parser = run.build_parser()
    args = parser.parse_args(["--input-dir", "/input", "--output-dir", "/output"])
    assert args.report_subject_id is None
    command = build_surface_command(
        "merged",
        "run",
        tmp_path / "baseline",
        tmp_path / "merged",
        tmp_path / "python",
        tmp_path / "inputs.json",
        tmp_path / "results",
        tmp_path / "evidence",
    )
    assert "--report-subject-id" not in command
