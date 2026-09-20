from __future__ import annotations

from pathlib import Path

from run import build_parser
from src.capture_profile import CaptureProfile
from src.core.config import Settings
from src.nine_analysis.orchestrator import NineAnalysisOrchestrator
from src.pipeline import DermaVisionPipeline


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--input-dir",
        str(tmp_path / "input"),
        "--output-dir",
        str(tmp_path / "output"),
    ]


def test_local_cli_defaults_to_consumer(tmp_path: Path) -> None:
    args = build_parser().parse_args(_required_args(tmp_path))

    assert args.capture_profile == CaptureProfile.CONSUMER.value


def test_settings_defaults_to_consumer(monkeypatch) -> None:
    monkeypatch.delenv("DERMAVISION_CAPTURE_PROFILE", raising=False)

    configured = Settings(_env_file=None)

    assert configured.capture_profile is CaptureProfile.CONSUMER


def test_local_cli_accepts_explicit_consumer(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        *_required_args(tmp_path),
        "--capture-profile",
        "consumer",
    ])

    assert args.capture_profile == CaptureProfile.CONSUMER.value


def test_pipeline_and_orchestrator_receive_typed_profile(tmp_path: Path) -> None:
    pipeline = DermaVisionPipeline(capture_profile=CaptureProfile.CONSUMER)
    orchestrator = NineAnalysisOrchestrator(
        tmp_path / "runtime",
        capture_profile=CaptureProfile.CONSUMER,
    )

    assert pipeline.capture_profile is CaptureProfile.CONSUMER
    assert pipeline.style_strategy.requires_vendor is False
    assert orchestrator.capture_profile is CaptureProfile.CONSUMER


def test_settings_reads_launcher_profile_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DERMAVISION_CAPTURE_PROFILE", "consumer")

    configured = Settings(_env_file=None)

    assert configured.capture_profile is CaptureProfile.CONSUMER
