from __future__ import annotations

from pathlib import Path

import src.nine_analysis.resident_worker as resident_worker_module
from src.nine_analysis.resident_worker import ServiceRuntime


def test_local_nine_analysis_requests_formal_fast(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given: a resident acne service with a recording pipeline fake.
    received_algorithms: list[list[str] | None] = []

    class RecordingPipeline:
        def process_single(
            self,
            _image_path: str,
            algorithms: list[str] | None = None,
        ) -> dict[str, str]:
            received_algorithms.append(algorithms)
            return {"status": "success"}

    runtime = object.__new__(ServiceRuntime)
    runtime.service = "acne"
    runtime.pipeline = RecordingPipeline()
    runtime._configure_output = lambda _output_root: None
    monkeypatch.setattr(
        resident_worker_module,
        "gpu_snapshot",
        lambda reset_peak=False: {"available": False, "reset_peak": reset_peak},
    )

    # When: NineAnalysis dispatches its acne resident request.
    response = runtime.analyze(
        tmp_path / "fixture.jpg",
        tmp_path / "output",
        algorithms=["acne"],
    )

    # Then: only the local resident adds the formal-fast policy token.
    assert received_algorithms == [["acne", "formal-fast"]]
    assert response["algorithms"] == ["acne", "formal-fast"]
