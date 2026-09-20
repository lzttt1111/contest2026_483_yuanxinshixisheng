from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cloud import simulate_cloud_request


def test_sequential_service_policy_never_overlaps_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    active = 0
    peak = 0
    order: list[str] = []

    class FakeProcess:
        def __init__(self, service: str) -> None:
            nonlocal active, peak
            self.service = service
            self._cloud_log_handle = SimpleNamespace(close=lambda: None)
            active += 1
            peak = max(peak, active)

        def wait(self) -> int:
            nonlocal active
            order.append(self.service)
            active -= 1
            return 0

    monkeypatch.setattr(
        simulate_cloud_request,
        "_run_service",
        lambda service, _input, _output: FakeProcess(service),
    )

    failures = simulate_cloud_request._execute_services(
        tmp_path / "input.jpg",
        tmp_path / "output",
        parallel=False,
    )

    assert failures == []
    assert order == ["dermavision", "acne", "wrinkle"]
    assert peak == 1
