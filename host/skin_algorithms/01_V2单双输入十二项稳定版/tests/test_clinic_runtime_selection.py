from __future__ import annotations

from pathlib import Path

import pytest

from src.detection_runtime import orchestrator as runtime_orchestrator
from src.detection_runtime.contracts import TWELVE_DETECTION_ITEMS
from src.detection_runtime.persistent_slot import _ConsumerResidentView
from src.detection_runtime.route_plan import CLINIC_ROUTE_PLAN, ItemProducer


EXPECTED_ORDER = tuple(item.item_id for item in TWELVE_DETECTION_ITEMS)
CLINIC_RUNTIME_ALGORITHMS = (
    *CLINIC_ROUTE_PLAN.dermavision_algorithms,
    "wrinkle",
    "acne",
)


def _success_item(item_id: str) -> dict[str, str]:
    return {
        "项目": item_id,
        "状态": "success",
        "主结果图": f"/{item_id}.jpg",
        "量化JSON": f"/{item_id}.json",
        "量化CSV": f"/{item_id}.csv",
    }


def test_clinic_runtime_requests_only_owned_base_algorithms(tmp_path: Path) -> None:
    # Given: the real institution runtime base.
    runtime = runtime_orchestrator._ClinicBaseOrchestrator(
        tmp_path,
        cuda_device="0",
        stage_input=False,
    )

    # When: its resident request selection is inspected before startup.
    selected = runtime.algorithms

    # Then: route-sensitive heads are absent and acne/wrinkle remain present.
    assert selected == CLINIC_RUNTIME_ALGORITHMS


def test_consumer_view_keeps_full_twelve_selection_on_shared_residents(
    tmp_path: Path,
) -> None:
    # Given: one institution-owned resident bundle shared with consumer jobs.
    base = runtime_orchestrator._ClinicBaseOrchestrator(
        tmp_path,
        cuda_device="0",
        stage_input=False,
    )

    # When: the consumer view is created over those resident processes.
    consumer = _ConsumerResidentView(base)

    # Then: consumer still requests all twelve canonical items exactly once.
    assert consumer.algorithms == EXPECTED_ORDER


def test_clinic_base_producer_map_defers_route_sensitive_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the clinic base produced only its four RGB heads plus acne/wrinkle.
    runtime = runtime_orchestrator.TwelveAnalysisOrchestrator.__new__(
        runtime_orchestrator.TwelveAnalysisOrchestrator
    )
    runtime.route_plan = CLINIC_ROUTE_PLAN
    available = {
        item_id: _success_item(item_id)
        for item_id in CLINIC_RUNTIME_ALGORITHMS
    }
    monkeypatch.setattr(runtime, "_canonical_items", lambda _responses: available)

    # When: base responses are grouped before four-light heads are attached.
    produced = runtime._items_by_producer({})

    # Then: the route-sensitive owner is absent until the modality provider runs.
    assert set(produced) == {
        ItemProducer.DERMAVISION,
        ItemProducer.WRINKLE,
        ItemProducer.ACNE,
    }
