from __future__ import annotations

from pathlib import Path

import pytest

from src.detection_runtime.contracts import TWELVE_DETECTION_ITEMS
from src.detection_runtime.orchestrator import TwelveAnalysisOrchestrator
from src.detection_runtime.route_plan import (
    CLINIC_ROUTE_PLAN,
    DetectionItem,
    DuplicateProducerError,
    IncompleteProducerOutputError,
    ItemProducer,
    canonical_join,
)


def _successful_item(item_id: str) -> DetectionItem:
    definition = next(
        item for item in TWELVE_DETECTION_ITEMS if item.item_id == item_id
    )
    return {
        "项目": definition.display_name,
        "状态": "success",
        "主结果图": f"/{item_id}.jpg",
        "量化JSON": f"/{item_id}.json",
        "量化CSV": f"/{item_id}.csv",
    }


def _complete_outputs() -> dict[ItemProducer, dict[str, DetectionItem]]:
    return {
        producer: {
            item_id: _successful_item(item_id)
            for item_id in CLINIC_ROUTE_PLAN.items_for(producer)
        }
        for producer in ItemProducer
    }


def test_missing_producer_output_blocks_canonical_index_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: one planned Clinic result is absent before the canonical join.
    produced = _complete_outputs()
    del produced[ItemProducer.CLINIC_MODALITY]["vascular"]
    orchestrator = TwelveAnalysisOrchestrator.__new__(TwelveAnalysisOrchestrator)
    orchestrator.route_plan = CLINIC_ROUTE_PLAN
    publications: list[str] = []
    monkeypatch.setattr(
        orchestrator,
        "_items_by_producer",
        lambda _responses: produced,
    )
    monkeypatch.setattr(
        orchestrator,
        "_write_canonical_indexes",
        lambda **_values: publications.append("canonical_index"),
    )
    # When: the real orchestrator publication gate tries to join the route.
    with pytest.raises(IncompleteProducerOutputError) as raised:
        orchestrator._write_indexes(
            sample_root=tmp_path,
            image_path=tmp_path / "RGB_M.jpg",
            responses={},
            wall_seconds=1.0,
        )

    # Then: the typed join failure happens before any public index is visible.
    assert type(raised.value).__name__ == "IncompleteProducerOutputError"
    assert publications == []
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "twelve_metrics.json").exists()


def test_failed_producer_output_blocks_canonical_join() -> None:
    # Given: all planned items are present but one reports failure.
    produced = _complete_outputs()
    produced[ItemProducer.CLINIC_MODALITY]["vascular"] = {
        "项目": "血管样结构",
        "状态": "failed",
    }
    # When/Then: the failed node closes the join instead of becoming a placeholder.
    with pytest.raises(IncompleteProducerOutputError, match="vascular"):
        canonical_join(CLINIC_ROUTE_PLAN, produced)


def test_failed_item_cannot_bypass_join_and_reach_index_writer(
    tmp_path: Path,
) -> None:
    # Given: a caller bypasses canonical_join with an ordered but failed item map.
    produced = _complete_outputs()
    items = {
        definition.item_id: produced[
            CLINIC_ROUTE_PLAN.item_producer(definition.item_id)
        ][definition.item_id]
        for definition in TWELVE_DETECTION_ITEMS
    }
    items["vascular"] = {"项目": "血管样结构", "状态": "failed"}
    orchestrator = TwelveAnalysisOrchestrator.__new__(TwelveAnalysisOrchestrator)
    orchestrator.run_id = "fail-closed"

    # When: the canonical writer receives the invalid mapping directly.
    with pytest.raises(IncompleteProducerOutputError, match="vascular"):
        orchestrator._write_canonical_indexes(
            sample_root=tmp_path,
            image_path=tmp_path / "RGB_M.jpg",
            items=items,
            responses={},
            wall_seconds=1.0,
            route="clinic_four_light",
            input_images={"RGB_M": str(tmp_path / "RGB_M.jpg")},
        )

    # Then: no public-facing index or metrics file is made visible.
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "twelve_metrics.json").exists()
    assert not (tmp_path / "nine_metrics.json").exists()
    assert not (tmp_path / "timing.json").exists()


def test_duplicate_runtime_producer_blocks_canonical_join() -> None:
    # Given: a second runtime producer attempts to supply Clinic-owned redness.
    complete = _complete_outputs()
    produced = {
        ItemProducer.CLINIC_MODALITY: complete[ItemProducer.CLINIC_MODALITY],
        ItemProducer.DERMAVISION: {
            **complete[ItemProducer.DERMAVISION],
            "redness": _successful_item("redness"),
        },
        ItemProducer.WRINKLE: complete[ItemProducer.WRINKLE],
        ItemProducer.ACNE: complete[ItemProducer.ACNE],
    }

    # When/Then: duplication is rejected before an index can be assembled.
    with pytest.raises(DuplicateProducerError, match="redness"):
        canonical_join(CLINIC_ROUTE_PLAN, produced)


def test_successful_join_publishes_canonical_index_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: every route producer supplies exactly its successful planned items.
    produced = _complete_outputs()
    orchestrator = TwelveAnalysisOrchestrator.__new__(TwelveAnalysisOrchestrator)
    orchestrator.route_plan = CLINIC_ROUTE_PLAN
    publications: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        orchestrator,
        "_items_by_producer",
        lambda _responses: produced,
    )
    monkeypatch.setattr(
        orchestrator,
        "_write_canonical_indexes",
        lambda **values: publications.append(tuple(values["items"])),
    )

    # When: the real orchestrator publication gate joins the complete route.
    orchestrator._write_indexes(
        sample_root=tmp_path,
        image_path=tmp_path / "RGB_M.jpg",
        responses={},
        wall_seconds=1.0,
    )

    # Then: one and only one canonical index publication occurs.
    assert publications == [
        tuple(item.item_id for item in TWELVE_DETECTION_ITEMS)
    ]
