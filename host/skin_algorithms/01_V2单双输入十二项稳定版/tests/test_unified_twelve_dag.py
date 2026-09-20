from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import TypedDict

import pytest

from src.detection_runtime import orchestrator as runtime_orchestrator
from src.detection_runtime.contracts import RuntimeRoute, TWELVE_DETECTION_ITEMS
from src.detection_runtime.route_plan import (
    CLINIC_ROUTE_PLAN,
    CONSUMER_ROUTE_PLAN,
    DuplicateProducerError,
    ItemProducer,
    ProducerPlan,
    RoutePlan,
    canonical_join,
)


EXPECTED_ORDER = tuple(item.item_id for item in TWELVE_DETECTION_ITEMS)
CLINIC_BASE_ALGORITHMS = ("spots", "texture", "pores", "contour_firmness")
CONSUMER_BASE_ALGORITHMS = CLINIC_BASE_ALGORITHMS
ROUTE_MODALITY_ITEMS = (
    "redness",
    "brown",
    "uv_spots",
    "porphyrin",
    "surface_gloss",
    "vascular",
)


def _success_item(item_id: str) -> dict[str, str]:
    definition = next(item for item in TWELVE_DETECTION_ITEMS if item.item_id == item_id)
    return {
        "项目": definition.display_name,
        "状态": "success",
        "主结果图": f"/{item_id}.jpg",
        "量化JSON": f"/{item_id}.json",
        "量化CSV": f"/{item_id}.csv",
    }


def test_clinic_route_does_not_schedule_six_replaced_rgb_items() -> None:
    assert CLINIC_ROUTE_PLAN.dermavision_algorithms == CLINIC_BASE_ALGORITHMS
    assert CLINIC_ROUTE_PLAN.item_producer("redness").value == "clinic_modality"
    assert CLINIC_ROUTE_PLAN.item_producer("brown").value == "clinic_modality"
    assert CLINIC_ROUTE_PLAN.item_producer("uv_spots").value == "clinic_modality"
    assert CLINIC_ROUTE_PLAN.item_producer("porphyrin").value == "clinic_modality"
    assert CLINIC_ROUTE_PLAN.item_producer("surface_gloss").value == "clinic_modality"
    assert CLINIC_ROUTE_PLAN.item_producer("vascular").value == "clinic_modality"


def test_consumer_route_schedules_each_item_once_through_shared_producers() -> None:
    # Given: the unified consumer route plan.
    # When: its base request and producer ownership are inspected.
    # Then: the RGB base owns only the four retained heads and the shared
    # modality provider owns the six route-sensitive heads exactly once.
    assert CONSUMER_ROUTE_PLAN.dermavision_algorithms == CONSUMER_BASE_ALGORITHMS
    assert CONSUMER_ROUTE_PLAN.items_for(ItemProducer.DERMAVISION) == (
        "spots",
        "texture",
        "pores",
        "contour_firmness",
    )
    assert CONSUMER_ROUTE_PLAN.items_for(
        ItemProducer.CLINIC_MODALITY
    ) == ROUTE_MODALITY_ITEMS


def test_route_plan_rejects_duplicate_item_producers() -> None:
    with pytest.raises(DuplicateProducerError, match="redness"):
        RoutePlan(
            route=RuntimeRoute.CONSUMER_RGB,
            dermavision_algorithms=("redness",),
            producers=(
                ProducerPlan(ItemProducer.DERMAVISION, ("redness",)),
                ProducerPlan(ItemProducer.CLINIC_MODALITY, ("redness",)),
            ),
        )


def test_canonical_join_is_fixed_order_when_producers_finish_out_of_order() -> None:
    produced = {
        ItemProducer.CLINIC_MODALITY: {
            item_id: _success_item(item_id)
            for item_id in reversed(
                CLINIC_ROUTE_PLAN.items_for(ItemProducer.CLINIC_MODALITY)
            )
        },
        ItemProducer.WRINKLE: {"wrinkle": _success_item("wrinkle")},
        ItemProducer.ACNE: {"acne": _success_item("acne")},
        ItemProducer.DERMAVISION: {
            item_id: _success_item(item_id)
            for item_id in reversed(
                CLINIC_ROUTE_PLAN.items_for(ItemProducer.DERMAVISION)
            )
        },
    }

    joined = canonical_join(CLINIC_ROUTE_PLAN, produced)

    assert tuple(joined) == EXPECTED_ORDER
    assert all(item["状态"] == "success" for item in joined.values())


@dataclass(frozen=True, slots=True)
class _Channel:
    role: str
    path: Path


@dataclass(frozen=True, slots=True)
class _Capture:
    capture_alias: str
    source_group_alias: str
    fixture_manifest_sha256: str
    signed_manifest_sha256: str
    channels: tuple[_Channel, ...]

    def channel(self, role: str) -> _Channel:
        return next(channel for channel in self.channels if channel.role == role)


class _FakeBaseBatch(TypedDict):
    输入图片: str
    服务原始响应: dict[str, "_FakeServiceResponse"]
    基础并行墙钟时间秒: float


class _FakeServiceResponse(TypedDict):
    status: str
    seconds: float


class _FakeReceipt(TypedDict):
    timing_seconds: dict[str, float]
    registration: dict[str, str]
    red_brown_provider: dict[str, str]


def test_clinic_prepare_overlaps_base_but_heads_wait_for_base(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    prepare_started = threading.Event()
    base_done = threading.Event()

    class FakeBase:
        run_id = "fake-run"
        run_root = tmp_path / "runtime" / "fake-run"

        def analyze(self, image_path: Path, sample_id: str) -> _FakeBaseBatch:
            events.append("base_start")
            assert prepare_started.wait(timeout=2)
            self.run_root.joinpath("repeat_1", sample_id).mkdir(
                parents=True,
                exist_ok=True,
            )
            events.append("base_done")
            base_done.set()
            return {
                "输入图片": str(image_path),
                "服务原始响应": {
                    "dermavision": {"status": "success", "seconds": 0.42},
                    "wrinkle": {"status": "success", "seconds": 0.31},
                    "acne": {"status": "success", "seconds": 0.27},
                },
                "基础并行墙钟时间秒": 0.73,
            }

        def _items_by_producer(
            self,
            _responses: dict[str, _FakeServiceResponse],
        ) -> dict[ItemProducer, dict[str, dict[str, str]]]:
            return {
                owner: {
                    item_id: _success_item(item_id)
                    for item_id in CLINIC_ROUTE_PLAN.items_for(owner)
                }
                for owner in (
                    ItemProducer.DERMAVISION,
                    ItemProducer.ACNE,
                    ItemProducer.WRINKLE,
                )
            }

        def _write_canonical_indexes(self, **values):
            events.append("canonical_join")
            items = values["items"]
            assert tuple(items) == EXPECTED_ORDER
            return {
                "状态": "success",
                "成功项目数": 12,
                "十二项结果": items,
                "服务原始响应": values["responses"],
                "阶段耗时秒": values["phase_timings"],
                **(values.get("manifest_metadata") or {}),
            }

        def close(self) -> None:
            return None

    class FakeProvider:
        def prepare(self, _capture: _Capture, _output_root: Path) -> str:
            events.append("clinic_prepare")
            prepare_started.set()
            return "prepared"

        def run_prepared(
            self,
            prepared: str,
            _output_root: Path,
        ) -> tuple[dict[str, dict[str, str]], _FakeReceipt]:
            assert prepared == "prepared"
            assert base_done.is_set(), "Clinic CUDA heads started before base completion"
            events.append("clinic_heads")
            items = {
                item_id: _success_item(item_id)
                for item_id in CLINIC_ROUTE_PLAN.items_for(
                    ItemProducer.CLINIC_MODALITY
                )
            }
            return items, {
                "timing_seconds": {
                    "clinic_prepare": 0.1,
                    "clinic_style": 0.1,
                    "clinic_modality_total": 0.4,
                },
                    "registration": {},
                    "red_brown_provider": {},
                    "vascular_red_provider": {},
                }

    capture = _Capture(
        capture_alias="sample",
        source_group_alias="fixture",
        fixture_manifest_sha256="a" * 64,
        signed_manifest_sha256="b" * 64,
        channels=tuple(
            _Channel(role, tmp_path / f"{role}.jpg")
            for role in ("RGB_M", "PP_M", "CP_M", "365_M")
        ),
    )
    orchestrator = runtime_orchestrator.ClinicTwelveAnalysisOrchestrator.__new__(
        runtime_orchestrator.ClinicTwelveAnalysisOrchestrator
    )
    orchestrator.base = FakeBase()
    orchestrator.provider = FakeProvider()

    result = orchestrator.analyze_capture(capture)

    assert result["状态"] == "success"
    assert events.index("clinic_prepare") < events.index("base_done")
    assert events.index("base_done") < events.index("clinic_heads")
    assert events.index("clinic_heads") < events.index("canonical_join")
    assert result["阶段耗时秒"]["base_parallel_wall"] == 0.73
    assert result["阶段耗时秒"]["base_parallel_wall"] != max(
        response["seconds"]
        for service, response in result["服务原始响应"].items()
        if service != "clinic_modality"
    )
