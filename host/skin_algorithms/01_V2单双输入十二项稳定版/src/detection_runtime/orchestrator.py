from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.detection_runtime.capture_manifest import ClinicCaptureInput
from src.detection_runtime.clinic_provider import ClinicFourLightProvider
from src.detection_runtime.contracts import LEGACY_NINE_IDS, TWELVE_DETECTION_ITEMS
from src.detection_runtime.route_plan import (
    CLINIC_ROUTE_PLAN,
    CONSUMER_ROUTE_PLAN,
    CanonicalItems,
    DetectionItem,
    DuplicateProducerError,
    IncompleteProducerOutputError,
    ItemProducer,
    ProducerPlan,
    RoutePlan,
    canonical_join,
)
from src.nine_analysis.json_utils import json_safe
from src.nine_analysis.metrics import extract_item
from src.nine_analysis.orchestrator import NineAnalysisOrchestrator


DERMAVISION_CONSUMER_ALGORITHMS = list(
    CONSUMER_ROUTE_PLAN.dermavision_algorithms
)
_JOB_NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ClinicCloseFailure:
    resource: str
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class ClinicOrchestratorCloseError(Exception):
    failures: tuple[ClinicCloseFailure, ...]

    def __str__(self) -> str:
        resources = ", ".join(failure.resource for failure in self.failures)
        return f"clinic orchestrator close failed: {resources}"


class JobNamespaceError(ValueError):
    """A persistent job namespace cannot be used as one path component."""


def _job_sample_id(job_namespace: str | None, sample_alias: str) -> str:
    if job_namespace is None:
        return sample_alias
    if not _JOB_NAMESPACE_PATTERN.fullmatch(job_namespace):
        raise JobNamespaceError("invalid job namespace")
    if not sample_alias or Path(sample_alias).name != sample_alias:
        raise JobNamespaceError("invalid sample alias")
    return f"{job_namespace}--{sample_alias}"


def _scalar_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if not isinstance(value, dict):
        return output
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            output[name] = float(item)
        elif isinstance(item, dict):
            output.update(_scalar_metrics(item, name))
    return output


def _consumer_red_brown_receipt(responses: dict[str, Any]) -> dict[str, Any] | None:
    derma = responses.get("dermavision") or {}
    result = (derma.get("result") or {}).get("results") or {}
    receipt = result.get("red_brown_provider_receipt")
    return dict(receipt) if isinstance(receipt, dict) else None


class TwelveAnalysisOrchestrator(NineAnalysisOrchestrator):
    """Consumer RGB runtime: established nine results plus three additions."""

    def __init__(
        self,
        output_root: Path,
        cuda_device: str = "0",
        *,
        stage_input: bool = True,
        route_plan: RoutePlan = CONSUMER_ROUTE_PLAN,
    ) -> None:
        super().__init__(
            output_root,
            cuda_device=cuda_device,
            stage_input=stage_input,
        )
        self.route_plan = route_plan

    def _request_algorithms(self, service_name: str) -> list[str] | None:
        if service_name == "dermavision":
            return list(self.route_plan.dermavision_algorithms)
        return None

    def _request_route_profile(self, service_name: str) -> str | None:
        if service_name == "dermavision":
            return self.route_plan.route.value
        return None

    def _canonical_items(self, responses: dict[str, Any]) -> dict[str, dict[str, Any]]:
        items = super()._canonical_items(responses)
        derma = responses.get("dermavision", {})
        result = derma.get("result", {}).get("results", {}) if derma.get("status") == "success" else {}
        medical = derma.get("result", {}).get("medical_v2_results", {})
        for key, label in (
            ("surface_gloss", "油光"),
            ("vascular", "血管样结构"),
            ("contour_firmness", "轮廓紧致度"),
        ):
            overlay = result.get(key)
            item = {"项目": label, "状态": "failed"}
            if overlay and Path(overlay).is_file():
                item.update({
                    "状态": "success",
                    "主结果图": overlay,
                    "量化JSON": result.get(f"{key}_metrics"),
                    "完整量化JSON": result.get(f"{key}_full_metrics") or result.get(f"{key}_metrics"),
                    "量化CSV": result.get(f"{key}_report"),
                    "医学V2CSV": (medical.get(key) or {}).get("csv"),
                })
            items[key] = item
        return items

    def _items_by_producer(
        self,
        responses: dict[str, Any],
    ) -> dict[ItemProducer, dict[str, DetectionItem]]:
        canonical = self._canonical_items(responses)
        produced: dict[ItemProducer, dict[str, DetectionItem]] = {}
        for producer_plan in self.route_plan.producers:
            available = {
                item_id: canonical[item_id]
                for item_id in producer_plan.item_ids
                if item_id in canonical
            }
            if available:
                produced[producer_plan.producer] = available
        return produced

    def _write_indexes(
        self,
        sample_root: Path,
        image_path: Path,
        responses: dict[str, Any],
        wall_seconds: float,
    ) -> dict[str, Any]:
        items = canonical_join(
            self.route_plan,
            self._items_by_producer(responses),
        )
        return self._write_canonical_indexes(
            sample_root=sample_root,
            image_path=image_path,
            items=items,
            responses=responses,
            wall_seconds=wall_seconds,
            route="consumer_rgb",
            input_images={"RGB_M": str(image_path)},
        )

    def _write_canonical_indexes(
        self,
        *,
        sample_root: Path,
        image_path: Path,
        items: dict[str, dict[str, Any]],
        responses: dict[str, Any],
        wall_seconds: float,
        route: str,
        input_images: dict[str, str],
        phase_timings: dict[str, float] | None = None,
        manifest_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        expected = tuple(item.item_id for item in TWELVE_DETECTION_ITEMS)
        if tuple(items) != expected:
            raise RuntimeError("十二项内部顺序或ID发生漂移")
        failed = tuple(
            item_id
            for item_id, item in items.items()
            if item["状态"] != "success"
        )
        if failed:
            raise IncompleteProducerOutputError(missing=(), failed=failed)
        legacy = {key: items[key] for key in LEGACY_NINE_IDS}
        manifest = {
            "运行ID": self.run_id,
            "路线": route,
            "输入图片": str(image_path),
            "输入图像": input_images,
            "状态": "success",
            "成功项目数": 12,
            "项目总数": 12,
            "十二项结果": items,
            "九项结果": legacy,
            "服务原始响应": responses,
            "评分状态": "partially_legacy_calibrated",
        }
        provider_receipt = _consumer_red_brown_receipt(responses)
        if provider_receipt is not None:
            manifest["红棕底图生成回执"] = provider_receipt
        if phase_timings:
            manifest["阶段耗时秒"] = phase_timings
        if manifest_metadata:
            manifest.update(manifest_metadata)
        (sample_root / "manifest.json").write_text(
            json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        compact: dict[str, Any] = {}
        scoring: dict[str, Any] = {}
        for key, item in items.items():
            raw = self._load_json(item.get("量化JSON")) or {}
            if key in LEGACY_NINE_IDS:
                summary, features = extract_item(key, raw)
            else:
                summary = _scalar_metrics(raw)
                features = {"engineering_metrics": summary}
            compact[key] = {**item, "核心总体指标": summary, "评分输入": features}
            scoring[key] = features
        payload = {
            "指标版本": "twelve_metrics_v1",
            "路线": route,
            "评分状态": "partially_legacy_calibrated",
            "十二项": compact,
        }
        (sample_root / "twelve_metrics.json").write_text(
            json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (sample_root / "nine_metrics.json").write_text(
            json.dumps(
                json_safe({
                    "指标版本": "nine_metrics_compact_v2",
                    "评分状态": "uncalibrated",
                    "九项": {key: compact[key] for key in LEGACY_NINE_IDS},
                }),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (sample_root / "scoring_features.json").write_text(
            json.dumps(
                json_safe({"指标版本": "twelve_scoring_features_v1", "十二项评分输入": scoring}),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with (sample_root / "twelve_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("检测项目", "状态", "主结果图", "量化JSON", "量化CSV"))
            for item in items.values():
                writer.writerow((item["项目"], item["状态"], item.get("主结果图", ""), item.get("量化JSON", ""), item.get("量化CSV", "")))
        timing = {
            "运行ID": self.run_id,
            "路线": route,
            "图片": str(image_path),
            "输入图像": input_images,
            "整次墙钟时间秒": round(wall_seconds, 4),
            "服务": {
                key: {
                    "状态": value.get("status"),
                    "总耗时秒": value.get("seconds"),
                    "算法耗时": value.get("result", {}).get("metadata", {}).get("timing_seconds"),
                    "GPU": value.get("gpu_after"),
                }
                for key, value in responses.items()
            },
        }
        if phase_timings:
            timing["阶段耗时秒"] = phase_timings
        (sample_root / "timing.json").write_text(
            json.dumps(json_safe(timing), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (sample_root / "timing.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("阶段", "状态", "耗时秒", "峰值显存MiB"))
            for key, value in responses.items():
                writer.writerow((key, value.get("status"), value.get("seconds"), value.get("gpu_after", {}).get("peak_allocated_mib", "")))
            writer.writerow(("parallel_wall", "success", round(wall_seconds, 4), ""))
        return manifest


class _ClinicBaseOrchestrator(TwelveAnalysisOrchestrator):
    """Run Clinic's retained RGB services without publishing an interim index."""

    def __init__(
        self,
        output_root: Path,
        cuda_device: str,
        *,
        stage_input: bool,
    ) -> None:
        super().__init__(
            output_root,
            cuda_device=cuda_device,
            stage_input=stage_input,
            route_plan=CLINIC_ROUTE_PLAN,
        )
        self.algorithms = (
            *CLINIC_ROUTE_PLAN.dermavision_algorithms,
            "wrinkle",
            "acne",
        )

    def _write_indexes(
        self,
        sample_root: Path,
        image_path: Path,
        responses: dict[str, Any],
        wall_seconds: float,
    ) -> dict[str, Any]:
        return {
            "输入图片": str(image_path),
            "服务原始响应": responses,
            "基础并行墙钟时间秒": round(wall_seconds, 4),
        }

    def _request_route_profile(self, service_name: str) -> str | None:
        """Clinic modality remains in the main-process compatibility adapter."""

        return None


class ClinicTwelveAnalysisOrchestrator:
    """Reuse the established RGB services, then replace clinic-only heads."""

    def __init__(self, output_root: Path, cuda_device: str = "0", *, stage_input: bool = False) -> None:
        self.base = _ClinicBaseOrchestrator(
            output_root,
            cuda_device=cuda_device,
            stage_input=stage_input,
        )
        self.provider = ClinicFourLightProvider()

    @property
    def run_id(self) -> str:
        return self.base.run_id

    @property
    def run_root(self) -> Path:
        return self.base.run_root

    def start(self) -> dict[str, Any]:
        return self.base.start()

    def job_runtime_root(self, job_namespace: str, sample_alias: str) -> Path:
        """Return the isolated internal directory used by one persistent job."""

        sample_id = _job_sample_id(job_namespace, sample_alias)
        return self.run_root / "repeat_1" / sample_id

    def analyze_capture(
        self,
        capture: ClinicCaptureInput,
        *,
        job_namespace: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        rgb = capture.channel("RGB_M").path
        sample_id = _job_sample_id(job_namespace, capture.capture_alias)
        sample_root = self.run_root / "repeat_1" / sample_id
        clinic_root = sample_root / "clinic_modality"
        sample_root.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=1) as executor:
            prepared_future = executor.submit(
                self.provider.prepare,
                capture,
                clinic_root,
            )
            base_batch = self.base.analyze(
                rgb,
                sample_id=sample_id,
            )
            prepared = prepared_future.result()
        replacements, receipt = self.provider.run_prepared(
            prepared,
            clinic_root,
        )
        responses = dict(base_batch.get("服务原始响应") or {})
        produced = self.base._items_by_producer(responses)
        produced[ItemProducer.CLINIC_MODALITY] = replacements
        items: CanonicalItems = canonical_join(CLINIC_ROUTE_PLAN, produced)
        responses["clinic_modality"] = {
            "status": "success",
            "seconds": receipt["timing_seconds"]["clinic_modality_total"],
            "result": {"metadata": receipt},
        }
        input_images = {channel.role: str(channel.path) for channel in capture.channels}
        analysis_wall = time.perf_counter() - started
        receipt_timings = receipt["timing_seconds"]
        phase_timings = {
            "base_parallel_wall": float(
                base_batch["基础并行墙钟时间秒"]
            ),
            "clinic_prepare": float(receipt_timings["clinic_prepare"]),
            "clinic_style": float(receipt_timings["clinic_style"]),
            "analysis_wall": round(analysis_wall, 4),
        }
        metadata = {
            "采集别名": capture.capture_alias,
            "来源组": capture.source_group_alias,
            "固定数据ManifestSHA256": capture.fixture_manifest_sha256,
            "签名权威ManifestSHA256": capture.signed_manifest_sha256,
            "通道状态": receipt["registration"],
            "红棕底图生成回执": receipt["red_brown_provider"],
            "血管RED辅助生成回执": receipt["vascular_red_provider"],
        }
        if job_namespace is not None:
            metadata["任务命名空间"] = job_namespace
        merged = self.base._write_canonical_indexes(
            sample_root=sample_root,
            image_path=rgb,
            items=items,
            responses=responses,
            wall_seconds=analysis_wall,
            route="clinic_four_light",
            input_images=input_images,
            phase_timings=phase_timings,
            manifest_metadata=metadata,
        )
        return merged

    def close(self) -> None:
        failures: list[ClinicCloseFailure] = []
        for resource_name, resource in (
            ("base", self.base),
            ("provider", self.provider),
        ):
            try:
                resource.close()
            except Exception as exc:  # noqa: BROAD_EXCEPT_OK - resource owner boundary
                failures.append(
                    ClinicCloseFailure(
                        resource=resource_name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
        if failures:
            raise ClinicOrchestratorCloseError(tuple(failures))


__all__ = [
    "CLINIC_ROUTE_PLAN",
    "ClinicCloseFailure",
    "ClinicOrchestratorCloseError",
    "CONSUMER_ROUTE_PLAN",
    "DERMAVISION_CONSUMER_ALGORITHMS",
    "DuplicateProducerError",
    "IncompleteProducerOutputError",
    "ItemProducer",
    "ProducerPlan",
    "RoutePlan",
    "ClinicTwelveAnalysisOrchestrator",
    "TwelveAnalysisOrchestrator",
    "canonical_join",
]
