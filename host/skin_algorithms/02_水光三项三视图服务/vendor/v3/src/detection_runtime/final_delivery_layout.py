from __future__ import annotations

"""Collapse runtime/report projections into one scoring-ready delivery tree."""

import json
import hashlib
import shutil
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeAlias

from src.capture_profile import CaptureProfile
from src.detection_runtime.exporter import split_institution_provider_receipts
from src.nine_analysis.final_output import build_complete_document
from src.nine_analysis.public_sanitize import sanitize_public_document


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

DERIVED_IMAGES = {
    "红褐混合印记": "01_红褐混合印记结果图.jpg",
    "重点色斑": "02_重点色斑区域结果图.jpg",
    "表面不规则": "03_表面不规则分布.jpg",
}
INSTITUTION_REDUNDANT_FILES = (
    "九项检测结果索引.json",
    "九项核心量化指标.json",
    "九项核心量化指标.csv",
    "九项医学量化指标_V2.csv",
    "十二项核心量化指标.json",
    "十二项核心量化指标.csv",
    "十二项医学量化指标_V2.json",
    "十二项医学量化指标_V2.csv",
    "运行耗时.json",
    "运行耗时.csv",
    "红棕底图生成回执.json",
)


@dataclass(frozen=True, slots=True)
class DeliveryLayoutError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _read_json(path: Path) -> dict[str, JsonValue]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise DeliveryLayoutError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, JsonValue]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_institution_media_hashes(
    result_root: Path,
    index: dict[str, JsonValue],
) -> None:
    """Bind every institution media role to one contained regular file SHA."""
    exported = index.get("十二项结果")
    if not isinstance(exported, dict):
        raise DeliveryLayoutError("institution twelve-item index is missing")
    resolved_root = result_root.resolve()
    for item_id, raw_item in exported.items():
        if not isinstance(raw_item, dict):
            raise DeliveryLayoutError(f"invalid exported item: {item_id}")
        paths = [raw_item.get("主结果图"), *(raw_item.get("附加结果图") or [])]
        images: list[dict[str, str]] = []
        for relative in paths:
            candidate = result_root / str(relative)
            if candidate.is_symlink():
                raise DeliveryLayoutError(f"institution media escapes result root: {relative}")
            path = candidate.resolve(strict=True)
            if not path.is_relative_to(resolved_root):
                raise DeliveryLayoutError(f"institution media escapes result root: {relative}")
            images.append({"path": str(relative), "sha256": _sha256(path)})
        raw_item["images"] = images


def build_institution_complete_document(
    result_root: Path,
    runtime_items: Mapping[str, Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    index = _read_json(result_root / "十二项检测结果索引.json")
    exported = index["十二项结果"]
    if not isinstance(exported, dict) or len(exported) != 12:
        raise DeliveryLayoutError("institution twelve-item index is incomplete")
    compact_document = _read_json(result_root / "十二项核心量化指标.json")
    compact_items = compact_document.get("十二项")
    if not isinstance(compact_items, dict) or set(compact_items) != set(exported):
        raise DeliveryLayoutError("institution runtime summary is incomplete")
    detector_results: dict[str, dict] = {}
    for item_id, runtime_item in runtime_items.items():
        exported_item = exported[item_id]
        if not isinstance(exported_item, dict):
            raise DeliveryLayoutError(f"invalid exported item: {item_id}")
        metric_path = Path(str(
            runtime_item.get("完整量化JSON")
            or runtime_item["量化JSON"]
        ))
        public_path = result_root / str(exported_item["量化JSON"])
        metrics = sanitize_public_document(_read_json(metric_path))
        compact_item = compact_items[item_id]
        if not isinstance(compact_item, dict):
            raise DeliveryLayoutError(f"invalid runtime summary: {item_id}")
        runtime_summary = compact_item.get("核心总体指标")
        if not isinstance(runtime_summary, dict):
            raise DeliveryLayoutError(f"missing runtime summary: {item_id}")
        metrics["runtime_summary"] = sanitize_public_document(runtime_summary)
        detector_results[item_id] = {
            "status": "success",
            "images": [
                exported_item["主结果图"],
                *(exported_item.get("附加结果图") or []),
            ],
            "public_metrics": sanitize_public_document(_read_json(public_path)),
            "metrics": metrics,
        }
    receipt = _read_json(result_root / "红棕底图生成回执.json")
    red_brown_receipt, vascular_receipt = split_institution_provider_receipts(
        receipt
    )
    return build_complete_document(
        detector_results=detector_results,
        capture_profile=CaptureProfile.INSTITUTION,
        input_signature={"channels": index.get("输入通道", {})},
        provenance={
            "red_brown_provider": red_brown_receipt,
            "vascular_red_provider": vascular_receipt,
        },
        quality_control={},
    )


def _publish_derived_images(result_root: Path, evidence_root: Path) -> None:
    if not evidence_root.is_dir():
        return
    target = result_root / "十二项检测" / "13_报告衍生图"
    target.mkdir(parents=True, exist_ok=False)
    for token, target_name in DERIVED_IMAGES.items():
        matches = tuple(
            path for path in evidence_root.rglob("*")
            if path.is_file() and token in path.name
        )
        if len(matches) != 1:
            raise FileNotFoundError(f"报告衍生图数量异常: {token}={len(matches)}")
        shutil.copy2(matches[0], target / target_name)


def remove_institution_report_transients(result_root: Path) -> None:
    """Remove structured payloads whose staged media are deleted at finalization."""

    for path in result_root.glob("*正式结构化数据.json"):
        path.unlink()


def finalize_consumer_delivery(result_root: Path, input_image: Path) -> None:
    """Keep one input, one thirteen-folder image tree, and scoring truth."""
    input_target = result_root / f"00_输入图片{input_image.suffix.lower()}"
    shutil.copy2(input_image, input_target)
    evidence = result_root / "正式报告衍生图"
    _publish_derived_images(result_root, evidence)
    if evidence.exists():
        shutil.rmtree(evidence)
    for path in result_root.glob("*结构化数据.json"):
        path.unlink()
    for path in result_root.glob("*评分追溯.json"):
        path.unlink()
    (result_root / "十二项简版量化指标.csv").unlink(missing_ok=True)
    index = _read_json(result_root / "十二项检测结果索引.json")
    index["input_image"] = input_target.name
    _write_json(result_root / "十二项检测结果索引.json", index)
    complete_path = result_root / "十二项完整量化指标.json"
    complete = _read_json(complete_path)
    detector_results = complete.get("detector_results")
    if not isinstance(detector_results, dict):
        raise DeliveryLayoutError("consumer detector truth is missing")
    rebuilt = build_complete_document(
        detector_results=detector_results,
        capture_profile=CaptureProfile.CONSUMER,
        input_signature={"relative_path": input_target.name},
        provenance=complete.get("provenance", {}),
        quality_control=complete.get("quality_control", {}),
    )
    _write_json(complete_path, rebuilt)


def _scalar_rows(prefix: str, value: JsonValue) -> list[tuple[str, JsonScalar]]:
    if isinstance(value, dict):
        rows: list[tuple[str, JsonScalar]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            rows.extend(_scalar_rows(child_prefix, child))
        return rows
    if isinstance(value, list):
        return []
    return [(prefix, value)]


def materialize_consumer_scoring_artifacts(result_root: Path) -> None:
    """Restore compact root-level JSON/CSV needed for paired score calibration."""

    complete = _read_json(result_root / "十二项完整量化指标.json")
    detector_results = complete.get("detector_results")
    if not isinstance(detector_results, dict):
        raise DeliveryLayoutError("consumer detector truth is missing")
    compact: dict[str, JsonValue] = {}
    rows: list[tuple[str, str, JsonScalar]] = []
    for item_id, raw_result in detector_results.items():
        if not isinstance(raw_result, dict):
            raise DeliveryLayoutError(f"invalid consumer detector result: {item_id}")
        public_metrics = raw_result.get("public_metrics")
        if not isinstance(public_metrics, dict):
            raise DeliveryLayoutError(f"missing public metrics: {item_id}")
        compact[item_id] = public_metrics
        rows.extend(
            (item_id, metric_path, value)
            for metric_path, value in _scalar_rows("", public_metrics)
        )
    _write_json(
        result_root / "十二项核心量化指标.json",
        {
            "指标版本": "twelve_core_from_complete_v1",
            "评分状态": "bridge_pending",
            "十二项": compact,
        },
    )
    with (result_root / "十二项核心量化指标.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("检测项目", "指标路径", "数值"))
        writer.writerows(rows)


def finalize_institution_delivery(
    result_root: Path,
    runtime_items: Mapping[str, Mapping[str, JsonValue]],
) -> None:
    """Add the shared scoring truth and remove legacy/report duplicates."""
    index = _read_json(result_root / "十二项检测结果索引.json")
    attach_institution_media_hashes(result_root, index)
    index["capture_profile"] = CaptureProfile.INSTITUTION.value
    _write_json(result_root / "十二项检测结果索引.json", index)
    receipt_path = result_root / "红棕底图生成回执.json"
    timing_path = result_root / "运行耗时.json"
    receipt = _read_json(receipt_path)
    timing = _read_json(timing_path)
    complete = build_institution_complete_document(result_root, runtime_items)
    _write_json(result_root / "十二项完整量化指标.json", complete)
    _write_json(
        result_root / "运行回执.json",
        build_institution_run_receipt(receipt, timing),
    )
    evidence_root = result_root / "正式报告衍生图"
    _publish_derived_images(result_root, evidence_root)
    if evidence_root.exists():
        shutil.rmtree(evidence_root)
    report_root = result_root / "正式报告"
    if report_root.exists():
        shutil.rmtree(report_root)
    remove_institution_report_transients(result_root)
    for name in INSTITUTION_REDUNDANT_FILES:
        (result_root / name).unlink(missing_ok=True)


def build_institution_run_receipt(
    provider_container: Mapping[str, JsonValue],
    timing: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Project both mandatory institution style views under stable final keys."""

    red_brown_receipt, vascular_receipt = split_institution_provider_receipts(
        provider_container
    )
    return {
        "capture_profile": "institution",
        "route": "clinic_four_light",
        "timing": dict(timing),
        "red_brown_provider": red_brown_receipt,
        "vascular_red_provider": vascular_receipt,
    }


__all__ = [
    "attach_institution_media_hashes",
    "build_institution_run_receipt",
    "build_institution_complete_document",
    "finalize_consumer_delivery",
    "materialize_consumer_scoring_artifacts",
    "finalize_institution_delivery",
    "remove_institution_report_transients",
]
