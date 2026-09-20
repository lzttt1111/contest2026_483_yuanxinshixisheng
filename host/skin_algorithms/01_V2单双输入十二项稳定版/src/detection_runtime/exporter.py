from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from typing_extensions import assert_never

from src.detection_runtime.contracts import (
    LEGACY_NINE_IDS,
    TWELVE_DETECTION_ITEMS,
    RuntimeRoute,
)
from src.detection_runtime.public_metrics import (
    write_added_item_public_metrics,
    write_uv_or_porphyrin_public_metrics,
)
from src.detection_runtime.public_contract import validate_public_delivery
from src.nine_analysis.clinical_reports import (
    build_acne_report,
    build_wrinkle_report,
    write_medical_v2_report,
    write_report_csv,
)
from src.nine_analysis.medical_v2_schema import to_english_document


_V2_SOURCE_NAMES = {
    "redness": "红区医学量化指标_V2.csv",
    "spots": "02_Spots医学量化指标_V2.csv",
    "brown": "棕色斑医学量化指标_V2.csv",
    "texture": "纹理医学量化指标_V2.csv",
    "pores": "毛孔医学量化指标_V2.csv",
    "uv_spots": "紫区医学量化指标_V2.csv",
    "porphyrin": "紫区医学量化指标_V2.csv",
}

_ADDITIONAL_IMAGE_NAMES = {
    "redness": ("00_厂家RED底图",),
    "brown": ("00_厂家BROWN底图",),
    "wrinkle": (
        "08_稳定性线性皱纹全脸分区结果图",
        "09_结构性沟纹全脸分区结果图",
    ),
}

_VASCULAR_RECEIPT_KEY = "vascular_auxiliary_red"


def _validate_style_receipt(
    value: Any,
    *,
    consumer: str,
    geometry_mask_source_role: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"missing {consumer} provider receipt")
    expected = {
        "consumer": consumer,
        "pixel_source_role": "CP_M",
        "geometry_mask_source_role": geometry_mask_source_role,
        "fallback_allowed": False,
        "object_aliasing_allowed": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError(f"invalid {consumer} provider receipt")
    generated = value.get("generated_sha256")
    if not isinstance(generated, dict) or set(generated) != {
        "RED_M.jpg", "BROWN_M.jpg", "REDNE_M.jpg"
    }:
        raise RuntimeError(f"invalid {consumer} vendor generation receipt")
    return dict(value)


def split_institution_provider_receipts(
    container: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(container, dict):
        raise RuntimeError("missing institution provider receipt container")
    vascular_raw = container.get(_VASCULAR_RECEIPT_KEY)
    red_raw = {
        key: value
        for key, value in container.items()
        if key != _VASCULAR_RECEIPT_KEY
    }
    red = _validate_style_receipt(
        red_raw,
        consumer="redness_brown",
        geometry_mask_source_role="RGB_M",
    )
    vascular = _validate_style_receipt(
        vascular_raw,
        consumer="vascular_auxiliary_red",
        geometry_mask_source_role="CP_M",
    )
    if red == vascular or red["generated_sha256"] != vascular["generated_sha256"]:
        raise RuntimeError("institution provider receipts are aliased or not one generation")
    return red, vascular


def provider_receipt_for_export(
    manifest: dict[str, Any],
    route: RuntimeRoute,
) -> dict[str, Any]:
    red = manifest.get("红棕底图生成回执")
    if not isinstance(red, dict):
        raise RuntimeError("缺少红棕底图生成回执")
    if route is RuntimeRoute.CONSUMER_RGB:
        return red
    vascular = manifest.get("血管RED辅助生成回执")
    if red is vascular or _VASCULAR_RECEIPT_KEY in red:
        raise RuntimeError("institution provider receipts must be distinct")
    container = {**red, _VASCULAR_RECEIPT_KEY: dict(vascular or {})}
    split_institution_provider_receipts(container)
    return container


class TwelveOutputExporter:
    """Materialize one canonical twelve-item tree and legacy index aliases."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _copy(source: str | Path | None, target: Path) -> Path:
        if source is None:
            raise FileNotFoundError(f"missing source for {target.name}")
        path = Path(source)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return target

    @staticmethod
    def _relative(path: Path, root: Path) -> str:
        return path.relative_to(root).as_posix()

    def _medical_v2(
        self,
        item_id: str,
        item: dict[str, Any],
        item_root: Path,
        display_name: str,
    ) -> Path:
        target = item_root / f"{display_name}医学量化指标_V2.csv"
        explicit = item.get("医学V2CSV")
        if explicit and Path(explicit).is_file():
            return self._copy(explicit, target)
        if item_id in _V2_SOURCE_NAMES:
            candidate = Path(item["主结果图"]).parent / _V2_SOURCE_NAMES[item_id]
            if candidate.is_file():
                return self._copy(candidate, target)
        if item_id in {"wrinkle", "acne"}:
            raw = json.loads(Path(item["量化JSON"]).read_text(encoding="utf-8-sig"))
            report = build_wrinkle_report(raw) if item_id == "wrinkle" else build_acne_report(raw)
            write_medical_v2_report(report, item_id, target, None)
            return target
        # New engineering modules currently expose the same scalar set in
        # both compact and V2 CSVs.  Keep a physical V2 file without changing
        # the values or inventing a score.
        return self._copy(item["量化CSV"], target)

    def _export_item(
        self,
        definition,
        item: dict[str, Any],
        temporary: Path,
        route: RuntimeRoute,
    ) -> dict[str, Any]:
        if item.get("状态") != "success":
            raise RuntimeError(f"{definition.display_name} did not succeed")
        item_root = temporary / "十二项检测" / definition.directory_name
        source_image = Path(item["主结果图"])
        suffix = source_image.suffix.lower() or ".jpg"
        main = self._copy(
            source_image,
            item_root / f"01_{definition.display_name}检测结果图{suffix}",
        )
        metric_json = item_root / f"{definition.display_name}量化指标.json"
        metric_csv = item_root / f"{definition.display_name}量化指标.csv"
        v2_csv = item_root / f"{definition.display_name}医学量化指标_V2.csv"
        if definition.item_id in {"wrinkle", "acne"}:
            raw = json.loads(Path(item["量化JSON"]).read_text(encoding="utf-8-sig"))
            report = (
                build_wrinkle_report(raw)
                if definition.item_id == "wrinkle"
                else build_acne_report(raw)
            )
            write_report_csv(report, metric_csv)
            medical = write_medical_v2_report(
                report, definition.item_id, v2_csv, None
            )
            metric_json.write_text(
                json.dumps(
                    to_english_document(medical),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
        elif definition.item_id in {"uv_spots", "porphyrin"}:
            raw = json.loads(Path(item["量化JSON"]).read_text(encoding="utf-8-sig"))
            versioned = bool(raw.get("metrics_version")) and isinstance(raw.get("overall_metrics"), dict)
            match route:
                case RuntimeRoute.CONSUMER_RGB if versioned:
                    raw["metrics_version"] = {
                        "uv_spots": "UVSpots-RGBProxy-V1",
                        "porphyrin": "RGBProxyPorphyrin-Frozen-V1",
                    }[definition.item_id]
                    raw["imaging_and_units"]["imaging_type"] = "单RGB正面图像（工程代理）"
                    raw["medical_limitations"] = ["仅表示单RGB正面图像推断的工程代理候选，不代表真实UV/365成像或荧光信号，不等同于临床诊断。"]
                    metric_json.write_text(
                        json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                    metric_csv = self._copy(item["量化CSV"], metric_csv)
                    v2_csv = self._medical_v2(
                        definition.item_id, item, item_root, definition.display_name
                    )
                case RuntimeRoute.CLINIC_FOUR_LIGHT if versioned:
                    metric_json = self._copy(item["量化JSON"], metric_json)
                    metric_csv = self._copy(item["量化CSV"], metric_csv)
                    v2_csv = self._medical_v2(
                        definition.item_id, item, item_root, definition.display_name
                    )
                case RuntimeRoute.CONSUMER_RGB | RuntimeRoute.CLINIC_FOUR_LIGHT:
                    write_uv_or_porphyrin_public_metrics(
                        item_id=definition.item_id,
                        raw=raw,
                        json_path=metric_json,
                        csv_path=metric_csv,
                        v2_path=v2_csv,
                        route=route,
                    )
                case unreachable:
                    assert_never(unreachable)
        elif definition.item_id in {"surface_gloss", "vascular", "contour_firmness"}:
            raw = json.loads(Path(item["量化JSON"]).read_text(encoding="utf-8-sig"))
            if raw.get("metrics_version") and isinstance(raw.get("overall_metrics"), dict):
                metric_json = self._copy(item["量化JSON"], metric_json)
                metric_csv = self._copy(item["量化CSV"], metric_csv)
                v2_csv = self._medical_v2(
                    definition.item_id, item, item_root, definition.display_name
                )
            else:
                write_added_item_public_metrics(
                    item_id=definition.item_id,
                    raw=raw,
                    json_path=metric_json,
                    csv_path=metric_csv,
                    v2_path=v2_csv,
                )
        else:
            metric_json = self._copy(item["量化JSON"], metric_json)
            metric_csv = self._copy(item["量化CSV"], metric_csv)
            v2_csv = self._medical_v2(
                definition.item_id, item, item_root, definition.display_name
            )
        exported = {
            "项目": definition.display_name,
            "状态": "success",
            "主结果图": self._relative(main, temporary),
            "量化JSON": self._relative(metric_json, temporary),
            "量化CSV": self._relative(metric_csv, temporary),
            "医学V2CSV": self._relative(v2_csv, temporary),
        }
        additional: list[str] = []
        fixed_names = _ADDITIONAL_IMAGE_NAMES.get(definition.item_id, ())
        if definition.item_id in {"uv_spots", "porphyrin"}:
            match route:
                case RuntimeRoute.CONSUMER_RGB:
                    fixed_names = (
                        ("00_RGB代理_UV底图",)
                        if definition.item_id == "uv_spots"
                        else ("00_RGB代理荧光底图",)
                    )
                case RuntimeRoute.CLINIC_FOUR_LIGHT:
                    fixed_names = (
                        ("00_真实365_UV底图",)
                        if definition.item_id == "uv_spots"
                        else ("00_真实365荧光底图",)
                    )
                case unreachable:
                    assert_never(unreachable)
        for index, source in enumerate(item.get("附加结果图") or (), start=1):
            source_path = Path(source)
            stem = fixed_names[index - 1] if index <= len(fixed_names) else f"{index + 1:02d}_附加结果图"
            suffix = source_path.suffix.lower() or ".jpg"
            copied = self._copy(source_path, item_root / f"{stem}{suffix}")
            additional.append(self._relative(copied, temporary))
        if additional:
            exported["附加结果图"] = additional
        return exported

    @staticmethod
    def _write_summary_csv(path: Path, items: dict[str, dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("检测项目", "状态", "主结果图", "量化JSON", "量化CSV", "医学V2CSV"))
            for item in items.values():
                writer.writerow(tuple(item.get(key, "") for key in ("项目", "状态", "主结果图", "量化JSON", "量化CSV", "医学V2CSV")))

    @staticmethod
    def _write_medical_index(path: Path, items: dict[str, dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("检测项目", "医学V2指标文件"))
            for item in items.values():
                writer.writerow((item["项目"], item["医学V2CSV"]))

    @staticmethod
    def _write_public_timing(source: Path, json_target: Path, csv_target: Path) -> None:
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        services = raw.get("服务") or {}
        public = {
            "路线": raw.get("路线"),
            "整次墙钟时间秒": raw.get("整次墙钟时间秒"),
            "模块耗时秒": {
                key: {
                    "状态": value.get("状态"),
                    "总耗时秒": value.get("总耗时秒"),
                    "算法耗时": value.get("算法耗时"),
                }
                for key, value in services.items()
            },
        }
        json_target.write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with csv_target.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("阶段", "状态", "耗时秒"))
            for key, value in services.items():
                writer.writerow((key, value.get("状态"), value.get("总耗时秒")))
            writer.writerow(("整次运行", "success", raw.get("整次墙钟时间秒")))

    def export(
        self,
        source_root: Path,
        manifest: dict[str, Any],
        destination_name: str,
    ) -> Path:
        if manifest.get("状态") != "success" or manifest.get("成功项目数") != 12:
            raise RuntimeError("十二项未全部成功，拒绝生成完整交付目录")
        route = RuntimeRoute(manifest["路线"])
        source_items = manifest.get("十二项结果") or {}
        expected = tuple(item.item_id for item in TWELVE_DETECTION_ITEMS)
        if tuple(source_items) != expected:
            raise RuntimeError("十二项结果索引不完整")
        if not destination_name or Path(destination_name).name != destination_name:
            raise ValueError("invalid destination name")
        temporary = self.output_root / f".{destination_name}.tmp-{uuid.uuid4().hex[:8]}"
        destination = self.output_root / destination_name
        backup = self.output_root / f".{destination_name}.backup-{uuid.uuid4().hex[:8]}"
        temporary.mkdir(parents=True)
        try:
            input_sources = manifest.get("输入图像") or {
                "RGB_M": manifest["输入图片"]
            }
            input_copies: dict[str, Path] = {}
            for role, source_value in input_sources.items():
                input_source = Path(source_value)
                input_copies[role] = self._copy(
                    input_source,
                    temporary / "00_输入图像" / f"{role}{input_source.suffix.lower() or '.img'}",
                )
            input_copy = input_copies["RGB_M"]
            exported: dict[str, dict[str, Any]] = {}
            for definition in TWELVE_DETECTION_ITEMS:
                exported[definition.item_id] = self._export_item(
                    definition,
                    source_items[definition.item_id],
                    temporary,
                    route,
                )
            legacy = {key: exported[key] for key in LEGACY_NINE_IDS}
            twelve_index = {
                "路线": route.value,
                "输入图像": self._relative(input_copy, temporary),
                "输入通道": {
                    role: self._relative(value, temporary)
                    for role, value in input_copies.items()
                },
                "状态": "success",
                "成功项目数": 12,
                "项目总数": 12,
                "十二项结果": exported,
            }
            (temporary / "十二项检测结果索引.json").write_text(
                json.dumps(twelve_index, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (temporary / "九项检测结果索引.json").write_text(
                json.dumps({
                    "路线": route.value,
                    "输入图片": self._relative(input_copy, temporary),
                    "状态": "success",
                    "成功项目数": 9,
                    "项目总数": 9,
                    "九项结果": legacy,
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            source_metrics = json.loads((source_root / "twelve_metrics.json").read_text(encoding="utf-8"))
            source_metrics["十二项"] = {
                key: {**value, **exported[key]}
                for key, value in source_metrics["十二项"].items()
            }
            (temporary / "十二项核心量化指标.json").write_text(
                json.dumps(source_metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._write_summary_csv(temporary / "十二项核心量化指标.csv", exported)
            medical_index = {key: {"项目": value["项目"], "医学V2CSV": value["医学V2CSV"]} for key, value in exported.items()}
            (temporary / "十二项医学量化指标_V2.json").write_text(
                json.dumps({"十二项医学量化": medical_index}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._write_medical_index(temporary / "十二项医学量化指标_V2.csv", exported)
            nine_metrics = {"指标版本": "nine_metrics_compact_v2", "九项": {key: source_metrics["十二项"][key] for key in LEGACY_NINE_IDS}}
            (temporary / "九项核心量化指标.json").write_text(
                json.dumps(nine_metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._write_summary_csv(temporary / "九项核心量化指标.csv", legacy)
            self._write_medical_index(temporary / "九项医学量化指标_V2.csv", legacy)
            self._write_public_timing(
                source_root / "timing.json",
                temporary / "运行耗时.json",
                temporary / "运行耗时.csv",
            )
            provider_receipt = provider_receipt_for_export(manifest, route)
            (temporary / "红棕底图生成回执.json").write_text(
                json.dumps(provider_receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            validate_public_delivery(temporary)
            if destination.exists():
                os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except Exception:
                if backup.exists() and not destination.exists():
                    os.replace(backup, destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            return destination
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise


__all__ = [
    "TwelveOutputExporter",
    "provider_receipt_for_export",
    "split_institution_provider_receipts",
]
