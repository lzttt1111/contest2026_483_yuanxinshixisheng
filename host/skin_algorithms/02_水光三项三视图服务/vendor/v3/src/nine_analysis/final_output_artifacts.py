from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

from .clinical_reports import (
    build_acne_report,
    build_wrinkle_report,
    write_medical_v2_report,
    write_report_csv,
)
from .medical_v2_schema import to_english_document
from .public_sanitize import sanitize_public_document
from .zero_target_metrics import (
    normalize_acne_report,
    normalize_zero_target_medical_csv,
)


@dataclass(frozen=True, slots=True)
class ItemDocumentNames:
    compact_csv: str
    public_json: str
    medical_csv: str


@dataclass(frozen=True, slots=True)
class ExportedItemDocuments:
    compact_csv: Path
    public_json: Path
    medical_csv: Path
    public_metrics: dict[str, Any]
    compact_rows: tuple[dict[str, str], ...]


ITEM_DOCUMENTS: Final = {
    "redness": ItemDocumentNames("红区量化指标.csv", "红区量化指标.json", "红区医学量化指标_V2.csv"),
    "spots": ItemDocumentNames("02_Spots量化指标.csv", "02_Spots量化指标.json", "02_Spots医学量化指标_V2.csv"),
    "brown": ItemDocumentNames("棕色斑量化指标.csv", "棕色斑量化指标.json", "棕色斑医学量化指标_V2.csv"),
    "texture": ItemDocumentNames("纹理量化指标.csv", "纹理量化指标.json", "纹理医学量化指标_V2.csv"),
    "pores": ItemDocumentNames("毛孔量化指标.csv", "毛孔量化指标.json", "毛孔医学量化指标_V2.csv"),
    "uv_spots": ItemDocumentNames("紫区量化指标.csv", "UV色斑量化指标.json", "紫区医学量化指标_V2.csv"),
    "porphyrin": ItemDocumentNames("紫区量化指标.csv", "卟啉量化指标.json", "紫区医学量化指标_V2.csv"),
    "wrinkle": ItemDocumentNames("皱纹量化指标.csv", "皱纹量化指标.json", "皱纹医学量化指标_V2.csv"),
    "acne": ItemDocumentNames("痤疮量化指标.csv", "痤疮量化指标.json", "痤疮医学量化指标_V2.csv"),
    "surface_gloss": ItemDocumentNames("表面油光量化指标.csv", "表面油光量化指标.json", "表面油光医学量化指标_V2.csv"),
    "vascular": ItemDocumentNames("血管样结构量化指标.csv", "血管样结构量化指标.json", "血管样结构医学量化指标_V2.csv"),
    "contour_firmness": ItemDocumentNames("轮廓紧致度量化指标.csv", "轮廓紧致度量化指标.json", "轮廓紧致度医学量化指标_V2.csv"),
}
PURPLE_COMPACT_LABELS: Final = {
    "uv_spots": "紫外线色斑",
    "porphyrin": "紫质",
}
PURPLE_MEDICAL_LABELS: Final = {
    "uv_spots": "标准化UV紫外线色斑工程代理",
    "porphyrin": "标准化荧光UV紫质工程代理",
}
PURPLE_PUBLIC_LABELS: Final = {"uv_spots": "UV色斑", "porphyrin": "卟啉"}
COMPACT_HEADER_MARKERS: Final = frozenset({"检测项目", "检测范围", "指标", "总计"})
MEDICAL_SCOPE_HEADERS: Final = frozenset({"检测项目", "检测范围"})


class InvalidMetricsDocumentError(TypeError):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__(f"量化 JSON 顶层必须为对象: {self.path}")


def _required_file(source: str | Path | None, target_name: str) -> Path:
    if not source:
        raise FileNotFoundError(f"缺少正式结果来源: {target_name}")
    path = Path(source)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"正式结果不存在或为空: {path}")
    return path


def _required_csv(
    source: str | Path | None,
    target_name: str,
    *,
    medical: bool,
) -> Path:
    path = _required_file(source, target_name)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = frozenset(reader.fieldnames or ())
        first_row = next(reader, None)
        if first_row is None or not any(first_row.values()):
            raise FileNotFoundError(f"正式结果不存在或为空: {path}")
    valid_headers = (
        "评估状态" in headers and bool(headers & MEDICAL_SCOPE_HEADERS)
        if medical
        else bool(headers & COMPACT_HEADER_MARKERS)
    )
    if not valid_headers:
        raise FileNotFoundError(f"正式 CSV 表头不符合合同: {path}")
    return path


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise InvalidMetricsDocumentError(path)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_selected_csv(source: Path, target: Path, label: str) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [row for row in reader if row and row[0] == label]
    if header[0] != "检测项目" or not rows:
        raise FileNotFoundError(f"正式结果不存在或为空: {source}")
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _compact_rows(key: str, path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = tuple(dict(row) for row in csv.DictReader(stream))
    selected = PURPLE_COMPACT_LABELS.get(key)
    if selected and any(row.get("检测项目") for row in rows):
        return tuple(row for row in rows if row.get("检测项目") == selected)
    return rows


def _convert_clinical(
    key: str,
    source_json: Path,
    destination: Path,
) -> dict[str, Any]:
    names = ITEM_DOCUMENTS[key]
    raw = _load_json(source_json)
    report = build_wrinkle_report(raw) if key == "wrinkle" else normalize_acne_report(
        build_acne_report(raw)
    )
    write_report_csv(report, destination / names.compact_csv)
    medical = write_medical_v2_report(report, key, destination / names.medical_csv, None)
    public = sanitize_public_document(to_english_document(medical))
    _write_json(destination / names.public_json, public)
    return public


def export_item_documents(
    key: str,
    item: Mapping[str, Any],
    destination: Path,
) -> ExportedItemDocuments:
    names = ITEM_DOCUMENTS[key]
    compact_target = destination / names.compact_csv
    public_target = destination / names.public_json
    medical_target = destination / names.medical_csv
    source_json = _required_file(item.get("量化JSON"), names.public_json)
    destination.mkdir(parents=True, exist_ok=True)
    if key in {"wrinkle", "acne"}:
        public = _convert_clinical(key, source_json, destination)
    else:
        compact_source = _required_csv(
            item.get("量化CSV"), names.compact_csv, medical=False
        )
        medical_source = _required_csv(
            item.get("医学V2CSV")
            or Path(str(item["主结果图"])).parent / names.medical_csv,
            names.medical_csv,
            medical=True,
        )
        if key in PURPLE_COMPACT_LABELS:
            _write_selected_csv(
                compact_source, compact_target, PURPLE_COMPACT_LABELS[key]
            )
            _write_selected_csv(
                medical_source, medical_target, PURPLE_MEDICAL_LABELS[key]
            )
        else:
            shutil.copy2(compact_source, compact_target)
            shutil.copy2(medical_source, medical_target)
        public = sanitize_public_document(_load_json(source_json))
        if key in PURPLE_COMPACT_LABELS:
            prefix = "uv_spots_" if key == "uv_spots" else "porphyrin_"
            public = {
                name: value for name, value in public.items() if name.startswith(prefix)
            }
            if not public:
                raise FileNotFoundError(
                    f"{PURPLE_PUBLIC_LABELS[key]}量化 JSON 缺少项目指标"
                )
        _write_json(public_target, public)
    normalize_zero_target_medical_csv(medical_target)
    _required_csv(compact_target, names.compact_csv, medical=False)
    _required_csv(medical_target, names.medical_csv, medical=True)
    return ExportedItemDocuments(
        compact_csv=compact_target,
        public_json=public_target,
        medical_csv=medical_target,
        public_metrics=public,
        compact_rows=_compact_rows(key, compact_target),
    )


def write_compact_aggregate(
    path: Path,
    documents: Mapping[str, ExportedItemDocuments],
    labels: Mapping[str, str],
) -> None:
    rows: list[dict[str, str]] = []
    columns = ["项目ID", "检测项目"]
    for key, exported in documents.items():
        for source_row in exported.compact_rows:
            row = dict(source_row)
            source_item = row.pop("检测项目", "")
            if source_item:
                row["来源子项目"] = source_item
            row = {"项目ID": key, "检测项目": labels[key], **row}
            rows.append(row)
            columns.extend(name for name in row if name not in columns)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["export_item_documents", "write_compact_aggregate"]
