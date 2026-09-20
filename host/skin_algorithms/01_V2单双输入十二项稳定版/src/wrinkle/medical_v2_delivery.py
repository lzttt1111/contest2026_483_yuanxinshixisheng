"""医学 V2 旁路产物校验；不修改皱纹旧量化合同。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from src.wrinkle.medical_v2_schema import (
    is_english_document,
    to_chinese_document,
    to_english_document,
)


REQUIRED_KEYS = {
    "检测项目", "指标版本", "评分状态", "成像与单位说明", "总体指标",
    "分区指标", "左右比较", "质量控制", "医学局限性",
}
REQUIRED_FULL_PUBLIC_KEYS = {
    "project", "metrics_version", "scoring_status", "imaging_and_units",
    "overall_metrics", "region_metrics", "left_right_comparison",
    "quality_control", "medical_limitations",
}
REQUIRED_PUBLIC_KEYS = {
    "metrics_version", "scoring_status", "units", "overall_metrics",
    "region_metrics", "left_right_comparison", "medical_limitations",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"医学 V2 JSON 包含非法数值: {value}")


def _assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"医学 V2 JSON 包含非有限数值: {path}")
    if isinstance(value, dict):
        for name, child in value.items():
            _assert_finite(child, f"{path}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _flatten(groups: Any, prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not isinstance(groups, dict):
        return output
    ignored = {"检测范围", "评估状态", "有效皮肤面积（像素）"}
    for values in groups.values():
        if isinstance(values, dict):
            output.update({
                f"{prefix}-{name}": value
                for name, value in values.items() if name not in ignored
            })
    return output


def _rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    overall = document["总体指标"]
    analysis = overall.get("辅助指标", {}).get("分析范围", {})
    source = [{
        "检测范围": analysis.get("检测范围", "不可评估"),
        "评估状态": analysis.get("评估状态", "不可评估"),
        "有效皮肤面积（像素）": analysis.get("有效皮肤面积（像素）", "不可评估"),
        **overall,
    }, *document["分区指标"]]
    return [{
        "检测范围": row.get("检测范围", "不可评估"),
        "评估状态": row.get("评估状态", "不可评估"),
        "有效皮肤面积（像素）": row.get("有效皮肤面积（像素）", "不可评估"),
        **_flatten(row.get("核心指标", {}), "核心"),
        **_flatten(row.get("辅助指标", {}), "辅助"),
    } for row in source]


def load_and_validate(json_path: str | Path, csv_path: str | Path) -> dict[str, Any]:
    with Path(json_path).open("r", encoding="utf-8") as handle:
        document = json.load(handle, parse_constant=_reject_constant)
    if isinstance(document, dict) and "medical_metrics_v2" in document:
        document = document["medical_metrics_v2"]
    if not isinstance(document, dict):
        raise ValueError("医学 V2 JSON 结构不完整")
    public_document = (
        document if is_english_document(document) else to_english_document(document)
    )
    required = REQUIRED_PUBLIC_KEYS if "units" in public_document else REQUIRED_FULL_PUBLIC_KEYS
    if required - set(public_document):
        raise ValueError("医学 V2 JSON 英文字段结构不完整")
    if public_document.get("metrics_version") != "medical_metrics_v2_20260728":
        raise ValueError("医学 V2 指标版本不匹配")
    if public_document.get("scoring_status") != "uncalibrated":
        raise ValueError("医学 V2 评分状态不匹配")
    if not isinstance(public_document.get("overall_metrics"), dict) or not isinstance(public_document.get("region_metrics"), list):
        raise ValueError("医学 V2 总体/分区指标类型错误")
    _assert_finite(public_document)
    json.dumps(public_document, ensure_ascii=False, allow_nan=False)
    expected_rows = _rows(to_chinese_document(public_document))
    expected_columns: list[str] = []
    for row in expected_rows:
        expected_columns.extend(name for name in row if name not in expected_columns)
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_rows = list(reader)
        actual_columns = list(reader.fieldnames or [])
    missing_columns = [name for name in expected_columns if name not in actual_columns]
    if missing_columns:
        raise ValueError(f"医学 V2 CSV 缺少增量字段: {missing_columns}")
    actual_by_scope = {row.get("检测范围"): row for row in actual_rows}
    for expected in expected_rows:
        actual = actual_by_scope.get(str(expected.get("检测范围")))
        if actual is None:
            raise ValueError(f"医学 V2 CSV 缺少检测范围: {expected.get('检测范围')}")
        for name in expected_columns:
            expected_cell = "" if expected.get(name) is None else str(expected.get(name, ""))
            if actual.get(name, "") != expected_cell:
                raise ValueError(f"医学 V2 CSV 与 JSON 值不一致: {name}")
    return public_document
