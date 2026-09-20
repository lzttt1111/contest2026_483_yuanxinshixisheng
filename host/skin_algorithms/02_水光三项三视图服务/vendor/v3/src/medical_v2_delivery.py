"""医学 V2 旁路产物的严格校验与紫区合并工具。

该模块只处理已经生成的报告文件，不参与检测，也不修改旧 metrics。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from src.medical_v2_schema import (
    is_english_document,
    to_chinese_document,
    to_english_document,
)


REQUIRED_DOCUMENT_KEYS = {
    "检测项目",
    "指标版本",
    "评分状态",
    "成像与单位说明",
    "总体指标",
    "分区指标",
    "左右比较",
    "质量控制",
    "医学局限性",
}
REQUIRED_PUBLIC_DOCUMENT_KEYS = {
    "project",
    "metrics_version",
    "scoring_status",
    "imaging_and_units",
    "overall_metrics",
    "region_metrics",
    "left_right_comparison",
    "quality_control",
    "medical_limitations",
}
REQUIRED_INCREMENTAL_DOCUMENT_KEYS = {
    "metrics_version", "scoring_status", "units", "overall_metrics",
    "region_metrics", "left_right_comparison", "medical_limitations",
}


def _reject_json_constant(value: str) -> None:
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


def validate_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("医学 V2 JSON 顶层必须是对象")
    if is_english_document(document):
        required = (
            REQUIRED_INCREMENTAL_DOCUMENT_KEYS
            if "units" in document
            else REQUIRED_PUBLIC_DOCUMENT_KEYS
        )
        missing = required - set(document)
        if missing:
            raise ValueError(f"医学 V2 JSON 缺少英文字段: {sorted(missing)}")
        if document.get("metrics_version") != "medical_metrics_v2_20260728":
            raise ValueError("医学 V2 指标版本不匹配")
        if document.get("scoring_status") != "uncalibrated":
            raise ValueError("医学 V2 评分状态必须为 uncalibrated")
        if not isinstance(document.get("overall_metrics"), dict):
            raise ValueError("医学 V2 overall_metrics 必须是对象")
        if not isinstance(document.get("region_metrics"), list):
            raise ValueError("医学 V2 region_metrics 必须是数组")
        _assert_finite(document)
        json.dumps(document, ensure_ascii=False, allow_nan=False)
        return document
    missing = REQUIRED_DOCUMENT_KEYS - set(document)
    if missing:
        raise ValueError(f"医学 V2 JSON 缺少字段: {sorted(missing)}")
    if document.get("指标版本") != "medical_metrics_v2_20260728":
        raise ValueError("医学 V2 指标版本不匹配")
    if document.get("评分状态") != "uncalibrated":
        raise ValueError("医学 V2 评分状态必须为 uncalibrated")
    if not isinstance(document.get("总体指标"), dict):
        raise ValueError("医学 V2 总体指标必须是对象")
    if not isinstance(document.get("分区指标"), list):
        raise ValueError("医学 V2 分区指标必须是数组")
    _assert_finite(document)
    # 重新编码一次，阻止 NumPy 标量或其他非标准 JSON 类型混入。
    json.dumps(document, ensure_ascii=False, allow_nan=False)
    return document


def load_document(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle, parse_constant=_reject_json_constant)
    if isinstance(document, dict) and "medical_metrics_v2" in document:
        document = document["medical_metrics_v2"]
    document = validate_document(document)
    return document if is_english_document(document) else to_english_document(document)


def _flatten_groups(groups: Any, prefix: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not isinstance(groups, dict):
        return output
    ignored = {"检测范围", "评估状态", "有效皮肤面积（像素）"}
    for values in groups.values():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if name not in ignored:
                output[f"{prefix}-{name}"] = value
    return output


def document_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    overall = document["总体指标"]
    purple_projects = {
        "标准化UV紫外线色斑工程代理": "uv_spots",
        "标准化荧光UV紫质工程代理": "porphyrin",
    }
    if isinstance(overall, dict) and any(
        project_name in overall for project_name in purple_projects
    ):
        region_groups = {
            str(group.get("subproject")): group.get("regions", [])
            for group in document.get("分区指标", [])
            if isinstance(group, dict)
        }
        combined_rows: list[dict[str, Any]] = []
        for project_name, subproject in purple_projects.items():
            project_overall = overall.get(project_name)
            if not isinstance(project_overall, dict):
                continue
            analysis = project_overall.get("辅助指标", {}).get("分析范围", {})
            source_rows = [
                {
                    "检测项目": project_name,
                    "检测范围": analysis.get("检测范围", "不可评估"),
                    "评估状态": analysis.get("评估状态", "不可评估"),
                    "有效皮肤面积（像素）": analysis.get(
                        "有效皮肤面积（像素）", "不可评估"
                    ),
                    **project_overall,
                },
                *[
                    {"检测项目": project_name, **row}
                    for row in region_groups.get(subproject, [])
                    if isinstance(row, dict)
                ],
            ]
            combined_rows.extend(
                {
                    "检测项目": row["检测项目"],
                    "检测范围": row.get("检测范围", "不可评估"),
                    "评估状态": row.get("评估状态", "不可评估"),
                    "有效皮肤面积（像素）": row.get(
                        "有效皮肤面积（像素）", "不可评估"
                    ),
                    **_flatten_groups(row.get("核心指标", {}), "核心"),
                    **_flatten_groups(row.get("辅助指标", {}), "辅助"),
                }
                for row in source_rows
            )
        return combined_rows
    analysis = (
        overall.get("辅助指标", {}).get("分析范围", {})
        if isinstance(overall, dict)
        else {}
    )
    source_rows = [
        {
            "检测范围": analysis.get("检测范围", "不可评估"),
            "评估状态": analysis.get("评估状态", "不可评估"),
            "有效皮肤面积（像素）": analysis.get(
                "有效皮肤面积（像素）", "不可评估"
            ),
            **overall,
        },
        *document["分区指标"],
    ]
    return [
        {
            "检测范围": row.get("检测范围", "不可评估"),
            "评估状态": row.get("评估状态", "不可评估"),
            "有效皮肤面积（像素）": row.get(
                "有效皮肤面积（像素）", "不可评估"
            ),
            **_flatten_groups(row.get("核心指标", {}), "核心"),
            **_flatten_groups(row.get("辅助指标", {}), "辅助"),
        }
        for row in source_rows
    ]


def _cell(value: Any) -> str:
    return "" if value is None else str(value)


def validate_csv_matches_document(
    csv_path: str | Path,
    document: dict[str, Any],
) -> None:
    expected_rows = document_rows(document)
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
    use_project = "检测项目" in expected_columns
    actual_by_scope = {
        (
            row.get("检测项目"),
            row.get("检测范围"),
        )
        if use_project
        else row.get("检测范围"): row
        for row in actual_rows
    }
    for row_index, expected in enumerate(expected_rows, start=2):
        lookup_key = (
            str(expected.get("检测项目")),
            str(expected.get("检测范围")),
        ) if use_project else str(expected.get("检测范围"))
        actual = actual_by_scope.get(lookup_key)
        if actual is None:
            raise ValueError(
                f"医学 V2 CSV 缺少检测范围: {expected.get('检测范围')}"
            )
        for name in expected_columns:
            if actual.get(name, "") != _cell(expected.get(name)):
                raise ValueError(
                    f"医学 V2 CSV 与 JSON 不一致: row={row_index} field={name}"
                )


def load_and_validate(
    json_path: str | Path,
    csv_path: str | Path,
) -> dict[str, Any]:
    document = load_document(json_path)
    validate_csv_matches_document(
        csv_path,
        to_chinese_document(document) if is_english_document(document) else document,
    )
    return document


def combine_documents(
    artifacts: Iterable[tuple[str | Path, str | Path]],
    combined_csv_path: str | Path,
) -> dict[str, Any]:
    """把紫外线色斑和紫质两个 V2 文档合成一个紫区旁路返回。"""
    # load_and_validate() 对外统一返回英文字段；紫区合并与中文 CSV 生成仍
    # 在内部使用中文结构，最终再由 embed_medical_metrics_v2() 转成英文。
    documents: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    columns = ["检测项目"]
    for json_path, csv_path in artifacts:
        public_document = load_and_validate(json_path, csv_path)
        document = to_chinese_document(public_document)
        documents.append(document)
        for row in document_rows(document):
            combined = {"检测项目": document["检测项目"], **row}
            combined_rows.append(combined)
            columns.extend(name for name in combined if name not in columns)
    if not documents:
        raise ValueError("紫区医学 V2 子项目为空")
    output_path = Path(combined_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined_rows)
    limitations: list[Any] = []
    for document in documents:
        for limitation in document.get("医学局限性", []):
            if limitation not in limitations:
                limitations.append(limitation)
    return validate_document(
        {
            "检测项目": "紫区（紫外线色斑与紫质）",
            "指标版本": "medical_metrics_v2_20260728",
            "评分状态": "uncalibrated",
            "成像与单位说明": {
                "子项目": [document["检测项目"] for document in documents],
                "说明": "两个子项目沿用各自普通白光 RGB 代理口径",
            },
            "总体指标": {
                document["检测项目"]: document["总体指标"]
                for document in documents
            },
            "分区指标": [
                {
                    "检测项目": document["检测项目"],
                    "分区指标": document["分区指标"],
                }
                for document in documents
            ],
            "左右比较": {
                document["检测项目"]: document["左右比较"]
                for document in documents
            },
            "质量控制": {
                document["检测项目"]: document["质量控制"]
                for document in documents
            },
            "医学局限性": limitations,
        }
    )
