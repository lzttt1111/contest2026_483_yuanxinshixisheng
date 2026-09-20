"""Freshness and public-contract inspection for branch-local results."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Final, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

from scripts.acceptance.acceptance_types import LocalInspection
from scripts.acceptance.acceptance_io import sha256_file
from scripts.acceptance.safe_paths import (
    UnsafeArtifactPathError,
    resolve_regular_file,
)
from scripts.acceptance.surface_score_receipt import (
    public_metric_truth_errors,
    score_and_receipt_errors,
)
from scripts.acceptance.surface_csv_truth import compact_truth_passed


Profile = Literal["baseline", "merged"]
JSON_ADAPTER = TypeAdapter(JsonValue)
MERGED_ITEM_DIRS: Final = (
    "01_红区",
    "02_可见斑点",
    "03_棕区",
    "04_纹理",
    "05_毛孔",
    "06_UV色斑",
    "07_卟啉",
    "08_皱纹",
    "09_痤疮",
    "10_油光",
    "11_血管样结构",
    "12_轮廓紧致度",
)
MERGED_ROOT_FILES: Final = {
    "十二项检测结果索引.json",
    "十二项简版量化指标.csv",
    "十二项完整量化指标.json",
    "运行回执.json",
}


def _mapping(value: JsonValue | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _case_id(path: Path, expected_cases: tuple[str, ...]) -> str | None:
    text = path.as_posix()
    return next((case_id for case_id in expected_cases if case_id in text), None)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _row_counter(rows: list[dict[str, str]]) -> Counter[tuple[tuple[str, str], ...]]:
    return Counter(
        tuple(sorted((key, value) for key, value in row.items() if value))
        for row in rows
    )


def _media_is_trusted(sample_root: Path, item: dict[str, JsonValue]) -> bool:
    main = item.get("主结果图")
    extras = item.get("附加结果图")
    images = item.get("images")
    if not isinstance(main, str) or not isinstance(extras, list) or not isinstance(images, list):
        return False
    expected = [main, *(value for value in extras if isinstance(value, str))]
    if len(expected) != 1 + len(extras) or len(set(expected)) != len(expected):
        return False
    declared: dict[str, str] = {}
    for value in images:
        image = _mapping(value)
        relative = image.get("path")
        digest = image.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str) or relative in declared:
            return False
        declared[relative] = digest
    if set(declared) != set(expected):
        return False
    return all(
        sha256_file(resolve_regular_file(sample_root, relative)) == digest
        for relative, digest in declared.items()
    )


def _validate_merged_sample(sample_root: Path, index: dict[str, JsonValue], require_word: bool) -> list[str]:
    errors: list[str] = []
    if index.get("schema_version") != "single_rgb_twelve_index_v3" or index.get("status") != "success":
        errors.append("invalid_index_v3")
    item_root = sample_root / "十二项检测"
    actual_dirs = {
        path.name for path in item_root.iterdir() if path.is_dir()
    } if item_root.is_dir() else set()
    if actual_dirs != set(MERGED_ITEM_DIRS):
        errors.append("invalid_item_directories")
    immediate_files = {path.name for path in sample_root.iterdir() if path.is_file()}
    if not MERGED_ROOT_FILES <= immediate_files:
        errors.append("missing_root_files")
    if any(
        name not in MERGED_ROOT_FILES and not name.lower().endswith(".xlsx")
        for name in immediate_files
    ):
        errors.append("unexpected_root_files")
    items = _mapping(index.get("items"))
    if len(items) != 12:
        errors.append("invalid_item_count")
    expected_compact_rows: list[dict[str, str]] = []
    expected_item_files: set[str] = set()
    public_paths: dict[str, Path] = {}
    compact_rows_by_item: dict[str, list[dict[str, str]]] = {}
    for item_key, item_value in items.items():
        item = _mapping(item_value)
        if item.get("状态") != "success":
            errors.append("failed_item")
        resolved_documents: dict[str, Path] = {}
        for key, suffix in (("量化CSV", ".csv"), ("医学V2CSV", ".csv"), ("量化JSON", ".json")):
            relative = item.get(key)
            if not isinstance(relative, str) or not relative.endswith(suffix):
                errors.append(f"missing_{key}")
                continue
            try:
                resolved_documents[key] = resolve_regular_file(sample_root, relative)
                expected_item_files.add(Path(relative).as_posix())
            except UnsafeArtifactPathError:
                errors.append("unsafe_artifact_path")
        compact_relative = item.get("量化CSV")
        medical_relative = item.get("医学V2CSV")
        metrics_relative = item.get("量化JSON")
        if isinstance(compact_relative, str) and "量化CSV" in resolved_documents:
            compact_path = resolved_documents["量化CSV"]
            compact_rows = _csv_rows(compact_path)
            compact_rows_by_item[item_key] = compact_rows
            if not compact_rows:
                errors.append("header_only_compact_csv")
            for source_row in compact_rows:
                row = dict(source_row)
                source_item = row.pop("检测项目", "")
                if source_item:
                    row["来源子项目"] = source_item
                label = item.get("项目")
                if isinstance(label, str):
                    expected_compact_rows.append(
                        {"项目ID": item_key, "检测项目": label, **row}
                    )
        if isinstance(medical_relative, str) and "医学V2CSV" in resolved_documents:
            if not _csv_rows(resolved_documents["医学V2CSV"]):
                errors.append("header_only_medical_csv")
        if isinstance(metrics_relative, str) and "量化JSON" in resolved_documents:
            try:
                public_value = JSON_ADAPTER.validate_json(
                    resolved_documents["量化JSON"].read_text(encoding="utf-8")
                )
                if not compact_truth_passed(
                    item_key,
                    compact_rows_by_item.get(item_key, []),
                    _mapping(public_value),
                ):
                    errors.append("compact_metric_drift")
                public_paths[item_key] = resolved_documents["量化JSON"]
            except ValidationError:
                errors.append("invalid_item_json")
        try:
            media_is_trusted = _media_is_trusted(sample_root, item)
        except UnsafeArtifactPathError:
            errors.append("unsafe_artifact_path")
            media_is_trusted = True
        if not media_is_trusted:
            errors.append("invalid_item_media")
        for image_value in item.get("images", []) if isinstance(item.get("images"), list) else []:
            image = _mapping(image_value)
            relative = image.get("path")
            if isinstance(relative, str):
                expected_item_files.add(Path(relative).as_posix())
    actual_item_files = {
        path.relative_to(sample_root).as_posix()
        for path in item_root.rglob("*")
        if path.is_file()
    } if item_root.is_dir() else set()
    if actual_item_files != expected_item_files:
        errors.append("unexpected_item_files")
    root_compact = sample_root / "十二项简版量化指标.csv"
    if root_compact.is_file() and expected_compact_rows:
        root_rows = _csv_rows(root_compact)
        if _row_counter(root_rows) != _row_counter(expected_compact_rows):
            errors.append("root_compact_csv_drift")
    errors.extend(score_and_receipt_errors(sample_root))
    errors.extend(public_metric_truth_errors(sample_root, public_paths))
    if require_word:
        reports = _mapping(index.get("formal_reports"))
        report_paths = [reports.get("user_docx"), reports.get("doctor_docx")]
        try:
            reports_are_safe = all(
                isinstance(value, str)
                and resolve_regular_file(sample_root, value).suffix.lower() == ".docx"
                for value in report_paths
            )
        except UnsafeArtifactPathError:
            errors.append("unsafe_artifact_path")
            reports_are_safe = True
        if not reports_are_safe:
            errors.append("missing_dual_word")
        if len(list(sample_root.rglob("*.docx"))) != 2:
            errors.append("unexpected_word_count")
    return errors


def inspect_local_results(
    *,
    profile: Profile,
    result_root: Path,
    expected_cases: tuple[str, ...],
    started_ns: int,
    require_word: bool,
) -> LocalInspection:
    """Inspect branch-specific local outputs without trusting the process exit code."""

    match profile:
        case "baseline":
            index_name = "九项检测结果索引.json"
        case "merged":
            index_name = "十二项检测结果索引.json"
        case unreachable:
            assert_never(unreachable)
    indices = sorted(result_root.rglob(index_name))
    by_case = {
        case_id: path
        for path in indices
        if (case_id := _case_id(path, expected_cases)) is not None
    }
    discovered = tuple(case_id for case_id in expected_cases if case_id in by_case)
    errors: list[str] = []
    required_paths: list[Path] = list(by_case.values())
    for case_id in discovered:
        index_path = by_case[case_id]
        try:
            value = JSON_ADAPTER.validate_json(index_path.read_text(encoding="utf-8"))
            index = _mapping(value)
            if profile == "merged":
                errors.extend(_validate_merged_sample(index_path.parent, index, require_word))
            elif require_word and not list(index_path.parent.rglob("*.docx")):
                errors.append("missing_baseline_word")
        except (OSError, UnicodeError, ValidationError, csv.Error, ValueError, TypeError) as exc:
            errors.append(f"inspection_error:{type(exc).__name__}")
    if list(result_root.rglob("*.xlsx")):
        errors.append("unexpected_xlsx")
    oldest = min(
        (path.stat().st_mtime_ns for path in required_paths),
        default=0,
    )
    if oldest <= started_ns:
        errors.append("stale_output")
    return LocalInspection(
        discovered_cases=discovered,
        oldest_required_output_ns=oldest,
        contract_errors=tuple(dict.fromkeys(errors)),
    )
