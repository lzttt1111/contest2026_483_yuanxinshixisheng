#!/usr/bin/env python3
"""校验十二项 Pydantic 云端模拟输出是否可交付验收。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloud_contracts import validate_worker_envelope


ALGORITHMS = (
    "redness",
    "spots",
    "brown",
    "texture",
    "pores",
    "purple",
    "surface_gloss",
    "vascular",
    "contour_firmness",
    "acne_v2",
    "wrinkle",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PURPLE_PROJECT_PREFIX = {"紫外线色斑": "uv_spots", "紫质": "porphyrin"}
PURPLE_COLUMN_SUFFIX = {
    "总计": "total",
    "额头": "forehead",
    "左脸颊": "left_cheek",
    "右脸颊": "right_cheek",
    "鼻部": "nose",
    "下巴": "chin",
}


def _walk_numbers(value: Any, location: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"JSON包含NaN或Infinity: {location}")
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_numbers(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_numbers(item, f"{location}[{index}]")


def _oss_file(root: Path, oss_key: str) -> Path:
    if oss_key.startswith(("/", "\\")) or ":\\" in oss_key:
        raise AssertionError(f"raw_result出现本地绝对路径: {oss_key}")
    path = root / "simulated_oss" / oss_key
    if not path.is_file():
        raise AssertionError(f"OSS对象不存在: {oss_key}")
    return path


def verify(root: Path) -> dict[str, Any]:
    public_path = root / "cloud_response_bundle.json"
    public_bundle = json.loads(public_path.read_text(encoding="utf-8"))
    internal_path = root / "logs/cloud_response_bundle.internal.json"
    bundle_path = internal_path if internal_path.is_file() else public_path
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    _walk_numbers(bundle)
    tasks = bundle.get("tasks", {})
    if set(tasks) != set(ALGORITHMS):
        raise AssertionError(f"任务集合错误: {sorted(tasks)}")

    images: set[Path] = set()
    csv_files: set[Path] = set()
    json_files: set[Path] = set(root.rglob("*.json"))
    for algorithm in ALGORITHMS:
        item = tasks[algorithm]
        response = item["response"]
        if response.get("status") != "success":
            raise AssertionError(f"{algorithm}任务未成功")
        if item.get("pydantic_validation") not in {"passed", "pydantic_passed"}:
            raise AssertionError(f"{algorithm}未记录Pydantic通过状态")
        validate_worker_envelope(algorithm, response)
        raw = response["raw_result"]
        for key, value in raw.items():
            if key in {"medical_report_csv_v2"} and isinstance(value, str):
                csv_files.add(_oss_file(root, value))
            elif isinstance(value, str) and key not in {
                "quality_status",
                "grading_status",
                "grading_reason",
                "detector_status",
                "detector_reason",
                "input_mode",
                "detection_scope",
                "run_preset",
                "device",
                "region_analysis_status",
            }:
                path = _oss_file(root, value)
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    images.add(path)
                elif path.suffix.lower() == ".csv":
                    csv_files.add(path)
                elif path.suffix.lower() == ".json":
                    json_files.add(path)
        report_csv = response.get("debug_info", {}).get("report_csv")
        if isinstance(report_csv, str):
            csv_files.add(_oss_file(root, report_csv))

    purple = tasks["purple"]["response"]["raw_result"]
    if {"medical_metrics_v2", "medical_report_csv_v2"} & set(purple):
        raise AssertionError("紫区不得返回医学宽表可选字段")
    if not all(key.isascii() and type(value) is int for key, value in purple["metrics"].items()):
        raise AssertionError("紫区metrics必须是扁平英文整数结构")
    for key in (
        "uv_base",
        "uv_spots_overlay",
        "fluorescence_base",
        "porphyrin_overlay",
    ):
        images.add(_oss_file(root, purple[key]))
    purple_csv = _oss_file(
        root, tasks["purple"]["response"]["debug_info"]["report_csv"]
    )
    with purple_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2 or [row.get("检测项目") for row in rows] != ["紫外线色斑", "紫质"]:
        raise AssertionError("紫区用户CSV必须恰好包含紫外线色斑和紫质两行")
    metrics_from_csv = {
        f"{PURPLE_PROJECT_PREFIX[row['检测项目']]}_{PURPLE_COLUMN_SUFFIX[column]}": int(value)
        for row in rows
        for column, value in row.items()
        if column != "检测项目"
    }
    if metrics_from_csv != purple["metrics"]:
        raise AssertionError("紫区JSON与两行用户CSV不一致")

    review_html = (root / "index.html").read_text(encoding="utf-8")
    if "<th>模型</th>" in review_html:
        raise AssertionError("验收网页仍在使用Pydantic内部模型名作为第一列")
    for required_text in (
        "<th>JSON路径</th>",
        "<th>中文名称</th>",
        "<th>当前值</th>",
        "<th>单位</th>",
        "raw_result.metrics.uv_spots_total",
        "UV色斑总数",
        "raw_result.metrics.porphyrin_left_cheek",
        "画面左脸颊紫质数量",
        "raw_result.metrics.油光面积占比.总计",
        "油光面积占比",
        "raw_result.metrics.血管样结构数量.总计",
        "血管样结构数量",
        "raw_result.metrics.中面部曲面连续性",
        "中面部曲面连续性",
        "raw_result.metrics.疑似痤疮数量.总计",
        "疑似痤疮数量",
    ):
        if required_text not in review_html:
            raise AssertionError(f"验收网页缺少逐字段中文说明: {required_text}")
    public_text = json.dumps(public_bundle, ensure_ascii=False)
    for forbidden in (
        "stage1_candidates", "vote_heatmap", "face_filter_debug",
        "/home/", "/tmp/",
    ):
        if forbidden in public_text:
            raise AssertionError(f"公开验收bundle泄露内部字段: {forbidden}")

    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise AssertionError(f"图片无法解码: {path}")
    for path in csv_files:
        if path.stat().st_size <= 0:
            raise AssertionError(f"CSV为空: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            if not any(csv.reader(handle)):
                raise AssertionError(f"CSV没有可读行: {path}")
    for path in json_files:
        value = json.loads(path.read_text(encoding="utf-8"))
        _walk_numbers(value, str(path))

    return {
        "status": "passed",
        "worker_tasks": len(tasks),
        "analysis_items": 12,
        "decoded_images": len(images),
        "nonempty_csv": len(csv_files),
        "parsed_json": len(json_files),
        "public_bundle_sanitized": True,
        "wall_clock_seconds": bundle.get("wall_clock_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
