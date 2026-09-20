from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import is_zipfile

import cv2
import numpy as np


EXPECTED_FILES = {
    "": {
        "00_输入图片.png",
        "九项检测结果索引.json",
        "九项核心量化指标.json",
        "九项核心量化指标.csv",
        "运行耗时.json",
        "运行耗时.csv",
    },
    "七项检测/红区": {
        "03_RBX红区结果图.jpg", "06_VISIA红色区实例图.jpg",
        "红区量化指标.csv", "红区量化指标.json",
    },
    "七项检测/斑点": {
        "01_Spots斑点结果图.jpg", "02_Spots量化指标.csv", "02_Spots量化指标.json",
    },
    "七项检测/棕区": {
        "01_RBX棕区结果图.jpg", "02_VISIA棕色斑实例图.jpg",
        "棕色斑量化指标.csv", "棕色斑量化指标.json",
    },
    "七项检测/纹理": {
        "01_纹理检测结果图.jpg", "纹理量化指标.csv", "纹理量化指标.json",
    },
    "七项检测/毛孔": {
        "01_毛孔检测结果图.jpg", "毛孔量化指标.csv", "毛孔量化指标.json",
    },
    "七项检测/紫区": {
        "01_紫外线色斑底图.png", "02_紫外线色斑检测结果.jpg",
        "03_紫质荧光底图.png", "04_紫质检测结果.jpg",
        "紫区量化指标.csv", "紫区量化指标.json",
    },
    "皱纹": {"01_皱纹检测结果图.jpg", "皱纹量化指标.csv", "皱纹量化指标.json"},
    "痤疮": {"01_痤疮检测结果图.jpg", "痤疮量化指标.csv", "痤疮量化指标.json"},
}
EXPECTED_DIRS = {"七项检测", *EXPECTED_FILES.keys()} - {""}
EXPECTED_ITEMS = {
    "redness", "spots", "brown", "texture", "pores",
    "uv_spots", "porphyrin", "wrinkle", "acne",
}


def _decode(path: Path) -> bool:
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED) is not None


def _validate_image(root: Path) -> list[str]:
    errors: list[str] = []
    actual_dirs = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}
    allowed_dirs = EXPECTED_DIRS | ({"医学报告"} if (root / "医学报告").is_dir() else set())
    if actual_dirs != allowed_dirs:
        errors.append(f"目录不匹配: actual={sorted(actual_dirs)}")
    expected_all: set[Path] = set()
    for relative_dir, names in EXPECTED_FILES.items():
        base = root / relative_dir
        for name in names:
            expected_all.add(base / name)
    optional_reports = set()
    report_dir = root / "医学报告"
    if report_dir.is_dir():
        optional_reports = {path for path in report_dir.iterdir() if path.is_file()}
        invalid = [path.name for path in optional_reports if path.suffix.lower() not in {".json", ".docx"}]
        if invalid:
            errors.append(f"医学报告目录含未知文件: {sorted(invalid)}")
        if not any(path.suffix.lower() == ".json" for path in optional_reports):
            errors.append("医学报告缺少汇总JSON")
        if not any(path.suffix.lower() == ".docx" for path in optional_reports):
            errors.append("医学报告缺少DOCX")
    actual_all = {path for path in root.rglob("*") if path.is_file()}
    allowed_all = expected_all | optional_reports
    if actual_all != allowed_all:
        missing = sorted(str(path.relative_to(root)) for path in expected_all - actual_all)
        extra = sorted(str(path.relative_to(root)) for path in actual_all - allowed_all)
        errors.append(f"文件白名单不匹配: missing={missing}, extra={extra}")

    for path in sorted(expected_all):
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"文件缺失或为空: {path.relative_to(root)}")
            continue
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"} and not _decode(path):
            errors.append(f"图片不可解码: {path.relative_to(root)}")
        elif suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                errors.append(f"JSON不可解析: {path.relative_to(root)}: {exc}")
        elif suffix == ".csv":
            try:
                if not list(csv.reader(path.open(encoding="utf-8-sig"))):
                    errors.append(f"CSV为空: {path.relative_to(root)}")
            except Exception as exc:
                errors.append(f"CSV不可解析: {path.relative_to(root)}: {exc}")

    for path in sorted(optional_reports):
        if path.stat().st_size <= 0:
            errors.append(f"医学报告为空: {path.relative_to(root)}")
        elif path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                errors.append(f"医学报告JSON不可解析: {path.relative_to(root)}: {exc}")
        elif path.suffix.lower() == ".docx" and not is_zipfile(path):
            errors.append(f"医学报告DOCX结构无效: {path.relative_to(root)}")

    index_path = root / "九项检测结果索引.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8-sig"))
        items = index.get("九项结果") or {}
        if index.get("成功项目数") != 9 or set(items) != EXPECTED_ITEMS:
            errors.append("九项索引不是9/9 success")
        for item_name, item in items.items():
            for field in ("主结果图", "量化JSON", "量化CSV"):
                relative = Path(item.get(field, ""))
                if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file():
                    errors.append(f"{item_name}.{field} 不是有效相对路径: {relative}")
            for value in item.get("附加结果图", []):
                relative = Path(value)
                if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file():
                    errors.append(f"{item_name}.附加结果图 不是有效相对路径: {relative}")
    for folder, prefix in (("皱纹", "皱纹"), ("痤疮", "痤疮")):
        report_json = root / folder / f"{prefix}量化指标.json"
        report_csv = root / folder / f"{prefix}量化指标.csv"
        if not report_json.is_file() or not report_csv.is_file():
            continue
        report = json.loads(report_json.read_text(encoding="utf-8-sig"))
        csv_rows = list(csv.DictReader(report_csv.open(encoding="utf-8-sig")))
        json_rows = [report.get("总体指标") or {}, *(report.get("分区指标") or [])]
        normalized = [{key: str(row.get(key, "")) for key in report.get("报告列", [])} for row in json_rows]
        if csv_rows != normalized:
            errors.append(f"{folder} CSV与JSON的总体/分区指标不一致")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="校验九项人工验收输出目录")
    parser.add_argument("--output", type=Path, default=Path("output-test"))
    parser.add_argument("--expected-images", nargs="+", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    errors: list[str] = []
    visible_dirs = sorted(path.name for path in output.iterdir() if path.is_dir() and not path.name.startswith("."))
    if visible_dirs != sorted(args.expected_images):
        errors.append(f"图片目录不匹配: actual={visible_dirs}")
    hidden = sorted(path.name for path in output.iterdir() if path.name.startswith("."))
    if hidden:
        errors.append(f"存在未清理运行目录: {hidden}")
    for name in args.expected_images:
        root = output / name
        if not root.is_dir():
            errors.append(f"缺少图片目录: {name}")
        else:
            errors.extend(f"{name}: {message}" for message in _validate_image(root))
    summary = {"状态": "passed" if not errors else "failed", "图片数": len(visible_dirs), "错误": errors}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
