"""Exact structural allowlist for intentional unmatched public artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final


ITEM_DOCUMENTS: Final = {
    "01_红区": {"红区量化指标.csv", "红区量化指标.json", "红区医学量化指标_V2.csv"},
    "02_可见斑点": {"02_Spots量化指标.csv", "02_Spots量化指标.json", "02_Spots医学量化指标_V2.csv"},
    "03_棕区": {"棕色斑量化指标.csv", "棕色斑量化指标.json", "棕色斑医学量化指标_V2.csv"},
    "04_纹理": {"纹理量化指标.csv", "纹理量化指标.json", "纹理医学量化指标_V2.csv"},
    "05_毛孔": {"毛孔量化指标.csv", "毛孔量化指标.json", "毛孔医学量化指标_V2.csv"},
    "06_UV色斑": {"紫区量化指标.csv", "UV色斑量化指标.json", "紫区医学量化指标_V2.csv"},
    "07_卟啉": {"紫区量化指标.csv", "卟啉量化指标.json", "紫区医学量化指标_V2.csv"},
    "08_皱纹": {"皱纹量化指标.csv", "皱纹量化指标.json", "皱纹医学量化指标_V2.csv"},
    "09_痤疮": {"痤疮量化指标.csv", "痤疮量化指标.json", "痤疮医学量化指标_V2.csv"},
    "10_油光": {"表面油光量化指标.csv", "表面油光量化指标.json", "表面油光医学量化指标_V2.csv"},
    "11_血管样结构": {"血管样结构量化指标.csv", "血管样结构量化指标.json", "血管样结构医学量化指标_V2.csv"},
    "12_轮廓紧致度": {"轮廓紧致度量化指标.csv", "轮廓紧致度量化指标.json", "轮廓紧致度医学量化指标_V2.csv"},
}
ITEM_IMAGE_STEMS: Final = {
    "01_红区": {"01_红区检测结果图", "02_VISIA红色区实例图"},
    "02_可见斑点": {"01_可见斑点检测结果图"},
    "03_棕区": {"01_棕区检测结果图", "02_VISIA棕色斑实例图"},
    "04_纹理": {"01_纹理检测结果图"},
    "05_毛孔": {"01_毛孔检测结果图"},
    "06_UV色斑": {"00_紫外线色斑底图", "01_UV色斑检测结果图"},
    "07_卟啉": {"00_紫质荧光底图", "01_卟啉检测结果图"},
    "08_皱纹": {"01_皱纹检测结果图", "02_全脸皱纹分区结果图", "03_皱纹分区汇总图"},
    "09_痤疮": {"01_痤疮检测结果图"},
    "10_油光": {"01_油光检测结果图"},
    "11_血管样结构": {"01_血管样结构检测结果图"},
    "12_轮廓紧致度": {"01_轮廓紧致度检测结果图"},
}
ROOT_FILES: Final = {
    "十二项检测结果索引.json",
    "十二项完整量化指标.json",
    "十二项简版量化指标.csv",
    "运行回执.json",
}
IMAGE_SUFFIXES: Final = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLOUD_MIGRATION_ALGORITHMS: Final = {
    "acne",
    "acne_v2",
    "surface_gloss",
    "vascular",
    "contour_firmness",
}


def _item_tail(path: str) -> tuple[str, str] | None:
    parts = Path(path).parts
    try:
        index = parts.index("十二项检测")
    except ValueError:
        return None
    tail = parts[index + 1:]
    return (tail[0], tail[1]) if len(tail) == 2 else None


def _root_file(path: str) -> bool:
    parts = Path(path).parts
    return bool(parts) and parts[-1] in ROOT_FILES and any(
        re.search(r"(?:^|_)clinic28-\d+", part) is not None for part in parts[:-1]
    )


def _formal_evidence(path: str) -> bool:
    parts = Path(path).parts
    return "正式报告" in parts and "evidence" in parts and bool(
        re.fullmatch(r"[0-9a-f]{12}_.+", parts[-1])
    )


def _cloud_artifact(path: str, section: str) -> bool:
    parts = Path(path).parts
    try:
        index = parts.index("simulated_oss")
    except ValueError:
        return False
    tail = parts[index + 1:]
    if (
        len(tail) != 4
        or tail[0] != "report"
        or not tail[1].startswith("visia2-clinic28-")
        or tail[2] not in CLOUD_MIGRATION_ALGORITHMS
        or any(token in tail[3].lower() for token in ("debug", "internal", "tmp"))
    ):
        return False
    suffix = Path(tail[3]).suffix.lower()
    allowed_suffixes = (
        IMAGE_SUFFIXES
        if section == "images"
        else {".csv"}
        if section == "csv_xlsx"
        else {".json"}
        if section == "json"
        else set()
    )
    return suffix in allowed_suffixes


def allowed_unmatched(path: str, section: str, channel: str) -> bool:
    if channel == "cloud":
        return (
            section == "json" and Path(path).name == "cloud_response_bundle.json"
        ) or _cloud_artifact(path, section)
    tail = _item_tail(path)
    if tail is not None:
        directory, name = tail
        if section == "images":
            return (
                Path(name).suffix.lower() in IMAGE_SUFFIXES
                and Path(name).stem in ITEM_IMAGE_STEMS.get(directory, set())
            )
        return name in ITEM_DOCUMENTS.get(directory, set())
    if section in {"json", "csv_xlsx"} and _root_file(path):
        return True
    if "_batch" in Path(path).parts and section in {"json", "csv_xlsx"}:
        return Path(path).suffix.lower() in {".json", ".csv"}
    if section == "json" and Path(path).name.endswith("正式结构化数据.json"):
        return "正式报告" in Path(path).parts
    return section == "images" and _formal_evidence(path)
