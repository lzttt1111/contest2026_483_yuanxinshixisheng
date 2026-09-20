from __future__ import annotations
"""Local report legacy-feature extraction, shared by CPU summary and Word.

This module changes neither scoring formulas nor reference distributions.
"""
from typing import Any, Iterable, Mapping
from src.nine_analysis.metrics import extract_item

def uv_detail_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    for row in rows:
        if row.get("检测项目") == "标准化UV紫外线色斑工程代理" and row.get("检测范围") == "全面部":
            return {
                "density": float(row["核心-单位面积密度（个/10万有效皮肤像素）"]),
                "area_ratio": float(row["核心-特征面积占比"]),
                "p90_intensity": float(row["核心-P90强度（0～1）"]),
                "high_intensity_ratio": float(row["核心-高强度目标比例"]),
                "high_intensity_area_ratio": float(row["辅助-高强度区域面积占比"]),
            }
    raise ValueError("紫区医学CSV缺少全面部UV指标")

def local_legacy_features(item: str, public: Mapping[str, Any], *,
                          capture_profile: str, uv_document: Mapping[str, Any] | None = None):
    features = extract_item(item, dict(public))[1]
    if item == "uv_spots" and capture_profile == "institution" and uv_document is not None:
        from src.medical_v2_delivery import document_rows
        from src.medical_v2_schema import is_english_document, to_chinese_document
        document = dict(uv_document)
        if is_english_document(document):
            document = to_chinese_document(document)
        row = next(row for row in document_rows(document) if row.get("检测范围") == "全面部")
        density = row.get("核心-单位面积密度（个/10万有效皮肤像素）")
        if density is None:
            valid = float(row["有效皮肤面积（像素）"])
            if valid <= 0:
                raise ValueError("UV评分缺少有效面积")
            density = round(float(row["核心-特征数量（个）"]) * 100000.0 / valid, 4)
        # Same density supplement as integration._scoring_features and the
        # institution combined-UV CSV adapter; not consumer's extra UV groups.
        features.setdefault("UV样色素范围", {})["density"] = float(density)
    if item == "uv_spots" and capture_profile == "consumer" and uv_document is not None:
        from src.medical_v2_delivery import document_rows
        from src.medical_v2_schema import is_english_document, to_chinese_document
        document = dict(uv_document)
        if is_english_document(document):
            document = to_chinese_document(document)
        rows = [{"检测项目": document.get("检测项目"), **row} for row in document_rows(document)]
        uv = uv_detail_from_rows(rows)
        features.setdefault("UV样色素范围", {}).update(density=uv["density"], area_ratio=uv["area_ratio"])
        features.setdefault("UV样色素强度", {}).update(p90_intensity=uv["p90_intensity"], high_intensity_ratio=uv["high_intensity_ratio"])
    return features
