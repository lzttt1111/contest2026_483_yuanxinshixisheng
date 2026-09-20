"""Exact module and doctor-group media role validation for formal payloads."""

from __future__ import annotations

from pathlib import Path

from scripts.acceptance.acceptance_types import JsonDict, JsonValue
from src.aisia_medical_report.twelve_delivery_contract import resolve_artifact, sha256


def _mapping(value: JsonValue | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def indexed_hashes(index: JsonDict) -> dict[str, list[str]]:
    items = _mapping(index.get("items"))
    return {
        item_id: [
            str(_mapping(image).get("sha256"))
            for image in _mapping(item_value).get("images", [])
            if isinstance(image, dict)
        ]
        for item_id, item_value in items.items()
    }


def module_hashes(indexed: dict[str, list[str]]) -> dict[str, list[str]]:
    wrinkle = indexed["wrinkle"]
    return {
        "01": [indexed["pores"][0]],
        "02": [indexed["surface_gloss"][0], indexed["porphyrin"][0]],
        "03": [indexed["brown"][0], indexed["spots"][0], indexed["uv_spots"][0]],
        "04": [indexed["redness"][0]],
        "05": [indexed["vascular"][0]],
        "06": [indexed["acne"][0]],
        "07": [wrinkle[0]],
        "08": [wrinkle[1]],
        "09": [wrinkle[2]],
        "10": [indexed["texture"][0]],
        "11": [indexed["contour_firmness"][0]],
    }


def _path_hashes(report_root: Path, values: JsonValue | None) -> list[str]:
    if not isinstance(values, list):
        return []
    return [
        sha256(resolve_artifact(report_root, value))
        for value in values
        if isinstance(value, str)
    ]


def _expected_group_hashes(
    module_id: str,
    title: str,
    expected: dict[str, list[str]],
) -> list[str]:
    if module_id == "02":
        return expected[module_id][:1] if "油光" in title else expected[module_id][1:] if "荧光" in title else []
    if module_id == "03":
        return expected[module_id][1:2] if "可见" in title else expected[module_id][2:] if "UV" in title else expected[module_id][:1] if "Brown" in title else []
    if module_id == "04":
        return expected[module_id] if "强度" in title else []
    if module_id in {"05", "07", "11"}:
        return expected[module_id]
    if module_id == "10":
        return expected[module_id] if "纹理" in title else []
    return []


def structured_media_passed(
    report_root: Path,
    structured: JsonDict,
    expected: dict[str, list[str]],
) -> bool:
    modules_value = structured.get("检测模块")
    if not isinstance(modules_value, list):
        return False
    modules = {
        str(_mapping(value).get("模块编号")): _mapping(value)
        for value in modules_value
    }
    if set(modules) != set(expected):
        return False
    for module_id, hashes in expected.items():
        module = modules[module_id]
        if _path_hashes(report_root, module.get("结果图")) != hashes:
            return False
        groups = module.get("医生结果分组")
        if not isinstance(groups, list):
            continue
        for group_value in groups:
            group = _mapping(group_value)
            images = group.get("images")
            paths = [
                _mapping(image).get("path")
                for image in images
                if isinstance(images, list) and isinstance(image, dict)
            ] if isinstance(images, list) else []
            if _path_hashes(report_root, paths) != _expected_group_hashes(
                module_id,
                str(group.get("title", "")),
                expected,
            ):
                return False
    return True
