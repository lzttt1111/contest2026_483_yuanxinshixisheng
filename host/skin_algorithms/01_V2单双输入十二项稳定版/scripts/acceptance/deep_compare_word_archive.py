"""Dynamic-role and frozen-static media checks for formal DOCX archives."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.acceptance.acceptance_types import JsonDict, JsonValue
from scripts.acceptance.deep_compare_word import inspect_docx
from src.aisia_medical_report.twelve_delivery_contract import (
    WordDeliveryContractError,
    resolve_artifact,
    sha256,
)


def _mapping(value: JsonValue | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _hashes(media: JsonValue | None) -> list[str]:
    return [
        str(_mapping(value).get("sha256"))
        for value in media
        if isinstance(media, list) and isinstance(value, dict)
    ] if isinstance(media, list) else []


def media_roles(payload: JsonDict, root: Path) -> dict[str, str]:
    roles: dict[str, str] = {}
    source = _mapping(payload.get("标准采集图像"))
    source_path = source.get("path")
    source_sha = source.get("sha256")
    if not isinstance(source_path, str) or not isinstance(source_sha, str):
        raise WordDeliveryContractError("formal source media is incomplete")
    if sha256(resolve_artifact(root, source_path)) != source_sha:
        raise WordDeliveryContractError("formal source media SHA256 mismatch")
    roles[source_sha] = "source"
    modules = payload.get("检测模块")
    if not isinstance(modules, list):
        raise WordDeliveryContractError("formal modules are incomplete")
    for module_value in modules:
        module = _mapping(module_value)
        module_id = str(module.get("模块编号", ""))
        images = module.get("结果图")
        if not isinstance(images, list):
            raise WordDeliveryContractError("formal module media is incomplete")
        for index, relative in enumerate(images, start=1):
            if not isinstance(relative, str):
                raise WordDeliveryContractError("formal module media path is invalid")
            digest = sha256(resolve_artifact(root, relative))
            role = f"module:{module_id}:{index}"
            if digest in roles and roles[digest] != role:
                raise WordDeliveryContractError("formal module media roles are duplicated")
            roles[digest] = role
    return roles


def frozen_static_hashes(baseline_root: Path, report_label: str) -> set[str]:
    reports = sorted(
        baseline_root.glob(f"clinic28-*/正式交付/*_{report_label}.docx")
    )
    if len(reports) != 3:
        raise WordDeliveryContractError("frozen formal report set is incomplete")
    hash_sets = [set(_hashes(inspect_docx(path).get("media"))) for path in reports]
    return set.intersection(*hash_sets)


def archive_media_passed(
    generated_media: JsonValue | None,
    approved_media: JsonValue | None,
    generated_roles: dict[str, str] | None,
    approved_static_hashes: set[str] | None,
) -> bool:
    generated = _hashes(generated_media)
    approved = _hashes(approved_media)
    if generated_roles is None or approved_static_hashes is None:
        return Counter(generated) == Counter(approved)
    generated_static = Counter(
        digest for digest in generated if digest not in generated_roles
    )
    approved_static = Counter(
        digest for digest in approved if digest in approved_static_hashes
    )
    return generated_static == approved_static
