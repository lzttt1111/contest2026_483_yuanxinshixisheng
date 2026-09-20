"""Frozen semantic, structural, and media checks for merged formal reports."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Final

from scripts.acceptance.acceptance_types import JsonDict, JsonValue
from scripts.acceptance.deep_compare_word import docx_content_signature, inspect_docx
from scripts.acceptance.deep_compare_word_archive import (
    archive_media_passed,
    frozen_static_hashes,
    media_roles,
)
from scripts.acceptance.deep_compare_word_media import (
    indexed_hashes,
    module_hashes,
    structured_media_passed,
)
from src.aisia_medical_report.twelve_delivery_contract import (
    WordDeliveryContractError,
    resolve_artifact,
    validate_index_media,
)


CASES: Final = ("clinic28-25", "clinic28-09", "clinic28-23")
MODULE_BOOKMARKS: Final = {f"module_{index:02d}" for index in range(1, 12)}
LEGACY_DOCTOR_ONLY_GROUP_TITLES: Final = frozenset(
    {
        "",
        "红褐混合印记",
        "重点色斑区域",
        "表面不规则分布",
    }
)


def _mapping(value: JsonValue | None) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WordDeliveryContractError("formal structured JSON must be an object")
    return value


def _subject_identity_passed(payload: JsonDict, case_id: str) -> bool:
    return str(_mapping(payload.get("受检者信息")).get("姓名或编号", "")) == case_id


def semantic_signature(payload: JsonDict) -> str:
    """Hash frozen wording/scores/recommendations while excluding bound media."""

    normalized = copy.deepcopy(payload)
    subject = _mapping(normalized.get("受检者信息"))
    subject.pop("姓名或编号", None)
    source = _mapping(normalized.get("标准采集图像"))
    for key in ("path", "sha256", "size_bytes"):
        source.pop(key, None)
    modules = normalized.get("检测模块")
    if isinstance(modules, list):
        for module_value in modules:
            module = _mapping(module_value)
            result_images = module.get("结果图")
            if isinstance(result_images, list):
                module["结果图"] = ["<BOUND_MEDIA>" for _ in result_images]
            groups = module.get("医生结果分组")
            if isinstance(groups, list):
                for group_value in groups:
                    group = _mapping(group_value)
                    images = group.get("images")
                    if isinstance(images, list):
                        for image_value in images:
                            image = _mapping(image_value)
                            for key in ("path", "sha256", "size_bytes"):
                                image.pop(key, None)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _doc_contract(
    generated: Path,
    approved: Path,
    role: str,
    all_dynamic: set[str],
    expected_dynamic: set[str],
    subjects: tuple[str, ...] = (),
    generated_roles: dict[str, str] | None = None,
    approved_static_hashes: set[str] | None = None,
) -> JsonDict:
    report = inspect_docx(generated)
    report["path"] = generated.name
    baseline = inspect_docx(approved)
    expected_bookmarks = {
        "user_overview" if role == "user" else "doctor_overview",
        "report_toc",
        *MODULE_BOOKMARKS,
    }
    media = report.get("media")
    media_hashes = [
        str(_mapping(value).get("sha256"))
        for value in media
        if isinstance(media, list) and isinstance(value, dict)
    ] if isinstance(media, list) else []
    expected_counter = Counter({digest: 1 for digest in expected_dynamic})
    actual_counter = Counter(
        digest for digest in media_hashes if digest in all_dynamic
    )
    media_multiset_passed = archive_media_passed(
        media,
        baseline.get("media"),
        generated_roles,
        approved_static_hashes,
    )
    ignored_visible = (
        LEGACY_DOCTOR_ONLY_GROUP_TITLES if role == "doctor" else frozenset()
    )
    return {
        **report,
        "approved_bookmarks_passed": set(report.get("bookmark_names", [])) == expected_bookmarks,
        "table_layout_passed": report.get("table_shapes") == baseline.get("table_shapes"),
        "embedded_dynamic_media_passed": (
            actual_counter == expected_counter and media_multiset_passed
        ),
        "embedded_media_multiset_passed": media_multiset_passed,
        "content_signature_passed": docx_content_signature(
            generated,
            subjects,
            normalize_image_targets=True,
            ignored_visible=ignored_visible,
        ) == docx_content_signature(
            approved,
            subjects,
            normalize_image_targets=True,
            ignored_visible=ignored_visible,
        ),
        "allowed_legacy_group_title_omissions": sorted(ignored_visible - {""}),
    }


def inspect_formal_word_tree(run_root: Path, baseline_root: Path) -> JsonDict:
    samples: list[JsonValue] = []
    reports: list[JsonValue] = []
    index_paths = sorted(run_root.rglob("十二项检测结果索引.json"))
    user_static = frozen_static_hashes(baseline_root, "用户精简版") if index_paths else set()
    doctor_static = frozen_static_hashes(baseline_root, "医生详细版") if index_paths else set()
    for index_path in index_paths:
        sample_root = index_path.parent
        case_id = next((case for case in CASES if case in sample_root.name), "")
        index = _load_json(index_path)
        subject_identity_passed = False
        try:
            validate_index_media(index, sample_root)
            report_root = sample_root / "正式报告"
            generated_structured = next(report_root.glob("*_正式结构化数据.json"))
            structured = _load_json(generated_structured)
            generated_roles = media_roles(structured, report_root)
            indexed = indexed_hashes(index)
            expected_modules = module_hashes(indexed)
            source = _mapping(structured.get("标准采集图像"))
            source_sha = str(source.get("sha256"))
            subject_identity_passed = _subject_identity_passed(structured, case_id)
            subjects = (case_id,)
            all_dynamic = {source_sha, *(digest for values in expected_modules.values() for digest in values)}
            user_dynamic = {source_sha, *(
                digest for module_id, values in expected_modules.items()
                for digest in (values[:2] if module_id == "02" else values[:1])
            )}
            formal = _mapping(index.get("formal_reports"))
            user = resolve_artifact(sample_root, str(formal.get("user_docx")))
            doctor = resolve_artifact(sample_root, str(formal.get("doctor_docx")))
            user_report = _doc_contract(
                user,
                baseline_root / case_id / "正式交付" / f"AISIA_{case_id}_用户精简版.docx",
                "user",
                all_dynamic,
                user_dynamic,
                subjects,
                generated_roles,
                user_static,
            )
            doctor_report = _doc_contract(
                doctor,
                baseline_root / case_id / "正式交付" / f"AISIA_{case_id}_医生详细版.docx",
                "doctor",
                all_dynamic,
                all_dynamic,
                subjects,
                generated_roles,
                doctor_static,
            )
            reports.extend((user_report, doctor_report))
            passed = (
                subject_identity_passed
                and structured_media_passed(
                    report_root, structured, expected_modules
                )
                and all(
                    _mapping(report).get(key) is True
                    for report in (user_report, doctor_report)
                    for key in (
                        "valid_docx",
                        "approved_bookmarks_passed",
                        "table_layout_passed",
                        "embedded_dynamic_media_passed",
                        "embedded_media_multiset_passed",
                        "content_signature_passed",
                    )
                )
            )
        except (
            OSError,
            StopIteration,
            WordDeliveryContractError,
            KeyError,
            IndexError,
            ValueError,
            TypeError,
        ):
            passed = False
        samples.append(
            {
                "case_id": case_id,
                "subject_identity_passed": subject_identity_passed,
                "formal_equivalence_passed": passed,
            }
        )
    return {
        "docx_count": len(list(run_root.rglob("*.docx"))),
        "reports": reports,
        "samples": samples,
        "merged_formal_contract_passed": len(samples) == 3
        and len(reports) == 6
        and all(_mapping(value).get("formal_equivalence_passed") is True for value in samples),
    }
