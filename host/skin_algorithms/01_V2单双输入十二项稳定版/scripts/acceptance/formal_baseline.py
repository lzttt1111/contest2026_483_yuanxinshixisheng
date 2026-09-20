"""Immutable formal-report baseline proof for explicit merged Word runs."""

from __future__ import annotations

from pathlib import Path
from typing import Final
from zipfile import BadZipFile, ZipFile

from pydantic import JsonValue, TypeAdapter

from scripts.acceptance.acceptance_io import AcceptanceInputError, sha256_file


EXPECTED_MANIFEST_SHA256: Final = "75d947b53895584132e63623cb42765fe70b77a2ef0ff624b10df96b6fff781f"
EXPECTED_TEMPLATE: Final = "9de65290471199a14690db5b0db1571257788a24c23168ccf077e3c3701ffd4c"
EXPECTED_RENDERER: Final = "84397fb5b8a6066c86b8542eec4007d5db2b70da3e89d6840f2b768c456ff0d1"
EXPECTED_ROLES: Final = {
    (sample, role)
    for sample in ("clinic28-09", "clinic28-23", "clinic28-25")
    for role in ("formal_user", "formal_doctor", "formal_structured_payload")
}
JSON_ADAPTER = TypeAdapter(JsonValue)


def verify_formal_baseline(project_root: Path, baseline_root: Path) -> None:
    root = baseline_root.resolve()
    manifest_path = root / "FROZEN_WORD_BASELINE_MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise AcceptanceInputError(detail=f"formal baseline manifest unavailable: {manifest_path}")
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise AcceptanceInputError(detail="formal baseline manifest SHA256 mismatch")
    value = JSON_ADAPTER.validate_json(manifest_path.read_text(encoding="utf-8"))
    manifest = value if isinstance(value, dict) else {}
    files = manifest.get("files")
    if (
        manifest.get("schema") != "aisia_frozen_formal_word_baseline_manifest_v2"
        or manifest.get("included_roles")
        != ["formal_user", "formal_doctor", "formal_structured_payload"]
        or manifest.get("excluded_roles") != ["engineering_debug"]
        or not isinstance(files, list)
        or len(files) != 9
    ):
        raise AcceptanceInputError(detail="formal baseline manifest contract mismatch")
    observed: set[tuple[str, str]] = set()
    expected_files = {"FROZEN_WORD_BASELINE_MANIFEST.json"}
    for file_value in files:
        item = file_value if isinstance(file_value, dict) else {}
        sample = item.get("sample")
        role = item.get("role")
        asset_class = item.get("asset_class")
        filename = item.get("filename")
        relative = item.get("relative_path")
        if not all(isinstance(field, str) for field in (sample, role, filename, relative)):
            raise AcceptanceInputError(detail="formal baseline file entry invalid")
        expected_relative = (
            f"{sample}/reports/formal_evidence/{filename}"
            if role == "formal_structured_payload"
            else f"{sample}/正式交付/{filename}"
        )
        expected_class = (
            "internal_render_source"
            if role == "formal_structured_payload"
            else "public_frozen_document"
        )
        candidate = root / str(relative)
        path = candidate.resolve()
        if (
            str(relative) != expected_relative
            or asset_class != expected_class
            or root not in path.parents
            or not path.is_file()
            or candidate.is_symlink()
            or path.stat().st_size != item.get("size_bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise AcceptanceInputError(detail="formal baseline DOCX mismatch")
        if role != "formal_structured_payload":
            try:
                with ZipFile(path) as archive:
                    if archive.testzip() is not None or "word/document.xml" not in archive.namelist():
                        raise AcceptanceInputError(detail="formal baseline DOCX CRC mismatch")
            except BadZipFile as exc:
                raise AcceptanceInputError(detail="formal baseline DOCX invalid") from exc
        observed.add((str(sample), str(role)))
        expected_files.add(str(relative))
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != EXPECTED_ROLES or actual_files != expected_files:
        raise AcceptanceInputError(detail="formal baseline file set mismatch")
    if any((root / relative).stat().st_mode & 0o222 for relative in expected_files):
        raise AcceptanceInputError(detail="formal baseline files must be readonly")
    from src.aisia_medical_report.twelve_delivery import _baseline_json

    for sample in ("clinic28-25", "clinic28-09", "clinic28-23"):
        _baseline_json(root, sample)
    template = project_root / "templates" / "AISIA_面部多指标检测汇总模板_V2.docx"
    renderer = project_root / "src" / "aisia_medical_report" / "report.py"
    if sha256_file(template) != EXPECTED_TEMPLATE:
        raise AcceptanceInputError(detail="formal template SHA256 mismatch")
    if sha256_file(renderer) != EXPECTED_RENDERER:
        raise AcceptanceInputError(detail="formal renderer SHA256 mismatch")
