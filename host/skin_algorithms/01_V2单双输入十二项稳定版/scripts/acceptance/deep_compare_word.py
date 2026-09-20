"""DOCX structural and leak inspection without rendering or mutation."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from pathlib import Path
from typing import Final
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from scripts.acceptance.acceptance_types import JsonDict, JsonValue


WORD_NS: Final = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
PATH_PATTERNS: Final = (
    re.compile(r"/home/[^\s<]+"),
    re.compile(r"/tmp/[^\s<]+"),
    re.compile(r"[A-Za-z]:\\[^\s<]+"),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _leaks(text: str) -> list[str]:
    return sorted({match.group(0) for pattern in PATH_PATTERNS for match in pattern.finditer(text)})


def docx_content_signature(
    path: Path,
    subjects: tuple[str, ...] = (),
    media_roles: dict[str, str] | None = None,
    normalize_image_targets: bool = False,
    ignored_visible: frozenset[str] = frozenset(),
) -> str:
    """Hash visible text, table structure, bookmarks, and relationships."""

    with ZipFile(path) as archive:
        document = ElementTree.fromstring(archive.read("word/document.xml"))
        relationship_rows: list[tuple[str, str, str, str]] = []
        if "word/_rels/document.xml.rels" in archive.namelist():
            relationships = ElementTree.fromstring(
                archive.read("word/_rels/document.xml.rels")
            )
            for relationship in relationships:
                relation_type = relationship.get("Type", "")
                target = relationship.get("Target", "")
                if relation_type.endswith("/image"):
                    if normalize_image_targets:
                        continue
                    elif media_roles is not None:
                        archive_name = posixpath.normpath(f"word/{target}")
                        digest = _sha256(archive.read(archive_name))
                        target = media_roles.get(digest, f"static:{digest}")
                relationship_rows.append(
                    (
                        relationship.get("Id", ""),
                        relation_type,
                        target,
                        relationship.get("TargetMode", ""),
                    )
                )
    visible = [
        "".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t"))
        for paragraph in document.iter(f"{WORD_NS}p")
    ]
    if ignored_visible:
        visible = [text for text in visible if text not in ignored_visible]
    for subject in subjects:
        if subject:
            visible = [text.replace(subject, "<SUBJECT>") for text in visible]
    bookmarks = sorted(
        item.get(f"{WORD_NS}name") or ""
        for item in document.iter(f"{WORD_NS}bookmarkStart")
        if (item.get(f"{WORD_NS}name") or "") != "_GoBack"
    )
    table_shapes = [
        [len(list(table.findall(f"{WORD_NS}tr"))), *[
            len(list(row.findall(f"{WORD_NS}tc")))
            for row in table.findall(f"{WORD_NS}tr")
        ]]
        for table in document.iter(f"{WORD_NS}tbl")
    ]
    payload = json.dumps(
        {
            "visible": visible,
            "bookmarks": bookmarks,
            "tables": table_shapes,
            "relationships": sorted(relationship_rows),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inspect_docx(path: Path) -> JsonDict:
    """Return chapters, tables, bookmarks, media hashes, and path leaks."""

    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            document_bytes = archive.read("word/document.xml")
            xml_texts = [
                archive.read(name).decode("utf-8", errors="replace")
                for name in names
                if name.endswith((".xml", ".rels"))
            ]
            media: list[JsonValue] = [
                {
                    "path": name,
                    "size_bytes": len(data),
                    "sha256": _sha256(data),
                }
                for name in sorted(names)
                if name.startswith("word/media/")
                for data in (archive.read(name),)
            ]
    except (BadZipFile, KeyError) as exc:
        return {"path": str(path), "valid_docx": False, "error": str(exc)}
    root = ElementTree.fromstring(document_bytes)
    chapters = 0
    for paragraph in root.iter(f"{WORD_NS}p"):
        styles = paragraph.findall(f".//{WORD_NS}pStyle")
        if any((style.get(f"{WORD_NS}val") or "").lower().startswith("heading") for style in styles):
            chapters += 1
    bookmarks = [
        item
        for item in root.iter(f"{WORD_NS}bookmarkStart")
        if (item.get(f"{WORD_NS}name") or "") != "_GoBack"
    ]
    table_shapes = [
        [len(list(table.findall(f"{WORD_NS}tr"))), *[
            len(list(row.findall(f"{WORD_NS}tc")))
            for row in table.findall(f"{WORD_NS}tr")
        ]]
        for table in root.iter(f"{WORD_NS}tbl")
    ]
    module_bookmarks = [
        item for item in bookmarks if (item.get(f"{WORD_NS}name") or "").startswith("module_")
    ]
    return {
        "path": str(path),
        "valid_docx": True,
        "sha256": _sha256(path.read_bytes()),
        "chapter_count": len(module_bookmarks) or chapters,
        "heading_count": chapters,
        "table_count": len(list(root.iter(f"{WORD_NS}tbl"))),
        "table_shapes": table_shapes,
        "bookmark_count": len(bookmarks),
        "bookmark_names": [item.get(f"{WORD_NS}name") or "" for item in bookmarks],
        "media": media,
        "path_leaks": sorted({leak for text in xml_texts for leak in _leaks(text)}),
    }


def inspect_word_tree(root: Path) -> JsonDict:
    reports: list[JsonDict] = []
    for path in sorted(root.rglob("*.docx")):
        report = inspect_docx(path)
        report["path"] = path.relative_to(root).as_posix()
        reports.append(report)
    output_image_hashes = {
        _sha256(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    for report in reports:
        media = report.get("media")
        media_hashes = {
            str(item.get("sha256"))
            for item in media
            if isinstance(media, list) and isinstance(item, dict)
        } if isinstance(media, list) else set()
        report["media_hashes_found_in_output"] = sorted(
            media_hashes & output_image_hashes
        )
        report["media_hash_match_count"] = len(media_hashes & output_image_hashes)
    return {
        "docx_count": len(reports),
        "reports": reports,
        "path_leak_count": sum(len(report.get("path_leaks", [])) for report in reports),
        "merged_formal_contract_passed": bool(reports)
        and all(
            report.get("chapter_count") == 11
            and report.get("bookmark_count") == 13
            and not report.get("path_leaks")
            for report in reports
        ),
    }


def scan_output_path_leaks(root: Path) -> list[JsonValue]:
    findings: list[JsonValue] = []
    text_suffixes = {".json", ".csv", ".md", ".txt", ".html", ".xml"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".docx":
            report = inspect_docx(path)
            leaks = report.get("path_leaks", [])
        elif path.suffix.lower() in text_suffixes:
            leaks = _leaks(path.read_text(encoding="utf-8", errors="replace"))
        else:
            continue
        if leaks:
            findings.append({"path": path.relative_to(root).as_posix(), "leaks": leaks})
    return findings
