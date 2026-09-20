from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.acceptance.deep_compare_word import docx_content_signature
from scripts.acceptance.deep_compare_word_formal import _doc_contract
from scripts.acceptance.deep_compare_verdict import evaluate_sections
from scripts.acceptance.surface_commands import build_surface_command
from test_acceptance_deep_verdict import _passing_sections


BOOKMARKS = (
    "user_overview",
    "report_toc",
    *(f"module_{index:02d}" for index in range(1, 12)),
)


def _write_docx(
    path: Path,
    visible_text: str,
    media_parts: tuple[bytes, ...],
    relationship_targets: tuple[str, ...] = (),
) -> None:
    bookmarks = "".join(
        f'<w:bookmarkStart w:id="{index}" w:name="{name}"/>'
        for index, name in enumerate(BOOKMARKS, start=1)
    )
    document = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{bookmarks}<w:p><w:r><w:t>{visible_text}</w:t></w:r></w:p>'
        '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>SCORE</w:t></w:r></w:p></w:tc>'
        '</w:tr></w:tbl></w:body></w:document>'
    )
    image_relationships = "".join(
        '<Relationship '
        f'Id="rId{index}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        f'Target="{target}"/>'
        for index, target in enumerate(relationship_targets, start=2)
    )
    relationships = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        f'{image_relationships}</Relationships>'
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/_rels/document.xml.rels", relationships)
        for index, data in enumerate(media_parts, start=1):
            archive.writestr(f"word/media/image{index}.png", data)


def test_doc_contract_rejects_visible_content_drift_with_same_structure_and_media(
    tmp_path: Path,
) -> None:
    # Given
    media = b"dynamic-media"
    digest = hashlib.sha256(media).hexdigest()
    approved = tmp_path / "approved.docx"
    generated = tmp_path / "generated.docx"
    _write_docx(approved, "APPROVED SCORE 81", (media,))
    _write_docx(generated, "WRONG SCORE 0 DEBUG", (media,))

    # When
    result = _doc_contract(
        generated,
        approved,
        "user",
        {digest},
        {digest},
    )

    # Then
    assert result["approved_bookmarks_passed"] is True
    assert result["table_layout_passed"] is True
    assert result["embedded_dynamic_media_passed"] is True
    assert result["content_signature_passed"] is False


def test_doc_contract_rejects_duplicate_dynamic_media_parts(tmp_path: Path) -> None:
    # Given
    media = b"dynamic-media"
    digest = hashlib.sha256(media).hexdigest()
    approved = tmp_path / "approved.docx"
    generated = tmp_path / "generated.docx"
    _write_docx(approved, "APPROVED", (media,))
    _write_docx(generated, "APPROVED", (media, media))

    # When
    result = _doc_contract(
        generated,
        approved,
        "user",
        {digest},
        {digest},
    )

    # Then
    assert result["embedded_dynamic_media_passed"] is False


def test_doc_contract_rejects_orphan_non_dynamic_media_part(
    tmp_path: Path,
) -> None:
    # Given
    dynamic = b"dynamic-media"
    orphan = b"internal-orphan"
    digest = hashlib.sha256(dynamic).hexdigest()
    approved = tmp_path / "approved.docx"
    generated = tmp_path / "generated.docx"
    _write_docx(approved, "APPROVED", (dynamic,))
    _write_docx(generated, "APPROVED", (dynamic, orphan))

    # When
    result = _doc_contract(
        generated,
        approved,
        "user",
        {digest},
        {digest},
    )

    # Then
    assert result["embedded_dynamic_media_passed"] is False
    assert result["embedded_media_multiset_passed"] is False


def test_content_signature_rejects_image_relationship_role_swap(tmp_path: Path) -> None:
    # Given
    first = b"first-media"
    second = b"second-media"
    approved = tmp_path / "approved.docx"
    generated = tmp_path / "generated.docx"
    _write_docx(
        approved,
        "APPROVED",
        (first, second),
        ("media/image1.png", "media/image2.png"),
    )
    _write_docx(
        generated,
        "APPROVED",
        (first, second),
        ("media/image2.png", "media/image1.png"),
    )

    # When / Then
    roles = {
        hashlib.sha256(first).hexdigest(): "module:01:1",
        hashlib.sha256(second).hexdigest(): "module:02:1",
    }
    assert docx_content_signature(generated, media_roles=roles) != (
        docx_content_signature(approved, media_roles=roles)
    )


def test_word_gate_applies_only_to_merged_local_surface(tmp_path: Path) -> None:
    # Given
    baseline_command = build_surface_command(
        "baseline",
        "run",
        tmp_path / "baseline",
        tmp_path / "merged",
        tmp_path / "python",
        tmp_path / "inputs.json",
        tmp_path / "results",
        tmp_path / "evidence",
    )
    sections = _passing_sections()
    sections["word"]["baseline"] = {"docx_count": 0, "reports": []}

    # When
    verdict = evaluate_sections(sections)

    # Then
    assert "--generate-medical-report" not in baseline_command
    assert verdict.status == "passed"
