from __future__ import annotations

import hashlib
from pathlib import Path

from src.detection_runtime import final_delivery_layout


def test_institution_index_records_each_media_sha(tmp_path: Path) -> None:
    first = tmp_path / "main.jpg"
    second = tmp_path / "extra.jpg"
    first.write_bytes(b"main")
    second.write_bytes(b"extra")
    index = {"十二项结果": {"wrinkle": {
        "主结果图": first.name,
        "附加结果图": [second.name],
    }}}
    attach = getattr(final_delivery_layout, "attach_institution_media_hashes", None)

    assert callable(attach)
    attach(tmp_path, index)

    assert index["十二项结果"]["wrinkle"]["images"] == [
        {"path": first.name, "sha256": hashlib.sha256(b"main").hexdigest()},
        {"path": second.name, "sha256": hashlib.sha256(b"extra").hexdigest()},
    ]


def test_institution_final_layout_removes_broken_structured_payloads(
    tmp_path: Path,
) -> None:
    structured = tmp_path / "AISIA_clinic28-25_正式结构化数据.json"
    trace = tmp_path / "AISIA_clinic28-25_评分追溯.json"
    structured.write_text("{}\n", encoding="utf-8")
    trace.write_text("{}\n", encoding="utf-8")

    final_delivery_layout.remove_institution_report_transients(tmp_path)

    assert not structured.exists()
    assert trace.exists()
