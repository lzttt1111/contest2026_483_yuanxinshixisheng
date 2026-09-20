from __future__ import annotations

"""Generate institution reports through the validated current-result chain."""

import json
import os
from pathlib import Path

from .single_rgb_v012_delivery import generate_single_rgb_v012_reports
from .identity import resolve_subject_id
from .twelve_delivery_contract import WordDeliveryContractError, resolve_artifact


def _write_current_complete(
    result_root: Path,
    runtime_items: dict[str, dict],
) -> None:
    from src.detection_runtime.final_delivery_layout import (
        build_institution_complete_document,
    )

    complete = build_institution_complete_document(result_root, runtime_items)
    (result_root / "十二项完整量化指标.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rename_outputs(
    result_root: Path,
    generated: dict[str, str],
) -> dict[str, str]:
    names = {
        "user_docx": f"AISIA_{result_root.name}_用户精简版.docx",
        "doctor_docx": f"AISIA_{result_root.name}_医生详细版.docx",
        "structured_json": f"AISIA_{result_root.name}_正式结构化数据.json",
        "trace_json": f"AISIA_{result_root.name}_评分追溯.json",
    }
    output: dict[str, str] = {}
    for role, source_value in generated.items():
        target = result_root / names[role]
        if target.exists():
            raise WordDeliveryContractError(f"机构正式报告已存在: {target.name}")
        os.replace(source_value, target)
        output[role] = str(target)
    return output


def generate_clinic_dual_reports_from_twelve_result(
    result_dir: str | Path,
    *,
    subject_id: str | None = None,
    runtime_items: dict[str, dict],
) -> dict[str, str]:
    """Render institution Word from this run, never from frozen metric values."""

    result_root = Path(result_dir).expanduser().resolve()
    index_path = result_root / "十二项检测结果索引.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("状态") != "success" or int(index.get("成功项目数", 0)) != 12:
        raise RuntimeError("正式双版只接受十二项全部成功的公开结果")
    from src.detection_runtime.final_delivery_layout import (
        attach_institution_media_hashes,
    )

    attach_institution_media_hashes(result_root, index)
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    source = resolve_artifact(result_root, str(index["输入图像"]))
    _write_current_complete(result_root, runtime_items)
    generated = generate_single_rgb_v012_reports(
        result_root,
        source_image=source,
        subject_id=resolve_subject_id(subject_id, source=result_root.name),
        runtime_items=runtime_items,
        report_id=f"AISIA-CLINIC-{result_root.name}",
    )
    output = _rename_outputs(result_root, generated)
    index["正式报告"] = {
        "用户精简版": Path(output["user_docx"]).name,
        "医生详细版": Path(output["doctor_docx"]).name,
    }
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["generate_clinic_dual_reports_from_twelve_result"]
