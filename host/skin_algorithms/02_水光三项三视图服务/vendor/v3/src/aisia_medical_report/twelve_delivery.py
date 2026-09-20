from __future__ import annotations

"""Render the approved dual Word reports from one canonical twelve-item run.

The accepted Clinic payload remains the wording/layout baseline.  This adapter
only rebinds its image slots to artifacts produced by the same ``run.py``
invocation; it never reruns or re-renders an algorithm image.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Final

from .integration import DEFAULT_TEMPLATE
from .report import render_doctor_docx, render_user_docx
from .twelve_delivery_contract import (
    WordDeliveryContractError,
    portable_payload,
    require_fresh_report_root,
    resolve_artifact,
    sha256,
    validate_index_media,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ALIASES = ("clinic28-25", "clinic28-09", "clinic28-23")
BASELINE_PAYLOAD_SHA256: Final = {
    "clinic28-25": "5b3fa9fc9b3a758a278fefb4da45d0828ea318a6f729af9fb819b33c81b3bcc4",
    "clinic28-09": "0f116ef9a0f6108df97ddf4175e126ff71c01de5c1cf2386497d94c3d0ff86eb",
    "clinic28-23": "b2a69a7dfae66233bbd9df0af29d134570eebd7130416b8fc221e168442c5e37",
}
DEFAULT_BASELINE_ROOT = (
    PROJECT_ROOT.parent / "00_runtime_assets/frozen_formal_word_baseline"
)


def _baseline_root() -> Path:
    configured = os.environ.get("AISIA_FORMAL_BASELINE_ROOT")
    root = Path(configured).expanduser() if configured else DEFAULT_BASELINE_ROOT
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            "定稿正式报告基线不存在；可用 AISIA_FORMAL_BASELINE_ROOT 指定: "
            f"{root}"
        )
    return root


def _baseline_json(root: Path, alias: str) -> Path:
    path = (
        root
        / alias
        / "reports/formal_evidence"
        / f"AISIA_{alias}_正式结构化数据.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = sha256(path)
    if actual_sha256 != BASELINE_PAYLOAD_SHA256[alias]:
        raise RuntimeError(
            f"定稿正式报告基线SHA256不匹配: {alias} {actual_sha256}"
        )
    return path


def _capture_alias(result_root: Path, source: Path, baseline_root: Path) -> str:
    hinted = tuple(alias for alias in CAPTURE_ALIASES if alias in result_root.name)
    candidates = hinted or CAPTURE_ALIASES
    source_sha = sha256(source)
    for alias in candidates:
        payload = json.loads(_baseline_json(baseline_root, alias).read_text(encoding="utf-8"))
        if str((payload.get("标准采集图像") or {}).get("sha256")) == source_sha:
            return alias
    raise WordDeliveryContractError(
        f"无法把固定RGB样本映射到Clinic-Fast3别名: {result_root.name}"
    )


def _stage(path: Path, report_root: Path, module_id: str) -> Path:
    target = report_root / "evidence" / module_id / f"{sha256(path)[:12]}_{path.name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or sha256(target) != sha256(path):
        shutil.copy2(path, target)
    return target


def _image_map(result_root: Path, index: dict[str, Any], report_root: Path) -> dict[str, Path]:
    items = index["items"]

    def main(item_id: str, module_id: str) -> Path:
        return _stage(
            resolve_artifact(result_root, items[item_id]["主结果图"]),
            report_root,
            module_id,
        )

    def extra(item_id: str, position: int, module_id: str) -> Path:
        values = items[item_id].get("附加结果图") or []
        if position >= len(values):
            raise ValueError(f"{item_id} 缺少第 {position + 1} 张正式附加图")
        return _stage(
            resolve_artifact(result_root, values[position]),
            report_root,
            module_id,
        )

    return {
        "01": main("pores", "01"),
        "02_oil": main("surface_gloss", "02"),
        "02_porphyrin": main("porphyrin", "02"),
        "03_visible": main("spots", "03"),
        "03_uv": main("uv_spots", "03"),
        "03_brown": main("brown", "03"),
        "04": main("redness", "04"),
        "05": main("vascular", "05"),
        "06": main("acne", "06"),
        "07": main("wrinkle", "07"),
        "08": extra("wrinkle", 0, "08"),
        "09": extra("wrinkle", 1, "09"),
        "10": main("texture", "10"),
        "11": main("contour_firmness", "11"),
    }


def _set_group_image(group: dict[str, Any], path: Path | None, caption: str) -> None:
    group["images"] = [] if path is None else [{"path": str(path), "caption": caption}]


def bind_twelve_report_images(
    payload: dict[str, Any],
    images: dict[str, Path],
    source: Path,
) -> None:
    payload["标准采集图像"] = {
        "path": str(source),
        "caption": "标准白光正面图",
        "sha256": sha256(source),
        "size_bytes": source.stat().st_size,
    }
    modules = {str(row["模块编号"]): row for row in payload["检测模块"]}
    brown_marker = images.get("03_brown_marker", images["03_brown"])
    redness_marker = images.get("04_marker", images["04"])
    pigment_images = [
        brown_marker,
        images["03_visible"],
        images["03_uv"],
        *([images["03_red_brown_mixed"]] if "03_red_brown_mixed" in images else []),
        *([images["03_priority_pigment"]] if "03_priority_pigment" in images else []),
    ]
    texture_images = [
        images["10"],
        *([images["10_surface_irregularity"]] if "10_surface_irregularity" in images else []),
    ]
    result_images = {
        "01": [images["01"]],
        "02": [images["02_oil"], images["02_porphyrin"]],
        "03": pigment_images,
        "04": [redness_marker],
        "05": [images["05"]],
        "06": [images["06"]],
        "07": [images["07"]],
        "08": [images["08"]],
        "09": [images["09"]],
        "10": texture_images,
        "11": [images["11"]],
    }
    for module_id, paths in result_images.items():
        modules[module_id]["结果图"] = [str(path) for path in paths]

    for group in modules["02"].get("医生结果分组") or []:
        if "油光" in str(group.get("title")):
            _set_group_image(group, images["02_oil"], "表面油光分布")
        elif "荧光" in str(group.get("title")):
            _set_group_image(group, images["02_porphyrin"], "毛囊荧光分布")
    for group in modules["03"].get("医生结果分组") or []:
        title = str(group.get("title"))
        if "可见" in title:
            _set_group_image(group, images["03_visible"], "可见色斑结果")
        elif "UV" in title:
            _set_group_image(group, images["03_uv"], "UV色斑结果")
        elif "Brown" in title:
            _set_group_image(
                group,
                brown_marker,
                "Brown综合色素带特征点结果",
            )
        elif "红褐混合" in title and "03_red_brown_mixed" in images:
            _set_group_image(
                group,
                images["03_red_brown_mixed"],
                "红褐混合印记",
            )
        elif "重点色斑" in title and "03_priority_pigment" in images:
            _set_group_image(
                group,
                images["03_priority_pigment"],
                "重点色斑区域",
            )
        else:
            _set_group_image(group, None, "")
    for group in modules["04"].get("医生结果分组") or []:
        if "强度" in str(group.get("title")):
            _set_group_image(
                group,
                redness_marker,
                "红区带特征点结果",
            )
        else:
            _set_group_image(group, None, "")
    for module_id, key, caption in (
        ("05", "05", "血管样结构分布"),
        ("07", "07", "干燥性细纹分布"),
        ("11", "11", "面部轮廓几何测量"),
    ):
        for group in modules[module_id].get("医生结果分组") or []:
            _set_group_image(group, images[key], caption)
    for group in modules["10"].get("医生结果分组") or []:
        title = str(group.get("title"))
        if "纹理" in title:
            _set_group_image(group, images["10"], "表面纹理分布")
        elif "不规则" in title and "10_surface_irregularity" in images:
            _set_group_image(
                group,
                images["10_surface_irregularity"],
                "表面不规则分布",
            )
        else:
            _set_group_image(group, None, "")


def generate_dual_reports_from_twelve_result(
    result_dir: str | Path,
    *,
    source_image: str | Path,
    subject_id: str | None = None,
) -> dict[str, str]:
    result_root = Path(result_dir).expanduser().resolve()
    index_path = result_root / "十二项检测结果索引.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    validate_index_media(index, result_root)

    source = Path(source_image).expanduser().resolve(strict=True)
    report_root = result_root / "正式报告"
    require_fresh_report_root(report_root)
    baseline_root = _baseline_root()
    alias = _capture_alias(result_root, source, baseline_root)
    payload = json.loads(_baseline_json(baseline_root, alias).read_text(encoding="utf-8"))
    payload["受检者信息"]["姓名或编号"] = subject_id or alias
    user_name = f"AISIA_{result_root.name}_用户精简版.docx"
    doctor_name = f"AISIA_{result_root.name}_医生详细版.docx"
    structured_name = f"AISIA_{result_root.name}_正式结构化数据.json"

    with tempfile.TemporaryDirectory(prefix=".aisia-formal-", dir=result_root) as temporary:
        working_root = Path(temporary)
        staged_source = _stage(source, working_root, "00")
        images = _image_map(result_root, index, working_root)
        bind_twelve_report_images(payload, images, staged_source)
        render_user_docx(payload, DEFAULT_TEMPLATE, working_root / user_name)
        render_doctor_docx(payload, DEFAULT_TEMPLATE, working_root / doctor_name)
        docx_names = {path.name for path in working_root.glob("*.docx")}
        if docx_names != {user_name, doctor_name}:
            raise WordDeliveryContractError(
                f"正式Word输出数量或文件名异常: {sorted(docx_names)}"
            )
        (working_root / structured_name).write_text(
            json.dumps(
                portable_payload(payload, working_root),
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        index["formal_reports"] = {
            "user_docx": (report_root / user_name).relative_to(result_root).as_posix(),
            "doctor_docx": (report_root / doctor_name).relative_to(result_root).as_posix(),
        }
        staged_index = working_root / ".twelve-index.json"
        staged_index.write_text(
            json.dumps(index, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(working_root, report_root)
        try:
            os.replace(report_root / staged_index.name, index_path)
        except OSError:
            shutil.rmtree(report_root)
            raise

    return {
        "user_docx": str(report_root / user_name),
        "doctor_docx": str(report_root / doctor_name),
        "structured_json": str(report_root / structured_name),
    }


__all__ = [
    "bind_twelve_report_images",
    "generate_dual_reports_from_twelve_result",
]
