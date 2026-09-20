from __future__ import annotations

"""Thin current-twelve to validated single-RGB V0.1.2 report adapter."""

import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Final

import cv2

from src.capture_profile import CaptureProfile
from src.nine_analysis.metrics import extract_item

from .controlled_evidence import (
    RuntimeItems,
    materialize_controlled_evidence,
    publish_controlled_report_assets,
)
from .identity import single_rgb_report_identity
from .twelve_delivery_contract import (
    WordDeliveryContractError,
    resolve_artifact,
    validate_index_media,
)
from .v012 import generate_front5_v012_reports


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
RUNTIME_ROOT: Final = PROJECT_ROOT / "00_runtime_assets/scoring_v011"
OFFICIAL_PROFILE: Final = RUNTIME_ROOT / "全量历史ECDF评分配置.json"
SHADOW_PROFILE: Final = RUNTIME_ROOT / "综合色素V0.1.2_shadow评分配置.json"
PROFILE_ASSETS: Final = {
    OFFICIAL_PROFILE.name: (
        11339357,
        "8cf292c2e87b21a2b9e3543bd2a031567835eca330b0fc52134aaebf4435b808",
    ),
    SHADOW_PROFILE.name: (
        19753820,
        "b047ff042b22e93cf98f3a85f7b333a120ffe73a801bd659edbd6b33902565fa",
    ),
}

OLD_PATHS: Final = {
    "redness": ("红区", "红区量化指标.json", "红区医学量化指标_V2.csv", ("03_RBX红区结果图.jpg", "06_VISIA红色区实例图.jpg")),
    "spots": ("斑点", "02_Spots量化指标.json", "02_Spots医学量化指标_V2.csv", ("01_Spots斑点结果图.jpg",)),
    "brown": ("棕区", "棕色斑量化指标.json", "棕色斑医学量化指标_V2.csv", ("01_RBX棕区结果图.jpg", "02_VISIA棕色斑实例图.jpg")),
    "texture": ("纹理", "纹理量化指标.json", "纹理医学量化指标_V2.csv", ("01_纹理检测结果图.jpg",)),
    "pores": ("毛孔", "毛孔量化指标.json", "毛孔医学量化指标_V2.csv", ("01_毛孔检测结果图.jpg",)),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise WordDeliveryContractError(f"JSON根结构无效: {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_paths() -> tuple[Path, Path]:
    configured = os.environ.get("AISIA_SCORING_V011_ROOT", "").strip()
    root = Path(configured).expanduser().resolve() if configured else RUNTIME_ROOT
    paths = tuple(root / name for name in PROFILE_ASSETS)
    for path in paths:
        expected_size, expected_sha = PROFILE_ASSETS[path.name]
        if not path.is_file() or path.stat().st_size != expected_size:
            raise WordDeliveryContractError(f"正式评分配置缺失或大小异常: {path.name}")
        if _sha256(path) != expected_sha:
            raise WordDeliveryContractError(f"正式评分配置SHA异常: {path.name}")
    return paths[0], paths[1]


def _copy_indexed_images(
    result_root: Path,
    item: dict[str, Any],
    target: Path,
    names: tuple[str, ...],
) -> None:
    sources = [item["主结果图"], *(item.get("附加结果图") or [])]
    for source, name in zip(sources, names):
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_root / source, target / name)


def _formal_images(
    result_root: Path,
    index: dict[str, Any],
) -> dict[str, Path]:
    items = index.get("items") or index.get("十二项结果")
    if not isinstance(items, dict):
        raise WordDeliveryContractError("十二项结果索引缺少项目映射")

    def main(item_id: str) -> Path:
        return resolve_artifact(result_root, items[item_id]["主结果图"])

    def extra(item_id: str, position: int) -> Path:
        values = items[item_id]["附加结果图"]
        return resolve_artifact(result_root, values[position])

    institution = index.get("路线") == "clinic_four_light"
    brown_base = extra("brown", 0) if institution else main("brown")
    brown_marker = main("brown") if institution else extra("brown", 0)
    redness_base = extra("redness", 0) if institution else main("redness")
    redness_marker = main("redness") if institution else extra("redness", 0)

    return {
        "01": main("pores"),
        "02_oil": main("surface_gloss"),
        "02_porphyrin": main("porphyrin"),
        "03_visible": main("spots"),
        "03_uv": main("uv_spots"),
        "03_brown": brown_marker,
        "03_brown_base": brown_base,
        "03_brown_marker": brown_marker,
        "04": redness_marker,
        "04_base": redness_base,
        "04_marker": redness_marker,
        "05": main("vascular"),
        "06": main("acne"),
        "07": main("wrinkle"),
        "08": extra("wrinkle", 0),
        "09": extra("wrinkle", 1),
        "10": main("texture"),
        "11": main("contour_firmness"),
    }


def _write_combined_purple_csv(
    result_root: Path,
    items: dict[str, Any],
    target: Path,
) -> None:
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    density_field = "核心-单位面积密度（个/10万有效皮肤像素）"
    labels = {
        "uv_spots": "标准化UV紫外线色斑工程代理",
        "porphyrin": "标准化荧光UV紫质工程代理",
    }
    for item_id in ("uv_spots", "porphyrin"):
        with (result_root / items[item_id]["医学V2CSV"]).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fields:
                    fields.append(field)
            for source in reader:
                row = dict(source)
                if not row.get("检测项目"):
                    row["检测项目"] = labels[item_id]
                if not row.get(density_field):
                    count = row.get("核心-特征数量（个）")
                    valid_area = row.get("有效皮肤面积（像素）")
                    if count and valid_area and float(valid_area) > 0:
                        row[density_field] = str(
                            round(float(count) * 100000.0 / float(valid_area), 4)
                        )
                if item_id == "porphyrin":
                    row.setdefault(
                        "核心-实例P50强度（0～1）",
                        row.get("核心-P50强度（0～1）", ""),
                    )
                    row.setdefault(
                        "核心-实例P90强度（0～1）",
                        row.get("核心-P90强度（0～1）", ""),
                    )
                rows.append(row)
    for required in (
        "检测项目",
        density_field,
        "核心-实例P50强度（0～1）",
        "核心-实例P90强度（0～1）",
    ):
        if required not in fields:
            fields.append(required)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _normalized_gloss_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("full_face"), dict):
        return raw
    overall = raw.get("overall_metrics")
    overall = overall if isinstance(overall, dict) else {}
    core = overall.get("core_metrics")
    core = core if isinstance(core, dict) else {}
    scope = core.get("scope_and_morphology")
    signal = core.get("signal_intensity")
    counts = core.get("count_and_density")
    scope = scope if isinstance(scope, dict) else {}
    signal = signal if isinstance(signal, dict) else {}
    counts = counts if isinstance(counts, dict) else {}
    full = {
        **scope,
        **signal,
        "patch_count": counts.get("gloss_patch_count"),
        "largest_gloss_component_area_ratio": 0.0,
    }
    regions: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw.get("region_metrics") or []):
        if not isinstance(value, dict):
            continue
        row_core = value.get("core_metrics")
        row_core = row_core if isinstance(row_core, dict) else {}
        row_scope = row_core.get("scope_and_morphology")
        row_signal = row_core.get("signal_intensity")
        row_counts = row_core.get("count_and_density")
        row_scope = row_scope if isinstance(row_scope, dict) else {}
        row_signal = row_signal if isinstance(row_signal, dict) else {}
        row_counts = row_counts if isinstance(row_counts, dict) else {}
        regions[f"institution_{index}"] = {
            "label": value.get("analysis_region", f"分区{index + 1}"),
            "status": value.get("evaluation_status", "ASSESSABLE"),
            **row_scope,
            **row_signal,
            "patch_count": row_counts.get("gloss_patch_count"),
            "largest_gloss_component_area_ratio": 0.0,
        }
    zone = {
        "status": "ASSESSABLE",
        "gloss_area_ratio": full.get("gloss_area_ratio"),
        "p90_gloss_intensity": full.get("p90_gloss_intensity"),
    }
    return {
        "full_face": full,
        "region_metrics": regions,
        "t_zone_summary": dict(zone),
        "cheek_zone_summary": dict(zone),
        "left_right_summary": {},
    }


def _normalized_vascular_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("vascular_count"), (int, float)):
        return raw
    overall = raw.get("overall_metrics")
    overall = overall if isinstance(overall, dict) else {}
    core = overall.get("core_metrics")
    core = core if isinstance(core, dict) else {}
    counts = core.get("count_and_density")
    scope = core.get("scope_and_morphology")
    signal = core.get("signal_intensity")
    counts = counts if isinstance(counts, dict) else {}
    scope = scope if isinstance(scope, dict) else {}
    signal = signal if isinstance(signal, dict) else {}
    analysis = overall.get("auxiliary_metrics")
    analysis = analysis if isinstance(analysis, dict) else {}
    analysis_scope = analysis.get("analysis_scope")
    analysis_scope = analysis_scope if isinstance(analysis_scope, dict) else {}
    valid_area = analysis_scope.get("valid_skin_area_px")
    branch_count = counts.get("branch_point_count")
    branch_density = (
        float(branch_count) * 10000.0 / float(valid_area)
        if isinstance(branch_count, (int, float))
        and isinstance(valid_area, (int, float))
        and valid_area > 0
        else None
    )
    regions: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw.get("region_metrics") or []):
        if not isinstance(value, dict):
            continue
        row_core = value.get("core_metrics")
        row_core = row_core if isinstance(row_core, dict) else {}
        row_counts = row_core.get("count_and_density")
        row_scope = row_core.get("scope_and_morphology")
        row_signal = row_core.get("signal_intensity")
        row_counts = row_counts if isinstance(row_counts, dict) else {}
        row_scope = row_scope if isinstance(row_scope, dict) else {}
        row_signal = row_signal if isinstance(row_signal, dict) else {}
        regions[f"institution_{index}"] = {
            "count": row_counts.get("vascular_count"),
            "total_length_px": row_scope.get("vascular_total_length_px"),
            "total_area_px": row_scope.get("vascular_area_px"),
            "p90_redness": row_signal.get("p90_redness_response"),
        }
    return {
        "vascular_count": counts.get("vascular_count"),
        "vascular_area_px": scope.get("vascular_area_px"),
        "vascular_area_ratio": scope.get("vascular_area_ratio"),
        "vascular_total_length_px": scope.get("vascular_total_length_px"),
        "vascular_line_density": counts.get(
            "vascular_line_density_per_10k_face_px"
        ),
        "p50_width_px": scope.get("p50_vascular_width_px"),
        "p90_width_px": scope.get("p90_vascular_width_px"),
        "p50_redness": signal.get("p50_redness_response"),
        "p90_redness": signal.get("p90_redness_response"),
        "branch_point_count": branch_count,
        "branch_point_density": branch_density,
        "network_ratio": None,
        "max_continuous_network_length_px": scope.get(
            "max_continuous_network_length_px"
        ),
        "region_distribution": regions,
        "left_right_summary": {},
    }


def _stage_current_result(
    result_root: Path,
    source_image: Path,
    stage_root: Path,
    *,
    complete_document: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, str]]:
    index = _load(result_root / "十二项检测结果索引.json")
    complete = complete_document or _load(result_root / "十二项完整量化指标.json")
    items = index.get("items") or index.get("十二项结果")
    if not isinstance(items, dict):
        raise WordDeliveryContractError("十二项结果索引缺少项目映射")
    review = stage_root / result_root.name
    candidate = stage_root / "candidate" / result_root.name
    review.mkdir(parents=True)
    shutil.copy2(source_image, review / f"00_输入图片{source_image.suffix.lower()}")

    compact: dict[str, Any] = {}
    old_items: dict[str, Any] = {}
    for item_id in (
        "redness", "spots", "brown", "texture", "pores",
        "uv_spots", "porphyrin", "wrinkle", "acne",
    ):
        public = _load(result_root / items[item_id]["量化JSON"])
        summary, features = extract_item(item_id, public)
        compact[item_id] = {"状态": "success", "核心总体指标": summary, "评分输入": features}
        old_items[item_id] = {"状态": "success"}

    for item_id, (folder, json_name, medical_csv_name, image_names) in OLD_PATHS.items():
        target = review / "七项检测" / folder
        public = _load(result_root / items[item_id]["量化JSON"])
        target.mkdir(parents=True, exist_ok=True)
        (target / json_name).write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(
            result_root / items[item_id]["医学V2CSV"],
            target / medical_csv_name,
        )
        _copy_indexed_images(result_root, items[item_id], target, image_names)

    purple = review / "七项检测" / "紫区"
    _write_combined_purple_csv(
        result_root,
        items,
        purple / "紫区医学量化指标_V2.csv",
    )
    _copy_indexed_images(
        result_root,
        items["porphyrin"],
        purple,
        ("04_紫质检测结果.jpg", "03_紫质荧光底图.png"),
    )

    for item_id, folder, json_name in (
        ("acne", "痤疮", "痤疮量化指标.json"),
        ("wrinkle", "皱纹", "皱纹量化指标.json"),
    ):
        target = review / folder
        target.mkdir()
        public = _load(result_root / items[item_id]["量化JSON"])
        medical = public.get("medical_metrics_v2", public)
        if not isinstance(medical, dict):
            raise WordDeliveryContractError(f"{folder}医学V2指标缺失")
        (target / json_name).write_text(
            json.dumps(medical, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _copy_indexed_images(
            result_root,
            items[item_id],
            target,
            (f"01_{folder}检测结果图.jpg",),
        )

    (review / "九项检测结果索引.json").write_text(
        json.dumps({"状态": "success", "成功项目数": 9, "九项结果": old_items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (review / "九项核心量化指标.json").write_text(
        json.dumps({"九项": compact}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    gloss = candidate / "02_油脂分泌倾向" / result_root.name
    gloss.mkdir(parents=True)
    gloss_metrics = _normalized_gloss_metrics(
        complete["detector_results"]["surface_gloss"]["metrics"]
    )
    (gloss / "表面油光量化指标.json").write_text(
        json.dumps(gloss_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(result_root / items["surface_gloss"]["主结果图"], gloss / "01_表面油光检测结果图.jpg")
    vascular = candidate / "05_血管样结构"
    vascular.mkdir(parents=True)
    vascular_metrics = _normalized_vascular_metrics(
        complete["detector_results"]["vascular"]["metrics"]
    )
    (vascular / "血管样结构量化指标.json").write_text(
        json.dumps(vascular_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(result_root / items["vascular"]["主结果图"], vascular / "05_血管样结构检测结果图.jpg")
    wrinkle_images: dict[str, str] = {}
    wrinkle_root = stage_root / "wrinkle_report_images"
    wrinkle_root.mkdir()
    for module_id, relative in zip(
        ("07", "08", "09"),
        [items["wrinkle"]["主结果图"], *(items["wrinkle"]["附加结果图"])],
    ):
        image = cv2.imread(str(result_root / relative))
        if image is None:
            raise WordDeliveryContractError(f"皱纹{module_id}报告图无法读取")
        target = wrinkle_root / f"{module_id}.jpg"
        cv2.imwrite(str(target), cv2.resize(image, (1024, 1024)))
        wrinkle_images[module_id] = str(target)
    return review, stage_root / "candidate", wrinkle_images


def generate_single_rgb_v012_reports(
    result_dir: str | Path,
    *,
    source_image: str | Path,
    subject_id: str | None = None,
    runtime_items: RuntimeItems,
    report_id: str | None = None,
) -> dict[str, str]:
    result_root = Path(result_dir).resolve()
    source = Path(source_image).resolve(strict=True)
    identity = single_rgb_report_identity(
        source_image=source,
        result_name=result_root.name,
        subject_id=subject_id,
        report_id=report_id,
    )
    index = _load(result_root / "十二项检测结果索引.json")
    validate_index_media(index, result_root)
    complete = _load(result_root / "十二项完整量化指标.json")
    provenance = complete.get("provenance")
    profile_name = (
        provenance.get("capture_profile")
        if isinstance(provenance, dict)
        else CaptureProfile.CONSUMER.value
    )
    capture_profile = CaptureProfile(str(profile_name))
    formal_images = _formal_images(result_root, index)
    official_profile, shadow_profile = _profile_paths()
    with tempfile.TemporaryDirectory(prefix=".single-rgb-v012-", dir=result_root) as temporary:
        workspace = Path(temporary)
        review, candidate, wrinkle_images = _stage_current_result(
            result_root,
            source,
            workspace / "stage",
        )
        controlled = materialize_controlled_evidence(
            source_image=source,
            runtime_items=runtime_items,
            candidate_sample_root=candidate / result_root.name,
            capture_profile=capture_profile,
        )
        generated = generate_front5_v012_reports(
            review,
            official_profile,
            shadow_profile,
            candidate,
            workspace / "generated",
            report_id=identity.report_id,
            subject_id=identity.subject_id,
            wrinkle_report_images=wrinkle_images,
            controlled_evidence=controlled,
            formal_images=formal_images,
            complete_document=complete,
        )
        publish_controlled_report_assets(
            Path(generated["structured_json"]),
            result_root,
        )
        output: dict[str, str] = {}
        for role, source_path in generated.items():
            target = result_root / Path(source_path).name
            if target.exists():
                raise WordDeliveryContractError(f"单RGB报告已存在: {target.name}")
            os.replace(source_path, target)
            output[role] = str(target)
        return output


__all__ = ["generate_single_rgb_v012_reports"]
