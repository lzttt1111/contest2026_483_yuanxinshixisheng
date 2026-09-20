from __future__ import annotations

"""Build the three controlled-report derived images from one live run."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import TypeAlias

import cv2
import numpy as np

from src.capture_profile import CaptureProfile
from src.engines.pigmentation_fusion_engine import (
    fuse_pigmentation,
    save_pigmentation_fusion,
)
from src.engines.visia_regions import (
    VISIA_BOUNDARY_COLOR,
    VisiaRegionSet,
    build_visia_regions,
    draw_region_boundaries,
)
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.preprocess.profile_mask_policy import apply_preprocess_mask_bundle
from src.utils.io_utils import cv_imread, cv_imwrite
from .controlled_report_explanations import attach_missing_metric_explanations


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
RuntimeItems: TypeAlias = Mapping[str, Mapping[str, JsonValue]]


@dataclass(frozen=True, slots=True)
class ControlledEvidencePaths:
    red_brown_mixed: Path
    priority_pigment: Path
    surface_irregularity: Path
    pigmentation_metrics: Path


@dataclass(frozen=True, slots=True)
class ControlledEvidenceError(RuntimeError):
    label: str
    path: Path

    def __str__(self) -> str:
        return f"正式报告内部证据缺失: {self.label}: {self.path}"


def _module(payload: dict[str, JsonValue], module_id: str) -> dict[str, JsonValue]:
    modules = payload.get("检测模块")
    if not isinstance(modules, list):
        raise ControlledEvidenceError("report-modules", Path("<missing>"))
    for value in modules:
        if isinstance(value, dict) and str(value.get("模块编号")) == module_id:
            return value
    raise ControlledEvidenceError(f"report-module-{module_id}", Path("<missing>"))


def _set_group_image(group: dict[str, JsonValue], path: Path, caption: str) -> None:
    group["images"] = [{"path": str(path), "caption": caption}]


def _append_result_image(module: dict[str, JsonValue], path: Path) -> None:
    images = module.get("结果图")
    if not isinstance(images, list):
        images = []
        module["结果图"] = images
    value = str(path)
    if value not in images:
        images.append(value)


def bind_controlled_evidence(
    payload: dict[str, JsonValue],
    evidence: ControlledEvidencePaths,
) -> None:
    """Bind the three approved 8/18 media roles to an already-formal payload."""
    pigment = _module(payload, "03")
    groups = pigment.get("医生结果分组")
    if not isinstance(groups, list) or len(groups) != 5:
        raise ControlledEvidenceError("pigmentation-five-groups", Path("<missing>"))
    document = json.loads(
        _required(
            evidence.pigmentation_metrics,
            "pigmentation-metrics",
        ).read_text(encoding="utf-8")
    )
    metrics = document.get("metrics") if isinstance(document, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    mixed = metrics.get("red_brown_mixed")
    mixed = mixed if isinstance(mixed, dict) else {}
    visible = metrics.get("visible_pigment")
    visible = visible if isinstance(visible, dict) else {}
    pure_red = metrics.get("pure_red_suspected")
    pure_red = pure_red if isinstance(pure_red, dict) else {}
    visible_area = float(visible.get("area_px") or 0)
    mixed_area = float(mixed.get("area_px") or 0)
    region_rows = document.get("region_distribution")
    region_rows = region_rows if isinstance(region_rows, list) else []
    current_regions = [row for row in region_rows if isinstance(row, dict) and row.get("available")]
    priority_count = sum(int(row.get("component_count") or 0) for row in current_regions)
    priority_area = sum(float(row.get("feature_area_px") or 0) for row in current_regions)
    priority_valid = sum(float(row.get("valid_area_px") or 0) for row in current_regions)
    for value in groups:
        if not isinstance(value, dict):
            continue
        title = str(value.get("title", ""))
        if "红褐混合" in title:
            _set_group_image(value, evidence.red_brown_mixed, "红褐混合印记")
            value["metrics"] = [
                {"name": "红褐混合目标数量", "value": mixed.get("component_count"), "unit": "个", "display_type": "COUNT"},
                {"name": "红褐混合总面积占比", "value": mixed.get("area_ratio"), "unit": "%", "display_type": "PERCENTAGE"},
                {"name": "红褐混合面积占可见色斑面积比例", "value": mixed_area / visible_area if visible_area else None, "unit": "%", "display_type": "PERCENTAGE"},
                {"name": "疑似纯红区域面积占比", "value": pure_red.get("area_ratio"), "unit": "%", "display_type": "PERCENTAGE"},
            ]
            value.pop("unavailable", None)
        elif "重点色斑" in title:
            _set_group_image(value, evidence.priority_pigment, "重点色斑区域")
            value["metrics"] = [
                {"name": "综合色素联合区域数量", "value": priority_count, "unit": "个", "display_type": "COUNT"},
                {"name": "综合色素联合区域面积占比", "value": priority_area / priority_valid if priority_valid else None, "unit": "%", "display_type": "PERCENTAGE"},
            ]
            value.pop("unavailable", None)
    _append_result_image(pigment, evidence.red_brown_mixed)
    _append_result_image(pigment, evidence.priority_pigment)
    texture = _module(payload, "10")
    texture_groups = texture.get("医生结果分组")
    if not isinstance(texture_groups, list):
        raise ControlledEvidenceError("texture-groups", Path("<missing>"))
    for value in texture_groups:
        if isinstance(value, dict) and "不规则" in str(value.get("title", "")):
            _set_group_image(
                value,
                evidence.surface_irregularity,
                "表面不规则分布",
            )
    _append_result_image(texture, evidence.surface_irregularity)
    attach_missing_metric_explanations(payload)


def publish_controlled_report_assets(
    structured_path: Path,
    result_root: Path,
) -> None:
    """Publish every formal image once and make the structured JSON portable."""
    payload = json.loads(structured_path.read_text(encoding="utf-8-sig"))
    modules = payload.get("检测模块")
    if not isinstance(modules, list):
        raise ControlledEvidenceError("structured-modules", structured_path)
    target_root = result_root / "正式报告衍生图"
    target_root.mkdir(parents=True, exist_ok=False)
    copied: dict[tuple[str, str], Path] = {}

    def publish(module_id: str, value: str) -> str:
        source = _required(Path(value), f"formal-image-{module_id}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        identity = (module_id, digest)
        target = copied.get(identity)
        if target is None:
            target = target_root / module_id / f"{digest[:12]}_{source.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[identity] = target
        return target.relative_to(result_root).as_posix()

    cover = payload.get("标准采集图像")
    if isinstance(cover, dict) and isinstance(cover.get("path"), str):
        cover["path"] = publish("00", cover["path"])
    for value in modules:
        if not isinstance(value, dict):
            continue
        module_id = str(value.get("模块编号", "unknown"))
        result_images = value.get("结果图")
        if isinstance(result_images, list):
            value["结果图"] = [
                publish(module_id, path)
                for path in result_images
                if isinstance(path, str)
            ]
        groups = value.get("医生结果分组")
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            images = group.get("images")
            if not isinstance(images, list):
                continue
            for image in images:
                if isinstance(image, dict) and isinstance(image.get("path"), str):
                    image["path"] = publish(module_id, image["path"])
    structured_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _main(items: RuntimeItems, item_id: str) -> Path:
    value = items[item_id].get("主结果图")
    if not isinstance(value, str):
        raise ControlledEvidenceError(item_id, Path("<missing-main-result>"))
    return Path(value).resolve(strict=True)


def _required(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ControlledEvidenceError(label, resolved)
    return resolved


def _mask(path: Path, label: str, shape: tuple[int, int]) -> np.ndarray:
    source = _required(path, label)
    value = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if value is None or value.shape != shape:
        raise ControlledEvidenceError(label, source)
    return value


def _surface_irregularity(
    display_image: np.ndarray,
    marker_path: Path,
    output_path: Path,
    *,
    regions: VisiaRegionSet,
) -> Path:
    marker = cv2.imdecode(
        np.fromfile(_required(marker_path, "texture-mask"), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if marker is None or marker.shape != display_image.shape:
        raise ControlledEvidenceError("texture-mask", marker_path)
    response = (np.max(marker, axis=2) > 0) & (regions.analysis_mask > 0)
    output = display_image.copy()
    output[response] = VISIA_BOUNDARY_COLOR
    output = _with_visia_boundary(output, regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv_imwrite(str(output_path), output)
    return _required(output_path, "surface-irregularity")


def _red_brown_overlay(
    display_image: np.ndarray,
    mask_path: Path,
    output_path: Path,
    *,
    regions: VisiaRegionSet,
) -> Path:
    mask = _mask(mask_path, "red-brown-mask", display_image.shape[:2]) > 0
    output = display_image.copy()
    color = np.full_like(output, (60, 30, 230))
    blended = cv2.addWeighted(output, 0.42, color, 0.58, 0)
    output[mask] = blended[mask]
    output = _with_visia_boundary(output, regions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv_imwrite(str(output_path), output)
    return _required(output_path, "red-brown-mixed")


def _with_visia_boundary(
    image: np.ndarray,
    regions: VisiaRegionSet,
) -> np.ndarray:
    return draw_region_boundaries(
        image,
        regions.display_regions,
        color=VISIA_BOUNDARY_COLOR,
        thickness=2,
        partial_face=regions.partial_face,
        closed_contour=regions.display_contour,
        separator_contour=regions.display_separator,
    )


def materialize_controlled_evidence(
    *,
    source_image: Path,
    runtime_items: RuntimeItems,
    candidate_sample_root: Path,
    capture_profile: CaptureProfile,
) -> ControlledEvidencePaths:
    """Create report-only evidence without changing detector metrics or masks."""
    decoded = cv_imread(str(source_image))
    if decoded is None:
        raise ControlledEvidenceError("source-image", source_image)
    preprocessor = ImagePreprocessor(capture_profile=capture_profile)
    try:
        preprocessed = preprocessor.preprocess_image(decoded)
        apply_preprocess_mask_bundle(capture_profile, preprocessed)
    finally:
        preprocessor.close()
    shape = preprocessed.analysis_image.shape[:2]
    regions = build_visia_regions(
        preprocessed.analysis_image,
        preprocessed.skin_mask,
        preprocessed.landmarks,
        preprocessed.quality_flags,
        include_chin=True,
        mode="full",
    )
    red_root = _main(runtime_items, "redness").parent
    spots_root = _main(runtime_items, "spots").parent
    brown_root = _main(runtime_items, "brown").parent
    uv_root = _main(runtime_items, "uv_spots").parent
    texture_root = _main(runtime_items, "texture").parent
    fusion = fuse_pigmentation(
        image=preprocessed.analysis_image,
        analysis_mask=preprocessed.skin_mask,
        landmarks=preprocessed.landmarks,
        visible_mask=_mask(spots_root / "05_Spots最终Mask.png", "visible-mask", shape),
        uv_mask=_mask(uv_root / "_debug/03_UV色斑Mask.png", "uv-mask", shape),
        brown_mask=_mask(brown_root / "03_棕色斑实例Mask.png", "brown-mask", shape),
        red_mask=_mask(red_root / "07_VISIA红色区实例Mask.png", "red-mask", shape),
    )
    fusion_root = candidate_sample_root / "03_综合色素" / "fusion"
    paths = save_pigmentation_fusion(fusion, fusion_root)
    derived_root = candidate_sample_root / "正式报告衍生图"
    red_brown = _red_brown_overlay(
        preprocessed.display_image,
        Path(paths["red_brown_mixed"]),
        derived_root / "03_红褐混合印记结果图.jpg",
        regions=regions,
    )
    priority = derived_root / "04_重点色斑区域结果图.jpg"
    priority.parent.mkdir(parents=True, exist_ok=True)
    cv_imwrite(
        str(priority),
        _with_visia_boundary(fusion["overlay"], regions),
    )
    surface = _surface_irregularity(
        preprocessed.display_image,
        texture_root / "03_纹理特征Mask.png",
        candidate_sample_root / "10_表面不规则" / "10_表面不规则分布.jpg",
        regions=regions,
    )
    return ControlledEvidencePaths(
        red_brown_mixed=red_brown,
        priority_pigment=_required(priority, "priority-pigment"),
        surface_irregularity=surface,
        pigmentation_metrics=_required(Path(paths["metrics"]), "pigmentation-metrics"),
    )


def materialize_delivery_evidence(
    *,
    source_image: Path,
    runtime_items: RuntimeItems,
    result_root: Path,
    capture_profile: CaptureProfile,
) -> Path:
    """Persist the three review images even when DOCX generation is disabled."""

    target = result_root / "正式报告衍生图"
    if target.exists():
        raise ControlledEvidenceError("delivery-evidence-exists", target)
    with tempfile.TemporaryDirectory(
        prefix=".delivery-evidence-",
        dir=result_root,
    ) as temporary:
        evidence = materialize_controlled_evidence(
            source_image=source_image,
            runtime_items=runtime_items,
            candidate_sample_root=Path(temporary),
            capture_profile=capture_profile,
        )
        target.mkdir()
        for source, name in (
            (evidence.red_brown_mixed, "03_红褐混合印记结果图.jpg"),
            (evidence.priority_pigment, "04_重点色斑区域结果图.jpg"),
            (evidence.surface_irregularity, "10_表面不规则分布.jpg"),
        ):
            shutil.copy2(source, target / name)
    return target


__all__ = [
    "ControlledEvidenceError",
    "ControlledEvidencePaths",
    "RuntimeItems",
    "bind_controlled_evidence",
    "materialize_controlled_evidence",
    "materialize_delivery_evidence",
    "publish_controlled_report_assets",
]
