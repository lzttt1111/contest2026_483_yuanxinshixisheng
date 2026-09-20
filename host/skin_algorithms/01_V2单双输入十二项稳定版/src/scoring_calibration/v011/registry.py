from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal


FORMAL_DIMENSIONS = (
    "visible_pores",
    "combined_pigmentation",
    "diffuse_redness",
    "surface_smoothness_decline",
)

DISPLAY_ONLY_DIMENSIONS = (
    "oiliness_tendency",
    "vascular_like_structures",
    "follicular_inflammation_acne",
    "dry_fine_lines",
    "stable_linear_wrinkles",
    "structural_grooves",
    "contour_firmness_decline",
)


@dataclass(frozen=True)
class MetricSpec:
    id: str
    source: str
    weight: float
    transform: Literal["identity", "log1p"] = "identity"
    direction: Literal["high", "low"] = "high"
    unit: str = ""
    zero_is_no_burden: bool = False
    zero_inflated: bool = False
    minimum_instances: int = 0
    evidence_version: str = "nine_analysis_medical_metrics_v2"


@dataclass(frozen=True)
class GroupSpec:
    id: str
    name: str
    weight: float
    metrics: tuple[MetricSpec, ...]
    required: bool = True


@dataclass(frozen=True)
class DimensionSpec:
    id: str
    name: str
    doctor_source: str
    groups: tuple[GroupSpec, ...]
    formal: bool = True


def m(id: str, source: str, weight: float, **kwargs: object) -> MetricSpec:
    return MetricSpec(id=id, source=source, weight=weight, **kwargs)


REGISTRY: tuple[DimensionSpec, ...] = (
    DimensionSpec(
        "visible_pores", "可见毛孔负担", "面部毛孔检测需求说明_V2",
        (
            GroupSpec("density", "毛孔密度", .20, (m("pore_density", "pores.毛孔密度负担.density", 1, transform="log1p", unit="个/10万有效皮肤像素", zero_is_no_burden=True, zero_inflated=True),)),
            GroupSpec("coverage", "毛孔面积覆盖", .25, (m("pore_area_ratio", "pores.毛孔面积负担.area_ratio", 1, unit="比例", zero_is_no_burden=True, zero_inflated=True),)),
            GroupSpec("size", "毛孔大小", .25, (m("pore_p50_area", "pores.毛孔尺度负担.p50_area", 1, transform="log1p", unit="像素", zero_is_no_burden=True, minimum_instances=5),)),
            GroupSpec("large_pores", "大毛孔负担", .20, (m("pore_p90_area", "pores.毛孔尺度负担.p90_area", 1, transform="log1p", unit="像素", zero_is_no_burden=True, minimum_instances=10),)),
            GroupSpec("shape", "毛孔形态不规则度", .10, (
                m("low_circularity_ratio", "pores.毛孔形态负担.low_circularity_ratio", .5, unit="比例", minimum_instances=10),
                m("elongated_ratio", "pores.毛孔形态负担.elongated_ratio", .5, unit="比例", minimum_instances=10),
            )),
        ),
    ),
    DimensionSpec(
        "combined_pigmentation", "综合色素问题", "面部色素检测需求说明_V2",
        (
            GroupSpec("visible_spots", "可见色斑负担", .40, (
                m("visible_spot_density", "spots.可见色斑范围.density", .25, transform="log1p", unit="个/10万有效皮肤像素", zero_is_no_burden=True, zero_inflated=True),
                m("visible_spot_area_ratio", "spots.可见色斑范围.area_ratio", .25, unit="比例", zero_is_no_burden=True, zero_inflated=True),
                m("visible_spot_p90_delta_e", "spots.可见色斑对比.p90_delta_e", .25, unit="DeltaE", zero_is_no_burden=True, minimum_instances=10),
                m("visible_spot_patch_ratio", "spots.可见色斑形态.patch_area_ratio", .25, unit="比例", zero_is_no_burden=True, minimum_instances=10),
            )),
            GroupSpec("uv_spots", "UV下更明显的色斑负担", .25, (
                m(
                    "uv_spot_density", "uv_spots.UV样色素范围.density", 1,
                    transform="log1p", unit="个/10万有效皮肤像素",
                    zero_is_no_burden=True, zero_inflated=True,
                ),
            )),
            GroupSpec("brown", "Brown综合色素负担", .35, (
                m("brown_area_ratio", "brown.Brown覆盖负担.area_ratio", .25, unit="比例", zero_is_no_burden=True, zero_inflated=True),
                m("brown_p90_intensity", "brown.Brown强度.p90_intensity", .25, unit="0-1", zero_is_no_burden=True, minimum_instances=10),
                m("brown_high_intensity_ratio", "brown.高强度Brown区域.high_intensity_area_ratio", .25, unit="比例", zero_is_no_burden=True, minimum_instances=10),
                m("brown_patch_ratio", "brown.Brown点片形态.patch_area_ratio", .25, unit="比例", zero_is_no_burden=True, minimum_instances=10),
            )),
        ),
    ),
    DimensionSpec(
        "diffuse_redness", "弥漫性泛红", "面部弥漫性泛红检测需求_V2",
        (
            GroupSpec("coverage", "泛红覆盖负担", .35, (
                m("diffuse_red_area_ratio", "redness.泛红覆盖负担.diffuse_red_area_ratio", .5, unit="比例", zero_is_no_burden=True, zero_inflated=True),
                m("high_red_area_ratio", "redness.泛红覆盖负担.high_red_area_ratio", .5, unit="比例", zero_is_no_burden=True, zero_inflated=True),
            )),
            GroupSpec("intensity", "泛红强度", .30, (
                m("mean_redness", "redness.泛红强度.mean_redness", .5, unit="0-1", zero_is_no_burden=True),
                m("p90_redness", "redness.泛红强度.p90_redness", .5, unit="0-1", zero_is_no_burden=True),
            )),
            GroupSpec("continuity", "背景红色连续性", .20, (
                m("max_continuous_region_ratio", "redness.背景红色连续性.max_continuous_region_ratio", .5, unit="比例", zero_is_no_burden=True),
                m("redness_continuity", "redness.背景红色连续性.redness_continuity", .5, unit="0-1", zero_is_no_burden=True),
            )),
            GroupSpec("boundary_uniformity", "边界与均匀性", .15, (
                m("boundary_gradient", "redness.边界与均匀性.boundary_gradient", .5, direction="low", unit="工程值"),
                m("uniformity", "redness.边界与均匀性.uniformity", .5, unit="0-1"),
            )),
        ),
    ),
    DimensionSpec(
        "surface_smoothness_decline", "面部表面平整度下降负担", "面部皮肤光滑度检测需求说明_V2",
        (
            GroupSpec("raised", "隆起性不平整", .4737, (
                m("raised_density", "texture.凸起样二维纹理.density", 1/3, transform="log1p", unit="个/10万有效皮肤像素", zero_is_no_burden=True, zero_inflated=True),
                m("raised_area_ratio", "texture.凸起样二维纹理.area_ratio", 1/3, unit="比例", zero_is_no_burden=True, zero_inflated=True),
                m("raised_p90_response", "texture.凸起样二维纹理.p90_response", 1/3, unit="0-1", zero_is_no_burden=True, minimum_instances=10),
            )),
            GroupSpec("depressed", "凹陷性不平整", .5263, (
                m("depressed_area_ratio", "texture.凹陷样二维纹理.area_ratio", 1, unit="比例", zero_is_no_burden=True, zero_inflated=True),
            )),
        ),
    ),
)


def validate_registry(registry: tuple[DimensionSpec, ...] = REGISTRY) -> None:
    sources: dict[str, str] = {}
    ids: set[str] = set()
    for dimension in registry:
        if dimension.id in ids:
            raise ValueError(f"重复维度ID: {dimension.id}")
        ids.add(dimension.id)
        if not dimension.formal:
            continue
        if abs(sum(group.weight for group in dimension.groups) - 1) > 1e-6:
            raise ValueError(f"维度组权重不等于1: {dimension.id}")
        for group in dimension.groups:
            if abs(sum(metric.weight for metric in group.metrics) - 1) > 1e-6:
                raise ValueError(f"指标权重不等于1: {dimension.id}.{group.id}")
            for metric in group.metrics:
                if metric.source in sources:
                    raise ValueError(
                        f"正式评分证据跨维/重复使用: {metric.source} "
                        f"({sources[metric.source]} 与 {dimension.id}.{metric.id})"
                    )
                sources[metric.source] = f"{dimension.id}.{metric.id}"


def registry_document() -> dict[str, object]:
    validate_registry()
    body = {
        "scoring_profile_version": "aisia_scoring_v0.1.1_integrity_20260806",
        "doctor_spec_version": "doctor_requirements_v2_20260728",
        "status": "candidate/uncalibrated",
        "score_direction": "higher_is_more_visible_burden",
        "formal_dimensions": list(FORMAL_DIMENSIONS),
        "display_only_dimensions": list(DISPLAY_ONLY_DIMENSIONS),
        "overall_score": None,
        "dimensions": [asdict(item) for item in REGISTRY],
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    body["registry_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body
