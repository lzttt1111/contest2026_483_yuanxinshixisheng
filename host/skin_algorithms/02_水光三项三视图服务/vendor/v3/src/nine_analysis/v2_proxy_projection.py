from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias

from typing_extensions import assert_never


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FormulaOperator: TypeAlias = Literal[
    "direct",
    "scale",
    "weighted_sum",
    "ratio",
    "one_minus_ratio",
    "product",
    "product_ratio",
    "spread",
    "maximum",
    "minimum",
    "fraction_above",
]
FORMULA_OPERATORS: dict[str, FormulaOperator] = {
    value: value
    for value in (
        "direct", "scale", "weighted_sum", "ratio", "one_minus_ratio",
        "product", "product_ratio", "spread", "maximum", "minimum",
        "fraction_above",
    )
}


class FormulaRegistryError(RuntimeError):
    def __init__(self, operator: JsonValue) -> None:
        self.operator = operator
        super().__init__(f"unsupported explicit formula operator: {operator}")


FORMULA_VERSION = "doctor_v2_explicit_proxy_v1"
FORMULA_PATH = (
    Path(__file__).resolve().parents[2]
    / "calibration"
    / "v2_explicit_formula_registry_v1.json"
)
MODULE_DETECTORS = {
    "pores": ("pores",),
    "oil_tendency": ("surface_gloss", "porphyrin"),
    "pigmentation": ("spots", "uv_spots", "brown"),
    "diffuse_redness": ("redness",),
    "vascular": ("vascular",),
    "acne_activity": ("acne", "redness", "texture", "porphyrin", "contour_firmness"),
    "dry_fine_lines": ("wrinkle", "contour_firmness"),
    "stable_wrinkles": ("wrinkle", "contour_firmness"),
    "structural_grooves": ("wrinkle", "contour_firmness", "texture"),
    "smoothness": ("texture", "contour_firmness", "spots"),
    "contour_firmness": ("contour_firmness", "wrinkle", "texture"),
}
WRINKLE_REPORT_GROUPS = {
    "dry_fine_lines": "07",
    "stable_wrinkles": "08",
    "structural_grooves": "09",
}


def _flatten(value: JsonValue, prefix: str = "") -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    match value:
        case dict():
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(child, bool):
                    continue
                if isinstance(child, (int, float)) and math.isfinite(float(child)):
                    rows.append((path, str(key), float(child)))
                else:
                    rows.extend(_flatten(child, path))
        case list():
            for index, child in enumerate(value):
                rows.extend(_flatten(child, f"{prefix}[{index}]"))
        case str() | int() | float() | bool() | None:
            pass
        case unreachable:
            assert_never(unreachable)
    return rows


def _pool(
    detector_results: Mapping[str, Mapping[str, JsonValue]],
    detectors: tuple[str, ...],
) -> dict[str, float]:
    rows: dict[str, float] = {}
    for detector in detectors:
        result = detector_results.get(detector, {})
        rows.update({
            f"{detector}.{path}": value
            for path, _, value in _flatten(result.get("metrics", {}))
        })
    return rows


def _load_formulas() -> dict[str, JsonValue]:
    return json.loads(FORMULA_PATH.read_text(encoding="utf-8"))


def _wrinkle_scope_unavailable_reason(
    module_id: str,
    detector_results: Mapping[str, Mapping[str, JsonValue]],
) -> str | None:
    report_group = WRINKLE_REPORT_GROUPS.get(module_id)
    if report_group is None:
        return None
    wrinkle_metrics = detector_results.get("wrinkle", {}).get("metrics", {})
    if not isinstance(wrinkle_metrics, dict):
        return None
    overlays = wrinkle_metrics.get("report_group_overlays")
    if not isinstance(overlays, dict):
        return None
    evidence = overlays.get(report_group)
    selected = evidence.get("selected_region_keys") if isinstance(evidence, dict) else None
    if isinstance(selected, list) and selected:
        return None
    return f"wrinkle_report_group_{report_group}_unavailable"


def _mark_groups_unavailable(groups: Mapping[str, dict]) -> None:
    for group in groups.values():
        for metric in group["metrics"].values():
            metric["value"] = None
            metric["availability"] = "unavailable"
            metric["source_metrics"] = []


def _formula_operator(raw: JsonValue) -> FormulaOperator:
    try:
        return FORMULA_OPERATORS[str(raw)]
    except KeyError as error:
        raise FormulaRegistryError(raw) from error


def _evaluate_formula(values: list[float], formula: Mapping[str, JsonValue]) -> float:
    operator = _formula_operator(formula["operator"])
    match operator:
        case "direct":
            result = values[0]
        case "scale":
            result = values[0] * float(formula["scale"])
        case "weighted_sum":
            coefficients = [float(value) for value in formula["coefficients"]]
            result = sum(value * coefficient for value, coefficient in zip(values, coefficients, strict=True))
            result += float(formula["offset"])
        case "ratio" | "one_minus_ratio":
            epsilon = float(formula["epsilon"])
            ratio = values[0] / max(abs(values[1]), epsilon) * float(formula["scale"])
            result = 1.0 - ratio if operator == "one_minus_ratio" else ratio
        case "product":
            adjusted = list(values)
            presence_index = formula.get("presence_index")
            if isinstance(presence_index, int):
                adjusted[presence_index] = float(adjusted[presence_index] > 0.0)
            complement_index = formula.get("complement_index")
            if isinstance(complement_index, int):
                adjusted[complement_index] = 1.0 - adjusted[complement_index]
            result = math.prod(adjusted) * float(formula["scale"])
        case "product_ratio":
            numerator_indices = [int(value) for value in formula["numerator_indices"]]
            denominator_index = int(formula["denominator_index"])
            numerator = math.prod(values[index] for index in numerator_indices)
            result = numerator / max(abs(values[denominator_index]), float(formula["epsilon"]))
            result *= float(formula.get("scale", 1.0))
        case "spread":
            result = (max(values) - min(values)) * float(formula["scale"])
        case "maximum":
            result = max(values) * float(formula["scale"])
        case "minimum":
            result = min(values) * float(formula["scale"])
        case "fraction_above":
            threshold = float(formula["threshold"])
            result = sum(value >= threshold for value in values) / len(values)
        case unreachable:
            assert_never(unreachable)
    clip = formula.get("clip")
    if isinstance(clip, list):
        result = min(float(clip[1]), max(float(clip[0]), result))
    return float(result)


def _select_explicit(
    rows: Mapping[str, float],
    definition: Mapping[str, JsonValue],
) -> tuple[float | None, list[str], str]:
    source_ids = [str(source) for source in definition["source_metric_ids"]]
    if any(source not in rows for source in source_ids):
        return None, [], "unavailable"
    values = [rows[source] for source in source_ids]
    formula = definition["formula"]
    value = _evaluate_formula(values, formula)
    availability = (
        "explicit_direct"
        if formula["operator"] == "direct"
        else "explicit_formula"
    )
    return value, source_ids, availability


def _project_groups(
    module_id: str,
    definitions: list[Mapping[str, JsonValue]],
    rows: Mapping[str, float],
    formula_registry: Mapping[str, JsonValue],
    *,
    region_scope: str,
) -> tuple[dict[str, dict], dict[str, dict[str, float | None]]]:
    groups: dict[str, dict] = {}
    scoring: dict[str, dict[str, float | None]] = {}
    formulas = formula_registry["metric_formulas"]
    for definition in definitions:
        group_id = str(definition["id"])
        metrics: dict[str, dict] = {}
        scoring_metrics: dict[str, float | None] = {}
        for metric_definition in definition["core_metrics"]:
            metric_id = str(metric_definition["id"])
            formula_definition = formulas[f"{module_id}.{group_id}.{metric_id}"]
            value, sources, availability = _select_explicit(
                rows, formula_definition
            )
            metrics[metric_id] = {
                "value": value,
                "unit": formula_definition["unit"],
                "direction": formula_definition["direction"],
                "availability": availability,
                "source_detector": list(formula_definition["source_detector"]),
                "source_metric_ids": list(formula_definition["source_metric_ids"]),
                "source_metrics": sources,
                "formula": formula_definition["formula"],
                "formula_version": FORMULA_VERSION,
                "metric_role": formula_definition["metric_role"],
                "region_scope": region_scope,
                "formula_region_scope": formula_definition["region_scope"],
                "group_weight": formula_definition["group_weight"],
                "proxy_boundary": formula_definition["proxy_boundary"],
                "algorithm_version": formula_definition["algorithm_version"],
                "roi_version": formula_definition["roi_version"],
                "medical_rationale": formula_definition["medical_rationale"],
                "metric_weight": formula_definition["metric_weight"],
            }
            scoring_metrics[metric_id] = value
        groups[group_id] = {"weight": definition["weight"], "metrics": metrics}
        scoring[group_id] = scoring_metrics
    return groups, scoring


def build_v2_proxy_modules(
    detector_results: Mapping[str, Mapping[str, JsonValue]],
    registry: Mapping[str, JsonValue],
) -> tuple[dict[str, dict], dict[str, dict[str, dict[str, float | None]]]]:
    formula_registry = _load_formulas()
    modules: dict[str, dict] = {}
    scoring: dict[str, dict[str, dict[str, float | None]]] = {}
    for module_id, module_value in registry["modules"].items():
        module = dict(module_value)
        rows = _pool(detector_results, MODULE_DETECTORS[module_id])
        groups, scoring_groups = _project_groups(
            module_id, module["groups"], rows, formula_registry, region_scope="full_face"
        )
        scope_unavailable_reason = _wrinkle_scope_unavailable_reason(
            module_id,
            detector_results,
        )
        if scope_unavailable_reason is not None:
            _mark_groups_unavailable(groups)
            for scoring_metrics in scoring_groups.values():
                for metric_id in scoring_metrics:
                    scoring_metrics[metric_id] = None
        missing = [
            f"{group_id}.{metric_id}"
            for group_id, group in groups.items()
            for metric_id, metric in group["metrics"].items()
            if metric["availability"] == "unavailable"
        ]
        region_groups: dict[str, dict] = {}
        for region in module["regions"]:
            region_id = region["id"]
            local_groups, _ = _project_groups(
                module_id,
                module["groups"],
                rows,
                formula_registry,
                region_scope="full_face_proxy_reused_for_region",
            )
            if scope_unavailable_reason is not None:
                _mark_groups_unavailable(local_groups)
            region_groups[region_id] = {"weight": region["weight"], "groups": local_groups}
        modules[module_id] = {
            "name": module["name"],
            "measurement_mode": module["measurement_mode"],
            "regions": module["regions"],
            "groups": groups,
            "region_groups": region_groups,
            "completeness": {
                "complete": not missing,
                "missing_required": missing,
                "scope_unavailable_reason": scope_unavailable_reason,
            },
        }
        scoring[module_id] = scoring_groups
    return modules, scoring


__all__ = ["build_v2_proxy_modules"]
