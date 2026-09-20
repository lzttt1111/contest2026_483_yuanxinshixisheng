from __future__ import annotations

import copy
from typing import Any


INSTITUTION_METRIC_ALLOWLIST = {
    "oil_tendency": {
        "surface_gloss_coverage": frozenset({"gloss_area_ratio", "high_gloss_area_ratio"}),
        "surface_gloss_intensity": frozenset({"p50_gloss_intensity", "p90_gloss_intensity"}),
        "porphyrin_density_area": frozenset({"porphyrin_density_per_100k_px", "porphyrin_area_ratio"}),
        "porphyrin_intensity": frozenset({"p50_porphyrin_intensity", "p90_porphyrin_intensity"}),
    },
    "vascular": {
        "count_length": frozenset({"vascular_density_per_100k_px", "vascular_length_density_per_10k_px"}),
        "red_intensity": frozenset({"p50_vascular_redness", "p90_vascular_redness"}),
        "branch_network": frozenset({"branch_density_per_10k_px"}),
    },
    "acne_activity": {
        "follicular_erythema": frozenset({
            "follicular_erythema_density", "follicular_erythema_area_ratio",
            "p90_follicular_redness", "erythema_halo_completeness",
        }),
        "follicular_papule": frozenset({"papule_density", "papule_area_ratio", "p90_papule_redness"}),
        "follicular_pustule": frozenset({
            "pustule_density", "pustule_area_ratio", "p90_pustule_redness",
            "p90_white_yellow_center_area_px",
        }),
    },
    "dry_fine_lines": {
        "coverage": frozenset({"fine_line_network_coverage_ratio", "high_density_fine_line_area_ratio"}),
        "density": frozenset({"fine_line_density_per_100k_px", "fine_line_length_density"}),
        "direction_network": frozenset({"direction_dispersion", "multidirectional_interweave", "network_texture_ratio"}),
        "visual_surface": frozenset({
            "p50_fine_line_contrast", "low_mid_contrast_ratio",
            "short_discontinuous_ratio", "large_groove_exclusion",
        }),
    },
    "stable_wrinkles": {
        "density": frozenset({"wrinkle_density_per_100k_px"}),
        "length": frozenset({"wrinkle_length_density", "p50_wrinkle_length_px", "p90_wrinkle_length_px"}),
        "linearity_continuity": frozenset({"wrinkle_linearity", "wrinkle_continuity"}),
    },
    "structural_grooves": {
        "curvature_valley": frozenset({"clear_valley_segment_ratio"}),
    },
    "contour_firmness": {
        "midface_support_decline": frozenset({"midface_surface_continuity"}),
        "lower_face_sagging": frozenset({"jawline_continuity"}),
    },
}


def _dictionary(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        current = _dictionary(current).get(key)
    return current


def _set(features: dict[str, Any], module: str, group: str, metric: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    features.setdefault(module, {}).setdefault(group, {})[metric] = float(value)


def _project_oil(features: dict[str, Any], detectors: dict[str, Any]) -> None:
    gloss = _dictionary(_dictionary(detectors.get("surface_gloss")).get("metrics"))
    scope = _dictionary(_nested(gloss, "overall_metrics", "core_metrics", "scope_and_morphology"))
    intensity = _dictionary(_nested(gloss, "overall_metrics", "core_metrics", "signal_intensity"))
    _set(features, "oil_tendency", "surface_gloss_coverage", "gloss_area_ratio", scope.get("gloss_area_ratio"))
    _set(features, "oil_tendency", "surface_gloss_coverage", "high_gloss_area_ratio", scope.get("high_gloss_area_ratio"))
    _set(features, "oil_tendency", "surface_gloss_intensity", "p50_gloss_intensity", intensity.get("p50_gloss_intensity"))
    _set(features, "oil_tendency", "surface_gloss_intensity", "p90_gloss_intensity", intensity.get("p90_gloss_intensity"))
    porphyrin = _dictionary(_dictionary(detectors.get("porphyrin")).get("metrics"))
    counts = _dictionary(_nested(porphyrin, "overall_metrics", "core_metrics", "count_and_density"))
    porphyrin_scope = _dictionary(_nested(porphyrin, "overall_metrics", "core_metrics", "scope_and_morphology"))
    porphyrin_intensity = _dictionary(_nested(porphyrin, "overall_metrics", "core_metrics", "signal_intensity"))
    valid = _nested(porphyrin, "overall_metrics", "auxiliary_metrics", "analysis_scope", "valid_skin_area_px")
    count = counts.get("feature_count")
    density = float(count) / float(valid) * 100000.0 if isinstance(count, (int, float)) and isinstance(valid, (int, float)) and valid else None
    _set(features, "oil_tendency", "porphyrin_density_area", "porphyrin_density_per_100k_px", density)
    _set(features, "oil_tendency", "porphyrin_density_area", "porphyrin_area_ratio", porphyrin_scope.get("feature_area_ratio"))
    _set(features, "oil_tendency", "porphyrin_intensity", "p50_porphyrin_intensity", porphyrin_intensity.get("p50_intensity"))
    _set(features, "oil_tendency", "porphyrin_intensity", "p90_porphyrin_intensity", porphyrin_intensity.get("p90_intensity"))


def _project_vascular(features: dict[str, Any], detectors: dict[str, Any]) -> None:
    vascular = _dictionary(_dictionary(detectors.get("vascular")).get("metrics"))
    counts = _dictionary(_nested(vascular, "overall_metrics", "core_metrics", "count_and_density"))
    scope = _dictionary(_nested(vascular, "overall_metrics", "core_metrics", "scope_and_morphology"))
    intensity = _dictionary(_nested(vascular, "overall_metrics", "core_metrics", "signal_intensity"))
    valid = _nested(vascular, "overall_metrics", "auxiliary_metrics", "analysis_scope", "valid_skin_area_px")
    if isinstance(valid, (int, float)) and valid:
        _set(features, "vascular", "count_length", "vascular_density_per_100k_px", float(counts.get("vascular_count") or 0) / float(valid) * 100000.0)
        _set(features, "vascular", "count_length", "vascular_length_density_per_10k_px", float(scope.get("vascular_total_length_px") or 0) / float(valid) * 10000.0)
        _set(features, "vascular", "branch_network", "branch_density_per_10k_px", float(counts.get("branch_point_count") or 0) / float(valid) * 10000.0)
    _set(features, "vascular", "red_intensity", "p50_vascular_redness", intensity.get("p50_redness_response"))
    _set(features, "vascular", "red_intensity", "p90_vascular_redness", intensity.get("p90_redness_response"))


def _project_contour(features: dict[str, Any], detectors: dict[str, Any]) -> None:
    contour = _dictionary(_dictionary(detectors.get("contour_firmness")).get("metrics"))
    _set(features, "contour_firmness", "midface_support_decline", "midface_surface_continuity", contour.get("中面部曲面连续性"))
    _set(features, "contour_firmness", "lower_face_sagging", "jawline_continuity", contour.get("下颌缘连续性"))


def project_institution_word_features(complete: dict[str, Any]) -> dict[str, Any]:
    """Project existing institution measurements into the frozen Word profile."""

    features = copy.deepcopy(_dictionary(complete.get("scoring_features")))
    detectors = _dictionary(complete.get("detector_results"))
    _project_oil(features, detectors)
    _project_vascular(features, detectors)
    _project_contour(features, detectors)
    return features


__all__ = [
    "INSTITUTION_METRIC_ALLOWLIST",
    "project_institution_word_features",
]
