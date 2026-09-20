"""Shared pure-data scoring entry: evidence -> two independent score systems."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Mapping

from src.capture_profile import CaptureProfile
from src.nine_analysis.v2_proxy_projection import FORMULA_VERSION
from src.nine_analysis.zero_target_metrics import normalize_zero_target_detector_results

from .assets import Assets, load_default_assets
from .completeness import module_word_missing
from .errors import ScoringInputError
from .proxy_scoring import load_metric_registry, score_proxy_modules
from .word_display import MODULE_ORDER, score_word_display

DISPLAY_POLICY_VERSION = "word_display_conservative_edge_v1"
MODULE_SCORING_IDS = {
    "01": "pores",
    "02": "oil_tendency",
    "03": "pigmentation",
    "04": "diffuse_redness",
    "05": "vascular",
    "06": "acne_activity",
    "07": "dry_fine_lines",
    "08": "stable_wrinkles",
    "09": "structural_grooves",
    "10": "smoothness",
    "11": "contour_firmness",
}


@lru_cache(maxsize=1)
def _assets() -> Assets:
    return load_default_assets()


def _system_result(
    *,
    score: Any,
    grade: Any,
    score_valid: bool,
    score_status: str,
    quality_gate: Any,
    missing_inputs: list[str],
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "score": score if score_valid else None,
        "grade": grade if score_valid else "不可评估",
        "score_valid": bool(score_valid),
        "score_status": score_status,
        "quality_gate": quality_gate,
        "missing_inputs": missing_inputs,
        "reason_codes": reason_codes,
    }


def _gate_status(gate: Mapping[str, Any] | None) -> str | None:
    if not isinstance(gate, Mapping):
        return None
    status = gate.get("status")
    return str(status) if status is not None else None


def _assert_finite(node: Any, path: str = "") -> None:
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, float):
        if not math.isfinite(node):
            raise ScoringInputError(f"non-finite score at {path or '<root>'}")
        return
    if isinstance(node, int):
        return
    if isinstance(node, Mapping):
        for key, value in node.items():
            _assert_finite(value, f"{path}.{key}")
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            _assert_finite(value, f"{path}[{index}]")


def score_report_from_evidence(
    evidence: Mapping[str, Any],
    *,
    capture_profile: str,
    input_route: str,
    quality: Mapping[str, Any] | None,
    input_quality_gate: Mapping[str, Any] | None,
    scoring_assets: Assets | None = None,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise ScoringInputError("evidence must be a detector_results mapping")
    try:
        profile = CaptureProfile(capture_profile)
    except ValueError as error:
        raise ScoringInputError(f"unknown capture_profile: {capture_profile!r}") from error
    assets = scoring_assets if scoring_assets is not None else _assets()
    detector_results: Mapping[str, Any]
    if profile is CaptureProfile.CONSUMER:
        detector_results = normalize_zero_target_detector_results(dict(evidence))
    else:
        detector_results = evidence

    proxy_info = score_proxy_modules(detector_results, load_metric_registry())
    word_info = score_word_display(
        detector_results=detector_results,
        scoring_features=proxy_info["scoring_features"],
        capture_profile=profile.value,
        input_quality_gate=input_quality_gate,
        quality=quality,
        assets=assets,
    )
    gate_status = _gate_status(input_quality_gate)
    algorithm_quality = (quality or {}).get("quality_status") or (quality or {}).get("status")

    modules: dict[str, Any] = {}
    completeness_modules: dict[str, Any] = {}
    for module_id in MODULE_ORDER:
        scoring_id = MODULE_SCORING_IDS[module_id]
        strategy = str(assets.field_list.module_word_bindings[module_id]["strategy"])
        word_missing = module_word_missing(
            assets.field_list, module_id, detector_results, profile.value
        )
        word = dict(word_info["modules"][module_id])
        if word_missing:
            word = _system_result(
                score=None, grade=None, score_valid=False,
                score_status="unavailable", quality_gate=gate_status,
                missing_inputs=word_missing,
                reason_codes=["missing_required_evidence"],
            )
        elif strategy == "legacy_v011" and gate_status != "PASS":
            word_missing = list(word_missing)
            code = (
                "missing_input_quality_gate"
                if gate_status is None
                else f"input_quality_gate_{gate_status.lower()}"
            )
            word = _system_result(
                score=None, grade=None, score_valid=False,
                score_status="unavailable", quality_gate=gate_status,
                missing_inputs=[],
                reason_codes=[code],
            )
        else:
            word = _system_result(
                score=word["score"], grade=word["grade"],
                score_valid=word["score_valid"], score_status=word["score_status"],
                quality_gate=gate_status, missing_inputs=[],
                reason_codes=(
                    [] if word["score_valid"] else ["insufficient_scoring_evidence"]
                ),
            )

        proxy = proxy_info["production"]["module_scores"][scoring_id]
        proxy_missing = proxy_info["missing_by_module"].get(scoring_id, [])
        proxy_valid = bool(proxy["score_valid"]) and not proxy_missing
        modules[module_id] = {
            "module_id": scoring_id,
            "word_display": word,
            "production_proxy_v1": _system_result(
                score=proxy["score"],
                grade=proxy["grade"],
                score_valid=proxy_valid,
                score_status="production_proxy_v1",
                quality_gate=algorithm_quality,
                missing_inputs=proxy_missing,
                reason_codes=[] if proxy_valid else ["insufficient_proxy_evidence"],
            ),
        }
        completeness_modules[module_id] = {
            "word_display": {
                "complete": not word_missing,
                "missing_inputs": list(word_missing),
            },
            "production_proxy_v1": {
                "complete": not proxy_missing,
                "missing_inputs": proxy_missing,
            },
        }

    result = {
        "capture_profile": profile.value,
        "input_route": str(input_route),
        "scoring_versions": {
            "code_version": FORMULA_VERSION,
            "display_policy_version": DISPLAY_POLICY_VERSION,
            "formula_registry_version": assets.field_list.formula_registry_version,
            "v011_profile_version": assets.v011_profile_version,
            "evidence_field_list_sha256": assets.field_list.source_sha256,
            "asset_sha256": dict(assets.asset_sha256),
        },
        "completeness": {
            "list_version": assets.field_list.list_version,
            "formula_registry_sha256": assets.field_list.formula_registry_sha256,
            "complete": all(
                entry["word_display"]["complete"]
                and entry["production_proxy_v1"]["complete"]
                for entry in completeness_modules.values()
            ),
            "modules": completeness_modules,
        },
        "modules": modules,
    }
    _assert_finite(result)
    return result


__all__ = ["DISPLAY_POLICY_VERSION", "score_report_from_evidence"]
