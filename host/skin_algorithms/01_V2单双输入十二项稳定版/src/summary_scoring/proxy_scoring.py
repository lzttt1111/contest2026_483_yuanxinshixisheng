"""Production proxy scoring orchestration (reuses the existing chains)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping

from src.nine_analysis.production_proxy_scoring import score_production_proxy
from src.nine_analysis.v2_proxy_projection import build_v2_proxy_modules

from .assets import METRIC_REGISTRY_PATH
from .errors import ScoringAssetError


@lru_cache(maxsize=1)
def load_metric_registry() -> dict[str, Any]:
    if not METRIC_REGISTRY_PATH.is_file():
        raise ScoringAssetError("metric registry asset is missing")
    return json.loads(METRIC_REGISTRY_PATH.read_text(encoding="utf-8"))


def canon(source_metric_id: str) -> str:
    detector, rest = source_metric_id.split(".", 1)
    return f"{detector}.metrics.{rest}"


def _missing_paths(medical_modules: Mapping[str, Any], module_id: str) -> list[str]:
    module = medical_modules.get(module_id) or {}
    missing: list[str] = []
    for group in (module.get("groups") or {}).values():
        for metric in (group.get("metrics") or {}).values():
            if metric.get("availability") != "unavailable":
                continue
            missing.extend(canon(str(source)) for source in metric["source_metric_ids"])
    return missing


def score_proxy_modules(
    detector_results: Mapping[str, Any],
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = registry if registry is not None else load_metric_registry()
    medical_modules, scoring_features = build_v2_proxy_modules(detector_results, registry)
    production = score_production_proxy(medical_modules)
    missing_by_module = {
        module_id: _missing_paths(medical_modules, module_id)
        for module_id in medical_modules
    }
    return {
        "production": production,
        "medical_modules": medical_modules,
        "scoring_features": scoring_features,
        "missing_by_module": missing_by_module,
    }


__all__ = ["canon", "load_metric_registry", "score_proxy_modules"]
