from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


LABELS_PATH = (
    Path(__file__).resolve().parents[2]
    / "calibration"
    / "metric_labels_v2_zh_20260728.json"
)


def load_v2_labels() -> dict:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


def metric_descriptor(
    module_name: str,
    metric_id: str,
    unit_id: str,
    labels: Mapping[str, object],
) -> dict[str, str]:
    terms = labels["metric_terms"]
    metric_name = metric_id
    for token in sorted(terms, key=len, reverse=True):
        metric_name = metric_name.replace(token, str(terms[token]))
    metric_name = " ".join(metric_name.replace("_", " ").split())
    unit_label = str(labels["units"][unit_id])
    description = str(labels["description_template"]).format(
        module_name=module_name,
        metric_name=metric_name,
    )
    return {
        "name": metric_name,
        "unit_label": unit_label,
        "description": description,
    }


def enrich_v2_modules(
    modules: Mapping[str, Mapping[str, object]],
    labels: Mapping[str, object],
) -> dict[str, dict]:
    enriched = json.loads(json.dumps(modules, ensure_ascii=False))
    for module in enriched.values():
        module_name = module["name"]
        for region in module["regions"]:
            region["name"] = labels["regions"][region["id"]]
        projections = [module["groups"]]
        projections.extend(
            projection["groups"]
            for projection in module["region_groups"].values()
        )
        for groups in projections:
            for group_id, group in groups.items():
                group["name"] = labels["groups"][group_id]
                for metric_id, metric in group["metrics"].items():
                    metric.update(metric_descriptor(
                        module_name,
                        metric_id,
                        metric["unit"],
                        labels,
                    ))
    return enriched


__all__ = ["LABELS_PATH", "enrich_v2_modules", "load_v2_labels", "metric_descriptor"]
