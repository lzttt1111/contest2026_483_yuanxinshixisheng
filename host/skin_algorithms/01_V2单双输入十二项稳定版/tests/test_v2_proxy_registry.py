from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.nine_analysis.v2_labels import load_v2_labels, metric_descriptor


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "calibration" / "metric_registry_v2_proxy_20260728.json"
LABELS = ROOT / "calibration" / "metric_labels_v2_zh_20260728.json"
MODULES = (
    "pores", "oil_tendency", "pigmentation", "diffuse_redness",
    "vascular", "acne_activity", "dry_fine_lines", "stable_wrinkles",
    "structural_grooves", "smoothness", "contour_firmness",
)


def _source_root() -> Path:
    return next(
        path for path in (ROOT, *ROOT.parents)
        if (path / "皮肤检测指标" / "面部检测需求md_V2_20260728").is_dir()
    )


def test_v2_registry_locks_all_sources_groups_and_regions() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert registry["registry_version"] == "doctor_v2_proxy_registry_20260728_v1"
    assert registry["scoring_status"] == "uncalibrated"
    assert list(registry["modules"]) == list(MODULES)
    assert len(registry["sources"]) == 11
    source_root = _source_root()
    for entry in registry["sources"]:
        source = source_root / entry["path"]
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"]
    for module_id, module in registry["modules"].items():
        assert module["measurement_mode"] == "single_rgb_2d_2_5d_proxy"
        assert module["groups"]
        assert abs(sum(float(group["weight"]) for group in module["groups"]) - 1.0) < 1e-9
        assert all(group["core_metrics"] for group in module["groups"])
        if module_id != "contour_firmness":
            assert module["regions"]
            assert abs(sum(float(region["weight"]) for region in module["regions"]) - 1.0) < 1e-9


def test_every_metric_has_stable_id_unit_direction_and_availability() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for module in registry["modules"].values():
        for group in module["groups"]:
            for metric in group["core_metrics"]:
                assert set(metric) == {"id", "unit", "direction", "required"}
                assert metric["id"].isascii()
                assert metric["unit"]
                assert metric["direction"] in {"higher_burden", "higher_health"}
                assert metric["required"] is True


def test_every_v2_region_group_metric_and_unit_has_chinese_label() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    labels = load_v2_labels()
    for module in registry["modules"].values():
        for region in module["regions"]:
            assert labels["regions"][region["id"]]
        for group in module["groups"]:
            assert labels["groups"][group["id"]]
            for metric in group["core_metrics"]:
                descriptor = metric_descriptor(
                    module["name"], metric["id"], metric["unit"], labels
                )
                assert descriptor["name"]
                assert descriptor["description"]
                assert descriptor["name"] != metric["id"]
                assert labels["units"][metric["unit"]]
