"""Field-list asset integrity and registry coverage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.summary_scoring.assets import (
    FIELD_LIST_PATH,
    FORMULA_PATH,
    WORD_PROFILE_PATH,
    load_field_list,
)

FIXTURE_MANIFEST = (
    Path(__file__).resolve().parent / "fixtures" / "scoring" / "manifest.json"
)


def _canon(source_metric_id: str) -> str:
    detector, rest = source_metric_id.split(".", 1)
    return f"{detector}.metrics.{rest}"


def test_field_list_sha_and_content_are_consistent() -> None:
    document = json.loads(FIELD_LIST_PATH.read_text(encoding="utf-8"))
    field_list = load_field_list()
    assert field_list.list_version == "scoring_input_field_list_v3"
    assert field_list.source_sha256 == hashlib.sha256(
        FIELD_LIST_PATH.read_bytes()
    ).hexdigest()
    assert document["formula_registry_sha256"] == hashlib.sha256(
        FORMULA_PATH.read_bytes()
    ).hexdigest()
    assert document["word_profile_sha256"] == hashlib.sha256(
        WORD_PROFILE_PATH.read_bytes()
    ).hexdigest()
    for algorithm in document["algorithms"].values():
        for field in algorithm["fields"]:
            assert field["unit"]
            assert field["consumers"]
            assert field["zero_semantics"] in {"legitimate_zero", "non_negative"}
            assert isinstance(field["required"], bool)


def test_field_list_covers_every_formula_registry_source_path() -> None:
    registry = json.loads(FORMULA_PATH.read_text(encoding="utf-8"))
    canonical = {
        _canon(source)
        for definition in registry["metric_formulas"].values()
        for source in definition["source_metric_ids"]
    }
    listed = {
        field.path
        for fields in load_field_list().algorithms.values()
        for field in fields
    }
    assert canonical <= listed, sorted(canonical - listed)


def test_field_list_pins_match_fixture_manifest() -> None:
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    document = json.loads(FIELD_LIST_PATH.read_text(encoding="utf-8"))
    assert document["formula_registry_sha256"] == manifest["formula_registry_sha256"]


def test_v011_instance_count_gate_is_recorded() -> None:
    field_list = load_field_list()
    paths = {entry["path"] for entry in field_list.v011_supplementary}
    assert paths == {
        "pores.instance_count",
        "spots.instance_count",
        "brown.instance_count",
        "texture.instance_count",
        "redness.instance_count",
    }


def test_vascular_redness_alias_registered_without_double_write() -> None:
    document = json.loads(FIELD_LIST_PATH.read_text(encoding="utf-8"))
    vascular = document["algorithms"]["vascular"]
    flat = {field["path"]: field for field in vascular["fields"]}
    assert "vascular.metrics.p50_redness" in flat
    alias = flat.get(
        "vascular.metrics.overall_metrics.core_metrics.signal_intensity."
        "p50_redness_response"
    )
    assert alias is not None
    assert alias["fallback_paths"] == ["vascular.metrics.p50_redness"]
    assert alias["profile_scope"] == "institution"
