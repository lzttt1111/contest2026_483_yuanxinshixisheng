"""Pinned scoring assets for the shared pure-data entry.

Every asset is loaded through the existing SHA-pinned loaders where one exists.
The V0.1.1 official ECDF profile has no public loader, so it is pinned here
against ``SCORING_ASSET_MANIFEST.json``; any drift fails closed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.aisia_medical_report.word_scoring import (
    load_word_acne_2d_reference,
    load_word_population_profile,
    load_word_wrinkle_2d_references,
)
from src.scoring_bridge.population_profile import PopulationProfile
from src.scoring_bridge.word_acne_2d import Acne2DReferences
from src.scoring_bridge.word_wrinkle_2d import Wrinkle2DReferences

from .errors import ScoringAssetError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIELD_LIST_PATH = PROJECT_ROOT / "calibration" / "scoring_input_field_list_v3.json"
FORMULA_PATH = PROJECT_ROOT / "calibration" / "v2_explicit_formula_registry_v1.json"
METRIC_REGISTRY_PATH = (
    PROJECT_ROOT / "calibration" / "metric_registry_v2_proxy_20260728.json"
)
WORD_PROFILE_PATH = (
    PROJECT_ROOT
    / "00_runtime_assets/scoring_word_provisional/word_population_reference_1000.json"
)
V011_ROOT = PROJECT_ROOT / "00_runtime_assets/scoring_v011"
V011_MANIFEST = V011_ROOT / "SCORING_ASSET_MANIFEST.json"
V011_OFFICIAL_NAME = "全量历史ECDF评分配置.json"
V011_OFFICIAL_VERSION = "aisia_scoring_v0.1.1_integrity_full_reference_20260807"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class InputField:
    path: str
    unit: tuple[str, ...]
    consumers: tuple[str, ...]
    zero_semantics: str
    required: bool
    profile_scope: str | None = None
    fallback_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FieldList:
    list_version: str
    formula_registry_sha256: str
    formula_registry_version: str
    word_profile_sha256: str
    algorithms: Mapping[str, tuple[InputField, ...]]
    algorithm_detectors: Mapping[str, tuple[str, ...]]
    module_word_bindings: Mapping[str, Mapping[str, Any]]
    quality_inputs: tuple[str, ...]
    v011_supplementary: tuple[Mapping[str, Any], ...]
    source_sha256: str

    def required_fields(self, algorithm: str) -> tuple[InputField, ...]:
        return tuple(
            field
            for field in self.algorithms.get(algorithm, ())
            if field.required and field.profile_scope is None
        )


@dataclass(frozen=True, slots=True)
class Assets:
    field_list: FieldList
    population_profile: PopulationProfile | None
    institution_profile: PopulationProfile | None
    wrinkle_references: Mapping[str, Wrinkle2DReferences] | None
    acne_reference: Acne2DReferences | None
    v011_references: Mapping[str, list[float]]
    v011_profile_version: str
    asset_sha256: Mapping[str, str]


def load_field_list(path: Path = FIELD_LIST_PATH) -> FieldList:
    if not path.is_file():
        raise ScoringAssetError(f"scoring input field list is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("list_version") not in {"scoring_input_field_list_v1", "scoring_input_field_list_v2", "scoring_input_field_list_v3"}:
        raise ScoringAssetError("scoring input field list version mismatch")
    registry_sha = _sha256(FORMULA_PATH)
    if document.get("formula_registry_sha256") != registry_sha:
        raise ScoringAssetError("field list formula registry SHA drifted")
    algorithms = {
        algorithm: tuple(
            InputField(
                path=str(entry["path"]),
                unit=tuple(str(value) for value in entry.get("unit", [])),
                consumers=tuple(str(value) for value in entry.get("consumers", [])),
                zero_semantics=str(entry.get("zero_semantics", "non_negative")),
                required=bool(entry.get("required", False)),
                profile_scope=entry.get("profile_scope"),
                fallback_paths=tuple(
                    str(value) for value in entry.get("fallback_paths", [])
                ),
            )
            for entry in value["fields"]
        )
        for algorithm, value in document["algorithms"].items()
    }
    return FieldList(
        list_version=document["list_version"],
        formula_registry_sha256=document["formula_registry_sha256"],
        formula_registry_version=document["formula_registry_version"],
        word_profile_sha256=document["word_profile_sha256"],
        algorithms=algorithms,
        algorithm_detectors={
            algorithm: tuple(value["detectors"])
            for algorithm, value in document["algorithms"].items()
        },
        module_word_bindings=document["module_word_bindings"],
        quality_inputs=tuple(document["quality_inputs"]),
        v011_supplementary=tuple(document["v011_supplementary"]),
        source_sha256=_sha256(path),
    )


def _load_v011_official() -> tuple[Mapping[str, list[float]], str]:
    if not V011_MANIFEST.is_file() or not (V011_ROOT / V011_OFFICIAL_NAME).is_file():
        raise ScoringAssetError("V0.1.1 scoring asset manifest or profile missing")
    manifest = json.loads(V011_MANIFEST.read_text(encoding="utf-8"))
    pinned = {
        entry["relative_path"]: entry["sha256"]
        for entry in manifest.get("files", [])
    }
    official_path = V011_ROOT / V011_OFFICIAL_NAME
    expected = pinned.get(V011_OFFICIAL_NAME)
    if expected is None or _sha256(official_path) != expected:
        raise ScoringAssetError("V0.1.1 official profile SHA mismatch")
    document = json.loads(official_path.read_text(encoding="utf-8"))
    if document.get("profile_ready") is not True:
        raise ScoringAssetError("V0.1.1 official profile is not ready")
    if document.get("scoring_profile_version") != V011_OFFICIAL_VERSION:
        raise ScoringAssetError("V0.1.1 official profile version mismatch")
    references = document.get("references")
    if not isinstance(references, dict):
        raise ScoringAssetError("V0.1.1 official profile references missing")
    return references, V011_OFFICIAL_VERSION


def load_default_assets() -> Assets:
    references, version = _load_v011_official()
    return Assets(
        field_list=load_field_list(),
        population_profile=load_word_population_profile(capture_profile="consumer"),
        institution_profile=load_word_population_profile(capture_profile="institution"),
        wrinkle_references=load_word_wrinkle_2d_references(),
        acne_reference=load_word_acne_2d_reference(),
        v011_references=references,
        v011_profile_version=version,
        asset_sha256={
            "scoring_input_field_list_v3.json": _sha256(FIELD_LIST_PATH),
            "v2_explicit_formula_registry_v1.json": _sha256(FORMULA_PATH),
            "metric_registry_v2_proxy_20260728.json": _sha256(METRIC_REGISTRY_PATH),
            "word_population_reference_1000.json": _sha256(WORD_PROFILE_PATH),
            V011_OFFICIAL_NAME: _sha256(V011_ROOT / V011_OFFICIAL_NAME),
        },
    )


__all__ = ["Assets", "FieldList", "InputField", "load_default_assets", "load_field_list"]
