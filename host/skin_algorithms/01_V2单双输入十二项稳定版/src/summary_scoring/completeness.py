"""Frozen-field-list completeness checks.

The entry runs these *before* any ``_number`` extractor so that a missing leaf
can never be silently converted to 0 by ``word_acne_2d``/``word_wrinkle_2d``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .assets import FieldList, InputField

_INDEX_RE = re.compile(r"^(?P<name>[^\[\]]*)(?P<selectors>(?:\[\d*\])*)$")
_SELECTOR_RE = re.compile(r"\[(\d*)\]")

STRATEGY_CONSUMERS = {
    "legacy_v011": "word-v011",
    "population_profile": "word-pop",
    "acne_2d": "word-acne2d",
    "wrinkle_2d": "word-wrinkle2d",
}


def _parse_head(part: str) -> tuple[str, list[int | None]]:
    match = _INDEX_RE.match(part)
    if match is None:
        return part, []
    selectors = [
        (int(value) if value else None)
        for value in _SELECTOR_RE.findall(match.group("selectors"))
    ]
    return match.group("name"), selectors


def _resolve(node: Any, parts: list[str]) -> tuple[bool, Any]:
    if not parts:
        return True, node
    name, selectors = _parse_head(parts[0])
    if name:
        if not isinstance(node, dict) or name not in node:
            return False, None
        node = node[name]
    for selector in selectors:
        if selector is None:
            if not isinstance(node, list) or not node:
                return False, None
            for item in node:
                found, value = _resolve(item, parts[1:])
                if found and value is not None:
                    return True, value
            return False, None
        if not isinstance(node, list) or not -len(node) <= selector < len(node):
            return False, None
        node = node[selector]
    return _resolve(node, parts[1:])


def resolve_path(evidence: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    found, value = _resolve(evidence, path.split("."))
    if not found or value is None:
        return False, None
    if isinstance(value, (list, dict)) and not value:
        return False, None
    return True, value


def field_present(evidence: Mapping[str, Any], field: InputField) -> bool:
    for candidate in (field.path, *field.fallback_paths):
        found, _ = resolve_path(evidence, candidate)
        if found:
            return True
    return False


def _scope_matches(field: InputField, capture_profile: str) -> bool:
    return field.profile_scope is None or field.profile_scope == capture_profile


def algorithm_missing(
    field_list: FieldList,
    algorithm: str,
    evidence: Mapping[str, Any],
    capture_profile: str,
) -> list[str]:
    missing: list[str] = []
    for field in field_list.algorithms.get(algorithm, ()):
        if not field.required or not _scope_matches(field, capture_profile):
            continue
        if not field_present(evidence, field):
            missing.append(field.path)
    return missing


def detector_owner(field_list: FieldList) -> dict[str, str]:
    owner: dict[str, str] = {}
    for algorithm, detectors in field_list.algorithm_detectors.items():
        for detector in detectors:
            owner.setdefault(detector, algorithm)
    return owner


def module_word_missing(
    field_list: FieldList,
    module_id: str,
    evidence: Mapping[str, Any],
    capture_profile: str,
) -> list[str]:
    binding = field_list.module_word_bindings[module_id]
    strategy = str(binding["strategy"])
    tags = {STRATEGY_CONSUMERS[strategy]}
    if strategy == "population_profile" and capture_profile == "institution":
        tags.add("word-inst")
    owner = detector_owner(field_list)
    missing: list[str] = []
    for detector in binding["detectors"]:
        algorithm = owner.get(detector)
        if algorithm is None:
            continue
        for field in field_list.algorithms.get(algorithm, ()):
            if not field.required or not _scope_matches(field, capture_profile):
                continue
            if tags.isdisjoint(field.consumers):
                continue
            if not field_present(evidence, field):
                missing.append(field.path)
    return missing


__all__ = [
    "algorithm_missing",
    "module_word_missing",
    "resolve_path",
]
