"""Frozen scoring release identity and pinned asset SHA table.

``RELEASE_ID`` binds the display policy, the production proxy policy and the
frozen field list to a code version. ``PINNED_ASSET_SHA256`` pins the exact
bytes of every scoring asset the summary entry loads. ``load_release``
re-hashes the real files and fails closed on any drift, so a summary run can
never silently switch to a different scoring configuration.

This module only imports ``hashlib``/``dataclasses``/``pathlib`` so it stays
usable from the lightweight worker entry point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Format: <word profile>+<proxy policy>+<field list>@<code version>. Bump the
# whole id whenever any pinned asset or scoring policy changes.
RELEASE_ID = "word_v0.1.1+proxy_v1+fieldlist_v3@dermavision-0.2-four-light"

ASSET_RELATIVE_PATHS: Mapping[str, str] = {
    "scoring_input_field_list_v3.json": "calibration/scoring_input_field_list_v3.json",
    "v2_explicit_formula_registry_v1.json": "calibration/v2_explicit_formula_registry_v1.json",
    "metric_registry_v2_proxy_20260728.json": (
        "calibration/metric_registry_v2_proxy_20260728.json"
    ),
    "word_population_reference_1000.json": (
        "00_runtime_assets/scoring_word_provisional/"
        "word_population_reference_1000.json"
    ),
    "全量历史ECDF评分配置.json": (
        "00_runtime_assets/scoring_v011/全量历史ECDF评分配置.json"
    ),
}

PINNED_ASSET_SHA256: Mapping[str, str] = {
    "scoring_input_field_list_v3.json": (
        "e4f4cc694ba48df6fe63e404ed53a6ff416deb900660958076da4c151c0190ff"
    ),
    "v2_explicit_formula_registry_v1.json": (
        "9ac2d51db8239679318d791f55f3519a69996596f489153a3779db983e059c27"
    ),
    "metric_registry_v2_proxy_20260728.json": (
        "5a6231ea936a38e1cbbfc28158defa66978b6ba03b7bc406bcfe8526365dbeb7"
    ),
    "word_population_reference_1000.json": (
        "a6933505ae50384612d28ed9b3cd788d5112566049bbcf8622f7bdb406a894ea"
    ),
    "全量历史ECDF评分配置.json": (
        "8cf292c2e87b21a2b9e3543bd2a031567835eca330b0fc52134aaebf4435b808"
    ),
}


class ReleaseAssetError(RuntimeError):
    """Raised when a pinned scoring asset is missing or has drifted."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class Release:
    release_id: str
    asset_sha256: Mapping[str, str]


def load_release() -> Release:
    """Verify every pinned asset and return the release identity + actual SHAs.

    Not cached: drift must be detected on every task, and the hashing cost is
    small compared with the scoring itself.
    """

    actual: dict[str, str] = {}
    for name, relative in ASSET_RELATIVE_PATHS.items():
        pinned = PINNED_ASSET_SHA256.get(name)
        if pinned is None:
            raise ReleaseAssetError(f"scoring asset not pinned: {name}")
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise ReleaseAssetError(f"scoring asset missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != pinned:
            raise ReleaseAssetError(
                f"scoring asset SHA drifted: {name} {digest} != {pinned}"
            )
        actual[name] = digest
    return Release(release_id=RELEASE_ID, asset_sha256=actual)


__all__ = [
    "ASSET_RELATIVE_PATHS",
    "PINNED_ASSET_SHA256",
    "RELEASE_ID",
    "Release",
    "ReleaseAssetError",
    "load_release",
]
