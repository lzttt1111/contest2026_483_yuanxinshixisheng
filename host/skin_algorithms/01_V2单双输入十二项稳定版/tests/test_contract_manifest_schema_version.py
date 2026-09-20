from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_MANIFEST = PROJECT_ROOT / "cloud" / "contracts" / "internal_dev_contracts.json"
SUCCESS_ENVELOPE = [
    "record_id",
    "status",
    "schema_version",
    "meta_data",
    "raw_result",
    "debug_info",
]


def test_manifest_matches_the_six_field_worker_envelope() -> None:
    manifest = json.loads(CONTRACT_MANIFEST.read_text(encoding="utf-8"))

    for repository in manifest["source_repositories"].values():
        assert repository["success_envelope"] == SUCCESS_ENVELOPE
