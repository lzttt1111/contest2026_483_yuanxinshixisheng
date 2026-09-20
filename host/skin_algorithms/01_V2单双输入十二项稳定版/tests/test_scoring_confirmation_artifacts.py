from __future__ import annotations

import json
from pathlib import Path

import src.scoring_bridge as scoring_bridge


def test_confirmation_report_is_sha_bound_and_path_free(tmp_path: Path) -> None:
    write = getattr(scoring_bridge, "write_confirmation_report", None)
    verify = getattr(scoring_bridge, "verify_confirmation_report", None)
    assert callable(write)
    assert callable(verify)

    receipt = write(
        output_path=tmp_path / "confirmation.json",
        attempted_count=250,
        compatibility_results={
            "visible_pores": {
                "status": "promoted",
                "evaluable_count": 230,
                "attrition_count": 20,
            }
        },
        population_results={
            "oil_tendency": {
                "status": "blocked",
                "evaluable_count": 249,
                "attrition_count": 1,
            }
        },
        provenance={
            "confirmation_manifest_sha256": "a" * 64,
            "confirmation_pairs_sha256": "b" * 64,
            "compatibility_profile_sha256": "c" * 64,
            "population_profile_sha256": "d" * 64,
            "code_sha": "e" * 40,
        },
    )

    document = json.loads(receipt.path.read_text(encoding="utf-8"))
    assert document["status"] == "blocked"
    assert verify(document) == receipt.sha256
    assert "/tmp/" not in json.dumps(document)


def test_confirmation_report_rejects_tampering(tmp_path: Path) -> None:
    write = getattr(scoring_bridge, "write_confirmation_report")
    verify = getattr(scoring_bridge, "verify_confirmation_report")
    receipt = write(
        output_path=tmp_path / "confirmation.json",
        attempted_count=250,
        compatibility_results={},
        population_results={},
        provenance={"code_sha": "f" * 40},
    )
    document = json.loads(receipt.path.read_text(encoding="utf-8"))
    document["attempted_count"] = 249

    try:
        verify(document)
    except ValueError as error:
        assert "SHA" in str(error)
    else:
        raise AssertionError("tampered confirmation report was accepted")
