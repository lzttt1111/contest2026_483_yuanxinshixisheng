from __future__ import annotations

import importlib

from cloud.simulated_contract_versions import install_simulated_contract_versions


def test_local_simulator_installs_all_locked_schema_versions() -> None:
    install_simulated_contract_versions()

    expected = {
        "redness.v3": "3",
        "spots.v3": "3",
        "brown.v3": "3",
        "texture.v3": "3",
        "pores.v3": "3",
        "purple.v2": "2",
        "acne.v1": "1",
        "wrinkle.v2": "2",
        "surface_gloss.v1": "1",
        "vascular.v1": "1",
        "contour_firmness.v1": "1",
    }
    for module_suffix, schema_version in expected.items():
        module = importlib.import_module(
            f"aisia_contracts.algorithms.{module_suffix}"
        )
        assert module.SCHEMA_VERSION == schema_version
