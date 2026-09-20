from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, ValidationError, create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "cloud" / "contracts" / "internal_dev_contracts.json"
OPTIONAL_FIELDS = {"medical_metrics_v2", "medical_report_csv_v2"}


def _strict_model(name: str, required_keys: set[str]):
    fields = {key: (Any, ...) for key in required_keys}
    fields.update({
        "medical_metrics_v2": (dict[str, Any] | None, None),
        "medical_report_csv_v2": (str | None, None),
    })
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


class MedicalV2BackendCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))[
            "source_repositories"
        ]

    def _assert_legacy_and_extended(self, name: str, keys: set[str]) -> None:
        model = _strict_model(f"{name.title()}RawResult", keys)
        legacy = {key: None for key in keys}
        parsed_legacy = model.model_validate(legacy)
        self.assertIsNone(parsed_legacy.medical_metrics_v2)

        extended = {
            **legacy,
            "medical_metrics_v2": {
                "project": name,
                "metrics_version": "medical_metrics_v2_20260728",
                "scoring_status": "uncalibrated",
            },
            "medical_report_csv_v2": f"{name}/医学量化指标_V2.csv",
        }
        parsed_extended = model.model_validate(extended)
        self.assertEqual(
            parsed_extended.medical_metrics_v2["scoring_status"],
            "uncalibrated",
        )
        with self.assertRaises(ValidationError):
            model.model_validate({**extended, "unexpected_field": 1})

    def test_dermavision_algorithms_accept_only_declared_additions(self) -> None:
        specs = self.manifest["dermavision"]["single_algorithm_raw_result"]
        for algorithm, keys in specs.items():
            with self.subTest(algorithm=algorithm):
                self._assert_legacy_and_extended(algorithm, set(keys))

    def test_acne_and_wrinkle_accept_only_declared_additions(self) -> None:
        for service in ("acne", "wrinkle"):
            spec = self.manifest[service]
            keys = set(spec["raw_result_oss_keys"]) | set(
                spec["raw_result_structured_keys"]
            )
            with self.subTest(service=service):
                self._assert_legacy_and_extended(service, keys)

    def test_optional_field_names_are_identical_across_services(self) -> None:
        for spec in self.manifest.values():
            self.assertEqual(
                set(spec["medical_v2_optional_raw_result_keys"]),
                OPTIONAL_FIELDS,
            )


if __name__ == "__main__":
    unittest.main()
