"""builder/信封：``scoring_input.quality.input_quality_gate`` 携带真实门禁并通过契约。"""

from __future__ import annotations

import json
from pathlib import Path

from aisia_contracts.scoring_input.v1 import ScoringInputV1

from src.scoring_input import build_scoring_input
from src.scoring_input.gate import compute_input_quality_gate

from input_gate_test_utils import StubPreprocessor, center_collage, clean_synthetic, encode_png

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scoring"


def _redness_runtime() -> dict:
    source = json.loads(
        (FIXTURES / "clinic28-25_r1.json").read_text(encoding="utf-8")
    )["detector_results"]
    metrics = dict(source["redness"]["metrics"])
    medical = metrics.pop("medical_metrics_v2", None)
    return {"metrics": metrics, "medical_metrics_v2": medical}


def _build(**kwargs) -> dict:
    return build_scoring_input(
        algorithm_name="redness",
        detection_schema_version="3",
        detection_impl_version="1",
        report_id="gate-report",
        detection_attempt_id="gate-attempt",
        source_image_sha256="a" * 64,
        capture_profile="institution",
        input_route="institution_queue",
        runtime=_redness_runtime(),
        quality={"quality_status": "PASS"},
        **kwargs,
    )


def test_real_reject_gate_is_persisted_and_validates() -> None:
    gate = compute_input_quality_gate(
        encode_png(center_collage()),
        preprocessor=StubPreprocessor(),
    )
    assert gate == {"status": "REJECT", "reason_codes": ["collage"]}

    document = _build(input_quality_gate=gate)

    assert document["quality"]["input_quality_gate"] == gate
    ScoringInputV1.model_validate(document)


def test_real_pass_gate_is_persisted_and_validates() -> None:
    gate = compute_input_quality_gate(
        encode_png(clean_synthetic()),
        preprocessor=StubPreprocessor(flags=()),
    )
    assert gate == {"status": "PASS", "reason_codes": []}

    document = _build(input_quality_gate=gate)

    assert document["quality"]["input_quality_gate"] == gate
    ScoringInputV1.model_validate(document)


def test_absent_gate_stays_none_for_backward_compatibility() -> None:
    document = _build()

    assert document["quality"]["input_quality_gate"] is None
    ScoringInputV1.model_validate(document)
