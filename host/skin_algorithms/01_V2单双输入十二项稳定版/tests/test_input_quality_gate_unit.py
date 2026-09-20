"""``compute_input_quality_gate`` 单元：真实调用 ``evaluate_image_path``。

零 GPU、零模型推理：人脸分支由注入 preprocessor 提供；接缝/热图/画布/解码
等图像派生完整性检查全部真实运行。
"""

from __future__ import annotations

import inspect

import pytest

from src.scoring_calibration.v011.quality import evaluate_image_path
from src.scoring_input.gate import compute_input_quality_gate

from input_gate_test_utils import (
    BoomPreprocessor,
    StubPreprocessor,
    center_collage,
    clean_synthetic,
    encode_png,
)


def _direct(image_bytes: bytes, tmp_path, preprocessor) -> dict:
    path = tmp_path / "direct.img"
    path.write_bytes(image_bytes)
    result = evaluate_image_path(path, preprocessor=preprocessor)
    return {
        "status": str(result["status"]),
        "reason_codes": [str(code) for code in result.get("reason_codes") or []],
    }


def test_clean_synthetic_matches_authoritative_function_output(tmp_path) -> None:
    image_bytes = encode_png(clean_synthetic())
    stub = StubPreprocessor(flags=())
    gate = compute_input_quality_gate(image_bytes, preprocessor=stub)
    direct = _direct(image_bytes, tmp_path, StubPreprocessor(flags=()))

    # 不预设 PASS：如实断言与权威函数直调一致，且只投影契约两字段。
    assert gate == direct
    assert set(gate) == {"status", "reason_codes"}
    assert gate["status"] in {"PASS", "REJECT"}


def test_center_vertical_seam_is_rejected_as_collage(tmp_path) -> None:
    image_bytes = encode_png(center_collage())
    gate = compute_input_quality_gate(image_bytes, preprocessor=StubPreprocessor())
    direct = _direct(image_bytes, tmp_path, StubPreprocessor())

    assert gate == direct
    assert gate["status"] == "REJECT"
    assert "collage" in gate["reason_codes"]


def test_corrupt_bytes_are_decode_failed_without_preprocessor_call() -> None:
    gate = compute_input_quality_gate(
        b"\x89PNG\r\n\x1a\nnot-an-image",
        preprocessor=BoomPreprocessor(),
    )

    assert gate == {"status": "REJECT", "reason_codes": ["decode_failed"]}


def test_missing_face_flag_is_rejected() -> None:
    gate = compute_input_quality_gate(
        encode_png(clean_synthetic()),
        preprocessor=StubPreprocessor(flags=("NO_FACE",)),
    )

    assert gate == {"status": "REJECT", "reason_codes": ["no_face"]}


def test_algorithm_quality_status_is_never_mapped_to_gate() -> None:
    # preprocessor 报 REJECT/quality_status，但 flags 为空：真实 evaluate_image_path
    # 只看 quality_flags 的结构性 reject 码，绝不做 quality_status → gate 映射。
    gate = compute_input_quality_gate(
        encode_png(clean_synthetic()),
        preprocessor=StubPreprocessor(flags=(), quality_status="REJECT"),
    )

    assert gate == {"status": "PASS", "reason_codes": []}
    source = inspect.getsource(compute_input_quality_gate)
    assert "quality_status" not in source
    assert "quality_flags" not in source


@pytest.mark.parametrize("value", [b"", bytearray(), None, "not-bytes"])
def test_unusable_input_returns_none(value) -> None:
    assert compute_input_quality_gate(value) is None
