from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = {
    "templates/AISIA_面部多指标检测汇总模板_V2.docx": "9de65290471199a14690db5b0db1571257788a24c23168ccf077e3c3701ffd4c",
    "src/aisia_medical_report/integration.py": "76181d51c1dbc34ffc5df24e094d34d911747f80b84de9a94aa4099c2821ac6e",
    "src/aisia_medical_report/aggregator.py": "faf3824fb4866474ed7f8f98c6606314583171f776b7a5b84c78705cbe28429c",
    "src/aisia_medical_report/report.py": "84397fb5b8a6066c86b8542eec4007d5db2b70da3e89d6840f2b768c456ff0d1",
}


def test_formal_word_template_and_pipeline_remain_frozen() -> None:
    actual = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in FROZEN_SHA256
    }
    assert actual == FROZEN_SHA256
