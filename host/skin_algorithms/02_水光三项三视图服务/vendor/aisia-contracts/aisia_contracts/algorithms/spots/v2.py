from typing import Any

from aisia_contracts.algorithms.spots.v1 import SpotsV1RawResult
from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope


class SpotsV2RawResult(SpotsV1RawResult):
    """spots v2: v1 全部字段（旧英文 metrics 不变）+ medical_metrics_v2 + medical_report_csv_v2。"""
    medical_metrics_v2: dict[str, Any] | None = None
    medical_report_csv_v2: str | None = None


class _SpotsV2Envelope(AlgorithmEnvelope):
    raw_result: SpotsV2RawResult


from aisia_contracts.registry import register
SCHEMA_VERSION = "2"
register("spots", SCHEMA_VERSION, _SpotsV2Envelope)
