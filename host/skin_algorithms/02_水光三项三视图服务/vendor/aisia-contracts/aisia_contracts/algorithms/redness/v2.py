from typing import Any

from aisia_contracts.algorithms.redness.v1 import RednessV1RawResult
from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope


class RednessV2RawResult(RednessV1RawResult):
    """redness v2: v1 全部字段（旧英文 metrics 不变）+ medical_metrics_v2 + medical_report_csv_v2。"""
    medical_metrics_v2: dict[str, Any] | None = None
    medical_report_csv_v2: str | None = None


class _RednessV2Envelope(AlgorithmEnvelope):
    raw_result: RednessV2RawResult


from aisia_contracts.registry import register
SCHEMA_VERSION = "2"
register("redness", SCHEMA_VERSION, _RednessV2Envelope)
