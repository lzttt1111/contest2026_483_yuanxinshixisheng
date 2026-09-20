"""pores v2 契约,源自 ai-skin-backend service/algorithm/implements/pores/v2.py。"""
from typing import Any

from aisia_contracts.algorithms.pores.v1 import PoresV1RawResult
from aisia_contracts.envelope import AlgorithmEnvelope


class PoresV2RawResult(PoresV1RawResult):
    """pores v2: v1 全部字段（旧 dict[str,int] metrics 不变）+ medical_metrics_v2 + medical_report_csv_v2。"""
    medical_metrics_v2: dict[str, Any] | None = None
    medical_report_csv_v2: str | None = None


class _PoresV2Envelope(AlgorithmEnvelope):
    raw_result: PoresV2RawResult



from aisia_contracts.registry import register
SCHEMA_VERSION = "2"
register("pores", SCHEMA_VERSION, _PoresV2Envelope)
