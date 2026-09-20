"""brown v2 契约,源自 ai-skin-backend service/algorithm/implements/brown/v2.py。"""
from typing import Any

from aisia_contracts.algorithms.brown.v1 import BrownV1RawResult
from aisia_contracts.envelope import AlgorithmEnvelope


class BrownV2RawResult(BrownV1RawResult):
    """brown v2: v1 全部字段（旧 dict[str,int] metrics 不变）+ medical_metrics_v2 + medical_report_csv_v2。"""
    medical_metrics_v2: dict[str, Any] | None = None
    medical_report_csv_v2: str | None = None


class _BrownV2Envelope(AlgorithmEnvelope):
    raw_result: BrownV2RawResult



from aisia_contracts.registry import register
SCHEMA_VERSION = "2"
register("brown", SCHEMA_VERSION, _BrownV2Envelope)
