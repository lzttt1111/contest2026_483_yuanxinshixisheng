"""texture v2 契约,源自 ai-skin-backend service/algorithm/implements/texture/v2.py。"""
from typing import Any

from aisia_contracts.algorithms.texture.v1 import TextureV1RawResult
from aisia_contracts.envelope import AlgorithmEnvelope


class TextureV2RawResult(TextureV1RawResult):
    """texture v2: v1 全部字段（旧 dict[str,int] metrics 不变）+ medical_metrics_v2 + medical_report_csv_v2。"""
    medical_metrics_v2: dict[str, Any] | None = None
    medical_report_csv_v2: str | None = None


class _TextureV2Envelope(AlgorithmEnvelope):
    raw_result: TextureV2RawResult



from aisia_contracts.registry import register
SCHEMA_VERSION = "2"
register("texture", SCHEMA_VERSION, _TextureV2Envelope)
