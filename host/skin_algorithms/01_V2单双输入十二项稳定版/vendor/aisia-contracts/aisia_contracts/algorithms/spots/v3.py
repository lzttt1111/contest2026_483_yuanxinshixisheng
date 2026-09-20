"""spots v3 契约。

已发布 v2 不可修改(发布即冻结铁律):本文件只新增 v3 并注册 ("spots", "3")。
v3 = v2 全部字段不变 + 版本化评分输入 ``scoring_input``。
"""
from aisia_contracts.algorithms.spots.v2 import SpotsV2RawResult
from aisia_contracts.envelope import AlgorithmEnvelope
from aisia_contracts.scoring_input.v1 import ScoringInputV1


class SpotsV3RawResult(SpotsV2RawResult):
    """spots v3: v2 全部字段 + scoring_input(可选)。"""

    scoring_input: ScoringInputV1 | None = None


class _SpotsV3Envelope(AlgorithmEnvelope):
    """drift 检测:校验 dermavision worker 对 spots v3 的返回结构。"""

    raw_result: SpotsV3RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "3"
register("spots", SCHEMA_VERSION, _SpotsV3Envelope)
