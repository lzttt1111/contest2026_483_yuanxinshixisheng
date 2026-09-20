"""purple v2 契约。

已发布 v1 不可修改(发布即冻结铁律):本文件只新增 v2 并注册 ("purple", "2")。
v2 = v1 全部字段不变 + 版本化评分输入 ``scoring_input``。
"""
from aisia_contracts.algorithms.purple.v1 import PurpleV1RawResult
from aisia_contracts.envelope import AlgorithmEnvelope
from aisia_contracts.scoring_input.v1 import ScoringInputV1


class PurpleV2RawResult(PurpleV1RawResult):
    """purple v2: v1 全部字段 + scoring_input(可选)。"""

    scoring_input: ScoringInputV1 | None = None


class _PurpleV2Envelope(AlgorithmEnvelope):
    """drift 检测:校验 dermavision worker 对 purple v2 的返回结构。"""

    raw_result: PurpleV2RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "2"
register("purple", SCHEMA_VERSION, _PurpleV2Envelope)
