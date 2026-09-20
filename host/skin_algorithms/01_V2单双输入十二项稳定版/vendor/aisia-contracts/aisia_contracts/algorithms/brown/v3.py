"""brown v3 契约。

已发布 v2 不可修改(发布即冻结铁律):本文件只新增 v3 并注册 ("brown", "3")。
v3 = v2 全部字段不变 + 版本化评分输入 ``scoring_input``。
"""
from aisia_contracts.algorithms.brown.v2 import BrownV2RawResult
from aisia_contracts.envelope import AlgorithmEnvelope
from aisia_contracts.scoring_input.v1 import ScoringInputV1


class BrownV3RawResult(BrownV2RawResult):
    """brown v3: v2 全部字段 + scoring_input(可选)。"""

    scoring_input: ScoringInputV1 | None = None


class _BrownV3Envelope(AlgorithmEnvelope):
    """drift 检测:校验 dermavision worker 对 brown v3 的返回结构。"""

    raw_result: BrownV3RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "3"
register("brown", SCHEMA_VERSION, _BrownV3Envelope)
