"""redness v3 契约。

已发布 v2 不可修改(发布即冻结铁律):本文件只新增 v3 并注册 ("redness", "3")。
v3 = v2 全部字段不变 + 版本化评分输入 ``scoring_input``(阶段 A 存储=inline 决策)。
"""
from aisia_contracts.algorithms.redness.v2 import RednessV2RawResult
from aisia_contracts.envelope import AlgorithmEnvelope
from aisia_contracts.scoring_input.v1 import ScoringInputV1


class RednessV3RawResult(RednessV2RawResult):
    """redness v3: v2 全部字段 + scoring_input(可选,历史旧数据可缺省)。"""

    scoring_input: ScoringInputV1 | None = None


class _RednessV3Envelope(AlgorithmEnvelope):
    """drift 检测:校验 dermavision worker 对 redness v3 的返回结构。"""

    raw_result: RednessV3RawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "3"
register("redness", SCHEMA_VERSION, _RednessV3Envelope)
