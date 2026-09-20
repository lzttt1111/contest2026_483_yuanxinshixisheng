"""总体评分聚合(overall-summary)契约包。

依据《十二项总体评分聚合接入方案_20260911.md》r3 §5.2/§5.4。
本包与算法信封 REGISTRY 解耦:summary 不是算法检测结果信封,
不注册到 REGISTRY,故不影响算法契约 REGISTRY 计数。

发布即冻结铁律:已发布的 schema_version 不可修改;
字段变动必须新增版本(如 v2),历史读取走旧版本 model。
"""
from aisia_contracts.overall_summary.v1 import (
    EXTERNAL_TO_SCORING_MODULE_ID,
    REQUEST_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    SUMMARY_MODULES,
    UNAVAILABLE_GRADE,
    AlgorithmOutcomeV1,
    SummaryCompleteness,
    SummaryError,
    SummaryModuleResult,
    SummaryRawResult,
    SummaryRequestV1,
    SummaryResultV1,
    SummaryScoringVersions,
    SummaryScoreEntry,
)

__all__ = [
    "EXTERNAL_TO_SCORING_MODULE_ID",
    "REQUEST_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SUMMARY_MODULES",
    "UNAVAILABLE_GRADE",
    "AlgorithmOutcomeV1",
    "SummaryCompleteness",
    "SummaryError",
    "SummaryModuleResult",
    "SummaryRawResult",
    "SummaryRequestV1",
    "SummaryResultV1",
    "SummaryScoringVersions",
    "SummaryScoreEntry",
]
