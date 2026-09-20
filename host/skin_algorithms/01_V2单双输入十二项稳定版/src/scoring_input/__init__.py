"""版本化评分输入（scoring_input）生成模块。

各 worker 在检测结束、清理中间文件前调用 ``build_scoring_input``，把单算法
运行时结果投影成 ``scoring_input_v1`` 形状的 dict，放入成功信封的
``raw_result.scoring_input``。evidence 只镜像冻结字段清单内的 ``detector_results``
子集，缺失字段诚实记为 partial/missing，绝不补零。
"""

from __future__ import annotations

from .builder import (
    FIELD_LIST_SHA256,
    ScoringInputBuildError,
    build_scoring_input,
    extract_evidence,
)
from .gate import compute_input_quality_gate

__all__ = [
    "FIELD_LIST_SHA256",
    "ScoringInputBuildError",
    "build_scoring_input",
    "compute_input_quality_gate",
    "extract_evidence",
]
