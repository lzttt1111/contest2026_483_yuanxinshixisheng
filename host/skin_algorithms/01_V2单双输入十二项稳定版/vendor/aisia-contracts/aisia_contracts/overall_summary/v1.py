"""overall_summary v1 契约(总体评分聚合结果)。

依据《十二项总体评分聚合接入方案_20260911.md》r3 §5.2/§5.4。
§5.4 输出模型 SummaryResultV1:顶层 schema_version / report_id / attempt_id /
input_fingerprint / capture_profile / input_route / scoring_release / status /
raw_result / error / debug_info;raw_result 含 scoring_versions / completeness /
固定十一模块 modules。

发布即冻结铁律:本文件已发布后不可修改;字段变动必须新建 v2,
历史 summary 记录读取走旧版本 model。

本文件同时落地输出侧 SummaryResultV1 与输入侧 SummaryRequestV1
(第三段补齐;ScoringInputV1 已在 aisia_contracts.scoring_input.v1 定义)。

task status 与模块有效性解耦:status 表示聚合任务是否正常完成,
模块缺证据(score_valid=false)仍可为 task success;两套分(word_display /
production_proxy_v1)各自独立判断有效性,不能用一套分的成功掩盖另一套失败。
"""
import math
from typing import Any, Literal

from pydantic import Field, model_validator

from aisia_contracts.base import ContractModel
from aisia_contracts.scoring_input.v1 import ScoringInputV1

# 输出/输入数据格式版本,用于定位 pydantic model(发布即冻结)。
RESULT_SCHEMA_VERSION = "overall_summary_result_v1"
# 输入侧 request 数据格式版本,用于定位 SummaryRequestV1(发布即冻结)。
REQUEST_SCHEMA_VERSION = "overall_summary_request_v1"

# 固定十一模块,顺序即对外 01–11;三元组 (module_no, 外部 key, 现有评分模块 ID)。
# 外部 key 是 SummaryResult 对外暴露的模块名,现有评分模块 ID 用于映射内部评分结果。
SUMMARY_MODULES: tuple[tuple[str, str, str], ...] = (
    ("01", "visible_pores", "pores"),
    ("02", "oil_tendency", "oil_tendency"),
    ("03", "combined_pigmentation", "pigmentation"),
    ("04", "diffuse_redness", "diffuse_redness"),
    ("05", "vascular_like_structures", "vascular"),
    ("06", "follicular_acne_activity", "acne_activity"),
    ("07", "dry_fine_lines", "dry_fine_lines"),
    ("08", "stable_linear_wrinkles", "stable_wrinkles"),
    ("09", "structural_grooves", "structural_grooves"),
    ("10", "surface_smoothness", "smoothness"),
    ("11", "contour_firmness", "contour_firmness"),
)

# 外部模块 key -> 现有评分模块 ID。
EXTERNAL_TO_SCORING_MODULE_ID: dict[str, str] = {
    external: scoring for _, external, scoring in SUMMARY_MODULES
}

# 不可评估的固定等级;原展示层空等级转接口"不可评估"属于显式输出适配。
UNAVAILABLE_GRADE = "不可评估"

# 合法的 (module_no, 外部 key) 组合,供单模块自校验。
_VALID_MODULE_PAIRS: frozenset[tuple[str, str]] = frozenset(
    (module_no, external) for module_no, external, _ in SUMMARY_MODULES
)


class SummaryScoreEntry(ContractModel):
    """单套评分(word_display 或 production_proxy_v1)。

    有限数值才可有效;不可评估固定 ``score=null / grade=不可评估 /
    score_valid=false``,禁止 NaN/Infinity(字段 allow_inf_nan=False 显式拒绝)。
    score_valid 与 score 严格等价:score_valid=False <-> score is None。
    """

    score: float | None = Field(allow_inf_nan=False)
    grade: str
    score_valid: bool
    score_status: str
    quality_gate: str
    missing_inputs: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_validity_invariants(self) -> "SummaryScoreEntry":
        """强制 score_valid / score / grade 三者一致,防止静默漂移。"""
        if self.score_valid:
            if self.score is None:
                raise ValueError("score_valid=true 要求 score 为非 None 有限数")
            if not math.isfinite(self.score):
                raise ValueError("score_valid=true 要求 score 为有限数")
            if not self.grade:
                raise ValueError("score_valid=true 要求 grade 非空")
        else:
            if self.score is not None:
                raise ValueError("score_valid=false 要求 score 为 None")
            if self.grade != UNAVAILABLE_GRADE:
                raise ValueError(
                    f"score_valid=false 要求 grade == {UNAVAILABLE_GRADE!r}"
                )
        return self


class SummaryModuleResult(ContractModel):
    """固定十一模块中的一项,承载两套分。

    module_no 为 "01"–"11",name 为外部 key,二者须匹配 SUMMARY_MODULES。
    """

    module_no: str
    name: str
    word_display: SummaryScoreEntry
    production_proxy_v1: SummaryScoreEntry

    @model_validator(mode="after")
    def _check_module_identity(self) -> "SummaryModuleResult":
        """拒绝未登记的 module_no / name 组合。"""
        if (self.module_no, self.name) not in _VALID_MODULE_PAIRS:
            raise ValueError(
                f"未登记的模块组合 module_no={self.module_no!r}, name={self.name!r}"
            )
        return self


class SummaryScoringVersions(ContractModel):
    """本次聚合使用的评分版本证据。

    记录真实代码版本、显示策略版本、公式注册表版本,
    以及每个实际资产的**完整** SHA256(asset_id -> 完整 hex)。
    禁止只写恒定的 ``meta_data.version=1`` 这类无区分度占位;
    资产哈希必须逐资产记录,可用于复现与漂移检测。
    """

    code_version: str
    display_policy_version: str
    formula_registry_version: str
    asset_sha256: dict[str, str]


class SummaryCompleteness(ContractModel):
    """期望集与终态摘要,用于区分"未请求/失败/成功"。"""

    expected_algorithms: list[str]
    terminal_success: list[str]
    terminal_failed: list[str]
    not_requested: list[str]
    word_display_valid_count: int
    production_proxy_valid_count: int


class SummaryError(ContractModel):
    """聚合任务级错误(非模块级缺证据)。"""

    code: str
    message: str
    retryable: bool
    details: dict[str, Any] | None = None


class SummaryRawResult(ContractModel):
    """聚合结果主体:版本、完整性与固定十一模块。

    模块必须恰好 11 项,且 module_no/name 严格按 SUMMARY_MODULES 顺序精确匹配;
    缺项、多项、乱序、错名一律拒绝。
    """

    scoring_versions: SummaryScoringVersions
    completeness: SummaryCompleteness
    modules: list[SummaryModuleResult]

    @model_validator(mode="after")
    def _check_modules_exact_order(self) -> "SummaryRawResult":
        """校验模块数量与 01–11 顺序/命名精确一致。"""
        expected = [(no, name) for no, name, _ in SUMMARY_MODULES]
        actual = [(m.module_no, m.name) for m in self.modules]
        if len(actual) != len(expected):
            raise ValueError(
                f"modules 必须恰好 {len(expected)} 项,实际 {len(actual)} 项"
            )
        for index, (got, want) in enumerate(zip(actual, expected)):
            if got != want:
                raise ValueError(
                    f"modules[{index}] 应为 {want},实际 {got};"
                    "缺项/多项/乱序/错名均不允许"
                )
        return self


class SummaryResultV1(ContractModel):
    """总体评分聚合 v1 顶层结果。

    status="success" -> raw_result 必填且 error 禁止;
    status="failed"  -> error 必填且 raw_result 禁止。
    模块缺证据仍可为 task success(见模块级 score_valid)。
    """

    schema_version: Literal[RESULT_SCHEMA_VERSION]
    report_id: str
    attempt_id: str
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_profile: str
    input_route: str
    scoring_release: str
    status: Literal["success", "failed"]
    raw_result: SummaryRawResult | None
    error: SummaryError | None
    debug_info: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_status_consistency(self) -> "SummaryResultV1":
        """强制 status 与 raw_result/error 的一致关系。"""
        if self.status == "success":
            if self.raw_result is None:
                raise ValueError("status=success 要求 raw_result 必填")
            if self.error is not None:
                raise ValueError("status=success 禁止携带 error")
        else:
            if self.error is None:
                raise ValueError("status=failed 要求 error 必填")
            if self.raw_result is not None:
                raise ValueError("status=failed 禁止携带 raw_result")
        return self


class AlgorithmOutcomeV1(ContractModel):
    """聚合请求中单个上游算法的终态。

    与 SummaryResultV1.status 一样,status 表示该算法检测任务是否正常完成,
    与 evidence_status 解耦:任务 success 仍可能 evidence_status=partial/missing。
    failed 时 error_code 记录失败原因;成功终态字段可缺省。
    """

    algorithm_name: str
    status: Literal["success", "failed"]
    detection_attempt_id: str | None = None
    detection_schema_version: str | None = None
    error_code: str | None = None
    evidence_status: Literal["present", "partial", "missing", "failed"] | None = None


class SummaryRequestV1(ContractModel):
    """聚合请求 v1:上游各算法终态 + 版本化评分输入(存储=inline 冻结)。

    冻结决策(阶段 A):评分输入以 ``raw_result.scoring_input`` inline 随算法
    信封持久化(见证据矩阵 §4/§6.1),本 request 再将其随聚合请求内联携带。
    未来若改为文件引用模式,必须新建新的 request 版本,不得原地改本模型语义。

    一致性约束:
    - expected_algorithms 无重复;
    - algorithm_outcomes 算法名唯一且必须 ⊆ expected_algorithms;
    - scoring_inputs 的键必须 ⊆ status=success 的 outcome 算法名
      (失败/未请求的算法不得携带评分输入);
    - 每个 scoring_input.algorithm_name 必须等于其字典键;
    - 每个 scoring_input.report_id 必须等于顶层 report_id。
    """

    schema_version: Literal[REQUEST_SCHEMA_VERSION]
    report_id: str
    attempt_id: str
    capture_profile: str
    input_route: str
    expected_algorithms: list[str]
    algorithm_outcomes: list[AlgorithmOutcomeV1]
    scoring_inputs: dict[str, ScoringInputV1]
    scoring_release: str
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _check_request_consistency(self) -> "SummaryRequestV1":
        """强制 expected/outcomes/scoring_inputs 的集合与身份一致。"""
        expected = self.expected_algorithms
        if len(set(expected)) != len(expected):
            raise ValueError("expected_algorithms 不得重复")

        names = [outcome.algorithm_name for outcome in self.algorithm_outcomes]
        if len(set(names)) != len(names):
            raise ValueError("algorithm_outcomes.algorithm_name 必须唯一")
        unexpected = [name for name in names if name not in expected]
        if unexpected:
            raise ValueError(
                f"algorithm_outcomes 含未在 expected_algorithms 中的算法: {unexpected}"
            )

        success_names = {
            outcome.algorithm_name
            for outcome in self.algorithm_outcomes
            if outcome.status == "success"
        }
        for key, scoring_input in self.scoring_inputs.items():
            if key not in success_names:
                raise ValueError(
                    f"scoring_inputs 键 {key!r} 不是 status=success 的 outcome"
                )
            if scoring_input.algorithm_name != key:
                raise ValueError(
                    f"scoring_inputs[{key!r}].algorithm_name="
                    f"{scoring_input.algorithm_name!r} 与键名不一致"
                )
            if scoring_input.report_id != self.report_id:
                raise ValueError(
                    f"scoring_inputs[{key!r}].report_id="
                    f"{scoring_input.report_id!r} 与顶层 report_id 不一致"
                )
        return self
