"""scoring_input v1 契约:版本化持久评分输入容器。

依据《十二项总体评分聚合接入方案_20260911.md》r3 §5.2/§5.4 与
《十二项评分输入字段证据矩阵_20260912.md》§4/§5。

关键决策(阶段 A 已冻结):
- 存储 = inline:评分输入随算法新版本 raw_result 一起持久化(第三段起);
  未来若改文件引用模式,必须新建 request/容器版本,不得原地改语义。
- evidence = 镜像 ``detector_results`` 子集的通用 dict:
  ``{detector_key: {"metrics": {...}}}``(purple 含 uv_spots + porphyrin 两键)。
  契约只定义容器;字段清单由 dermavision 侧冻结并用
  ``evidence_field_list_sha256`` 钉住,深度字段覆盖校验在聚合入口执行。
- ``evidence_status`` 门禁:missing/failed 不得携带 evidence,
  聚合侧据此拒绝把"未检测/检测失败"当成 0 参与评分。

发布即冻结铁律:已发布的 schema_version 不可修改,字段变动必须新建 v2。
"""
from typing import Any, Literal

from pydantic import Field, model_validator

from aisia_contracts.base import ContractModel

SCORING_INPUT_SCHEMA_VERSION = "scoring_input_v1"


class InputQualityGate(ContractModel):
    """原图派生输入门禁结果(如清晰度/光照等原图质量门)。"""

    status: str
    reason_codes: list[str] = Field(default_factory=list)


class ScoringInputQuality(ContractModel):
    """区分"算法输入质量"与"原图派生门禁"两类质量信息。

    云端普通检测没有原图派生的 ``input_quality_gate`` 时,必须诚实置 ``None``,
    由 V0.1.1 侧自行降级为 REVIEW;不得伪造一个假的 PASS/门禁结果。
    """

    algorithm_quality_status: str | None = None
    algorithm_quality_flags: list[str] = Field(default_factory=list)
    input_quality_gate: InputQualityGate | None = None


class ScoringInputV1(ContractModel):
    """单个算法一次检测的版本化完整评分输入。

    ``evidence`` 形状为 ``{detector_key: {"metrics": {...}}}``,与聚合侧
    ``detector_results`` 同一寻址语系(如 purple 需同时含 uv_spots 与 porphyrin)。
    ``evidence_status`` 与 evidence/missing_fields 必须一致:

    - ``missing`` / ``failed`` -> evidence 必须为空 dict;
    - ``partial`` -> ``missing_fields`` 必须非空(列出缺失的规范路径);
    - ``present`` -> ``missing_fields`` 必须为空且 evidence 非空。

    注意:合法零目标(如 0 个痤疮、0 条皱纹)是 evidence 内真实存在的 ``0`` 值,
    不是 missing;契约不做值语义判断,深度字段清单覆盖校验由聚合入口执行。
    """

    schema_version: Literal[SCORING_INPUT_SCHEMA_VERSION]
    algorithm_name: str
    detection_schema_version: str
    detection_impl_version: str
    report_id: str
    detection_attempt_id: str
    source_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capture_profile: str
    input_route: str
    evidence_status: Literal["present", "partial", "missing", "failed"]
    evidence_field_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality: ScoringInputQuality
    evidence: dict[str, Any]
    missing_fields: list[str] = Field(default_factory=list)
    missing_reason: str | None = None

    @model_validator(mode="after")
    def _check_evidence_consistency(self) -> "ScoringInputV1":
        """强制 evidence_status 与 evidence / missing_fields 一致。"""
        if self.evidence_status in ("missing", "failed"):
            if self.evidence:
                raise ValueError(
                    f"evidence_status={self.evidence_status} 要求 evidence 为空 dict"
                )
        elif self.evidence_status == "partial":
            if not self.missing_fields:
                raise ValueError(
                    "evidence_status=partial 要求 missing_fields 非空"
                )
        else:  # present
            if self.missing_fields:
                raise ValueError(
                    "evidence_status=present 要求 missing_fields 为空"
                )
            if not self.evidence:
                raise ValueError(
                    "evidence_status=present 要求 evidence 非空"
                )
        return self
