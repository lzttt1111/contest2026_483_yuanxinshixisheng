"""信封: worker 返回的统一结构,成功失败共用,status 区分。

设计:
- 一个 AlgorithmEnvelope 基类,所有字段 Optional(除 status)。
- 成功填 meta_data/raw_result/debug_info,失败填 error_*。
- model_validator 强制两套字段集互斥且各自必填。
- raw_result 不在基类声明;各算法版本子类声明 raw_result: XXXRawResult 强类型。
- 失败信封用基类校验(无 raw_result 字段,extra="forbid" 兜底)。
"""
from typing import Any

from pydantic import ConfigDict, model_validator

from aisia_contracts.base import AlgorithmMeta, ContractModel

_SUCCESS_FIELDS = ("meta_data", "debug_info")
_FAILURE_FIELDS = ("error_message", "error_code", "error_details")


class AlgorithmEnvelope(ContractModel):
    """统一信封结构。

    子类通过 ``raw_result: XXXRawResult`` 声明该算法该版本的 raw_result 强类型。
    validator 保证:
    - status="success": meta_data/raw_result/debug_info 必填,error_* 禁止
    - status="failed": error_message 必填,meta_data/raw_result/debug_info 禁止
    """

    model_config = ConfigDict(extra="forbid")
    record_id: str | None = None  # worker 返回的追踪字段
    status: str  # "success" / "failed"
    schema_version: str  # 数据格式版本(写死在 contract v{N}.py 的 SCHEMA_VERSION),worker 接到任务即填,落库单独列;与 meta_data.version(算法 git 版本)分开

    # 成功字段(成功时必填,失败时禁止)
    meta_data: AlgorithmMeta | None = None
    debug_info: dict[str, Any] | None = None
    # raw_result 不在此声明,由子类声明为 XXXRawResult 强类型

    # 失败字段(失败时 error_message 必填,成功时禁止)
    error_message: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_status_fields(self) -> "AlgorithmEnvelope":
        raw_result = getattr(self, "raw_result", None)
        if self.status == "success":
            missing = [f for f in _SUCCESS_FIELDS if getattr(self, f) is None]
            if raw_result is None:
                missing.append("raw_result")
            if missing:
                raise ValueError(f"success 信封缺少必填字段: {missing}")
            forbidden = [f for f in _FAILURE_FIELDS if getattr(self, f) is not None]
            if forbidden:
                raise ValueError(f"success 信封不应含失败字段: {forbidden}")
        elif self.status == "failed":
            if self.error_message is None:
                raise ValueError("failed 信封必须有 error_message")
            forbidden = [f for f in _SUCCESS_FIELDS if getattr(self, f) is not None]
            if raw_result is not None:
                forbidden.append("raw_result")
            if forbidden:
                raise ValueError(f"failed 信封不应含成功字段: {forbidden}")
        else:
            raise ValueError(f"未知 status: {self.status!r}(应为 'success' 或 'failed')")
        return self
