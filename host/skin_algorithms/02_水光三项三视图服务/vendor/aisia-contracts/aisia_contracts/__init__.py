"""aisia-contracts: dermavision worker 与 ai-skin-backend 共享的 pydantic 契约。

单一事实来源:每个算法每个 schema 版本一个 RawResult model,发布即冻结;
数据格式变动 -> 新增版本 -> bump SCHEMA_VERSION。两边都 import 本包,
不再各自维护 pydantic,杜绝 drift。

schema_version(数据格式版本,用于找 pydantic model)与
meta_data.version(算法 git 版本,信息性)分开 -- 算法代码变但格式没变时
schema_version 不变。

校验逻辑(validate)不在此包:只有 ai-skin-backend 校验,
流程: 读顶层 schema_version -> get_envelope(algo, schema_version) -> model(**json)。
"""
from aisia_contracts.base import AlgorithmMeta, ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope
from aisia_contracts.registry import (
    REGISTRY,
    get_envelope,
    register,
)

# 导入各算法版本,触发注册
from aisia_contracts import algorithms  # noqa: F401

__all__ = [
    "AlgorithmMeta",
    "AlgorithmEnvelope",
    "ContractModel",
    "REGISTRY",
    "get_envelope",
    "register",
]
