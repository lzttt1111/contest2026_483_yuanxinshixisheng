"""契约基类与算法元数据。"""
from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """禁止未登记字段,防止 worker 与后端合同静默漂移。"""

    model_config = ConfigDict(extra="forbid")


class AlgorithmMeta(ContractModel):
    """信封 meta_data:算法名 + 结构版本号(字段变动时 bump)。"""

    name: str
    version: str
