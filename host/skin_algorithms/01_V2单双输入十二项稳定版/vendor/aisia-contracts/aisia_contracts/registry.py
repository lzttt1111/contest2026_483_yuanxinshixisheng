"""算法版本注册表。

单一事实来源:每个 (algo, schema_version) 注册一个 envelope 子类。
worker 构造信封时填 schema_version(从 contract v{N}.py 的 SCHEMA_VERSION 常量取);
backend 收到后读顶层 schema_version -> get_envelope 找 pydantic model 校验。
schema_version(数据格式)与 meta_data.version(算法 git 版本)分开,
算法代码变但格式没变时 schema_version 不变。
"""
from aisia_contracts.envelope import AlgorithmEnvelope

# (algo, schema_version) -> envelope 子类(声明了 raw_result 强类型)
REGISTRY: dict[tuple[str, str], type[AlgorithmEnvelope]] = {}


def register(algo: str, schema_version: str, envelope_cls: type[AlgorithmEnvelope]) -> None:
    """注册某算法某 schema 版本的信封模型。重复注册抛错。"""
    key = (algo, schema_version)
    if key in REGISTRY:
        raise ValueError(f"契约重复注册: {algo} schema_version={schema_version}")
    REGISTRY[key] = envelope_cls


def get_envelope(algo: str, schema_version: str) -> type[AlgorithmEnvelope]:
    """按算法 + schema_version 取信封模型;未注册抛 KeyError。"""
    return REGISTRY[(algo, schema_version)]
