"""规范化 JSON 序列化与输入指纹(fingerprint)v1。

依据《十二项总体评分聚合接入方案_20260911.md》r3 §5.2/§5.4:
双方(后端与 worker)对同一待哈希 payload 必须得到逐字节一致的规范表示,
再取完整 SHA256 作为 input_fingerprint / 证据哈希。

固定序列化规则(双方共享,发布即冻结):
- JSON 编码,UTF-8;``ensure_ascii=False``(中文原样,不做 \\u 转义)。
- ``sort_keys=True``:所有 dict 键按 Unicode 码点升序排序,与插入顺序无关。
- ``separators=(",", ":")``:无多余空白,紧凑输出。
- ``allow_nan=False``:NaN/Infinity 不是合法 JSON,遇到直接抛错,绝不静默写坏指纹。
- list 顺序是语义的一部分,由生产方保证稳定;规范化**不重排 list**。
- 调用方负责在构造待哈希 payload 时排除易变字段(短期 URL、查询时间、
  调试耗时等),本模块只保证确定性序列化,不做语义裁剪。

dict 插入顺序不同得到相同字节;list 顺序不同得到不同字节。
"""
import hashlib
import json
from typing import Any

# 写入 SummaryRequest / 持久化记录时标识哈希规则版本;
# 规则一旦发布即冻结,变更须新增 v2。
FINGERPRINT_ALGORITHM = "sha256_canonical_json_v1"


def canonical_json_bytes(payload: Any) -> bytes:
    """把 payload 编码为双方约定的规范 JSON 字节串。

    非 JSON 安全类型(set/bytes/NaN/Infinity 等)直接抛错:
    ``allow_nan=False`` 对 NaN/Infinity 抛 ``ValueError``,
    其余不可序列化类型由 ``json.dumps`` 抛 ``TypeError``,原样透传。
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def fingerprint_v1(payload: Any) -> str:
    """返回 :func:`canonical_json_bytes` 结果的完整 SHA256(64 位小写 hex)。

    使用 FINGERPRINT_ALGORITHM 标识的规则版本;
    相同规范 payload -> 相同 fingerprint,任何字节差异 -> 不同 fingerprint。
    """
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
