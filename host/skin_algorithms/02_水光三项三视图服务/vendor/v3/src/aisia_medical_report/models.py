from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Metric:
    name: str
    value: Any
    unit: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegionResult:
    name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "分区名称": self.name,
            "评估状态": self.status,
            "指标": sanitize_json(self.metrics),
            "说明": self.summary,
        }


@dataclass
class ModuleResult:
    module_id: str
    title: str
    status: str
    sources: list[str]
    summary: str
    metrics: list[Metric] = field(default_factory=list)
    regions: list[RegionResult] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    daily_note: str = ""
    limitations: list[str] = field(default_factory=list)
    scoring: float | None = None
    evidence_status: str = "SHADOW"
    module_status: str = "PARTIAL"
    report_eligibility: str = "INTERNAL_ONLY"
    evidence_name: str = ""
    missing_evidence: list[str] = field(default_factory=list)
    medical_construct_satisfied: bool = False
    is_2d_supplement: bool = False

    def _display_status(self) -> str:
        return {
            "NOT_ASSESSABLE": "不可评估（证据不足）",
            "PARTIAL": "部分可用（完整V2证据不足）",
            "ENGINEERING_READY": "工程就绪（未医学通过）",
            "MEDICAL_READY": "医学就绪",
        }.get(self.module_status, "状态未定义")

    def to_dict(self) -> dict[str, Any]:
        display_status = self._display_status()
        return {
            "模块编号": self.module_id,
            "模块名称": self.title,
            "可用状态": self.status,
            "评估状态": display_status,
            "用户评估状态": display_status,
            "医生评估状态": display_status,
            "证据状态": self.evidence_status,
            "模块状态": self.module_status,
            "报告资格": self.report_eligibility,
            "证据名称": self.evidence_name,
            "缺失证据": self.missing_evidence,
            "医学构念满足": self.medical_construct_satisfied,
            "二维补充证据": self.is_2d_supplement,
            "数据来源": self.sources,
            "综合得分": self.scoring,
            "程度等级": None,
            "结果摘要": self.summary,
            "核心指标": [m.to_dict() for m in self.metrics],
            "分区指标": [r.to_dict() for r in self.regions],
            "结果图": self.images,
            "日常管理提示": self.daily_note,
            "医学局限性": self.limitations,
        }


def sanitize_json(value: Any) -> Any:
    """Return a deterministic JSON-safe value without NaN/Infinity."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "不可评估"
        return round(value, 6)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_json(v) for v in value]
    if hasattr(value, "item"):
        return sanitize_json(value.item())
    return str(value)
