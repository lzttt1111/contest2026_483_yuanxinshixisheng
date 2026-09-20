"""acne v1 契约,源自 ai-skin-backend service/algorithm/implements/acne_v1.py。
收紧必填性(对齐 dermavision)+ 加 medical_metrics_v2/medical_report_csv_v2 Optional。"""
from typing import Any

from aisia_contracts.base import ContractModel
from aisia_contracts.envelope import AlgorithmEnvelope

# ── Pydantic models ──────────────────────────────────────────────


class AcnePresence(ContractModel):
    status: str
    message: str


class AcneRegionCount(ContractModel):
    region: str
    count: int


class AcneSeverityLevel(ContractModel):
    等级: int | None
    注释: str


class AcneQuantification(ContractModel):
    疑似痤疮圈选数量: int | None
    痤疮严重程度等级: AcneSeverityLevel


class AcneRawResult(ContractModel):

    # 17 个 OSS key 字段（产物文件）
    acne_summary: str | None = None
    acne_circles: str | None = None
    acne_raw_boxes: str | None = None
    acne_detections: str | None = None
    acne_skin_mask: str | None = None
    acne_forbidden_mask: str | None = None
    acne_standardized: str | None = None
    max_recall_heatmap: str | None = None
    diffuse_erythema_heatmap: str | None = None
    focal_candidate_heatmap: str | None = None
    max_recall_circles: str | None = None
    combined_candidate_circles: str | None = None
    max_recall_debug: str | None = None
    max_recall_json: str | None = None
    original_yolo_circles: str | None = None
    original_unsupervised_circles: str | None = None
    original_combined_circles: str | None = None

    # 结构化数据（前端需要）
    acne_presence: AcnePresence | None = None
    acne_count: int | None = None
    region_counts: list[AcneRegionCount] = []
    grading_status: str | None = None
    grading_reason: str | None = None
    detector_status: str | None = None
    detector_reason: str | None = None
    input_mode: str | None = None
    detection_scope: str | None = None
    量化结果: AcneQuantification
    # medical_metrics_v2 (dermavision worker 已返回,Optional 兼容旧数据)
    medical_metrics_v2: dict[str, Any] | None = None
    medical_report_csv_v2: str | None = None


class _AcneV1Envelope(AlgorithmEnvelope):
    raw_result: AcneRawResult


from aisia_contracts.registry import register

SCHEMA_VERSION = "1"
register("acne", SCHEMA_VERSION, _AcneV1Envelope)