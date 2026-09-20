"""从单算法运行时结果构造版本化 ``scoring_input_v1``。

- ``evidence`` 只镜像冻结字段清单命中的 ``detector_results`` 子集。
- 必需叶子缺失诚实产出 ``partial``/``missing``，绝不补零。
- 医学 V2 旁路以“生成成功即纳入”为准，不依赖上传成功。
- ``input_quality_gate`` 由 worker 通过 ``compute_input_quality_gate`` 真实调用
  V0.1.1 ``evaluate_image_path`` 得到；不可计算时保持 ``None``，交聚合侧降级。
"""

from __future__ import annotations

import copy
import re
from functools import lru_cache
from typing import Any, Mapping

from aisia_contracts.scoring_input.v1 import (
    SCORING_INPUT_SCHEMA_VERSION,
    ScoringInputV1,
)

from src.summary_scoring.assets import FieldList, load_field_list
from src.summary_scoring.completeness import algorithm_missing

FIELD_LIST_SHA256 = "e4f4cc694ba48df6fe63e404ed53a6ff416deb900660958076da4c151c0190ff"
FAST_FIVE = frozenset({"redness", "spots", "brown", "texture", "pores"})

_MISSING = object()
_END = "__end__"
_WILDCARD = "__wildcard__"
_SEGMENT_RE = re.compile(r"^([^\[\]]*)((?:\[\d*\])*)$")
_SELECTOR_RE = re.compile(r"\[(\d*)\]")


class ScoringInputBuildError(Exception):
    """Raised when scoring_input cannot be built honestly."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


@lru_cache(maxsize=1)
def load_pinned_field_list() -> FieldList:
    field_list = load_field_list()
    if field_list.source_sha256 != FIELD_LIST_SHA256:
        raise ScoringInputBuildError(
            "scoring_input field list SHA drifted: "
            f"{field_list.source_sha256} != {FIELD_LIST_SHA256}"
        )
    return field_list


def _segments(path: str) -> list[Any]:
    segments: list[Any] = []
    for part in path.split("."):
        match = _SEGMENT_RE.match(part)
        if match is None:
            segments.append(part)
            continue
        if match.group(1):
            segments.append(match.group(1))
        for selector in _SELECTOR_RE.findall(match.group(2)):
            segments.append(_WILDCARD if selector == "" else int(selector))
    return segments


def _trie(paths: list[str]) -> dict[Any, Any]:
    root: dict[Any, Any] = {}
    for path in paths:
        node = root
        for segment in _segments(path):
            node = node.setdefault(segment, {})
        node[_END] = True
    return root


def _prune(node: Any, trie: Mapping[Any, Any]) -> Any:
    if _END in trie:
        return copy.deepcopy(node)
    if isinstance(node, Mapping):
        projected: dict[str, Any] = {}
        for key, child in trie.items():
            if key is _END or key is _WILDCARD or isinstance(key, int):
                continue
            if key in node:
                value = _prune(node[key], child)
                if value is not _MISSING:
                    projected[key] = value
        return projected or _MISSING
    if isinstance(node, list):
        wildcard = trie.get(_WILDCARD)
        indices = [(key, child) for key, child in trie.items() if isinstance(key, int)]
        if wildcard is not None:
            projected_list = []
            for item in node:
                value = _prune(item, wildcard)
                projected_list.append(None if value is _MISSING else value)
            return projected_list or _MISSING
        if indices:
            size = max(index for index, _ in indices) + 1
            projected_list = [None] * size
            found = False
            for index, child in indices:
                if 0 <= index < len(node):
                    value = _prune(node[index], child)
                    if value is not _MISSING:
                        projected_list[index] = value
                        found = True
            return projected_list if found else _MISSING
    return _MISSING


def extract_evidence(
    algorithm: str,
    detector_source: Mapping[str, Any],
    field_list: FieldList,
) -> dict[str, Any]:
    """把 detector_results 子集裁剪成冻结字段清单内的 evidence。"""

    fields = field_list.algorithms.get(algorithm, ())
    evidence: dict[str, Any] = {}
    for detector in field_list.algorithm_detectors.get(algorithm, ()):
        node = detector_source.get(detector)
        if not isinstance(node, Mapping):
            continue
        prefix = f"{detector}."
        relative = [field.path[len(prefix):] for field in fields if field.path.startswith(prefix)]
        if not relative:
            continue
        projected = _prune(node, _trie(relative))
        if projected is not _MISSING:
            evidence[detector] = projected
    return evidence


def _node(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"metrics": dict(metrics)}


def _runtime_detector_source(
    algorithm: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """把 worker 运行时产物收敛成 detector_results 语系的节点。"""

    if algorithm == "purple":
        full = runtime.get("full_metrics")
        if not isinstance(full, Mapping):
            return {}
        return {
            detector: {"metrics": {detector: full[detector]}}
            for detector in ("uv_spots", "porphyrin")
            if isinstance(full.get(detector), Mapping)
        }
    if algorithm == "acne":
        summary = runtime.get("summary")
        if not isinstance(summary, Mapping):
            return {}
        detection = summary.get("detection")
        metrics: dict[str, Any] = {}
        if isinstance(detection, Mapping):
            summary_detection = {
                key: detection[key] for key in ("count", "filtered_count") if key in detection
            }
            detections = detection.get("detections")
            if isinstance(detections, list):
                summary_detection["detections"] = [
                    {key: item[key] for key in ("box_area", "confidence") if key in item}
                    for item in detections
                    if isinstance(item, Mapping)
                ]
            metrics["detection"] = summary_detection
        preprocess = summary.get("preprocess")
        if isinstance(preprocess, Mapping) and "skin_pixels" in preprocess:
            metrics["preprocess"] = {"skin_pixels": preprocess["skin_pixels"]}
        node: dict[str, Any] = _node(metrics)
        medical = runtime.get("medical_metrics_v2")
        if isinstance(medical, Mapping):
            node["public_metrics"] = medical
        return {"acne": node}
    if algorithm == "wrinkle":
        summary = runtime.get("summary")
        return {"wrinkle": _node(summary)} if isinstance(summary, Mapping) else {}
    if algorithm in {"surface_gloss", "vascular", "contour_firmness"}:
        full = runtime.get("full_metrics")
        return {algorithm: _node(full)} if isinstance(full, Mapping) else {}
    metrics = runtime.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    source = dict(metrics)
    medical = runtime.get("medical_metrics_v2")
    if isinstance(medical, Mapping):
        source["medical_metrics_v2"] = medical
    return {algorithm: _node(source)}


def _missing_reason(
    algorithm: str,
    runtime: Mapping[str, Any],
    missing: list[str],
) -> str:
    if algorithm in FAST_FIVE and runtime.get("medical_metrics_v2") is None:
        if any("medical_metrics_v2" in path for path in missing):
            return "旁路失败"
    return "字段缺失"


def _normalize_gate(gate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(gate, Mapping):
        return None
    status = gate.get("status")
    if status is None:
        return None
    reasons = gate.get("reason_codes") or []
    return {
        "status": str(status),
        "reason_codes": [str(code) for code in reasons],
    }


def build_scoring_input(
    *,
    algorithm_name: str,
    detection_schema_version: str,
    detection_impl_version: str,
    report_id: str,
    detection_attempt_id: str,
    source_image_sha256: str,
    capture_profile: str,
    input_route: str = "institution_queue",
    runtime: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
    input_quality_gate: Mapping[str, Any] | None = None,
    field_list: FieldList | None = None,
) -> dict[str, Any]:
    """构造并校验一条 ``ScoringInputV1``，返回可 JSON 序列化的 dict。"""

    pinned = field_list if field_list is not None else load_pinned_field_list()
    runtime = runtime if isinstance(runtime, Mapping) else {}
    # Institution providers already produce the same detector nodes as the local
    # finalizer. Preserve that evidence instead of reading the compact cloud view.
    direct_source = runtime.get("detector_results")
    detector_source = (
        copy.deepcopy(dict(direct_source))
        if isinstance(direct_source, Mapping)
        else _runtime_detector_source(algorithm_name, runtime)
    )
    from src.aisia_medical_report.legacy_word_features import local_legacy_features
    if algorithm_name in FAST_FIVE and algorithm_name in detector_source:
        node = detector_source[algorithm_name]
        public = node.get("public_metrics", node.get("metrics"))
        if isinstance(public, Mapping):
            node["word_features"] = local_legacy_features(
                algorithm_name, public, capture_profile=capture_profile)
    elif algorithm_name == "purple":
        for detector in ("uv_spots", "porphyrin"):
            node = detector_source.get(detector)
            if not isinstance(node, dict):
                continue
            if capture_profile == "institution":
                node["word_features"] = local_legacy_features(
                    detector, node.get("metrics", {}), capture_profile=capture_profile,
                    uv_document=node.get("metrics") if detector == "uv_spots" else None)
            elif isinstance(runtime.get("metrics"), Mapping):
                public = {key: value for key, value in runtime["metrics"].items()
                          if key.startswith(detector + "_")}
                full = runtime.get("full_metrics") or {}
                node["word_features"] = local_legacy_features(
                    detector, public, capture_profile=capture_profile,
                    uv_document=full.get(detector) if detector == "uv_spots" else None)
    evidence = extract_evidence(algorithm_name, detector_source, pinned)
    missing = algorithm_missing(pinned, algorithm_name, evidence, capture_profile)
    if not detector_source:
        status = "missing"
        missing_fields: list[str] = []
        missing_reason: str | None = "算法证据缺失"
    elif missing:
        status = "partial"
        missing_fields = missing
        missing_reason = _missing_reason(algorithm_name, runtime, missing)
    else:
        status = "present"
        missing_fields = []
        missing_reason = None

    truthy_quality = dict(quality or {})
    flags = truthy_quality.get("quality_flags") or truthy_quality.get(
        "algorithm_quality_flags"
    )
    document = ScoringInputV1(
        schema_version=SCORING_INPUT_SCHEMA_VERSION,
        algorithm_name=str(algorithm_name),
        detection_schema_version=str(detection_schema_version),
        detection_impl_version=str(detection_impl_version),
        report_id=str(report_id),
        detection_attempt_id=str(detection_attempt_id),
        source_image_sha256=str(source_image_sha256),
        capture_profile=str(capture_profile),
        input_route=str(input_route),
        evidence_status=status,
        evidence_field_list_sha256=pinned.source_sha256,
        quality={
            "algorithm_quality_status": truthy_quality.get("quality_status")
            or truthy_quality.get("algorithm_quality_status"),
            "algorithm_quality_flags": [str(flag) for flag in (flags or [])],
            "input_quality_gate": _normalize_gate(input_quality_gate),
        },
        evidence=evidence,
        missing_fields=missing_fields,
        missing_reason=missing_reason,
    )
    return document.model_dump(mode="json")
