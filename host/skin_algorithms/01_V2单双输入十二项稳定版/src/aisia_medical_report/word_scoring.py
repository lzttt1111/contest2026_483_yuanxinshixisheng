from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.scoring_bridge.population_artifacts import NumericFeaturesDocument
from src.scoring_bridge.population_profile import (
    PopulationProfile,
    empirical_burden_percentile,
)
from src.scoring_bridge.population_runtime import score_population_modules
from src.scoring_bridge.population_serialization import population_profile_from_document
from src.scoring_bridge.word_acne_2d import (
    Acne2DReferences,
    acne_2d_reference_from_document,
    extract_acne_2d_metrics,
    score_acne_2d,
)
from src.scoring_bridge.word_wrinkle_2d import (
    Wrinkle2DReferences,
    extract_wrinkle_2d_metrics,
    score_wrinkle_2d,
    wrinkle_2d_references_from_document,
)
from src.aisia_medical_report.institution_word_features import (
    project_institution_word_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORD_PROFILE_PATH = (
    PROJECT_ROOT
    / "00_runtime_assets/scoring_word_provisional/word_population_reference_1000.json"
)
WORD_PROFILE_SHA256 = "10275a196372fa3af26605636fc3db6284017f5b1810ab6e5e2d498537fd7bb4"


MODULE_SCORE_IDS = {
    "01": "pores",
    "02": "oil_tendency",
    "03": "pigmentation",
    "04": "diffuse_redness",
    "05": "vascular",
    "06": "acne_activity",
    "07": "dry_fine_lines",
    "08": "stable_wrinkles",
    "09": "structural_grooves",
    "10": "smoothness",
    "11": "contour_firmness",
}
LEGACY_MODULE_IDS = frozenset({"01", "03", "04", "10"})
POPULATION_DRIVERS = {
    "oil_tendency": ["油光覆盖和强度", "油光连续性", "紫质密度和面积"],
    "vascular": ["线状结构数量和长度", "覆盖和宽度", "红色信号和网络结构"],
    "dry_fine_lines": ["细纹密度和长度", "细纹覆盖范围", "纹理方向和视觉对比"],
    "contour_firmness": ["中面部连续性", "沟槽负担", "下颌缘连续性和下脸部负担"],
}


def _load_pinned_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = document.get("profile_sha256")
    body = {
        key: value for key, value in document.items()
        if key != "profile_sha256"
    }
    actual = hashlib.sha256(json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    if expected != actual or actual != WORD_PROFILE_SHA256:
        raise ValueError("Word scoring profile SHA mismatch")
    return document


def load_word_population_profile(
    path: Path = WORD_PROFILE_PATH,
    *,
    capture_profile: str = "consumer",
) -> PopulationProfile:
    document = _load_pinned_document(path)
    profile_document = (
        document.get("institution_profile")
        if capture_profile == "institution"
        else document
    )
    if not isinstance(profile_document, dict):
        raise ValueError("Word scoring profile route is missing")
    expected_id = (
        "word_population_reference_1000_institution_alias"
        if capture_profile == "institution"
        else "word_population_reference_1000"
    )
    if profile_document.get("profile_id") != expected_id:
        raise ValueError("Word scoring profile identity mismatch")
    return population_profile_from_document({
        key: profile_document[key]
        for key in ("profile_id", "status", "development_count", "modules")
    })


def load_word_wrinkle_2d_references(
    path: Path = WORD_PROFILE_PATH,
) -> dict[str, Wrinkle2DReferences]:
    document = _load_pinned_document(path)
    wrinkle_document = document.get("wrinkle_2d_profile")
    if not isinstance(wrinkle_document, dict):
        raise ValueError("Word wrinkle 2D profile is missing")
    return wrinkle_2d_references_from_document(wrinkle_document)


def load_word_acne_2d_reference(
    path: Path = WORD_PROFILE_PATH,
) -> Acne2DReferences:
    document = _load_pinned_document(path)
    acne_document = document.get("acne_2d_profile")
    if not isinstance(acne_document, dict):
        raise ValueError("Word acne 2D profile is missing")
    return acne_2d_reference_from_document(acne_document)


def _grade(score: float) -> str:
    if score <= 25.0:
        return "未见明显"
    if score <= 45.0:
        return "轻度"
    if score <= 65.0:
        return "中度"
    if score <= 90.0:
        return "较明显"
    return "显著"


def conservative_word_score(score: float, *, allow_high: bool = True) -> float:
    """Compress provisional new-module percentiles for consumer-facing Word."""

    if score <= 0.0:
        return 0.0
    mapped = min(79.0, 5.0 + 0.72 * score)
    return mapped if allow_high else min(mapped, 64.9)


def has_population_evidence_consensus(
    features: dict[str, dict[str, dict[str, int | float | None]]],
    profile: PopulationProfile,
    module_id: str,
) -> bool:
    module = profile.modules.get(module_id)
    if module is None or module.status != "candidate":
        return False
    group_scores: list[float] = []
    for group in module.groups.values():
        group_score = 0.0
        for metric in group.metrics.values():
            if not metric.usable:
                continue
            value = features.get(module_id, {}).get(group.group_id, {}).get(metric.spec.metric_id)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            group_score += empirical_burden_percentile(
                float(value),
                metric.reference,
                direction=metric.spec.direction,
                zero_inflated=metric.zero_inflated,
            ) * metric.effective_weight
        group_scores.append(group_score)
    high_count = sum(score >= 75.0 for score in group_scores)
    return (
        len(group_scores) >= 2
        and high_count >= 2
        and high_count / len(group_scores) >= 0.60
    )


def _modules(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = payload.get("检测模块")
    if not isinstance(raw, list):
        raise ValueError("formal report modules are missing")
    return {
        str(module.get("模块编号")): module
        for module in raw
        if isinstance(module, dict)
    }


def _mark_displayed(
    module: dict[str, Any],
    *,
    score: float,
    profile_label: str,
    drivers: list[str] | None = None,
) -> None:
    evidence = list(drivers or [])
    driver_text = "、".join(evidence)
    module["综合得分"] = round(score, 2)
    module["程度等级"] = _grade(score)
    module["score_valid"] = True
    module["完整评分值"] = round(score, 6)
    module["Word评分展示"] = True
    module["评分Profile"] = profile_label
    module["评分状态说明"] = "参考评分已生成"
    module["正式评分说明"] = {
        "summary_text": (
            "本项分数依据参考样本分布中的相对位置计算；"
            "展示时采用保守区间，分数越高表示当前图像中相关表现越明显。"
            + (f"主要核对证据为{driver_text}。" if driver_text else "")
        ),
        "drivers": [],
    }


def _mark_wrinkle_2d_displayed(
    module: dict[str, Any],
    *,
    score: float,
    module_id: str,
) -> None:
    _mark_displayed(
        module,
        score=score,
        profile_label="二维可见纹路参考评分",
    )
    name = "稳定性线性皱纹" if module_id == "08" else "结构性沟纹"
    module["正式评分说明"] = {
        "summary_text": (
            f"{name}分数由当前结果中的可见线段数量（45%）、"
            "中心线总长度（35%）和最长单段（20%）综合得到；"
            "展示时采用保守区间，分数越高表示二维图像中可见纹路负担越明显。"
        ),
        "drivers": [],
    }


def _mark_acne_2d_displayed(module: dict[str, Any], score: float) -> None:
    _mark_displayed(
        module,
        score=score,
        profile_label="二维可见痤疮样特征参考评分",
        drivers=["候选数量", "单位面积密度", "候选面积占比", "候选置信度"],
    )


def _mark_legacy_displayed(module: dict[str, Any], score: float) -> None:
    if score > 90.0:
        module["程度等级"] = "显著"
    elif module.get("程度等级") == "显著":
        module["程度等级"] = "较明显"
    module["score_valid"] = True
    module["完整评分值"] = score
    module["Word评分展示"] = True
    module["评分Profile"] = "历史参考分布评分"
    module["评分状态说明"] = "当前结果评分已生成"


def _mark_unavailable(module: dict[str, Any]) -> None:
    module["综合得分"] = None
    module["程度等级"] = None
    module["score_valid"] = False
    module["完整评分值"] = None
    module["Word评分展示"] = False
    module["评分Profile"] = None
    module["评分状态说明"] = "评分证据不足"
    module["正式评分说明"] = None


def apply_word_scores(
    payload: dict[str, Any],
    complete: dict[str, Any],
    population_profile: PopulationProfile | None,
    wrinkle_2d_references: dict[str, Wrinkle2DReferences] | None = None,
    acne_2d_reference: Acne2DReferences | None = None,
) -> None:
    """Preserve official old scores and add Word-only population scores."""

    modules = _modules(payload)
    capture_profile = str(
        (complete.get("provenance") or {}).get("capture_profile") or "consumer"
    )
    raw_features = (
        project_institution_word_features(complete)
        if capture_profile == "institution"
        else complete.get("scoring_features") or {}
    )
    features = NumericFeaturesDocument.model_validate(raw_features).root
    population_scores = (
        score_population_modules(features=features, profile=population_profile)
        if population_profile is not None
        else {}
    )
    wrinkle_scores: dict[str, float] = {}
    wrinkle_metrics = (
        ((complete.get("detector_results") or {}).get("wrinkle") or {})
        .get("metrics")
    )
    if wrinkle_2d_references is not None and isinstance(wrinkle_metrics, dict):
        for module_id, scoring_id in (
            ("08", "stable_wrinkles"),
            ("09", "structural_grooves"),
        ):
            reference = wrinkle_2d_references.get(scoring_id)
            if reference is None:
                continue
            metrics = extract_wrinkle_2d_metrics(wrinkle_metrics, scoring_id)
            wrinkle_scores[module_id] = score_wrinkle_2d(metrics, reference)
    acne_score: float | None = None
    acne_result = (complete.get("detector_results") or {}).get("acne")
    if acne_2d_reference is not None and isinstance(acne_result, dict):
        acne_score = score_acne_2d(
            extract_acne_2d_metrics(acne_result),
            acne_2d_reference,
        )
    displayed: list[str] = []
    for module_id, scoring_id in MODULE_SCORE_IDS.items():
        module = modules[module_id]
        current = module.get("综合得分")
        if module_id in LEGACY_MODULE_IDS and isinstance(current, (int, float)):
            _mark_legacy_displayed(module, float(current))
            displayed.append(module_id)
            continue
        if module_id in wrinkle_scores:
            _mark_wrinkle_2d_displayed(
                module,
                score=conservative_word_score(wrinkle_scores[module_id]),
                module_id=module_id,
            )
            displayed.append(module_id)
            continue
        if module_id == "06" and acne_score is not None:
            _mark_acne_2d_displayed(module, conservative_word_score(acne_score))
            displayed.append(module_id)
            continue
        population = population_scores.get(scoring_id)
        if (
            population is not None
            and population.score_valid
            and population.score is not None
        ):
            population_score = float(population.score)
            if (
                scoring_id == "acne_activity"
                and isinstance(acne_result, dict)
                and extract_acne_2d_metrics(acne_result).candidate_count == 0.0
            ):
                population_score = 0.0
            _mark_displayed(
                module,
                score=conservative_word_score(
                    population_score,
                    allow_high=(
                        scoring_id not in {"vascular", "contour_firmness"}
                        and
                        population_profile is not None
                        and has_population_evidence_consensus(
                            features, population_profile, scoring_id
                        )
                    ),
                ),
                profile_label="参考人群分布评分",
                drivers=POPULATION_DRIVERS.get(scoring_id),
            )
            displayed.append(module_id)
        else:
            _mark_unavailable(module)
    info = payload.setdefault("报告信息", {})
    info["评分模式"] = "参考人群相对评分"
    info["评分配置版本"] = "当前参考人群分布"
    info["综合得分"] = None
    info["程度等级"] = None
    info["Word评分展示范围"] = "/".join(displayed)
    payload.pop("评分配置摘要", None)


__all__ = [
    "apply_word_scores",
    "load_word_population_profile",
    "load_word_wrinkle_2d_references",
    "load_word_acne_2d_reference",
    "conservative_word_score",
    "has_population_evidence_consensus",
]
