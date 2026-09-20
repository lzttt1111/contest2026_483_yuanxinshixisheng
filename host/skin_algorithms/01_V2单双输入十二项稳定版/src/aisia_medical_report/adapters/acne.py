from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from ..io_utils import existing_files, load_json
from ..models import Metric, ModuleResult, RegionResult


REGION_NAMES = {
    "forehead": "额头",
    "glabella": "眉间",
    "nose": "鼻部",
    "subject_left_cheek": "受检者左面颊",
    "subject_right_cheek": "受检者右面颊",
    "left_cheek": "画面左面颊",
    "right_cheek": "画面右面颊",
    "perioral": "口周",
    "chin": "下巴",
    "jaw": "下颌",
    "left_jaw": "画面左下颌",
    "right_jaw": "画面右下颌",
    "unknown": "其他可见区域",
}


def _resolve_image(summary_path: Path, value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = summary_path.parent / path
    return str(path.resolve()) if path.is_file() else None


def adapt_acne(summary_json: str | Path | None, focal_red_metrics: dict[str, Any] | None = None) -> ModuleResult:
    if not summary_json:
        return ModuleResult(
            module_id="06",
            title="毛囊炎症/痤疮样活动",
            status="本次未检测",
            sources=[],
            summary="本次报告未收到同一受检者的痤疮检测结果。",
            limitations=["缺少同图痤疮检测结果"],
        )
    path = Path(summary_json).expanduser().resolve()
    if not path.is_file():
        return ModuleResult(
            module_id="06",
            title="毛囊炎症/痤疮样活动",
            status="检测失败/不可评估",
            sources=[],
            summary=f"痤疮结果文件不存在: {path}",
            limitations=["痤疮结果文件缺失"],
        )

    raw = load_json(path)
    if isinstance(raw.get("overall_metrics"), dict):
        return _adapt_medical_v2(path, raw, focal_red_metrics)
    if isinstance(raw.get("总体指标"), dict):
        return _adapt_review_metrics(path, raw, focal_red_metrics)
    detection = raw.get("detection", {}) if isinstance(raw.get("detection"), dict) else {}
    grading = raw.get("grading", {}) if isinstance(raw.get("grading"), dict) else {}
    preprocess = raw.get("preprocess", {}) if isinstance(raw.get("preprocess"), dict) else {}
    detections = detection.get("detections", []) if isinstance(detection.get("detections"), list) else []
    count = int(detection.get("count", len(detections)) or 0)
    skin_pixels = int(preprocess.get("skin_pixels", 0) or 0)
    density = round(count * 100000.0 / skin_pixels, 6) if skin_pixels > 0 else "不可评估"
    confidences = [float(x.get("confidence", 0.0)) for x in detections if isinstance(x, dict)]
    mean_conf = round(statistics.fmean(confidences), 6) if confidences else 0.0
    p90_conf = _percentile(confidences, 0.9) if confidences else 0.0
    areas = [float(x.get("box_area", 0.0)) for x in detections if isinstance(x, dict)]

    region_counts = detection.get("region_counts", {})
    if not isinstance(region_counts, dict):
        region_counts = {}
    if not region_counts:
        for item in detections:
            if isinstance(item, dict):
                key = str(item.get("region", "unknown"))
                region_counts[key] = int(region_counts.get(key, 0)) + 1

    regions: list[RegionResult] = []
    for key, region_count in sorted(region_counts.items(), key=lambda pair: str(pair[0])):
        regions.append(
            RegionResult(
                name=REGION_NAMES.get(str(key), str(key)),
                status="可评估",
                metrics={"疑似痤疮候选数量（个）": int(region_count)},
            )
        )

    focal = (focal_red_metrics or {}).get("总体指标", {}) if focal_red_metrics else {}
    focal_count = focal.get("局灶红色实例数量（个）", focal.get("特征数量（个）", "不可评估"))
    probabilities = grading.get("severity_probabilities", "不可评估")
    if isinstance(probabilities, list):
        probabilities = {f"等级{i + 1}": round(float(v), 6) for i, v in enumerate(probabilities)}

    image_candidates: list[str] = []
    outputs = raw.get("outputs", {}) if isinstance(raw.get("outputs"), dict) else {}
    detection_outputs = raw.get("detection_outputs", {}) if isinstance(raw.get("detection_outputs"), dict) else {}
    for candidate in (
        detection_outputs.get("circle_image"),
        detection_outputs.get("original_circle_image"),
        outputs.get("final_result"),
        outputs.get("result_image"),
    ):
        resolved = _resolve_image(path, candidate)
        if resolved:
            image_candidates.append(resolved)

    primary_region = max(region_counts, key=lambda k: region_counts[k]) if region_counts else "unknown"
    status = "可评估" if raw.get("status") == "ok" and detection.get("status") == "ok" else "部分支持"
    return ModuleResult(
        module_id="06",
        title="毛囊炎症/痤疮样活动",
        status=status,
        sources=["痤疮YOLO候选检测", "Acne-LDS等级", "DermaVision局灶性红色实例"],
        summary=(
            f"检测到 {count} 个痤疮样特征，主要集中于 {REGION_NAMES.get(str(primary_region), str(primary_region))}；"
            f"Acne-LDS输出等级为 {grading.get('severity_level', '不可评估')}。"
        ),
        metrics=[
            Metric("有效皮肤面积", skin_pixels if skin_pixels else "不可评估", "像素"),
            Metric("痤疮样特征数量", count, "个"),
            Metric("单位面积密度", density, "个/10万有效皮肤像素"),
            Metric("候选框总面积", round(sum(areas), 3), "标准化图像像素²"),
            Metric("P50候选框面积", _percentile(areas, 0.5) if areas else 0.0, "标准化图像像素²"),
            Metric("P90候选框面积", _percentile(areas, 0.9) if areas else 0.0, "标准化图像像素²"),
            Metric("平均工程置信度", mean_conf, "0～1"),
            Metric("P90工程置信度", p90_conf, "0～1"),
            Metric("局灶性红色实例数量", focal_count, "个"),
            Metric("Acne-LDS等级", grading.get("severity_level", "不可评估"), "1～4级"),
            Metric("Acne-LDS等级概率", probabilities),
            Metric("主要集中区域", REGION_NAMES.get(str(primary_region), str(primary_region))),
        ],
        regions=regions,
        images=existing_files(image_candidates),
        daily_note="建议保持温和清洁并避免挤压；若出现持续疼痛、快速增多、明显红肿或脓性改变，应咨询专业人士。",
        limitations=[
            "候选检测不等同于医学确诊或真实痤疮数量",
            "当前模型类别不支持时，不细分丘疹、脓疱等病灶类型",
            "Acne-LDS等级仅为模型参考结果",
        ],
    )


def _adapt_review_metrics(
    path: Path,
    raw: dict[str, Any],
    focal_red_metrics: dict[str, Any] | None,
) -> ModuleResult:
    total = raw.get("总体指标", {})
    rows = [row for row in raw.get("分区指标", []) if isinstance(row, dict)]
    focal = (focal_red_metrics or {}).get("总体指标", {}) if focal_red_metrics else {}
    image = path.parent / "01_痤疮检测结果图.jpg"
    regions = [
        RegionResult(
            name=str(row.get("检测范围", "未命名分区")),
            status=str(row.get("评估状态", "不可评估")),
            metrics={key: value for key, value in row.items() if key not in {"检测范围", "评估状态"}},
        )
        for row in rows
    ]
    count = total.get("疑似痤疮候选数量（个）", 0)
    primary = total.get("主要集中区域", "不可评估")
    return ModuleResult(
        module_id="06",
        title="毛囊炎症/痤疮样活动",
        status=str(total.get("评估状态", "可评估")),
        sources=["痤疮YOLO候选检测", "Acne-LDS等级", "DermaVision局灶性红色实例"],
        summary=f"检测到 {count} 个痤疮样特征，主要集中于 {primary}。",
        metrics=[
            Metric("有效皮肤面积", total.get("有效皮肤面积（像素）", "不可评估"), "像素"),
            Metric("痤疮样特征数量", count, "个"),
            Metric("单位面积密度", total.get("单位面积密度（个/10万有效皮肤像素）", "不可评估"), "个/10万有效皮肤像素"),
            Metric("候选框总面积", total.get("候选框总面积（像素）", "不可评估"), "标准化图像像素²"),
            Metric("候选框面积占比", total.get("候选框面积占比", "不可评估"), "比例"),
            Metric("P50候选框面积", total.get("P50候选框面积（像素）", "不可评估"), "标准化图像像素²"),
            Metric("P90候选框面积", total.get("P90候选框面积（像素）", "不可评估"), "标准化图像像素²"),
            Metric("平均工程置信度", total.get("平均置信度（0～1）", "不可评估"), "0～1"),
            Metric("P90工程置信度", total.get("P90置信度（0～1）", "不可评估"), "0～1"),
            Metric("局灶性红色实例数量", focal.get("局灶红色实例数量（个）", focal.get("特征数量（个）", "不可评估")), "个"),
            Metric("Acne-LDS评估状态", total.get("Acne-LDS评估状态", "不可评估")),
            Metric("Acne-LDS等级", total.get("Acne-LDS等级", "不可评估")),
            Metric("主要集中区域", primary),
        ],
        regions=regions,
        images=existing_files([image]),
        daily_note="建议保持温和清洁并避免挤压；若出现持续疼痛、快速增多、明显红肿或脓性改变，应咨询专业人士。",
        limitations=list(raw.get("医学局限性", [])) or ["候选检测不等同于医学确诊或真实痤疮数量"],
    )


def _medical_sections(value: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if not isinstance(value, dict):
        return output
    for section in (value.get("core_metrics"), value.get("auxiliary_metrics")):
        if not isinstance(section, dict):
            continue
        for group in section.values():
            if isinstance(group, dict):
                output.update(group)
    return output


def _adapt_medical_v2(
    path: Path,
    raw: dict[str, Any],
    focal_red_metrics: dict[str, Any] | None,
) -> ModuleResult:
    total = _medical_sections(raw.get("overall_metrics"))
    count = int(total.get("suspected_acne_candidate_count", 0) or 0)
    primary = total.get("primary_concentration_region", "—")
    focal = (focal_red_metrics or {}).get("总体指标", {}) if focal_red_metrics else {}
    regions: list[RegionResult] = []
    for row in raw.get("region_metrics", []):
        if not isinstance(row, dict):
            continue
        values = _medical_sections(row)
        regions.append(RegionResult(
            name=str(row.get("analysis_region", values.get("analysis_region", "未命名分区"))),
            status=str(row.get("evaluation_status", values.get("evaluation_status", "可评估"))),
            metrics={"痤疮样特征数量（个）": int(values.get("suspected_acne_candidate_count", 0) or 0)},
        ))
    return ModuleResult(
        module_id="06",
        title="毛囊炎症/痤疮样活动",
        status=str(total.get("evaluation_status", "可评估")),
        sources=["痤疮样特征检测", "Acne-LDS等级", "DermaVision局灶性红色实例"],
        summary=f"检测到 {count} 个痤疮样特征，主要集中于 {primary}。",
        metrics=[
            Metric("有效皮肤面积", total.get("valid_skin_area_px", "不可评估"), "像素"),
            Metric("痤疮样特征数量", count, "个"),
            Metric("单位面积密度", total.get("feature_density_per_100k_skin_px", "不可评估"), "个/10万有效皮肤像素"),
            Metric("特征框总面积", total.get("candidate_box_total_area_px", "不可评估"), "标准化图像像素²"),
            Metric("特征框面积占比", total.get("candidate_box_area_ratio", "不可评估"), "比例"),
            Metric("P50特征框面积", total.get("p50_candidate_box_area_px", "不可评估"), "标准化图像像素²"),
            Metric("P90特征框面积", total.get("p90_candidate_box_area_px", "不可评估"), "标准化图像像素²"),
            Metric("平均工程置信度", total.get("mean_confidence", "不可评估"), "0～1"),
            Metric("P90工程置信度", total.get("p90_confidence", "不可评估"), "0～1"),
            Metric("局灶性红色实例数量", focal.get("局灶红色实例数量（个）", focal.get("特征数量（个）", "不可评估")), "个"),
            Metric("Acne-LDS评估状态", total.get("acne_lds_evaluation_status", "不可评估")),
            Metric("Acne-LDS等级", total.get("acne_lds_level", "不可评估")),
            Metric("主要集中区域", primary),
        ],
        regions=regions,
        images=existing_files([path.parent / "01_痤疮检测结果图.jpg"]),
        daily_note="建议保持温和清洁并避免挤压；持续疼痛、快速增多或明显红肿时建议进一步处理。",
        limitations=list(raw.get("medical_limitations", [])),
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 6)
