from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..io_utils import existing_files, load_json, value
from ..models import Metric, ModuleResult, RegionResult


def _resolve(root: Path, *candidates: str) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    return root / candidates[0]


def _load_metrics(root: Path, *candidates: str) -> dict[str, Any] | None:
    path = _resolve(root, *candidates)
    return load_json(path) if path.is_file() else None


def _csv_value(raw: str | None) -> Any:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _load_medical_csv(
    root: Path,
    *candidates: str,
    project: str | None = None,
) -> dict[str, Any] | None:
    """将医学V2宽表还原为报告适配器的总体/分区结构。

    云端精简JSON继续保持原合同；医生报告只在本层读取独立详细CSV。
    """
    path = _resolve(root, *candidates)
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for source in csv.DictReader(handle):
            if project is not None and source.get("检测项目") != project:
                continue
            row: dict[str, Any] = {}
            # 核心列在医生宽表中排在辅助列之前；同义列优先保留核心口径。
            for key, raw in source.items():
                normalized = key.removeprefix("核心-").removeprefix("辅助-")
                if normalized == "检测项目":
                    continue
                value_item = _csv_value(raw)
                if normalized not in row or row[normalized] in (None, ""):
                    row[normalized] = value_item
            rows.append(row)
    if not rows:
        return None
    overall = next((row for row in rows if row.get("检测范围") == "全面部"), rows[0])
    regions = [row for row in rows if row is not overall]
    return {"总体指标": overall, "分区指标": regions, "来源文件": path.name}


def _regions(raw: dict[str, Any] | None, keep: list[str]) -> list[RegionResult]:
    if not raw:
        return []
    output: list[RegionResult] = []
    for row in raw.get("分区指标", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("检测范围", "未命名分区"))
        status = str(row.get("评估状态", "不可评估"))
        metrics = {key: row.get(key, "不可评估") for key in keep}
        output.append(RegionResult(name=name, status=status, metrics=metrics))
    return output


def _unavailable(title: str, module_id: str, reason: str) -> ModuleResult:
    return ModuleResult(
        module_id=module_id,
        title=title,
        status="检测失败/不可评估",
        sources=[],
        summary=reason,
        limitations=[reason],
    )


def adapt_dermavision(result_dir: str | Path) -> tuple[dict[str, ModuleResult], dict[str, Any]]:
    root = Path(result_dir).expanduser().resolve()
    if not root.is_dir():
        reason = f"五项皮肤检测结果目录不存在: {root}"
        return {
            "pores": _unavailable("可见毛孔", "01", reason),
            "pigmentation": _unavailable("综合色素表现", "03", reason),
            "redness": _unavailable("弥漫性泛红", "04", reason),
            "texture": _unavailable("面部表面平滑度", "10", reason),
        }, {}

    redness = _load_medical_csv(
        root,
        "十二项检测/01_红区/红区医学量化指标_V2.csv",
        "七项检测/红区/红区医学量化指标_V2.csv",
    ) or _load_metrics(
        root, "十二项检测/01_红区/红区量化指标.json", "七项检测/红区/红区量化指标.json", "红区量化指标.json"
    )
    spots = _load_medical_csv(
        root,
        "十二项检测/02_可见斑点/可见斑点医学量化指标_V2.csv",
        "七项检测/斑点/02_Spots医学量化指标_V2.csv",
    ) or _load_metrics(
        root, "十二项检测/02_可见斑点/可见斑点量化指标.json", "七项检测/斑点/02_Spots量化指标.json", "02_Spots量化指标.json"
    )
    brown = _load_medical_csv(
        root,
        "十二项检测/03_棕区/棕区医学量化指标_V2.csv",
        "七项检测/棕区/棕色斑医学量化指标_V2.csv",
    ) or _load_metrics(
        root, "十二项检测/03_棕区/棕区量化指标.json", "七项检测/棕区/棕色斑量化指标.json", "棕色斑量化指标.json"
    )
    texture = _load_medical_csv(
        root,
        "十二项检测/04_纹理/纹理医学量化指标_V2.csv",
        "七项检测/纹理/纹理医学量化指标_V2.csv",
    ) or _load_metrics(
        root, "十二项检测/04_纹理/纹理量化指标.json", "七项检测/纹理/纹理量化指标.json", "纹理量化指标.json"
    )
    pores = _load_medical_csv(
        root,
        "十二项检测/05_毛孔/毛孔医学量化指标_V2.csv",
        "七项检测/毛孔/毛孔医学量化指标_V2.csv",
    ) or _load_metrics(
        root, "十二项检测/05_毛孔/毛孔量化指标.json", "七项检测/毛孔/毛孔量化指标.json", "毛孔量化指标.json"
    )
    purple_uv = _load_medical_csv(
        root, "十二项检测/06_UV色斑/UV色斑医学量化指标_V2.csv",
    ) or _load_medical_csv(
        root, "七项检测/紫区/紫区医学量化指标_V2.csv",
        project="标准化UV紫外线色斑工程代理",
    )
    purple = _load_medical_csv(
        root, "十二项检测/07_卟啉/卟啉医学量化指标_V2.csv",
    ) or _load_medical_csv(
        root, "七项检测/紫区/紫区医学量化指标_V2.csv",
        project="标准化荧光UV紫质工程代理",
    ) or _load_metrics(root, "十二项检测/07_卟啉/卟啉量化指标.json", "七项检测/紫区/紫区量化指标.json", "紫区量化指标.json")

    modules: dict[str, ModuleResult] = {}
    raw: dict[str, Any] = {
        "redness": redness,
        "spots": spots,
        "brown": brown,
        "texture": texture,
        "pores": pores,
        "purple": purple,
        "uv_spots": purple_uv,
        "result_root": str(root),
    }

    if pores:
        t = pores.get("总体指标", {})
        modules["pores"] = ModuleResult(
            module_id="01",
            title="可见毛孔",
            status=str(t.get("评估状态", "可评估")),
            sources=["DermaVision毛孔检测"],
            summary=(
                f"共检测到 {value(t, '特征数量（个）')} 个可见毛孔，"
                f"主要集中于 {value(t, '主要集中区域')}。"
                "结果描述标准化白光照片中的可见毛孔负担。"
            ),
            metrics=[
                Metric("有效皮肤面积", value(t, "有效皮肤面积（像素）"), "像素"),
                Metric("可见毛孔数量", value(t, "特征数量（个）"), "个"),
                Metric("单位面积密度", value(t, "单位面积密度（个/10万有效皮肤像素）"), "个/10万有效皮肤像素"),
                Metric("毛孔总面积", value(t, "特征总面积（像素）"), "像素"),
                Metric("毛孔面积占比", value(t, "特征面积占比"), "比例"),
                Metric("P50单体面积", value(t, "P50单体面积（像素）"), "像素"),
                Metric("P90单体面积", value(t, "P90单体面积（像素）"), "像素"),
                Metric("最大单体面积", value(t, "最大单体面积（像素）"), "像素"),
                Metric("P50等效直径", value(t, "P50等效直径（像素）"), "像素"),
                Metric("P90等效直径", value(t, "P90等效直径（像素）"), "像素"),
                Metric("P90中心—环带视觉对比度", value(t, "P90中心—环带视觉对比度（0～1）"), "0～1工程归一化值"),
                Metric("P50圆度", value(t, "P50圆度（0～1）"), "0～1"),
                Metric("P50长宽比", value(t, "P50长宽比"), "比值"),
                Metric("低圆度毛孔比例", value(t, "低圆度毛孔比例"), "比例"),
                Metric("椭圆形毛孔比例", value(t, "椭圆形毛孔比例"), "比例"),
                Metric("拉长样毛孔比例", value(t, "拉长样毛孔比例"), "比例"),
                Metric("主要集中区域", value(t, "主要集中区域")),
            ],
            regions=_regions(
                pores,
                [
                    "有效皮肤面积（像素）", "特征数量（个）", "单位面积密度（个/10万有效皮肤像素）",
                    "特征面积占比", "P50单体面积（像素）", "P90单体面积（像素）",
                    "P90等效直径（像素）", "P90中心—环带视觉对比度（0～1）",
                ],
            ),
            images=existing_files([_resolve(root, "十二项检测/05_毛孔/01_毛孔检测结果图.jpg", "七项检测/毛孔/01_毛孔检测结果图.jpg", "01_毛孔检测结果图.jpg")]),
            daily_note="复测时应保持光线、距离、焦距和清洁状态一致；结果不代表毛孔真实深度、堵塞程度或皮脂分泌量。",
            limitations=["普通白光图像中的可见毛孔代理指标", "未建立医学大毛孔阈值", "不测量毛孔真实深度"],
        )
    else:
        modules["pores"] = _unavailable("可见毛孔", "01", "缺少毛孔量化指标.json")

    if redness:
        t = redness.get("总体指标", {})
        valid_area = t.get("有效皮肤面积（像素）")
        diffuse_ratio = t.get("弥漫红区面积占比")
        diffuse_area: Any = "不可评估"
        if isinstance(valid_area, (int, float)) and isinstance(diffuse_ratio, (int, float)):
            diffuse_area = round(float(valid_area) * float(diffuse_ratio), 3)
        modules["redness"] = ModuleResult(
            module_id="04",
            title="弥漫性泛红",
            status=str(t.get("评估状态", "可评估")),
            sources=["DermaVision红区检测", "局灶性红色实例检测"],
            summary=(
                f"弥漫红区面积占比为 {value(t, '弥漫红区面积占比')}，"
                f"局灶性红色实例 {value(t, '局灶红色实例数量（个）', value(t, '特征数量（个）'))} 个，"
                f"主要集中于 {value(t, '主要集中区域')}。"
            ),
            metrics=[
                Metric("有效皮肤面积", value(t, "有效皮肤面积（像素）"), "像素"),
                Metric("弥漫红区面积", diffuse_area, "像素（由有效皮肤面积×弥漫红区占比计算）"),
                Metric("弥漫红区面积占比", value(t, "弥漫红区面积占比", value(t, "特征面积占比")), "比例"),
                Metric("平均红度", value(t, "平均强度（0～1）"), "0～1工程归一化值"),
                Metric("P50红度", value(t, "P50强度（0～1）"), "0～1工程归一化值"),
                Metric("P90红度", value(t, "P90强度（0～1）"), "0～1工程归一化值"),
                Metric("P95红度", value(t, "P95强度（0～1）"), "0～1工程归一化值"),
                Metric("高红度区域面积占比", value(t, "高强度区域面积占比"), "比例"),
                Metric("最大连续红区面积", value(t, "最大连续区域面积（像素）"), "像素"),
                Metric("红区连续性", value(t, "红区连续性（0～1）"), "0～1"),
                Metric("红区均匀度", value(t, "弥漫红区均匀度（0～1）"), "0～1"),
                Metric("红度综合负担", value(t, "红度综合负担"), "工程指标"),
                Metric("局灶性红色实例数量", value(t, "局灶红色实例数量（个）", value(t, "特征数量（个）")), "个"),
                Metric("局灶实例单位面积密度", value(t, "单位面积密度（个/10万有效皮肤像素）"), "个/10万有效皮肤像素"),
                Metric("局灶实例总面积", value(t, "特征总面积（像素）"), "像素"),
                Metric("局灶实例P50面积", value(t, "P50单体面积（像素）"), "像素"),
                Metric("局灶实例P90面积", value(t, "P90单体面积（像素）"), "像素"),
                Metric("中央面部集中比例", value(t, "中央面部实例集中比例"), "比例"),
                Metric("画面左右面颊差异", value(t, "画面左右面颊面积占比差异")),
                Metric("主要集中区域", value(t, "主要集中区域")),
            ],
            regions=_regions(
                redness,
                [
                    "有效皮肤面积（像素）", "弥漫红区面积占比", "平均强度（0～1）",
                    "P90强度（0～1）", "P95强度（0～1）", "高强度区域面积占比",
                    "特征数量（个）", "单位面积密度（个/10万有效皮肤像素）", "红度综合负担",
                ],
            ),
            images=existing_files([
                _resolve(root, "十二项检测/01_红区/01_红区检测结果图.png", "十二项检测/01_红区/01_红区检测结果图.jpg"),
                _resolve(root, "七项检测/红区/03_RBX红区结果图.jpg", "03_RBX红区结果图.jpg"),
                _resolve(root, "七项检测/红区/06_VISIA红色区实例图.jpg", "06_VISIA红色区实例图.jpg"),
            ]),
            daily_note="建议在相同光线和清洁状态下复测；若泛红长期不退或伴持续刺痛、灼热等不适，应咨询专业人士。",
            limitations=["可见红度工程指标，不等同于血流量", "不代表医学炎症诊断或严重程度", "拍摄曝光和白平衡会影响结果"],
        )
    else:
        modules["redness"] = _unavailable("弥漫性泛红", "04", "缺少红区量化指标.json")

    if spots or brown:
        st = spots.get("总体指标", {}) if spots else {}
        bt = brown.get("总体指标", {}) if brown else {}
        status = "可评估" if spots and brown else "部分支持"
        regions: list[RegionResult] = []
        region_map: dict[str, dict[str, Any]] = {}
        for prefix, source in (("可见斑点", spots), ("棕色实例", brown)):
            if not source:
                continue
            for row in source.get("分区指标", []):
                if not isinstance(row, dict):
                    continue
                name = str(row.get("检测范围", "未命名分区"))
                dest = region_map.setdefault(name, {"评估状态": row.get("评估状态", "不可评估")})
                for key in ("特征数量（个）", "单位面积密度（个/10万有效皮肤像素）", "特征面积占比", "P90强度（0～1）"):
                    dest[f"{prefix}{key}"] = row.get(key, "不可评估")
        for name, row in region_map.items():
            status_value = str(row.pop("评估状态", "不可评估"))
            regions.append(RegionResult(name=name, status=status_value, metrics=row))

        ut = purple_uv.get("总体指标", {}) if purple_uv else {}
        modules["pigmentation"] = ModuleResult(
            module_id="03",
            title="综合色素表现",
            status=status,
            sources=[x for x in [
                "DermaVision可见斑点检测" if spots else "",
                "DermaVision棕区检测" if brown else "",
                "DermaVision UV色斑检测" if purple_uv else "",
            ] if x],
            summary=(
                f"普通白光图像中检测到可见斑点 {value(st, '特征数量（个）')} 个、"
                f"棕色实例 {value(bt, '特征数量（个）')} 个。"
                f"主要集中区域为 {value(bt, '主要集中区域', value(st, '主要集中区域'))}。"
            ),
            metrics=[
                Metric("可见斑点数量", value(st, "特征数量（个）"), "个"),
                Metric("可见斑点密度", value(st, "单位面积密度（个/10万有效皮肤像素）"), "个/10万有效皮肤像素"),
                Metric("可见斑点面积占比", value(st, "特征面积占比"), "比例"),
                Metric("可见斑点P50面积", value(st, "P50单体面积（像素）"), "像素"),
                Metric("可见斑点P90面积", value(st, "P90单体面积（像素）"), "像素"),
                Metric("可见斑点最大面积", value(st, "最大单体面积（像素）"), "像素"),
                Metric("平均综合色差ΔE", value(st, "平均综合色差ΔE"), "ΔE工程值"),
                Metric("P90综合色差ΔE", value(st, "P90综合色差ΔE"), "ΔE工程值"),
                Metric("点状斑点数量", value(st, "点状斑点数量（个）"), "个"),
                Metric("片状斑点数量", value(st, "片状斑点数量（个）"), "个"),
                Metric("融合样候选数量", value(st, "融合样候选数量（个）"), "个"),
                Metric("棕色实例数量", value(bt, "特征数量（个）"), "个"),
                Metric("棕色实例密度", value(bt, "单位面积密度（个/10万有效皮肤像素）"), "个/10万有效皮肤像素"),
                Metric("棕色实例面积占比", value(bt, "特征面积占比"), "比例"),
                Metric("棕色实例P50面积", value(bt, "P50单体面积（像素）"), "像素"),
                Metric("棕色实例P90面积", value(bt, "P90单体面积（像素）"), "像素"),
                Metric("P90棕色强度", value(bt, "P90强度（0～1）"), "0～1工程归一化值"),
                Metric("连续棕色色素覆盖占比", value(bt, "连续棕色色素覆盖占比"), "比例"),
                Metric("最大连续棕色区域面积", value(bt, "最大连续区域面积（像素）"), "像素"),
                Metric("棕色色素综合负担", value(bt, "棕色色素综合负担"), "工程指标"),
                Metric("UV色斑数量", value(ut, "特征数量（个）"), "个"),
                Metric("UV色斑密度", value(ut, "单位面积密度（个/10万有效皮肤像素）"), "个/10万有效皮肤像素"),
                Metric("UV色斑面积占比", value(ut, "特征面积占比"), "比例"),
                Metric("UV色斑P90强度", value(ut, "P90强度（0～1）"), "0～1工程归一化值"),
                Metric("红棕重叠比例", "不可评估", note="当前正式输出未提供同坐标实例Mask"),
                Metric("主要集中区域", value(bt, "主要集中区域", value(st, "主要集中区域"))),
            ],
            regions=regions,
            images=existing_files([
                _resolve(root, "十二项检测/02_可见斑点/01_可见斑点检测结果图.jpg"),
                _resolve(root, "十二项检测/03_棕区/01_棕区检测结果图.png", "十二项检测/03_棕区/01_棕区检测结果图.jpg"),
                _resolve(root, "七项检测/斑点/01_Spots斑点结果图.jpg", "01_Spots斑点结果图.jpg"),
                _resolve(root, "七项检测/棕区/01_RBX棕区结果图.jpg", "01_RBX棕区结果图.jpg"),
                _resolve(root, "七项检测/棕区/02_VISIA棕色斑实例图.jpg", "02_VISIA棕色斑实例图.jpg"),
            ]),
            daily_note="应在相同设备、光照和防晒/底妆状态下复测。综合色素结果用于观察可见外观和变化趋势。",
            limitations=["仅为普通白光RGB综合色素代理", "不输出UV隐藏色斑", "不推断黑色素真实含量或表皮/真皮深度"],
        )
    else:
        modules["pigmentation"] = _unavailable("综合色素表现", "03", "缺少斑点和棕区量化指标")

    if texture:
        t = texture.get("总体指标", {})
        modules["texture"] = ModuleResult(
            module_id="10",
            title="面部表面平滑度",
            status=str(t.get("评估状态", "可评估")),
            sources=["DermaVision二维纹理检测"],
            summary=(
                f"共检测到 {value(t, '特征数量（个）')} 个二维可见纹理特征，"
                f"主要类型为 {value(t, '主要问题类型')}，主要集中于 {value(t, '主要集中区域')}。"
            ),
            metrics=[
                Metric("纹理特征总数", value(t, "特征数量（个）"), "个"),
                Metric("纹理特征密度", value(t, "单位面积密度（个/10万有效皮肤像素）"), "个/10万有效皮肤像素"),
                Metric("纹理总面积", value(t, "特征总面积（像素）"), "像素"),
                Metric("纹理面积占比", value(t, "特征面积占比"), "比例"),
                Metric("凸起样数量", value(t, "凸起样数量（个）"), "个"),
                Metric("凸起样比例", value(t, "凸起样比例"), "比例"),
                Metric("凸起样面积占比", value(t, "凸起样面积占比"), "比例"),
                Metric("凹陷样数量", value(t, "凹陷样数量（个）"), "个"),
                Metric("凹陷样比例", value(t, "凹陷样比例"), "比例"),
                Metric("凹陷样面积占比", value(t, "凹陷样面积占比"), "比例"),
                Metric("P50纹理响应强度", value(t, "P50强度（0～1）"), "0～1工程归一化值"),
                Metric("P90纹理响应强度", value(t, "P90强度（0～1）"), "0～1工程归一化值"),
                Metric("P50特征面积", value(t, "P50单体面积（像素）"), "像素"),
                Metric("P90特征面积", value(t, "P90单体面积（像素）"), "像素"),
                Metric("聚集区域数量", value(t, "聚集区域数量（个）"), "个"),
                Metric("最大聚集区域特征数", value(t, "最大聚集区域特征数（个）"), "个"),
                Metric("主要问题类型", value(t, "主要问题类型")),
                Metric("主要集中区域", value(t, "主要集中区域")),
            ],
            regions=_regions(
                texture,
                [
                    "有效皮肤面积（像素）", "特征数量（个）", "单位面积密度（个/10万有效皮肤像素）",
                    "特征面积占比", "凸起样数量（个）", "凹陷样数量（个）", "P90强度（0～1）",
                ],
            ),
            images=existing_files([_resolve(root, "十二项检测/04_纹理/01_纹理检测结果图.jpg", "七项检测/纹理/01_纹理检测结果图.jpg", "01_纹理检测结果图.jpg")]),
            daily_note="结果适合用于同条件下观察二维表面纹理趋势；清洁状态、护肤品残留和光线会影响结果。",
            limitations=["二维可见表面纹理代理", "不输出真实高度、深度或体积", "不能单独区分缺水、屏障问题或附着物成因"],
        )
    else:
        modules["texture"] = _unavailable("面部表面平滑度", "10", "缺少纹理量化指标.json")

    return modules, raw
