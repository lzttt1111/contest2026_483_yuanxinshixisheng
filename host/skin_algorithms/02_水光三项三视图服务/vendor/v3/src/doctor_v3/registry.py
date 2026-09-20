"""Versioned physician definitions; no invented population cutoffs."""
from dataclasses import dataclass
from . import SPEC_VERSION

@dataclass(frozen=True)
class Module:
    id: str
    title: str
    weights: dict
    regions: tuple
    note: str

PORE_REGIONS = ("forehead", "nose", "left_nasal", "right_nasal", "left_inner_cheek", "right_inner_cheek", "left_outer_cheek", "right_outer_cheek", "chin")
SKIN_REGIONS = ("forehead", "nose", "left_nasal", "right_nasal", "left_zygoma", "right_zygoma", "left_cheek", "right_cheek", "chin")
FINE_REGIONS = ("left_under_eye", "right_under_eye", "left_zygoma", "right_zygoma", "left_cheek", "right_cheek")
REGION_LABELS = dict(zip(
    ("full_face", "forehead", "nose", "left_nasal", "right_nasal", "left_inner_cheek", "right_inner_cheek", "left_outer_cheek", "right_outer_cheek", "chin", "left_zygoma", "right_zygoma", "left_cheek", "right_cheek", "glabella", "perioral", "left_under_eye", "right_under_eye", "left_crow_feet", "right_crow_feet", "crow_feet", "left_eye", "right_eye", "left_jaw", "right_jaw", "jaw", "left_midface", "right_midface", "left_lower_face", "right_lower_face"),
    ("全面部", "额部", "鼻部", "左鼻旁", "右鼻旁", "左内侧面颊", "右内侧面颊", "左外侧面颊", "右外侧面颊", "下巴", "左颧部", "右颧部", "左面颊", "右面颊", "眉间", "口周", "左眼下", "右眼下", "左鱼尾纹", "右鱼尾纹", "双侧鱼尾纹", "左眼周", "右眼周", "左下颌", "右下颌", "下颌", "左中面部", "右中面部", "左下脸", "右下脸")
))
GROOVE_REGIONS = tuple(side + "_" + area for area in ("nasolabial", "marionette", "midcheek", "tear_trough", "mouth_corner") for side in ("left", "right"))
for area, name in zip(("nasolabial", "marionette", "midcheek", "tear_trough", "mouth_corner"), ("法令纹", "木偶纹", "颊中沟", "泪沟", "口角相关凹陷")):
    for side, prefix in (("left", "左"), ("right", "右")):
        REGION_LABELS[side + "_" + area] = prefix + name

MODULES = (
    Module("01", "可见毛孔", {"density": .30, "area_p50": .30, "large_density": .40}, PORE_REGIONS, "仅评价可见毛孔外观，不代表真实毛囊总数或深度。"),
    Module("02", "油脂分泌倾向", {"gloss_area": .25, "gloss_high_area": .20, "gloss_mean": .15, "porphyrin_high_density": .28, "porphyrin_p90": .12}, SKIN_REGIONS, "油光和卟啉是图像表现，不直接代表皮脂分泌量或微生物数量。"),
    Module("03", "综合色素问题", {"spots": .40, "brown": .30, "uv": .30}, ("forehead", "left_eye", "right_eye", "left_zygoma", "right_zygoma", "nose", "left_cheek", "right_cheek", "perioral", "jaw"), "三个色素成像层允许重叠；不能判断色素深度、病因或具体疾病。"),
    Module("04", "弥漫性泛红", {"area": .35, "high_area": .20, "mean": .25, "p90": .20}, ("forehead", "glabella", *SKIN_REGIONS[1:-1], "perioral", "jaw"), "评价本次背景泛红，不据单次图像判断持续性、病因或疾病。"),
    Module("05", "线状血管样结构", {"clusters": .30, "affected_area": .40, "local_density": .30}, (*SKIN_REGIONS[:-1], "jaw"), "受累簇不是单根血管数量，标准化网格不是实测毫米面积。"),
    Module("06", "毛囊炎症及痤疮样活动", {"erythema_density": .20, "papule_density": .40, "pustule_density": .40}, (*SKIN_REGIONS[:-1], "perioral", "left_jaw", "right_jaw"), "图像表型不构成痤疮、毛囊炎或病原体的确定性诊断。"),
    Module("07", "干燥性细纹", {"area": .35, "high_area": .25, "density": .25, "contrast_p50": .15}, FINE_REGIONS, "细碎纹理外观不等于实际含水量；二维对比度不是三维深度。"),
    Module("08", "稳定性线性皱纹", {"forehead": .25, "glabella": .25, "crow_feet": .25, "perioral": .25}, ("forehead", "glabella", "left_crow_feet", "right_crow_feet", "crow_feet", "perioral"), "需面部放松；额纹和眉间纹的完整评分需要合格三维凹陷证据。"),
    Module("09", "结构性沟槽", {"extent": .25, "mean_depth": .30, "p90_depth": .25, "volume": .20}, GROOVE_REGIONS, "缺少合格三维数据时不从二维暗线推断凹陷深度、体积和病因。"),
    Module("10", "面部皮肤平整度", {"raised_area": .30, "raised_p90": .20, "depressed_area": .30, "depressed_p90": .20}, (*SKIN_REGIONS[:-1], "perioral", "left_jaw", "right_jaw"), "评价表面几何，排除毛孔、各类皱纹和结构性沟槽；需要三维数据。"),
    Module("11", "面部轮廓紧致度", {"smoothness": .25, "turning": .20, "jowl": .30, "jaw_continuity": .25}, ("left_midface", "right_midface", "left_lower_face", "right_lower_face", "left_jaw", "right_jaw"), "三维轮廓不直接代表皮肤弹性、脂肪室位移或韧带状态。"),
)
METRIC_LABELS = {
"density": "可见毛孔密度", "area_p50": "毛孔面积P50", "large_density": "大毛孔密度",
"gloss_area": "油光面积占比", "gloss_high_area": "高强度油光面积占比", "gloss_mean": "平均油光强度",
"porphyrin_high_density": "高强度卟啉目标密度", "porphyrin_p90": "卟啉强度P90",
"area": "受累面积占比", "high_area": "高强度或高密度区域面积占比", "mean": "平均强度", "p90": "强度P90",
"clusters": "独立受累区域数量", "affected_area": "受累网格面积占比", "roi_count": "受累网格数量", "local_density": "受累区域内部密度",
"erythema_density": "毛囊周围红斑样目标密度", "papule_density": "红色丘疹样目标密度", "pustule_density": "丘脓疱样目标密度",
"contrast_p50": "视觉对比度P50", "extent": "结构性凹陷范围", "mean_depth": "平均相对深度", "p90_depth": "相对深度P90", "volume": "标准化凹陷体积",
"raised_area": "隆起影响范围", "raised_p90": "隆起高度P90", "depressed_area": "凹陷影响范围", "depressed_p90": "凹陷深度P90",
"smoothness": "轮廓平顺性", "turning": "高低起伏", "jowl": "低位组织堆积", "jaw_continuity": "下颌缘连续性"
}


def stage1_weights(module_id):
    """Latest Word count policy is separate from retained historical density recipes."""
    if module_id == "06":
        return {"erythema_count": .20, "papule_count": .40, "pustule_count": .40}
    return dict(next(m.weights for m in MODULES if m.id == module_id))


def measurement(module, name, region, profile, value=None, unit="标准化指数", reason=None, source_kind="measured"):
    labels = {"main_count":"稳定主皱纹数量","length_burden":"稳定皱纹长度负担","depth":"三维凹陷程度",
              "erythema_count":"毛囊周围红斑样目标数量","papule_count":"红色丘疹样目标数量","pustule_count":"丘脓疱样目标数量"}
    label = labels.get(name,METRIC_LABELS.get(name,"相关指标"))
    if module == "03" and "." in name:
        layer, metric = name.split(".",1)
        label = {"spots":"色斑","brown":"棕色色素","uv":"UV色素"}[layer]+{"area":"影响范围","p90":"明显程度P90"}[metric]
    if module in ("07","08") and name == "density":
        label = "受累区域纹路密度"
    return {"metric_id": module + "." + name, "name": label, "module": module,
            "region": region, "capture_profile": profile, "definition_version": SPEC_VERSION,
            "direction": "higher_burden",
            "source_kind": source_kind, "value": value, "unit": unit,
            "status": "measured" if value is not None else "unavailable",
            "reason": reason if value is None else None}
