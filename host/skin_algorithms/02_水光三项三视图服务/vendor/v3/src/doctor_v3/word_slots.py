"""Explicit physician row names -> report metric keys. Unknown rows fail closed."""
import re
from .registry import REGION_LABELS
REGIONS={v:k for k,v in REGION_LABELS.items()}
REGIONS.update({"额纹":"forehead","眉间纹":"glabella","鱼尾纹":"crow_feet","双侧鱼尾纹整体":"crow_feet",
                "口周纹":"perioral","左下颌缘":"left_jaw","右下颌缘":"right_jaw","下颌及下巴":"jaw",
                "左口角相关凹陷":"left_mouth_corner","右口角相关凹陷":"right_mouth_corner"})
ROWS={
"01":{
"density":["可见毛孔数量","可见毛孔密度","毛孔密度"],
"area_p50":["典型毛孔大小","P50毛孔面积","毛孔面积P50"],
"large_density":["明显粗大毛孔数量","大毛孔密度"]},
"02":{
"surface_gloss":["表面油光"],"porphyrin":["毛囊卟啉"],
"gloss_area":["油光面积占比"],"gloss_high_area":["高强度油光面积占比"],
"gloss_mean":["平均油光强度"],"porphyrin_high_density":["高强度卟啉目标密度","高强度卟啉目标密度指数"],
"porphyrin_p90":["P90卟啉荧光强度","卟啉荧光强度P90"]},
"03":{
"spots":["色斑","色斑严重度"],"brown":["棕色色素","棕色色素严重度"],"uv":["UV色素","UV色素严重度"],
"spots.area":["综合色斑影响范围"],"spots.p90":["综合色斑明显程度"],
"brown.area":["棕色色素影响范围"],"brown.p90":["棕色色素明显程度"],
"uv.area":["UV色素影响范围"],"uv.p90":["UV色素明显程度"]},
"04":{
"area":["泛红覆盖范围","弥漫性泛红面积占比"],
"high_area":["明显泛红区域范围","高强度泛红面积占比"],
"mean":["整体泛红颜色","平均泛红强度"],"p90":["较红区域颜色","P90泛红强度"]},
"05":{
"clusters":["明显区域数量","线状血管样受累区域数量"],
"affected_area":["可见范围"],"roi_count":["线状血管样受累ROI数量"],
"local_density":["增生血管密集程度","受累ROI血管密集程度"]},
"06":{
"erythema_count":["毛囊周围发红","毛囊周围红斑样目标数量","毛囊周围红斑样目标"],
"papule_count":["红色丘疹样目标","毛囊性红色丘疹样目标数量","毛囊性红色丘疹样目标"],
"pustule_count":["丘脓疱样目标","毛囊性丘脓疱样目标数量","毛囊性丘脓疱样目标"]},
"07":{
"area":["干燥性细纹受累面积占比"],"high_area":["高密度细纹区域面积占比"],
"density":["受累区域细纹密度","受累区域细纹密度指数"],
"contrast_p50":["P50细纹视觉对比度"]},
"08":{
"main_count":["稳定主皱纹数量","额纹稳定主皱纹数量"],
"length_burden":["稳定皱纹长度负担","额纹长度负担"],
"depth":["三维凹陷程度","额纹三维凹陷"],
"area":["放射状皱纹受累面积占比","鱼尾纹受累面积"],
"density":["受累区域皱纹长度密度","鱼尾纹密度"],
"high_area":["高密度放射纹区域面积占比","鱼尾纹高密度区域"],
"contrast_p50":["P50皱纹视觉对比度"]},
"09":{
"extent":["凹陷累及范围","结构性凹陷范围状态","凹陷范围"],
"mean_depth":["整体凹陷程度","平均相对深度"],
"p90_depth":["明显区域凹陷程度","P90相对深度"],
"volume":["整体三维凹陷负担","标准化凹陷体积"]},
"10":{
"raised_area":["凸起影响范围"],"raised_p90":["明显凸起程度","凸起程度"],
"depressed_area":["凹陷影响范围"],"depressed_p90":["明显凹陷程度","凹陷程度"]},
"11":{
"smoothness":["轮廓平顺性","中下脸轮廓平顺性"],
"turning":["高低起伏程度","中下脸高低起伏"],
"jowl":["低位组织堆积"],"jaw_continuity":["下颌线连续状态","下颌缘连续状态"]}
}
def clean(value):
    return re.sub(r"[\s:：/（）()\-]","",value).replace("⻥","鱼").replace("⻆","角")

def resolve(module,label,region="full_face"):
    label=clean(label)
    if label in {clean(x) for x in ("综合状态","状态评分","综合状态评分","整体油脂状态",
        "面部油脂分泌状态评分","面部综合色素状态","面部综合色素状态评分","干燥性细纹状态评分",
        "稳定性线性皱纹综合状态","综合平整度状态","综合轮廓紧致状态")}:
        return "_module",region
    for name,r in REGIONS.items():
        if label in (clean(name),clean(name)+"状态评分"):
            return "_module",r
    without=label.removesuffix("状态评分").removesuffix("评分")
    for metric,aliases in ROWS[module].items():
        if without in {clean(x) for x in aliases} or label in {clean(x) for x in aliases}:
            if module=="08" and without.startswith("额纹"):
                region="forehead"
            if module=="08" and without.startswith("鱼尾纹"):
                region="crow_feet"
            return metric,region
    raise ValueError(f"unmapped physician row {module}: {label}")

def score_text(item):
    return "暂无评分（"+item.get("no_score_reason","缺少对应评分依据")+"）" if item.get("score") is None else str(item["score"])+"分"

def grade_text(item):
    return "暂无评分" if item.get("score") is None else item["grade"]

def metric(model,module,key,region):
    m=model["modules"][module]
    if key=="_module":
        return m if region=="full_face" else m.get("regions",{}).get(region,{})
    return m.get("metrics",{}).get(region,{}).get(key,{})
