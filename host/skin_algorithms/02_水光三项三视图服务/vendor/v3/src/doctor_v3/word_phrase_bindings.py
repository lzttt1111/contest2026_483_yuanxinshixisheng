"""Specific variable prose slots in the 20260908 doctor's examples."""
from .word_slots import metric,score_text,grade_text
from .registry import REGION_LABELS

def metric_sentence(model,mid,key,label,region="full_face"):
    x=metric(model,mid,key,region)
    if model["modules"][mid].get("zero_state") and region=="full_face":
        return label+"：已评估区域未见明显相关表现。"
    if x.get("score") is None:return label+"："+score_text(x)+"。"
    return label+"状态评分为"+score_text(x)+"，"+grade_text(x)+"。"

def region_list(model,mid,regions,metric_key="_module"):
    def shown(region):
        item=metric(model,mid,metric_key,region)
        return grade_text(item) if item.get("score") is not None else score_text(item)
    return "；".join(REGION_LABELS.get(r,r)+"："+shown(r) for r in regions)+"。"

def layer_location(model,key):
    from .pigment_distribution import regions_text
    return "覆盖分布："+regions_text(model,key)+"。"

def bind(model,mid,text):
    if mid=="02":
        if text.startswith("整体以T区"):
            return region_list(model,mid,("forehead","nose","chin"),"surface_gloss")
        if text.startswith(("表面油光主要集中于","鼻部最明显")):
            return "表面油光："+region_list(model,mid,("forehead","nose","chin"),"surface_gloss")
        if text.startswith("当前不仅存在") or text.startswith("表面油光提示"):
            return "毛囊卟啉："+region_list(model,mid,("nose","left_nasal","right_nasal","chin"),"porphyrin")
        if text.startswith("因此，本次油脂相关问题"):
            return "结合本次结果，可关注以下方面："
    if mid=="03":
        if text.startswith(("主要分布在","Brown图中可见")):
            return layer_location(model,"brown")
        if text.startswith(("UV图中","UV图像")):
            return layer_location(model,"uv")
        if text.startswith("部分区域在普通照片"):
            return "不同色素图像分别展示相应的色素表现，不据此判断色素的实际深度。"
        if text.startswith(("其中右侧颧部","当前明确色斑")):
            return metric_sentence(model,mid,"spots","色斑")+layer_location(model,"spots")
        if text.startswith("综合色素问题主要体现"):
            from .word_tables import distribution
            return distribution(model,mid,"主要严重度来源")+"。"
    if mid=="07":
        if text.startswith("你的干燥性细纹主要有以下特点"):return text
        if text.startswith(("细纹覆盖了一定范围","本次表现主要为")):
            return metric_sentence(model,mid,"area","细纹受累范围")
        if text.startswith("已经出现细纹的区域内部"):
            return metric_sentence(model,mid,"density","受累区域内部细纹密度")
        if text.startswith("细纹影响范围较广"):
            return metric_sentence(model,mid,"high_area","高密度细纹范围")
        if text.startswith("总体来看，目前的问题"):
            return "本次细纹的范围与密集程度分别见以上结果。"
        if text.startswith("本次结果提示"):
            return region_list(model,mid,("left_under_eye","right_under_eye","left_zygoma","right_zygoma"))
    if mid=="08":
        if text.startswith("眉间纹为"):
            return region_list(model,mid,("glabella","perioral"))
        if text.startswith(("部分眼尾区域","鱼尾纹影响范围")):
            return metric_sentence(model,mid,"area","鱼尾纹影响范围","crow_feet")
        if text.startswith("本次主要严重度表现"):
            return "各类皱纹的本次表现如下："
    if mid=="09":
        if text.startswith("本次结构性沟槽主要表现"):
            return "各结构性沟槽的本次表现如下："
        if text.startswith("当前总体严重度主要来源"):
            from .word_tables import distribution
            return distribution(model,mid,"主要严重度来源")+"。"
    return None
