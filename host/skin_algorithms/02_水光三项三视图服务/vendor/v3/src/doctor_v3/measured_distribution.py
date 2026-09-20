"""Describe saved regional quantities independently of score calibration."""
import math
from .registry import REGION_LABELS


def measured(model, module, label):
    metrics = model["modules"][module].get("metrics", {})

    def values(key):
        result = []
        for region, data in metrics.items():
            if region == "full_face":
                continue
            declared = model["modules"][module].get("regions", {})
            if declared and region not in declared:
                continue
            item = data.get(key, {})
            value = item.get("value")
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                continue
            if item.get("status") in ("REJECT", "FAILED"):
                continue
            if "有效区域不足" in str(item.get("no_score_reason", "")):
                continue
            area = item.get("effective_area")
            if area is not None and area < 2000:
                continue
            # Score availability is independent of a valid saved measurement.
            # Ratios are stored on 0..1; percentage output is on 0..100.
            if item.get("unit") == "比例":
                value *= 100
            result.append((region, float(value)))
        return sorted(result, key=lambda pair: -pair[1])

    def locations(key):
        data = values(key)
        positive = [(r, v) for r, v in data if v > 0]
        if positive:
            return "、".join(REGION_LABELS.get(r, r) for r, _ in positive[:2])
        return "已评估区域未检出相应目标" if data else "暂无分布结论（缺少对应区域量化）"

    def bilateral(key, unit):
        data = dict(values(key))
        for left in ("left_cheek", "left_zygoma", "left_nasal", "left_under_eye"):
            right = left.replace("left_", "right_", 1)
            if left in data and right in data:
                return (f"{REGION_LABELS.get(left, left)}{data[left]:.2f}{unit}；"
                        f"{REGION_LABELS.get(right, right)}{data[right]:.2f}{unit}")
        return "暂无左右比较（缺少同口径双侧量化）"

    if module == "05":
        if "左右" in label:
            return bilateral("clusters", "处")
        if "形态" in label:
            return "暂无形态分类结论（未保存形态分类指标）"
        if "区域" in label or "分布" in label:
            return "按受累区域数量：" + locations("clusters")
    if module == "02":
        if "油光区域" in label:
            return "按油光面积占比：" + locations("gloss_area")
        if "卟啉区域" in label:
            return "按高强度目标密度：" + locations("porphyrin_high_density")
        if "左右" in label:
            return bilateral("gloss_area", "%（油光面积占比）")
        if "T区" in label.replace(" ", ""):
            data = dict(values("gloss_area"))
            chosen = [(r, data[r]) for r in ("forehead", "nose", "chin") if r in data]
            return "；".join(f"{REGION_LABELS.get(r, r)}油光面积占比{v:.2f}%" for r, v in chosen) or "暂无T区分布（缺少对应区域量化）"
        if "分布" in label:
            return "油光：" + locations("gloss_area") + "；卟啉：" + locations("porphyrin_high_density")
        if "组合" in label:
            return "表面油光与卟啉分别记录；当前不作未评分子项的严重度比较"
    if module == "07":
        if not values("area"):
            return None
        if "高密度" in label:
            return "按高密度面积占比：" + locations("high_area")
        if "左右" in label:
            return bilateral("area", "%（受累面积占比）")
        if "面颊分布" not in label and ("区域" in label or "分布" in label):
            if model["modules"][module].get("detected") is False:
                return "已评估区域未见明显细纹"
            return "按受累面积占比：" + locations("area")
    if module == "11" and ("区域" in label or "表现" in label or "受累侧" in label):
        keys = ("smoothness", "turning", "jowl", "jaw_continuity")
        metrics = model["modules"][module].get("metrics", {})
        scored = [(region, node["score"]) for region, items in metrics.items() if region != "full_face"
                  for key, node in items.items() if key in keys and node.get("score") is not None]
        if "左右" in label or "受累侧" in label:
            for left in ("left_midface", "left_lower_face", "left_jaw"):
                right = left.replace("left_", "right_", 1)
                for key in keys:
                    a, b = metrics.get(left, {}).get(key, {}), metrics.get(right, {}).get(key, {})
                    if a.get("score") is not None and b.get("score") is not None:
                        return f"{a.get('name', key)}：{REGION_LABELS[left]}{a['score']}分；{REGION_LABELS[right]}{b['score']}分"
        elif scored:
            regions = list(dict.fromkeys(r for r, score in sorted(scored, key=lambda v:v[1]) if score <= 80))
            return "、".join(REGION_LABELS.get(r, r) for r in regions[:2]) if regions else "未见明显受累区域"
        present = any(values(key) for key in keys)
        return "暂无分区结论（对应区域参考数据尚未建立）" if present else "暂无分区结论（本次区域几何信息不足）"
    return None
