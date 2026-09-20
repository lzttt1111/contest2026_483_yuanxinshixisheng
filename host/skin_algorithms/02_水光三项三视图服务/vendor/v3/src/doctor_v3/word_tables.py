"""Bind physician table cells to a single saved report model."""
from .word_slots import REGIONS, clean, resolve, metric, score_text, grade_text
import math


def _scored(item):
    value=item.get('score')
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=100


def _current_groove_regions(module):
    from .registry import REGION_LABELS
    scored=[(region,node) for region,node in module.get('regions',{}).items() if _scored(node)]
    affected=[]
    for region,node in scored:
        values=module.get('metrics',{}).get(region,{})
        volume=values.get('volume',{}).get('value')
        if (node.get('zero_state') is True or node['score']>=81 or isinstance(volume,bool)
                or not isinstance(volume,(int,float)) or not math.isfinite(volume) or volume<=0):
            continue
        affected.append((region,node))
    affected.sort(key=lambda pair:pair[1]['score'])
    if not affected:
        return '暂无主要区域结论（缺少有目标区域的评分依据）'
    if len(scored)<2:
        region,node=affected[0]
        return '暂无主要区域结论；已评分的'+REGION_LABELS.get(region,region)+'表现为'+grade_text(node)
    return '已评分区域中，'+ '、'.join(REGION_LABELS.get(region,region) for region,_ in affected[:2])+'相关表现相对明显'

def table_kind(rows):
    header=rows[0]
    if any("上次" in h or "变化" in h for h in header):
        return "history"
    if header==["项目","检测结果"]:
        return "quality"
    if header[0]=="派生结果":
        return "distribution"
    if header[0] in ("面部分区","区域","皱纹类型/区域","结构性沟槽"):
        return "regional"
    if header[0]=="皱纹类型" and "核心指标" not in header:
        return "regional"
    return "metric"

def description(item,label):
    if item.get("score") is None:
        return score_text(item)
    grade=item["grade"]
    if grade=="未见明显":
        return label+"未见明显异常。"
    return label+"相关表现为"+grade+"。"

def distribution(model,module,label):
    m=model["modules"][module]
    if m.get('zero_state') is True:
        return '已评估区域未见明显相关表现'
    if module=='09' and '区域' in label and any(word in label for word in ('主要','次要')):
        return _current_groove_regions(m)
    if module=="03":
        from .pigment_distribution import describe
        measured=describe(model,label)
        if measured is not None:return measured
    explicit=m.get("narratives",{}).get(label)
    if explicit:
        return explicit
    if module in ("02", "05", "07", "11"):
        from .measured_distribution import measured
        quantitative = measured(model, module, label)
        if quantitative is not None:
            return quantitative
    if any(x in label for x in ("多图共同","额外明显","形态特征","下半面结构","口角相关")):
        return "—"
    subkey=None
    if module=="02" and "区域" in label:
        subkey="porphyrin" if "卟啉" in label else "surface_gloss" if "油光" in label else None
    if module=="03" and "主要区域" in label:
        subkey="brown" if label.startswith("棕色") else "uv" if label.startswith("UV") else "spots" if label.startswith("色斑") else None
    if subkey:
        from .registry import REGION_LABELS
        values=[(r,v.get(subkey,{})) for r,v in m["metrics"].items() if r!="full_face" and v.get(subkey,{}).get("score") is not None]
        values.sort(key=lambda x:x[1]["score"])
        affected=[(r,v) for r,v in values if v["score"]<81]
        return "、".join(REGION_LABELS.get(r,r) for r,v in affected[:2]) if affected else "未见明显异常区域" if values and len(values)==len(m["regions"]) else "—"
    if module=="07" and "高密度" in label:
        from .registry import REGION_LABELS
        values=[(r,v.get("high_area",{}).get("value")) for r,v in m["metrics"].items() if r!="full_face"]
        positive=[(r,v) for r,v in values if v is not None and v>0]
        positive.sort(key=lambda x:-x[1])
        return "、".join(REGION_LABELS.get(r,r) for r,v in positive[:2]) if positive else "未见明显高密度区域" if all(v is not None for r,v in values) else "—"
    if module=="07" and "面颊分布" in label:
        from .registry import REGION_LABELS
        import math
        areas={r:m["metrics"].get(r,{}).get("area",{}).get("value") for r in ("left_cheek","right_cheek")}
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in areas.values()):
            return "本次暂不提供面颊分布结果"
        selected=[r for r,v in areas.items() if v>0]
        return "、".join(REGION_LABELS[r] for r in selected) if selected else "未见明显面颊受累"
    if module=="10" and label in ("主要隆起区域","主要凹陷区域"):
        from .registry import REGION_LABELS
        prefix="raised" if "隆起" in label else "depressed"
        ranked=[]
        for r,values in m["metrics"].items():
            if r=="full_face":continue
            a,b=values.get(prefix+"_area",{}),values.get(prefix+"_p90",{})
            if a.get("score_raw") is None or b.get("score_raw") is None or not (a.get("value") or 0)>0:continue
            score=.6*a["score_raw"]+.4*b["score_raw"]
            if score<80.5:ranked.append((r,score))
        ranked.sort(key=lambda x:x[1])
        return "、".join(REGION_LABELS.get(r,r) for r,s in ranked[:2]) if ranked else "—"
    if module=="08" and "来源" in label and "鱼尾纹" in label:
        return "—" if m["regions"].get("crow_feet",{}).get("score") is None else "鱼尾纹范围与密集程度"
    if module=="08" and "来源" in label and "额纹" in label:
        metrics=m["metrics"].get("forehead",{})
        weights=m["regions"].get("forehead",{}).get("weights",{})
        labels={"main_count":"稳定主皱纹数量","length":"纹路延伸负担","maximum":"最长纹路线段"}
        values=[(k,v) for k,v in metrics.items() if k in labels and v.get("score") is not None and v["score"]<81]
        values.sort(key=lambda x:-(100-x[1]["score"])*weights.get(x[0],0))
        return "、".join(labels[k] for k,v in values[:2]) if values else "—"
    if module=="06" and ("受累区域" in label or "主要受累区域" in label):
        from .registry import REGION_LABELS
        targets=[]
        for r,items in m["metrics"].items():
            if r=="full_face":continue
            count=sum(items.get(k,{}).get("value") or 0 for k in ("erythema_count","papule_count","pustule_count"))
            if count>0:targets.append((r,count))
        targets.sort(key=lambda x:-x[1])
        chosen=targets[2:4] if "次要" in label else targets[:2]
        return "、".join(REGION_LABELS.get(r,r) for r,n in chosen) if chosen else "—"
    regions=[(r,s) for r,s in m.get("regions",{}).items() if s.get("score") is not None]
    regions.sort(key=lambda x:x[1]["score"])
    from .registry import REGION_LABELS
    names=dict(REGION_LABELS)
    if module=="08":names.update(forehead="额纹",glabella="眉间纹",perioral="口周纹")
    is_driver=any(x in label for x in ("来源","主要类型","主要问题","组合","目标类型","不平整类型","轮廓问题"))
    if regions and all(s["score"]>=81 for _,s in regions) and "较轻" not in label and "较平整" not in label and not is_driver:
        return "未见明显异常区域" if len(regions)==len(m.get("regions",{})) else "—"
    if "左右" in label or "受累侧" in label:
        pairs=[]
        for r,s in regions:
            if r.startswith("left_"):
                other=m.get("regions",{}).get(r.replace("left_","right_",1),{})
                if other.get("score") is not None:
                    pairs.append((abs(s["score"]-other["score"]),r,s["score"],other["score"]))
        if not pairs:
            return "本次未提供左右比较结果"
        difference,r,left,right=max(pairs)
        if difference==0:
            return "左右对应区域状态评分相同"
        return names.get(r if left<right else r.replace("left_","right_",1),"相应区域")+"表现相对明显"
    if "较轻" in label or "较平整" in label:
        regions=list(reversed(regions))
    else:
        regions=[(r,s) for r,s in regions if s["score"]<81]
    if is_driver:
        preferred={"01":["density","area_p50","large_density"],"02":["surface_gloss","porphyrin"],
          "03":["spots","brown","uv"],"04":["area","high_area","mean","p90"],
          "05":["clusters","affected_area","local_density"],"06":["erythema_count","papule_count","pustule_count"],
          "07":["area","high_area","density","contrast_p50"],"09":["extent","mean_depth","p90_depth","volume"],
          "10":["raised_area","raised_p90","depressed_area","depressed_p90"],"11":["smoothness","turning","jowl","jaw_continuity"]}
        if module=='07' and model.get('stage1_measurement_config',{}).get('imaging',{}).get('high_precision_texture') is not True:
            preferred['07'].remove('contrast_p50')
        items=[(k,x) for k,x in m.get("metrics",{}).get("full_face",{}).items()
               if _scored(x) and k in preferred.get(module,[])]
        expected=preferred.get(module,[])
        if expected and {k for k,_ in items} != set(expected):
            from .word_slots import ROWS
            known=dict(items)
            facts=[ROWS.get(module,{}).get(k,[known[k].get('name',k)])[0]+'为'+grade_text(known[k])
                   for k in expected if k in known]
            return '已评分项目：'+'；'.join(facts) if facts else '暂无主要程度来源结论（缺少核心评分依据）'
        weights=m.get("weights",{})
        if module=="02":weights={"surface_gloss":.6,"porphyrin":.4}
        items.sort(key=lambda x:-(100-x[1]["score"])*weights.get(x[0],weights.get(x[0].replace("_count","_density"),1)))
        if items:
            relevant=[(k,x) for k,x in items if x["score"]<81]
            if not relevant:return "各可评估项目未见明显异常表现"
            from .word_slots import ROWS
            labels=[]
            for k,x in relevant[:2]:
                name=ROWS.get(module,{}).get(k,[x.get("name",k)])[0]
                if module=="11" and k in ("smoothness","jaw_continuity"):name+="下降"
                labels.append(name)
            return "、".join(labels)+"相对突出"
    if regions:
        selected=regions[:2] if "次要" not in label else regions[2:4]
        if selected:
            return "、".join(names.get(r,r) for r,_ in selected)
    return "—"

def quality_value(model,module,label):
    if "纵向" in label:
        return "无历史记录" if not model.get("previous") else "见纵向变化"
    if "3D" in label or "三维" in label or "曲面" in label:
        return "暂无对应分区估计" if module=="08" else "2.5D图像估计（非实测）" if module in ("09","10","11") else "未采集"
    if "图像质量" in label or "照片质量" in label:
        if "UV" in label and model["capture_profile"]=="consumer":
            return "未采集UV照片"
        return model.get("quality_label","未提供")
    if "面部放松" in label:
        return "未提供"
    if "图像配准" in label:
        return model.get("registration_label","未提供" if model["capture_profile"]=="institution" else "不适用")
    if "表型" in label or "表现" in label:
        m=model["modules"][module]
        if m.get("detected") is None:return "本次暂不评价"
        return "已检出" if m["detected"] else "未见明显"
    if label in ("色斑","棕色色素","UV色素"):
        key={"色斑":"spots.area","棕色色素":"brown.area","UV色素":"uv.area"}[label]
        value=metric(model,module,key,"full_face").get("value")
        return "本次暂不评价" if value is None else "已检出" if value>0 else "未见明显"
    if "左右" in label:
        regions=model["modules"][module].get("regions",{})
        paired=any(r.startswith("left_") and x.get("score") is not None and regions.get(r.replace("left_","right_",1),{}).get("score") is not None for r,x in regions.items())
        return "可评价" if paired else "本次暂不评价"
    if "有效" in label:
        return "可用" if model["modules"][module].get("has_measurements") else "不足"
    if "P50" in label or "高精度" in label or "纹理成像" in label:
        high = model.get("stage1_measurement_config", {}).get("imaging", {}).get("high_precision_texture") is True
        if "P50" in label:
            return "已纳入评价" if high else "本次未纳入评价"
        if high:
            return "高精度纹理成像"
        return "普通图像评估"
    if "分类" in label or "定位" in label or "下颌缘" in label:
        return "已完成图像分析" if model["modules"][module].get("has_measurements") else "本次暂不评价"
    if label in ("隆起性不平整","凹陷性不平整"):
        return "可进行图像评估" if model["modules"][module].get("has_measurements") else "本次暂不评价"
    raise ValueError("unmapped quality row: "+label)

def bind_table(model,module,rows):
    kind=table_kind(rows)
    output=[rows[0]]
    region="full_face"
    for row in rows[1:]:
        if kind=="quality":
            output.append([row[0],quality_value(model,module,row[0])]);continue
        if kind=="distribution":
            output.append([row[0],distribution(model,module,row[0])]);continue
        if kind=="regional":
            region=REGIONS.get(row[0])
            if region is None:raise ValueError("unmapped region: "+row[0])
            item=metric(model,module,"_module",region)
            result=[row[0]+("（图像估计）" if item.get("assessment_basis")=="historical_named_roi_image_estimate" else "")]
            for header in rows[0][1:]:
                if "评分" in header:result.append(score_text(item))
                elif header in ("等级","程度"):result.append(grade_text(item))
                elif header=="受累区域数量":result.append(metric(model,module,"clusters",region).get("display","—"))
                elif header=="受累ROI数量":result.append(metric(model,module,"roi_count",region).get("display","—"))
                elif header in ("轮廓平顺性","高低起伏","低位组织堆积","连续状态"):
                    k={"轮廓平顺性":"smoothness","高低起伏":"turning","低位组织堆积":"jowl","连续状态":"jaw_continuity"}[header]
                    result.append(grade_text(metric(model,module,k,region)))
                elif header in ("色斑状态","棕色色素状态","UV色素状态"):
                    value=metric(model,module,{"色斑状态":"spots","棕色色素状态":"brown","UV色素状态":"uv"}[header],region)
                    result.append(grade_text(value) if value.get("score") is not None else score_text(value))
                elif module=="03":
                    from .pigment_distribution import regional_description
                    result.append(regional_description(model,region))
                else:
                    metrics=model["modules"][module]["metrics"].get(region,{})
                    restored=[v for v in metrics.values() if v.get("score") is not None and v.get("restored_additive")]
                    if item.get("score") is None and restored:
                        result.append("；".join(v["name"]+"："+score_text(v) for v in restored)+"；其余"+score_text(item))
                    else:result.append(description(item,model["modules"][module]["title"]))
            output.append(result);continue
        label=row[0]
        prefix=[]
        if rows[0][0]=="皱纹类型" and "核心指标" in rows[0]:
            if row[0]:region=REGIONS[row[0]]
            label=row[1];prefix=[row[0]]
        k,r=resolve(module,label,region)
        item=metric(model,module,k,r)
        if item.get("measurement_basis") in ("user_approved_historical_v2_image_estimate","historical_named_roi_image_estimate"):
            label=label+"（图像估计）"
        result=prefix+[label]
        for header in rows[0][len(result):]:
            if kind=="history" and ("上次" in header or "变化" in header or header=="结果"):
                previous=model.get("previous")
                old=metric(previous,module,k,r) if previous else {}
                use_score=k=="_module" or "评分" in label or any("评分" in h for h in rows[0])
                if "上次" in header:
                    result.append(score_text(old) if use_score and old.get("score") is not None else old.get("display","—"))
                elif header=="结果":result.append("—")
                else:
                    field="score" if use_score else "value"
                    a,b=item.get(field),old.get(field)
                    if a is None or b is None or (not use_score and item.get("unit")!=old.get("unit")):result.append("—")
                    else:
                        delta=a-b
                        if not use_score and item.get("unit")=="比例":delta*=100
                        suffix="分" if use_score else "个百分点" if item.get("unit")=="比例" else ""
                        result.append(f"{delta:+.5g}"+suffix)
            elif "评分" in header:
                result.append(score_text(item))
            elif header in ("等级","程度","结果"):
                result.append(grade_text(item))
            elif "解释" in header or header=="主要结果":
                result.append(description(item,item.get("name",label)))
            elif header=="本次表现":
                result.append(grade_text(item))
            elif kind=="history":
                use_score=k=="_module" or "评分" in label or any("评分" in h for h in rows[0])
                result.append(score_text(item) if use_score else item.get("display","—"))
            elif header in ("本次结果","检测结果","本次检测","检测值"):
                result.append(item.get("display","—") if k!="_module" else grade_text(item))
            else:
                raise ValueError("unmapped physician column: "+header)
        output.append(result)
    return output
