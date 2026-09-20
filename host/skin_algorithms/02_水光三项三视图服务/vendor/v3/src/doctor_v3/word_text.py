"""Physician prose binding: static instructions retained, example findings replaced."""
import re
import math
from .word_slots import metric, score_text, grade_text, REGIONS
from .word_tables import distribution, description

ENGINEERING_NOTES=("完善疾病库后","后台核心指标","后续设备标定","标定结果为准",
                   "具体单位及标准化","此处","示例中","系统验证结果进行标定","重复拍摄稳定性")
def heading(block,edition):
    text=block.get("text","")
    if block["kind"]=="chapter":return True
    if re.match(r"^[一二三四五六七八九十]+、",text):return True
    return edition=="home" and block.get("font_size",0)>=17 and not re.search(r"评分.*分|程度:|状态:",text)

def bind_text(model,module,block,state,edition):
    text=block["text"]
    m=model["modules"][module]
    if m.get("assessment_title"):
        m={**m,"title":m["assessment_title"]}
    if block["kind"]=="chapter":
        return text
    if module=='09' and text.strip()=='主要沟槽区域':
        state.update(section=text,medical=False,history=False,current_region_section=True,current_region_written=False)
        return text
    if heading(block,edition):
        state['current_region_section']=False
        state["section"]=text
        state["medical"]="医学说明" in text or "温馨提示" in text
        state["history"]=any(x in text for x in ("纵向","上次","变化总结","追踪结论"))
        state["history_written"]=False
        return re.sub(r"[（(]完善疾病库后.*?[）)]","",text)
    if any(x in text for x in ENGINEERING_NOTES):
        state.setdefault("exceptions",[]).append({"id":block["id"],"reason":"physician_authoring_note"})
        return None
    if re.search(r"(后续|具体).*(标定|专家标注|验证结果)",text):
        return None
    if state.get("medical"):
        if module=="09" and text.startswith("本报告中的结构性凹陷范围"):
            return "本报告根据结构性沟槽的可见范围与2.5D图像估计进行状态评价，估计值不代表实测毫米深度或体积。"
        if module=="10" and text.startswith("本结果反映"):
            return "本结果反映标准化面部图像中的皮肤表面平整外观。"
        # Preserve medical limitations. Do not assert that 3D capture took place.
        if ("3D" in text or "三维" in text) and any(x in text for x in ("本报告评价的是","本结果反映","本报告中的")):
            text=re.sub(r"及\s*3D[^，。中的]*","",text)
            text=text.replace("整体三维负担","整体形态负担").replace("三维轮廓形态","轮廓形态")
            state.setdefault("exceptions",[]).append({"id":block["id"],"reason":"actual_imaging_source"})
        return text.replace("1 mm × 1 mm标准化局部皮肤ROI","标准化图像网格").replace("V2中同样明确指出图像结果不能直接完成痤疮与毛囊炎的确定性诊断及鉴别。","")
    if state.get("history"):
        if not state.get("history_written"):
            state["history_written"]=True
            if model.get("previous"):
                previous=model["previous"]["modules"][module]
                return "上次状态评分为"+score_text(previous)+"，本次为"+score_text(m)+"。"
            return "暂无历史检测记录。"
        return None
    if (module=='09' and state.get('current_region_section')
            and re.search(r'^[•·\-\s]*(?:左|右|双侧).*?(?:法令|泪沟|木偶|颊中|口角)',text)):
        state.setdefault('exceptions',[]).append({'id':block['id'],'reason':'current_region_selection'})
        if state.get('current_region_written'):
            return None
        state['current_region_written']=True
        return distribution(model,module,'主要沟槽区域')+'。'
    if "评分" in text and re.search(r"[:：]",text) and ("分" in text):
        prefix=re.split(r"[:：]",text,1)[0]
        if module=="02" and prefix in ("表面油光评分","毛囊卟啉评分"):
            sub="surface_gloss" if prefix=="表面油光评分" else "porphyrin"
            value=metric(model,module,sub,"full_face")
            return prefix+"："+score_text(value)+"｜"+grade_text(value)
        return prefix+"："+score_text(m)
    if re.search(r"(程度|结果等级|色素状态|当前程度)[:：]",text):
        return re.split(r"[:：]",text,1)[0]+"："+grade_text(m)
    if any(x in text for x in ("分数越高","评分越高","100分制","越高,表示","越低,表示","越高，表示","越低，表示")):
        return text
    if "反映“" in text or ("越" in text and "表示" in text):
        return text.replace("每个ROI对应1 mm × 1 mm标准化局部皮肤范围","每个ROI对应一个标准化图像网格")
    if text.startswith("三个二级结果属于") or text.endswith("结论："):
        return text
    if ("高精度" in text and "可见" in text) or text.startswith("在高精度检测下"):
        if model.get("stage1_measurement_config", {}).get("imaging", {}).get("high_precision_texture") is True:
            from .word_phrase_bindings import metric_sentence
            return metric_sentence(model,module,"contrast_p50","细纹视觉对比度")
        return "本次未纳入高精度纹理对比度评价。"
    if "三维凹陷" in text and not state.get("history"):
        if module=="08":
            return "额纹凹陷："+score_text(metric(model,module,"depth","forehead"))+"；眉间纹凹陷："+score_text(metric(model,module,"depth","glabella"))+"。"
        return "本次采用2.5D图像估计，结果不代表三维实测。"
    if module=="05" and text.startswith("本次共检测到"):
        def count(name):
            value=metric(model,module,name,'full_face').get('value')
            return str(int(value)) if not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and value>=0 and int(value)==value else None
        a,b=count('clusters'),count('roi_count')
        if a is not None and b is not None:return '本次独立受累区域'+a+'处，受累图像网格'+b+'个。'
        if a is not None:return '本次独立受累区域'+a+'处；受累图像网格数量暂无结果。'
        if b is not None:return '本次受累图像网格'+b+'个；独立受累区域数量暂无结果。'
        return '本次暂无独立受累区域及受累图像网格数量结果。'
    if module=="05" and text.startswith(("主要形态表现","从外观上看")):
        return "线状血管样结构的分布及密集程度见本章检测结果。"
    if module=="09" and text.startswith(("双侧法令纹","双侧泪沟")):
        rs=("left_nasolabial","right_nasolabial") if "法令" in text else ("left_tear_trough","right_tear_trough")
        from .registry import REGION_LABELS
        return "；".join(REGION_LABELS.get(r,r)+"："+grade_text(metric(model,module,"_module",r)) for r in rs)+"。"
    if text in ("说明:","说明：") or text.endswith("越多;") or text.endswith("越大;"):
        return text
    from .word_phrase_bindings import bind
    specific=bind(model,module,text)
    if specific:return specific
    from .word_narratives import region_sentence,mentioned_metric,oil_combination
    regional=region_sentence(model,module,text)
    if regional:return regional
    if module=="02" and "表面油光" in text and "卟啉" in text:
        return oil_combination(model)
    if module=="03" and text.startswith(("当前棕色","当前UV")):
        sub="brown" if "棕色" in text else "uv"
        value=metric(model,module,sub,"full_face")
        return ("棕色色素" if sub=="brown" else "UV色素")+"表现为"+grade_text(value)+"。"
    section=state.get("section","")
    if text.startswith(("本次检测发现","当前检测到","当前存在")):
        if m.get("score") is None:return "本次"+m["title"]+"暂不评分。"
        if m["grade"]=="未见明显":return "本次检测中，"+m["title"]+"未见明显异常。"
        return "本次检测发现，"+m["title"]+"表现为"+m["grade"]+"。"
    if text.startswith(("从结果来看","总体来看","本次主要表现","当前主要表现")):
        if module=="06":return distribution(model,module,"主要严重度来源")+"。"
        mentioned=mentioned_metric(model,module,text)
        if mentioned:return mentioned
        value=distribution(model,module,"主要严重度来源")
        return value+"。" if value!="—" else "本次暂不提供主要程度来源结论。"
    if "左右" in text or ("右" in text and "左" in text and any(x in text for x in ("更","高于","低于","较"))):
        return distribution(model,module,"左右对应区域差异")+"。"
    for name,region in sorted(REGIONS.items(),key=lambda x:-len(x[0])):
        if text.startswith(name+("：")) or text.startswith(name+":"):
            return name+"："+description(metric(model,module,"_module",region),m["title"])
    if any(x in text for x in ("主要","集中","分布","区域","额部","鼻部","面颊","眼下","颧部","下颌","口周")):
        location=distribution(model,module,"主要受累区域")
        if location=="—":return "本次暂不提供分区结论。"
        if location=="未见明显异常区域":return "可评估区域未见明显异常表现。"
        if location.startswith(("已评估区域", "暂无")):
            return location+"。"
        return "主要表现区域为"+location+"。"
    if text.endswith(("以下特点:","以下特点：","主要表现为:","主要表现为：")):
        return text
    # Preserve physician definitions, not generic replacement prose.
    mentioned=mentioned_metric(model,module,text)
    if mentioned:return mentioned
    if not any(token in text for token in ("本次","当前","你的","较明显","中度","轻度","有所","突出","偏高","偏低","较多","粗大","偏大","不明显")):
        return text
    return description(m,m["title"])
