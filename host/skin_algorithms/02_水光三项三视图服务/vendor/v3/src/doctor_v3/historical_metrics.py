"""Normalize preserved regional measurements; no image reconstruction or inference."""
import math
from functools import lru_cache
REGION_ALIASES = {
    "全面部":"full_face","全脸":"full_face","额头":"forehead","额部":"forehead","眉间":"glabella",
    "鼻部":"nose","鼻旁":"nose_alar_nasal_side","画面左鼻旁":"left_nasal","画面右鼻旁":"right_nasal",
    "画面左颧部":"left_zygoma","画面右颧部":"right_zygoma","画面左面颊":"left_cheek","画面右面颊":"right_cheek",
    "左脸颊":"left_cheek","右脸颊":"right_cheek","左眼下":"left_under_eye","右眼下":"right_under_eye",
    "画面左眼下":"left_under_eye","画面右眼下":"right_under_eye","口周":"perioral","下巴":"chin",
    "左鼻旁":"left_nasal","右鼻旁":"right_nasal","左颧部":"left_zygoma","右颧部":"right_zygoma",
    "左面颊":"left_cheek","右面颊":"right_cheek","画面左眼周":"left_eye","画面右眼周":"right_eye",
    "额头纹":"forehead","眉间纹":"glabella","左鱼尾纹":"left_crow_feet","右鱼尾纹":"right_crow_feet",
    "左法令纹":"left_nasolabial","右法令纹":"right_nasolabial","左木偶纹":"left_marionette","右木偶纹":"right_marionette",
    "left_nasal_side":"left_nasal","right_nasal_side":"right_nasal","left_periocular":"left_eye","right_periocular":"right_eye",
}
FIELDS = {
 "mean_intensity","p50_intensity","diffuse_red_area_ratio","high_intensity_area_ratio","p90_delta_e",
 "feature_density_per_100k_skin_px","feature_area_ratio","p50_instance_area_px","p90_instance_area_px",
 "max_instance_area_px","feature_count","valid_skin_area_px","valid_area_px","valid_pixels",
 "red_area_ratio","high_red_area_ratio","mean_redness","p90_redness","continuous_brown_coverage_ratio",
 "p90_intensity","high_intensity_area_ratio","p90_delta_e","p90_deltaE","mean_deltaE",
 "gloss_area_ratio","high_gloss_area_ratio","mean_gloss_intensity","p50_gloss_intensity","p90_gloss_intensity",
 "vascular_count","vascular_area_ratio","vascular_total_length_px","vascular_line_density_per_10k_face_px",
 "count","total_length_px","total_area_px","line_density_per_10k_px","area_ratio","total_wrinkle_length_px",
 "wrinkle_segment_count","segment_count","wrinkle_pixels","max_segment_length_px","max_segment_length",
 "wrinkle_area_ratio","wrinkle_length_density","weighted_relative_response","mean_response",
 "raised_like_area_ratio","depressed_like_area_ratio","raised_area_ratio","depressed_area_ratio",
 "p90_raised_response","jaw_continuity_ratio","jaw_curve_p90","midface_surface_continuity_ratio","midface_relative_relief_p90",
 "面积占比","特征面积占比","连续异常面积占比","实例P90强度（0～1）","P90强度（0～1）",
 "单位面积密度（个/10万有效皮肤像素）","特征数量（个）","有效皮肤面积（像素）",
 "高强度目标比例","高强度异常面积占比","目标平均强度（0～1）",
}
def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        return None
    return float(value)

@lru_cache(maxsize=1)
def wrinkle_extractor():
    # Load the existing scalar-only helper without the scoring_bridge package
    # initializer, which unnecessarily imports scipy on the Windows reader.
    import importlib.util,sys
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"scoring_bridge/word_wrinkle_2d.py"
    spec=importlib.util.spec_from_file_location("_v3_wrinkle_scalar_reader",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.extract_wrinkle_2d_metrics

@lru_cache(maxsize=512)
def csv_field(label):
    from src.medical_v2_schema import to_english_document
    name=label.split("-",1)[-1]
    try:
        converted=to_english_document({name:0})
        return next(iter(converted))
    except (ValueError,KeyError):
        return name

def leaves(doc, prefix="", selected=True):
    result={}
    if not isinstance(doc,dict):
        return result
    for key,value in doc.items():
        if key in ("region_metrics","region_distribution","regions","runs","instances","detections"):
            continue
        path=prefix+"."+key if prefix else key
        if isinstance(value,dict):
            result.update(leaves(value,path,selected))
        elif number(value) is not None and (not selected or key in FIELDS):
            result[path]=float(value)
    return result

def medical(doc):
    if not isinstance(doc,dict):
        return None
    if "medical_metrics_v2" in doc and isinstance(doc["medical_metrics_v2"],dict):
        return doc["medical_metrics_v2"]
    if any(k in doc for k in ("overall_metrics","region_metrics","full_face","region_distribution","总体指标","分区指标")):
        return doc
    return None

def project_series(project,project_name=None):
    candidates=[project.get("json"),project.get("public_metrics"),project.get("metrics")]
    candidates=[x.get(project_name,x) if isinstance(x,dict) and project_name else x for x in candidates]
    document=next((medical(x) for x in candidates if medical(x) is not None),None)
    if document is None:
        document=next((x for x in candidates if isinstance(x,dict)),{})
    result={}
    overall=document.get("overall_metrics",document.get("总体指标",document.get("full_face",document)))
    result["full_face"]=leaves(overall)
    regions=document.get("region_metrics",document.get("分区指标",document.get("region_distribution",{})))
    if isinstance(regions,list):
        regions={x.get("analysis_region") or x.get("region_name") or x.get("检测范围") or str(i):x for i,x in enumerate(regions)}
    if isinstance(regions,dict):
        for key,value in regions.items():
            if not isinstance(value,dict):
                continue
            region=REGION_ALIASES.get(key,key)
            if value.get("evaluation_status",value.get("评估状态")) in ("不可评估","UNASSESSABLE") or value.get("status") in ("INVALID","UNAVAILABLE"):
                continue
            result[region]=leaves(value)
    # Match by exact leaf name only when there is no conflicting duplicate.
    normalized={}
    for region,values in result.items():
        grouped={}
        for path,value in values.items():
            grouped.setdefault(path.rsplit(".",1)[-1],[]).append((path,value))
        normalized[region]={k:{"value":vs[0][1],"source_fields":[p for p,_ in vs]}
                            for k,vs in grouped.items() if len({v for _,v in vs})==1}
        aliases={"segment_count":"wrinkle_segment_count","wrinkle_pixels":"total_wrinkle_length_px","max_segment_length":"max_segment_length_px"}
        for old,new in aliases.items():
            if old in normalized[region] and new not in normalized[region]:
                normalized[region][new]=normalized[region][old]
    if project.get("csv"):
        for row in project["csv"]:
            tag=str(row.get("检测项目",""))
            if project_name=="uv_spots" and any(x in tag for x in ("卟啉","紫质","porphyrin")):continue
            if project_name=="porphyrin" and any(x in tag for x in ("紫外","UV","uv_spots")):continue
            label=row.get("检测范围") or row.get("分区")
            if not label or row.get("评估状态")=="不可评估":continue
            region=REGION_ALIASES.get(label,label)
            local=normalized.setdefault(region,{})
            for name,value in row.items():
                try:value=float(value)
                except (TypeError,ValueError):continue
                field=csv_field(name)
                if math.isfinite(value) and field in FIELDS and field not in local:
                    local[field]={"value":value,"source_fields":["CSV."+name]}
    for region,values in project.get("legacy_metrics",{}).items():
        local=normalized.setdefault(region,{})
        for field,value in values.items():
            if number(value) is not None:
                local[field]={"value":float(value),"source_fields":["saved_v2_medical_scope."+field]}
    return normalized

def normalize(data):
    output={}
    for name,project in data.get("projects",{}).items():
        if project.get("status") not in (None,"success"):
            continue
        for region,values in project_series(project,name).items():
            if name=="porphyrin":
                count=values.get("特征数量（个）",{}).get("value")
                area=values.get("有效皮肤面积（像素）",{}).get("value")
                ratio=values.get("高强度目标比例",{}).get("value")
                if count is not None and area and ratio is not None and 0<=ratio<=1:
                    values["porphyrin_high_density"]={"value":round(count*ratio)*100000/area,
                        "source_fields":["feature_count","high_intensity_target_ratio","valid_skin_area"]}
            for field,item in values.items():
                output[name+"|"+region+"|"+field]=item
    for group,regions in (("wrinkle_stable",("forehead","glabella","left_crow_feet","right_crow_feet")),
                          ("wrinkle_grooves",("left_nasolabial","right_nasolabial","left_marionette","right_marionette"))):
        for field in ("wrinkle_segment_count","total_wrinkle_length_px","max_segment_length_px"):
            entries=[output.get("wrinkle|"+r+"|"+field) for r in regions]
            if all(e is not None for e in entries):
                numbers=[e["value"] for e in entries]
                output[group+"|full_face|"+field]={"value":max(numbers) if field.startswith("max") else sum(numbers),
                    "source_fields":[p for e in entries for p in e["source_fields"]]}
    raw=data.get("projects",{}).get("wrinkle",{}).get("metrics",{})
    if isinstance(raw,dict) and raw.get("region_analysis_status")=="ok":
        extract_wrinkle_2d_metrics=wrinkle_extractor()
        for group,old,slot in (("wrinkle_stable","stable_wrinkles","08"),("wrinkle_grooves","structural_grooves","09")):
            proof=raw.get("report_group_overlays",{}).get(slot,{})
            if not isinstance(proof,dict) or not proof.get("selected_region_keys"):continue
            v=extract_wrinkle_2d_metrics(raw,old)
            for field,value in (("wrinkle_segment_count",v.segment_count),("total_wrinkle_length_px",v.total_length_px),("max_segment_length_px",v.max_segment_length_px)):
                output[group+"|full_face|"+field]={"value":value,"source_fields":["validated_v2_named_region_aggregation."+field]}
    features=data.get("aggregate",{}).get("scoring_features",{})
    relief=output.get("contour_firmness|full_face|midface_relative_relief_p90",{}).get("value")
    if relief is not None:
        for key,entry in list(output.items()):
            if key.startswith("texture|") and key.endswith("|p90_intensity"):
                region=key.split("|")[1]
                for direction in ("raised","depressed"):
                    area=output.get("texture|"+region+"|"+direction+"_like_area_ratio",{}).get("value")
                    if area is not None:
                        output["texture|"+region+"|p90_"+direction+"_relief_estimate"]={
                            "value":relief*entry["value"]*area,
                            "source_fields":["contour.midface_relative_relief_p90",key,direction+"_like_area_ratio"],
                            "method":"v2_relief_formula_with_actual_regional_area_and_response"}
    for module,groups in features.items():
        for path,value in leaves(groups,selected=False).items():
            output["features."+module+"|full_face|"+path]={"value":value,"source_fields":["scoring_features."+module+"."+path]}
    return output
