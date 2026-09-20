"""Score saved inputs with explicit references and retain both measurement and score traces."""
from bisect import bisect_left,bisect_right
from datetime import datetime,timezone
import math
from .registry import MODULES,measurement,METRIC_LABELS
from .scoring import aggregate,grade,display_score
from .historical_metrics import normalize
from .compatibility_recipes import FEATURES,RAW,UNITS,HIGHER_HEALTH,VERSION
from .paired_calibration import predict
from .large_pore_estimation import estimate_large_density

PARENT_REFERENCE={"left_inner_cheek":"left_cheek","left_outer_cheek":"left_cheek",
                  "right_inner_cheek":"right_cheek","right_outer_cheek":"right_cheek",
                  "jaw":"chin"}
EMPTY={"score_raw":None,"score":None,"grade":"不可评估"}

def evaluate(value,reference,higher_health=False):
    if value is None or not reference:
        return dict(EMPTY)
    values=reference["values"]
    if not values or not math.isfinite(value):return dict(EMPTY)
    rank=(bisect_left(values,value)+bisect_right(values,value))/(2*len(values))
    raw=100*rank if higher_health else 100*(1-rank)
    if not higher_health and value==0 and values[0]>=0:raw=100.0
    return {"score_raw":raw,"score":display_score(raw),"grade":grade(raw)}

def formatted(value,unit):
    if value is None:return "—"
    if unit=="比例":return f"{value*100:.2f}%"
    if unit in ("个","处","条"):return str(int(round(value)))+unit
    return f"{value:.5g}"+(" "+unit if unit else "")

def build(payload,data,catalog):
    normalized=normalize(data)
    measured={(x["module"],x["metric_id"][3:],x["region"]):x for x in payload["measurements"]}
    strict=payload.get("score_trace",{})
    refs=catalog["references"]
    traces={}
    def item(module,name,region):
        key=(module,name,region);m=measured.get(key,{})
        extra_names={"length":"纹路总长度","maximum":"最长纹路线段","count":"纹路线段数量"}
        result={"name":m.get("name",METRIC_LABELS.get(name,extra_names.get(name,name))),"value":m.get("value"),
                "unit":m.get("unit",UNITS.get(name,"标准化指数")),**EMPTY}
        lookup=None;value=None;method=None;reference=None;ref_key=None
        raw=RAW.get(module,{}).get(name)
        ref_region=PARENT_REFERENCE.get(region,region)
        if raw:
            project,field=raw
            if module=="08" and region=="full_face":project="wrinkle_stable"
            source_key=project+"|"+region+"|"+field
            entry=normalized.get(source_key)
            if project=="texture" and region in ("left_jaw","right_jaw"):
                named={"left_jaw":"画面左下颌","right_jaw":"画面右下颌"}[region]
                source_key=project+"|"+named+"|"+field
                entry=normalized.get(source_key)
                ref_region=named
            if project=="wrinkle" and entry is None and region in ("left_under_eye","right_under_eye"):
                named={"left_under_eye":"左眼下细纹","right_under_eye":"右眼下细纹"}[region]
                source_key=project+"|"+named+"|"+field
                entry=normalized.get(source_key)
                if entry:ref_region=named
            transferred=False
            if entry is None and ref_region!=region and m.get("value") is not None and module=="01":
                entry={"value":m["value"],"source_fields":[m["metric_id"]+":"+region]}
                transferred=True
            if entry is None and m.get("value") is not None and module in ("02","03","04","05","07","08","10"):
                entry={"value":m["value"],"source_fields":[m["metric_id"]+":"+region]}
                transferred=True
            if entry:
                lookup=source_key;value=entry["value"]
                for cohort in ("development1000","legacy"):
                    candidate=cohort+"|"+project+"|"+ref_region+"|"+field
                    if candidate in refs:
                        ref_key=candidate;reference=refs[candidate];break
                if reference is None and module=="02" and name=="gloss_mean":
                    candidate="development1000|surface_gloss|"+ref_region+"|mean_gloss_intensity_estimate"
                    if catalog.get("gloss_mean_estimator",{}).get("accepted") and candidate in refs:
                        ref_key=candidate;reference=refs[candidate]
                method="same_metric_reference" if ref_region==region else "parent_region_reference_transfer"
                if transferred:method="v3_measurement_compatible_reference_transfer"
                mapping=catalog.get("paired_mappings",{}).get(project+"|"+region+"|"+field,{})
                if mapping.get("accepted") and payload["capture_profile"]=="consumer" and not transferred:
                    old_key="legacy|"+project+"|"+region+"|"+field
                    if old_key in refs:
                        value=predict(mapping["knots"],value);ref_key=old_key;reference=refs[old_key]
                        method="held_out_validated_monotone_mapping"
        if reference is None and region=="full_face":
            feature=FEATURES.get(module,{}).get(name)
            if feature:
                family,path=feature.split(".",1)
                source_key="features."+family+"|full_face|"+path
                entry=normalized.get(source_key)
                candidate="development1000|"+source_key
                if entry and candidate in refs:
                    lookup=source_key;value=entry["value"];reference=refs[candidate];ref_key=candidate
                    method="v2_explicit_formula_with_v3_report_weights"
        if name=="large_density" and module=="01":
            for cohort in ("development1000","legacy"):
                threshold=catalog.get("large_pore_thresholds",{}).get(cohort+"|"+ref_region)
                candidate=cohort+"|pores|"+ref_region+"|large_density_estimate"
                if threshold and candidate in refs:
                    values=[normalized.get("pores|"+region+"|"+f,{}).get("value") for f in
                            ("feature_density_per_100k_skin_px","p50_instance_area_px","p90_instance_area_px","max_instance_area_px")]
                    estimate=estimate_large_density(*values,threshold)
                    if estimate is not None:
                        value=estimate;reference=refs[candidate];ref_key=candidate
                        method="quantile_piecewise_large_density_estimate";lookup="pores_size_summary:"+region
                        result.update(value=estimate,unit="密度指数");break
        if reference:
            scored=evaluate(value,reference,(module,name) in HIGHER_HEALTH)
            result.update(scored)
            if module in ("07","08","09") and name in ("count","length","maximum") and result["value"] is None and lookup and lookup.startswith("wrinkle|"):
                # These are explicitly named V2 line-summary fields, not missing
                # V3 area/depth measurements. Preserve their actual saved values.
                result["value"]=entry["value"]
                result["unit"]="条" if name=="count" else "标准化px"
                result["measurement_basis"]="saved_v2_named_wrinkle_summary"
            if module=="02" and name=="porphyrin_high_density" and result["value"] is None and raw:
                result["value"]=value;result["unit"]="个/10万有效像素"
            if module=="10" and result["value"] is None:
                result["value"]=value
                result["unit"]="比例" if name.endswith("_area") else "图像指数"
            # A compatible score is not permitted to fabricate a missing measurement.
            if result["value"] is None and method=="v2_explicit_formula_with_v3_report_weights":
                result["value"]=value;result["unit"]="图像指数"
            traces[module+"."+name+":"+region]={"source":lookup,"method":method,"score_input":value,
                "reference":ref_key,"population_n":reference["n"],"reference_region":ref_region,
                "measurement_region":region,"profile":payload["capture_profile"],
                "cross_profile_reference":payload["capture_profile"]=="institution",**scored}
        else:
            old=strict.get(module+"."+name+":"+region,{})
            if old.get("score") is not None:
                result.update({k:old[k] for k in ("score","grade","score_raw")})
                traces[module+"."+name+":"+region]={"method":"previous_same_definition_reference",**old}
        if module=="06" and name.endswith("_density") and m.get("status")=="measured" and m.get("value")==0:
            result.update(score_raw=100.0,score=100,grade="未见明显")
            traces[module+"."+name+":"+region]={"method":"observed_zero_target_anchor","value":0,"score":100}
        result["display"]=formatted(result["value"],result["unit"])
        return result
    modules={}
    for definition in MODULES:
        mid=definition.id;regions={};metrics={}
        names=list(definition.weights)
        if mid=="03":names=[layer+"."+x for layer in ("spots","brown","uv") for x in ("area","p90")]
        if mid=="08":names=["main_count","length_burden","depth","area","density","high_area","contrast_p50","count","length","maximum"]
        if mid=="07":names+=["count","length","maximum"]
        if mid=="09":names+=["count","length","maximum"]
        if mid=="05":names+=["roi_count"]
        if mid=="06":names+=["erythema_count","papule_count","pustule_count"]
        for region in ("full_face",*definition.regions):
            local={name:item(mid,name,region) for name in names}
            zero_terms=[]
            if mid=="01" and local["density"].get("value")==0:zero_terms=["area_p50","large_density"]
            if mid=="02":
                if local["gloss_area"].get("value")==0:zero_terms+=["gloss_mean"]
                count=normalized.get("porphyrin|"+region+"|特征数量（个）",{}).get("value")
                if count==0:zero_terms+=["porphyrin_p90"]
            if mid=="07" and local["area"].get("value")==0:zero_terms=list(definition.weights)
            if mid=="05" and local["clusters"].get("value")==0 and local["affected_area"].get("value")==0:
                zero_terms=["local_density"]
            if mid=="08" and region!="full_face" and local["area"].get("value")==0:zero_terms=["main_count","length","maximum"]
            for term in zero_terms:
                local[term].update(score_raw=100.0,score=100,grade="未见明显")
                traces[mid+"."+term+":"+region]={"method":"no_target_burden_in_valid_measured_scope",
                                               "value":local[term].get("value"),"score":100}
            metrics[region]=local
            weights=definition.weights
            if mid=="03":
                for layer in ("spots","brown","uv"):
                    local[layer]={"name":{"spots":"色斑","brown":"棕色色素","uv":"UV色素"}[layer],
                                  **aggregate({n:local[layer+"."+n] for n in ("area","p90")},{"area":.55,"p90":.45})}
                    local[layer]["display"]=local[layer]["grade"]
                components={k:local[k] for k in weights}
            elif mid=="08":
                weights={"count":.45,"length":.35,"maximum":.20}
                components={k:local[k] for k in weights}
            elif mid=="09" and region!="full_face":
                weights={"count":.45,"length":.35,"maximum":.20}
                components={k:local[k] for k in weights}
            else:
                components={k:local[k] for k in weights}
                if mid=="07" and local["contrast_p50"]["score"] is None:
                    weights={k:v/.85 for k,v in weights.items() if k!="contrast_p50"}
                    components={k:local[k] for k in weights}
            regions[region]=aggregate(components,weights)
            if mid=="08":
                regions[region]["compatibility_definition"]="saved_v2_named_segment_state_not_v3_mainline_measurement"
            if mid=="07" and region!="full_face" and regions[region]["score"] is None:
                alternative=aggregate({k:local[k] for k in ("count","length","maximum")},{"count":.45,"length":.35,"maximum":.2})
                if alternative["score"] is not None:
                    alternative["method"]="v2_named_regional_wrinkle_compatibility"
                    regions[region]=alternative
            if mid=="06":
                for kind in ("erythema","papule","pustule"):
                    for k in ("score","score_raw","grade"):local[kind+"_count"][k]=local[kind+"_density"][k]
            if mid=="05":
                for k in ("score","score_raw","grade"):local["roi_count"][k]=local["affected_area"][k]
            if mid=="02":
                for sub,weights2 in (("surface_gloss",{"gloss_area":.25/.6,"gloss_high_area":.2/.6,"gloss_mean":.15/.6}),
                                     ("porphyrin",{"porphyrin_high_density":.7,"porphyrin_p90":.3})):
                    local[sub]={"name":{"surface_gloss":"表面油光","porphyrin":"毛囊卟啉"}[sub],**aggregate({k:local[k] for k in weights2},weights2)}
                    local[sub]["display"]=local[sub]["grade"]
        has=any(x.get("value") is not None for rs in metrics.values() for x in rs.values())
        positive=[x["value"] for rs in metrics.values() for x in rs.values() if x.get("value") is not None]
        modules[mid]={"title":definition.title,**regions.pop("full_face"),"regions":regions,
                      "metrics":metrics,"has_measurements":has,"detected":any(v>0 for v in positive) if positive else None}
    model={"schema_version":"doctor_v3_word_model_v1","subject_id":payload["subject_id"],
           "capture_profile":payload["capture_profile"],"captured_at":payload.get("captured_at"),
           "generated_at":datetime.now(timezone.utc).isoformat(),"modules":modules,
           "quality_label":"合格" if data.get("quality",{}).get("status")=="PASS" else "需关注",
           "calibration_version":VERSION,"dataset_sha256":catalog["dataset_sha256"],"score_trace":traces}
    return model
