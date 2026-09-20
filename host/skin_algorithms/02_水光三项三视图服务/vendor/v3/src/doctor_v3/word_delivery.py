"""Prepare a portable, V2-shaped result directory for text-only doctor V3 Word."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from .bundle import sha256,write_json,contained

def sanitize(value):
    if isinstance(value,dict):
        return {k:sanitize(v) for k,v in value.items() if k not in
                ("weights","source","output_dir","platform","python_version","device","runs")}
    if isinstance(value,list):return [sanitize(x) for x in value]
    if isinstance(value,str) and (value.startswith(("/home/","/mnt/","/tmp/","\\\\")) or (len(value)>2 and value[1:3] in (":\\",":/"))):
        return None
    return value

def prepare(source,bundle_root,catalog_path,destination,cloud=False,additional_metrics=None,frozen_v2_root=None,in_place=False):
    from .common_result_features import common_features
    from .clinical_scoring import build
    source,bundle_root,destination=Path(source),Path(bundle_root),Path(destination)
    if destination.exists() and not in_place:raise FileExistsError("new result destination required")
    if in_place and (source.resolve()!=destination.resolve() or (destination/"报告数据").exists()):
        raise ValueError("in-place report requires current result without prior report data")
    with gzip.open(catalog_path,"rt",encoding="utf-8") as stream:catalog=json.load(stream)
    payload,data=common_features(bundle_root,source,cloud,additional_metrics)
    model=build(payload,data,catalog)
    original_scoring=model
    from .protected_scoring import protect
    model=protect(model,payload,data,catalog,bundle_root)
    from .additive_scoring import supplement as supplement_additive
    model=supplement_additive(model,data,catalog,bundle_root)
    destination.mkdir(parents=True,exist_ok=in_place)
    if not cloud:
        for filename in ("十二项检测结果索引.json","十二项完整量化指标.json","运行回执.json","00_输入图片.jpg"):
            p=source/filename
            if p.is_file() and not in_place:shutil.copyfile(p,destination/filename)
        for dirname in ("十二项检测","00_输入图像"):
            p=source/dirname
            if p.is_dir() and not in_place:shutil.copytree(p,destination/dirname)
        complete=json.loads((destination/"十二项完整量化指标.json").read_text(encoding="utf-8"))
    else:
        complete={"schema_version":"doctor_v3_cloud_result_projection_v1",
                  "provenance":{"capture_profile":"consumer","source":"actual_worker_results"},
                  "detector_results":sanitize(data["projects"])}
        bundle=json.loads((source/"cloud_response_bundle.json").read_text(encoding="utf-8"))
        items={}
        names={"redness":"01_红区","spots":"02_斑点","brown":"03_棕区","texture":"04_纹理",
               "pores":"05_毛孔","uv_spots":"06_UV色斑","porphyrin":"07_卟啉","wrinkle":"08_皱纹",
               "acne":"09_痤疮","surface_gloss":"10_表面油光","vascular":"11_血管","contour_firmness":"12_轮廓紧致度"}
        for task_name,task in bundle["tasks"].items():
            raw=task["response"]["raw_result"]
            roles={"uv_spots":["uv_base","uv_spots_overlay"],"porphyrin":["fluorescence_base","porphyrin_overlay"]} if task_name=="purple" else {
                "acne" if task_name=="acne_v2" else task_name:
                ["stage2_overlay","region_overlay"] if task_name=="wrinkle" else
                ["overlay","red_areas_overlay"] if task_name=="redness" else
                ["overlay","brown_spots_overlay"] if task_name=="brown" else ["overlay"]}
            for project,keys in roles.items():
                folder=destination/"十二项检测"/names[project];folder.mkdir(parents=True)
                images=[]
                for i,k in enumerate(keys,1):
                    origin=contained(source,"simulated_oss/"+raw[k])
                    target=folder/f"{i:02d}_{k}{origin.suffix}"
                    shutil.copyfile(origin,target)
                    images.append({"path":target.relative_to(destination).as_posix(),"sha256":sha256(target),"role":k})
                metrics_path=folder/"量化指标.json"
                write_json(metrics_path,sanitize(data["projects"].get(project,raw.get("metrics",{}))))
                items[project]={"状态":"success","主结果图":images[0]["path"],"附加结果图":[x["path"] for x in images[1:]],
                    "images":images,"量化JSON":metrics_path.relative_to(destination).as_posix()}
                report_key=raw.get("medical_report_csv_v2") or task["response"].get("debug_info",{}).get("report_csv")
                if report_key:
                    origin=contained(source,"simulated_oss/"+report_key)
                    target=folder/"医学量化指标_V2.csv";shutil.copyfile(origin,target)
                    items[project]["医学V2CSV"]=target.relative_to(destination).as_posix()
        write_json(destination/"十二项检测结果索引.json",{"schema_version":"single_rgb_twelve_index_v3","capture_profile":"consumer","status":"success","items":items})
        write_json(destination/"运行回执.json",{"capture_profile":"consumer","new_inference_executed":False,"source_task_count":len(bundle["tasks"])})
        origin=source/"00_输入图片.jpg"
        if origin.is_file():shutil.copyfile(origin,destination/origin.name)
    report_data=destination/"报告数据";report_data.mkdir()
    if True:  # Compatibility is replayed from this detection, not a frozen patient.
        def input_hashes(root):
            root=Path(root)
            inputs=list(root.glob("00_输入图片.*"))+list((root/"00_输入图像").glob("*.jpg"))
            return sorted(sha256(p) for p in inputs if p.is_file())
        actual=input_hashes(destination)
        expected=input_hashes(frozen_v2_root) if frozen_v2_root is not None else actual
        if not actual or actual!=expected:raise ValueError("V2 quality reference input mismatch")
        from .v2_saved_replay import replay
        from .legacy_state_compatibility import supplement
        replay_complete=dict(complete)
        if cloud:
            replay_complete.update(detector_results=data["projects"],scoring_features=data["aggregate"]["scoring_features"])
        v2=replay(destination,replay_complete,data.get("quality",{}).get("status")=="PASS")
        model=supplement(model,v2)
        model["legacy_input_proof"]={"input_sha256":actual,"frozen_input_identical":frozen_v2_root is not None,
                                    "proof_kind":"current_input_and_quality"}
        write_json(report_data/"v2_compatible_scoring.json",v2)
        from .v2_saved_replay import assets
        legacy_profile,legacy_sha,_,_,_=assets()
        used_ids={metric["metric_id"] for row in v2["official_v011"].get("formal_dimension_scores",[])
                  for group in row.get("groups",[]) for metric in group.get("metrics",[])}
        legacy_reference={"source_sha256":legacy_sha,"references":{k:legacy_profile["references"][k]
                          for k in used_ids if k in legacy_profile["references"]}}
        with gzip.open(report_data/"legacy_v011_reference.json.gz","wt",encoding="utf-8") as stream:
            json.dump(legacy_reference,stream,ensure_ascii=False)
    from .named_roi_proxy import apply as apply_named_roi_proxy
    model=apply_named_roi_proxy(model,original_scoring,data,catalog)
    from .historical_proxy_view import apply as apply_historical_proxy_view
    model=apply_historical_proxy_view(model)
    from .native_measurement_provenance import restore
    model=restore(model,payload["measurements"])
    if catalog.get("additive_reference_version"):
        model["calibration_version"]="doctor_v3_saved_history_completion_20260909_v1"
    from .score_availability import annotate
    model=annotate(model)
    write_json(report_data/"report.json",model)
    write_json(report_data/"severity_policy.json",model["severity_policy"])
    write_json(report_data/"v3_measurements.json",payload["measurements"])
    # Retain the compact configuration actually used; not the full historical corpus.
    used={x.get("reference") for x in model["score_trace"].values() if x.get("reference")}
    def compatibility_references(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in ("reference","reference_key") and isinstance(item,str) and item in catalog["references"]:used.add(item)
                else:compatibility_references(item)
        elif isinstance(value,list):
            for item in value:compatibility_references(item)
    compatibility_references(model["modules"])
    subset={k:v for k,v in catalog.items() if k not in ("references","paired_mappings")}
    subset["references"]={k:catalog["references"][k] for k in sorted(used)}
    subset["paired_mappings"]={k:v for k,v in catalog.get("paired_mappings",{}).items() if v.get("accepted")}
    with gzip.open(report_data/"scoring_config.json.gz","wt",encoding="utf-8") as stream:
        json.dump(subset,stream,ensure_ascii=False,separators=(",",":"))
    templates=Path(__file__).resolve().parents[2]/"templates/doctor_v3_text"
    shutil.copytree(templates,report_data/"templates")
    # Evidence is for future result-level recalculation; no duplicate media tree.
    shutil.copytree(bundle_root/"evidence",report_data/"evidence")
    if additional_metrics:
        shutil.copyfile(additional_metrics,report_data/"evidence/contour_complete.json")
    complete["doctor_v3"]={"report_data":"报告数据/report.json","report_sha256":sha256(report_data/"report.json"),
                           "measurements":payload["measurements"],"scores":model["modules"],
                           "score_trace":model["score_trace"],"calibration_version":model["calibration_version"]}
    write_json(destination/"十二项完整量化指标.json",complete)
    rows=[]
    for m,mod in model["modules"].items():
        for region,values in mod["metrics"].items():
            for key,item in values.items():
                rows.append({"模块":m,"分区":region,"指标":key,"测量值":item.get("value"),
                             "单位":item.get("unit"),"状态评分":item.get("score"),"等级":item.get("grade")})
    with (report_data/"V3医学量化指标.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=[{"path":p.relative_to(destination).as_posix(),"sha256":sha256(p),"bytes":p.stat().st_size}
              for p in sorted(report_data.rglob("*")) if p.is_file()]
    write_json(report_data/"manifest.json",{"schema_version":"doctor_v3_word_package_v1","files":manifest})
    from .word_package import finalize
    finalize(destination)
    return model

def read_model(result_dir,previous_result=None):
    root=Path(result_dir);data=root/"报告数据"
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path=contained(root,entry["path"])
        if sha256(path)!=entry["sha256"]:raise ValueError("report data hash mismatch")
    model=json.loads((data/"report.json").read_text(encoding="utf-8"))
    if previous_result:
        previous=read_model(previous_result)
        if previous["subject_id"]!=model["subject_id"]:raise ValueError("different subject")
        from datetime import datetime
        compatible=all(previous.get(k)==model.get(k) for k in ("capture_profile","calibration_version","dataset_sha256","stage1_versions"))
        try:
            before=datetime.fromisoformat(previous["captured_at"])
            current=datetime.fromisoformat(model["captured_at"])
            compatible=compatible and before.tzinfo is not None and current.tzinfo is not None and before<current
        except (ValueError,TypeError,KeyError):
            compatible=False
        if compatible:model["previous"]=previous
        else:model["history_unavailable_reason"]="incompatible_configuration_or_acquisition"
    return model
