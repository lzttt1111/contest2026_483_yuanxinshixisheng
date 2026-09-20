"""Finalize V3 quantitative tables in the familiar V2 sample tree, without rescoring."""
import csv,json
from pathlib import Path
from .bundle import sha256,write_json
from .registry import REGION_LABELS

PROJECT_MODULES={"pores":["01"],"surface_gloss":["02"],"porphyrin":["02"],"spots":["03"],"brown":["03"],
 "uv_spots":["03"],"redness":["04"],"vascular":["05"],"acne":["06"],"wrinkle":["07","08","09"],
 "texture":["10"],"contour_firmness":["11"]}
def finalize(root):
    root=Path(root);data=root/"报告数据"
    model=json.loads((data/"report.json").read_text(encoding="utf-8"))
    rows=[]
    for mid,module in model["modules"].items():
        for region,metrics in module["metrics"].items():
            for key,item in metrics.items():
                trace=model["score_trace"].get(mid+"."+key+":"+region,{})
                rows.append({"模块编号":mid,"模块名称":module["title"],"分区":REGION_LABELS.get(region,region),
                  "指标ID":mid+"."+key,"指标名称":item.get("name",key),"测量值":item.get("value"),"单位":item.get("unit"),
                  "状态评分":item.get("score"),"等级":item.get("grade"),"评分输入值":trace.get("score_input",trace.get("value")),
                  "评分方法":trace.get("method"),"参考配置":trace.get("reference",trace.get("reference_version")),
                  "配置版本":trace.get("reference_version") or model["calibration_version"]})
    for mid,module in model["modules"].items():
        for region,item in [("full_face",module),*module["regions"].items()]:
            rows.append({"模块编号":mid,"模块名称":module["title"],"分区":REGION_LABELS.get(region,region),
                "指标ID":mid+"._module","指标名称":module["title"]+"状态评分","测量值":None,"单位":"分",
                "状态评分":item.get("score"),"等级":item.get("grade"),"评分输入值":item.get("raw_statistical_score"),
                "评分方法":item.get("method","guarded_composite"),"参考配置":item.get("assessment_basis"),
                "配置版本":model["calibration_version"]})
    def save(path,selected):
        with path.open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(selected)
    save(data/"V3医学量化指标.csv",rows)
    index_path=root/"十二项检测结果索引.json";index=json.loads(index_path.read_text(encoding="utf-8"))
    items=index.get("items",index.get("十二项结果"))
    for project,item in items.items():
        folder=(root/item["主结果图"]).parent
        path=folder/"医学量化指标_V3.csv"
        save(path,[r for r in rows if r["模块编号"] in PROJECT_MODULES[project]])
        item["医学V3CSV"]=path.relative_to(root).as_posix()
        item["医学V3CSV_SHA256"]=sha256(path)
    write_json(index_path,index)
    files=[p for p in data.rglob("*") if p.is_file() and p.name!="manifest.json"]
    files += [root/n for n in ("十二项检测结果索引.json","十二项完整量化指标.json","运行回执.json")]
    files += [root/item["医学V3CSV"] for item in items.values()]
    files += [root/entry["path"] for groups in model.get("module_figures",{}).values() for group in groups for entry in group]
    entries=[{"path":p.relative_to(root).as_posix(),"sha256":sha256(p),"bytes":p.stat().st_size} for p in sorted(set(files))]
    write_json(data/"manifest.json",{"schema_version":"doctor_v3_word_package_v1","files":entries})
