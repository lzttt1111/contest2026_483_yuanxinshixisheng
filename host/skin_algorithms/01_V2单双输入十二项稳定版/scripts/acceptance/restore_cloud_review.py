import argparse,copy,hashlib,html,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cloud.simulated_contract_versions import install_simulated_contract_versions
install_simulated_contract_versions()
from cloud.simulate_cloud_request import _write_html
from cloud_contracts import validate_worker_envelope
CASES=("clinic28-25","clinic28-09","clinic28-23")
ALGOS=("redness","spots","brown","texture","pores","purple","surface_gloss","vascular","contour_firmness","acne","wrinkle")
def strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from strings(v)
    elif isinstance(value,list):
        for v in value:yield from strings(v)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();dest=root/"web_review_r2"
    if dest.exists():raise FileExistsError("fresh review output required")
    dest.mkdir()
    receipts=[]
    for case in CASES:
        page=dest/case;page.mkdir()
        tasks={}
        institution=root/"word_crosscheck"/"cloud_institution"/case
        index=json.loads((institution/"十二项检测结果索引.json").read_text())
        for algo in ALGOS:
            service=algo if algo in ("acne","wrinkle") else ("dermavision25" if case=="clinic28-25" else "dermavision09_23")
            source=root/service
            original=json.loads((source/case/(algo+".json")).read_text())
            response=copy.deepcopy(original)
            for v in strings(original):
                path=source/"storage"/v
                try: exists=path.is_file()
                except OSError:exists=False
                if exists:
                    target=page/"simulated_oss"/v
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            if algo in ("redness","brown"):
                item=index["十二项结果"][algo]
                base=institution/item["附加结果图"][0]
                marked=institution/item["主结果图"]
                extra="red_areas_overlay" if algo=="redness" else "brown_spots_overlay"
                for field,path in (("overlay",base),(extra,marked)):
                    key=f"report/{case}-{algo}/{algo}/{path.name}"
                    target=page/"simulated_oss"/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
                    response["raw_result"][field]=key
                assert response["raw_result"]["metrics"]==original["raw_result"]["metrics"]
                receipts.append({"case":case,"algorithm":algo,"base_sha256":sha(base),"annotated_sha256":sha(marked),
                    "metrics_unchanged":True,"source":"existing cloud institution report aggregate; media-only republication"})
            validate_worker_envelope("acne_v2" if algo=="acne" else algo,response)
            elapsed=next(r["seconds"] for r in json.loads((source/"WORKER_RECEIPT.json").read_text()) if r["case"]==case and r["algorithm"]==algo)
            tasks[algo]={"queue":"acne_v2" if algo=="acne" else algo,"task":"dermavision.analyze_image" if algo!="wrinkle" else "wrinkle.analyze_image",
                         "elapsed_seconds":elapsed,"pydantic_validation":"passed","response":response}
        reports=[]
        for doc in sorted(institution.glob("*.docx")):
            reports.append(os.path.relpath(doc,page).replace(os.sep,"/"))
        bundle={"tasks":tasks,"task_count":11,"projected_result_count":12,"capture_profile":"institution",
                "medical_report_files":reports,"simulation_note":"Existing Worker results, corrected media projection; no inference rerun",
                "media_republished":True}
        (page/"cloud_response_bundle.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2))
        shutil.copy2(institution/"00_输入图像"/"RGB_M.jpg",page/"00_输入图片.jpg")
        _write_html(Path(case+".jpg"),page,bundle,sum(x["elapsed_seconds"] for x in tasks.values()))
        path=page/"index.html";text=path.read_text()
        text=text.replace("AISIA 单RGB12项云端返回模拟","AISIA 四光源机构12项云端返回模拟")
        text=text.replace('<a href="/docs" target="_blank">Pydantic Swagger接口文档</a> ·',"")
        note='<p class="hint">四光源输入。红棕图左侧overlay为纯底图，右侧实例字段为带线框特征点图。本页复用原版中文字段说明模板。基于已完成结果修正媒体投影，未重新执行检测；非生产部署回执。整体单双管线与Word外观终审尚未完成。</p>'
        text=text.replace('<div class="summary">','<div class="summary">'+note,1)
        path.write_text(text,encoding="utf-8")
    (dest/"MEDIA_REPROJECTION.json").write_text(json.dumps(receipts,ensure_ascii=False,indent=2))
    old=root/"index.html"
    shutil.copy2(old,dest/"previous_landing.html")
    body='<!doctype html><meta charset="utf-8"><title>云端验收入口</title><h1>四光源云端验收 原版展示</h1><p>修正红棕媒体映射，恢复中文指标及Worker字段说明。单RGB完整验收、Word外观终审尚未完成。</p><ul>'
    for case in CASES:body+=f'<li><a href="web_review_r2/{case}/index.html">{case}</a></li>'
    old.write_text(body+"</ul>",encoding="utf-8")
    print("PASS 3 pages, 33 envelopes, 6 media corrections; no inference")
if __name__=="__main__":main()
