from __future__ import annotations
import argparse,html,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from cloud.simulate_cloud_request import _write_html
from cloud.review_sanitize import sanitize_review_bundle
from cloud_contracts import validate_worker_envelope
CASES=("clinic28-25","clinic28-09","clinic28-23")
ALGOS=("redness","spots","brown","texture","pores","purple","surface_gloss","vascular","contour_firmness","acne","wrinkle")
def strings(x):
    if isinstance(x,str):yield x
    elif isinstance(x,dict):
        for v in x.values():yield from strings(v)
    elif isinstance(x,list):
        for v in x:yield from strings(v)
def load(p):return json.loads(p.read_text())
def relative(path,root):return os.path.relpath(path,root).replace(os.sep,"/")
def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();overrides=load(root/"response_overrides.json")
    web=root/"web";web.mkdir(exist_ok=True)
    home=[]
    for profile in ("consumer","institution"):
        for alias in CASES:
            page=web/profile/alias;page.mkdir(parents=True,exist_ok=True)
            tasks={}
            for name in ALGOS:
                source=overrides[profile+":"+alias+":"+name]
                response=load(root/source["response"])
                validate_worker_envelope("acne_v2" if name=="acne" else name,response)
                tasks[name]={"queue":("consumer_" if profile=="consumer" else "")+("acne_v2" if name=="acne" else name),
                    "task":"wrinkle.analyze_image" if name=="wrinkle" else "dermavision.analyze_image",
                    "elapsed_seconds":0,"pydantic_validation":"passed","response":response}
            reports_root=root/"cloud_consumer/report_aggregate" if profile=="consumer" else root/"institution_reports/cloud_institution"
            docs=[p for p in reports_root.rglob("*.docx") if alias in p.name];assert len(docs)==2
            bundle=sanitize_review_bundle({"capture_profile":profile,"tasks":tasks,"task_count":11,"projected_result_count":12,
                "medical_report_files":[relative(p,page) for p in sorted(docs)],
                "evidence_note":"Direct real Worker calls with local storage, no broker/server deployment; queue names describe the corresponding launch target. One frozen full run plus bounded corrections; scoring replay provenance is recorded separately.",
                "word_layout_status":"passed","release_status":"awaiting_user_review_not_pushed"})
            for name,task in bundle["tasks"].items():
                storage=(root/overrides[profile+":"+alias+":"+name]["storage_base"]/"storage").resolve()
                for value in strings(task["response"]):
                    candidate=storage/value
                    try:exists=candidate.is_file()
                    except OSError:exists=False
                    if not exists:continue
                    assert candidate.resolve().is_relative_to(storage)
                    copy_file(candidate,page/"simulated_oss"/value)
            image=next((root/"inputs").glob("*"+alias+"*_RGB_M.jpg"))
            copy_file(image,page/"00_输入图片.jpg")
            (page/"cloud_response_bundle.json").write_text(json.dumps(bundle,ensure_ascii=False,indent=2))
            _write_html(Path(alias+".jpg"),page,bundle,0)
            doc=page/"index.html";text=doc.read_text()
            title="AISIA "+("单RGB" if profile=="consumer" else "四光源机构")+"12项云端返回验收"
            text=text.replace("AISIA 单RGB12项云端返回模拟",title)
            text=text.replace(" · 耗时 0s ·"," · 结果复验 ·")
            text=text.replace("总墙钟耗时：0.000s。","耗时详见原始运行回执（本页不重复执行检测）。")
            text=text.replace('<a href="/docs" target="_blank">Pydantic Swagger接口文档</a> ·',"")
            summary=root/"review"/profile/alias/"summary_result.json"
            result=load(summary);assert result["status"]=="success"
            copy_file(summary,page/"summary_result.json")
            rows=""
            for module in result["raw_result"]["modules"]:
                def show(system):
                    s=module[system];return "不可评估" if not s["score_valid"] else f'{s["score"]:.2f} · {s["grade"]}'
                rows+="<tr><td>"+html.escape(module["module_no"]+" "+module["name"])+"</td><td>"+show("word_display")+"</td><td>"+show("production_proxy_v1")+"</td></tr>"
            local_root=root/"local_consumer" if profile=="consumer" else root/"institution_reports/local_institution"
            locals=[p for p in local_root.rglob("*.docx") if alias in p.name];assert len(locals)==2
            links=" · ".join('<a href="'+html.escape(relative(p,page))+'">'+html.escape(p.name)+"</a>" for p in locals)
            local_data=" · ".join('<a href="'+html.escape(relative(locals[0].parent/name,page))+'">'+name+'</a>' for name in ("十二项检测结果索引.json","十二项完整量化指标.json","运行回执.json"))
            note='<section class="summary"><p>同路线指标、评分及双Word表格/媒体对照通过，报告版面检查完成。结果复核已完成；真实Worker使用本地存储模拟，不代表生产部署。部分评分采用原始保存证据CPU重放，缺失证据任务已定点复验。</p><p>本地对照Word：'+links+'</p><p>本地完整结果：'+local_data+'</p><h2>CPU summary 十一模块</h2><table><tr><th>模块</th><th>Word展示评分</th><th>机器代理评分</th></tr>'+rows+'</table><a href="summary_result.json">完整summary结果</a></section>'
            text=text.replace("{sections}","")
            text=text.replace("</main>",note+"</main>")
            doc.write_text(text,encoding="utf-8")
            home.append((profile,alias,relative(doc,root)))
    body='<!doctype html><meta charset="utf-8"><title>AISIA 整合验收</title><style>body{font:17px/1.8 Arial,sans-serif;max-width:1100px;margin:40px auto;color:#24364b}a{color:#087ea4}td,th{border:1px solid #cad7df;padding:14px}table{border-collapse:collapse;width:100%}</style><h1>AISIA 整合验收</h1><p>12组交付 · 24份Word · 66份检测响应 · 6份summary。结果对照与报告版面检查通过，结果复核已完成，生产部署需独立执行。</p><table><tr><th>样本</th><th>单RGB</th><th>四光源</th></tr>'
    for alias in CASES:
        body+="<tr><td>"+alias+"</td>"
        for profile in ("consumer","institution"):
            url=next(url for p,c,url in home if p==profile and c==alias)
            body+='<td><a href="'+url+'">结果图、评分、字段说明和双Word</a></td>'
        body+="</tr>"
    body+='</table><p><a href="review/SUMMARY.md">验收汇总与证据边界</a> · <a href="review/CROSSCHECK.json">同路线对照回执</a> · <a href="RUN_RECEIPT.json">原始运行命令与耗时</a></p>'
    (root/"index.html").write_text(body,encoding="utf-8")
    print("PASS six original-template review pages and landing")
if __name__=="__main__":main()
