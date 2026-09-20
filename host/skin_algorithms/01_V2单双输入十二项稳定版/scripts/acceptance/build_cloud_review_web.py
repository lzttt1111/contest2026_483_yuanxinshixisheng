"""Build a local review website from existing outputs only; no model imports."""
import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

CASES=("clinic28-25","clinic28-09","clinic28-23")
ALGOS=("redness","spots","brown","texture","pores","purple","surface_gloss","vascular","contour_firmness","acne","wrinkle")
LABELS=("红区","可见斑点","棕区","纹理","毛孔","UV与卟啉","油光","血管样结构","轮廓紧致度","痤疮","皱纹")
def esc(value):return html.escape(str(value))
def link(page,target,label):
    assert target.is_file(),target
    url=quote(os.path.relpath(target,page.parent).replace(os.sep,"/"),safe="/")
    return f'<a href="{url}" target="_blank">{esc(label)}</a>'
def media_values(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for item in value.values():yield from media_values(item)
    elif isinstance(value,list):
        for item in value:yield from media_values(item)
def shell(title,body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+"""</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f4f6f9;color:#172638;max-width:1400px;margin:auto;padding:24px}
section{background:white;border:1px solid #dce2e9;border-radius:10px;padding:20px;margin:20px 0}
a{color:#075cab}nav a{margin-right:16px}.warning{background:#fff4d6;padding:16px;border-left:4px solid #bd8700}
.gallery{display:flex;flex-wrap:wrap;gap:16px}.gallery figure{margin:0;width:300px}.gallery img{width:100%;max-height:420px;object-fit:contain}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}h1{font-size:28px}h2{font-size:21px}small{color:#56657a}</style><body><h1>"""+esc(title)+"</h1>"+body+"</body></html>"
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    a=p.parse_args();root=a.root.resolve(strict=True)
    web=root/"web";web.mkdir(exist_ok=True)
    pages=[]
    warning='<p class="warning">阶段性验收：三套四光源Worker已完成；机构本地与云端聚合Word及主图一致性检查通过。单RGB完整双Word批次、Word页面外观及独立强审尚未完成，整体未标记最终PASS。此处云端为真实Worker本地存储模拟，非服务器生产部署。</p>'
    for case in CASES:
        page=web/(case+".html")
        body=warning+'<p>'+link(page,root/"review"/"WORD_CROSSCHECK.json","Word一致性回执")+'</p>'
        body+='<section><h2>四光源报告与本地对照</h2><p>报告由独立的完整机构聚合生成；下面的Worker图来自单项任务，两者不混称同一制品。</p>'
        for route,label in (("cloud_institution","云端机构聚合"),("local_institution","本地机构对照")):
            folder=root/"word_crosscheck"/route/case
            docs=sorted(folder.glob("*.docx"))
            assert len(docs)==2,(case,route)
            body+="<h3>"+label+"</h3><ul>"
            for doc in docs:body+="<li>"+link(page,doc,doc.name)+"</li>"
            body+="<li>"+link(page,folder/"十二项检测结果索引.json","完整结果索引")+"</li></ul>"
        body+="</section><nav>"
        for key,label in zip(ALGOS,LABELS):body+=f'<a href="#{key}">{label}</a>'
        body+="</nav>"
        for key,label in zip(ALGOS,LABELS):
            service=(key if key in ("acne","wrinkle") else ("dermavision25" if case=="clinic28-25" else "dermavision09_23"))
            folder=root/service
            response=folder/case/(key+".json")
            data=json.loads(response.read_text(encoding="utf-8"))
            assert data.get("status")=="success",(case,key)
            body+=f'<section id="{key}"><h2>{label} · success</h2><p>'+link(page,response,"Worker完整响应JSON")+'</p><div class="gallery">'
            seen=set()
            for value in media_values(data.get("raw_result",{})):
                target=(folder/"storage"/value)
                if target.suffix.lower() not in (".png",".jpg",".jpeg",".webp"):continue
                if not target.is_file():continue
                target=target.resolve()
                assert target.is_relative_to(root)
                if target in seen:continue
                seen.add(target)
                url=quote(os.path.relpath(target,page.parent).replace(os.sep,"/"),safe="/")
                body+=f'<figure><a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a><figcaption>{esc(target.name)}</figcaption></figure>'
            body+='</div><details><summary>查看量化指标及质量字段</summary><pre>'+esc(json.dumps(data.get("raw_result",{}),ensure_ascii=False,indent=2))+'</pre></details></section>'
        page.write_text(shell(case+" 四光源云端验收",body),encoding="utf-8")
        pages.append(page)
    index=root/"index.html"
    body=warning+"<section><h2>三套四光源</h2><ul>"
    for page in pages:body+="<li>"+link(index,page,page.stem+"：结果图、指标及双Word")+"</li>"
    body+="</ul></section><section><h2>单RGB</h2><p>模式切换回归已通过；本轮完整云端＋本地＋双Word批次尚未运行完成，因此不提供旧结果冒充本轮结果。</p></section>"
    index.write_text(shell("云端单RGB与四光源验收入口",body),encoding="utf-8")
    print(json.dumps({"index":str(index),"sample_pages":len(pages),"status":"website_generated_not_final_acceptance"},ensure_ascii=False))
if __name__=="__main__":main()
