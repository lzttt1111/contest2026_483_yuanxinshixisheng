"""Run copied stable V2 once for the front; return actual Word scores and spot subgroup."""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from .settings import VENDOR,ROOT

def grade(score):
    return "未见明显" if score<=20 else "轻度" if score<=40 else "中度" if score<=60 else "较明显" if score<=90 else "显著"

def score_front(front,job,device):
    source=job/"score_input"
    source.mkdir()
    shutil.copy2(front,source/("front"+front.suffix))
    output=job/"v2_score"
    env=os.environ.copy()
    env.pop("PYTHONPATH",None)
    env["PYTHONDONTWRITEBYTECODE"]="1"
    env["YOLO_OFFLINE"]="true"
    env["TMPDIR"]=str(job/"tmp")
    Path(env["TMPDIR"]).mkdir()
    # The baseline CLI owns device selection; keep its CUDA device visible, or explicitly hide GPUs.
    if device=="cpu":env["CUDA_VISIBLE_DEVICES"]=""
    elif device.startswith("cuda:"):env["CUDA_VISIBLE_DEVICES"]=device.split(":")[1]
    command=[sys.executable,"-B","run.py","--input-dir",str(source),"--output-dir",str(output),
        "--capture-profile","consumer","--algorithms","all","--generate-medical-report",
        "--report-subject-id",job.name,"--limit","1","--no-resume","--stop-on-error"]
    with (job/"v2_scoring.log").open("wb") as log:
        proc=subprocess.Popen(command,cwd=VENDOR/"v2",env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            raise RuntimeError("V2评分超时")
    if proc.returncode:raise RuntimeError("V2评分流程执行失败")
    reports=list(output.glob("batch_*/*/*_用户精简版.docx"))
    if len(reports)!=1:raise RuntimeError("V2评分报告未完整生成")
    report=reports[0]
    ns="{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(report) as z:xml=ET.fromstring(z.read("word/document.xml"))
    wanted={"可见毛孔":"pores","油脂分泌倾向":"surface_gloss"}
    scores={}
    for tr in xml.iter(ns+"tr"):
        cells=["".join(n.text or "" for n in cell.iter(ns+"t")) for cell in tr.findall(ns+"tc")]
        if len(cells)<2 or cells[0] not in wanted:continue
        match=re.search(r"([0-9]+(?:\.[0-9]+)?)分[·・]([^\s｜|]+)",cells[1])
        if match:scores[wanted[cells[0]]]={"score":float(match[1]),"severity":match[2]}
    if len(scores)!=2:raise RuntimeError("正面评分证据不足")
    # Execute the immutable V2 ECDF subgroup code in its own process to avoid src module collisions.
    with (job/"spot_subscore.log").open("wb") as log:
        proc=subprocess.run([sys.executable,"-B",str(ROOT/"score_subgroup.py"),
            "--baseline",str(VENDOR/"v2"),"--result",str(report.parent),"--output",str(job/"spot_score.json")],
            cwd=VENDOR/"v2",env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    if proc.returncode:raise RuntimeError("可见色斑评分失败")
    spot=json.loads((job/"spot_score.json").read_text())
    scores["spots"]={"score":round(spot["score"],1),"severity":grade(spot["score"])}
    (job/"score_evidence.json").write_text(json.dumps({"front_sha256":hashlib.sha256(front.read_bytes()).hexdigest(),
        "report_sha256":hashlib.sha256(report.read_bytes()).hexdigest(),"scores":scores,"spot":spot},
        ensure_ascii=False,indent=2),encoding="utf-8")
    return scores
