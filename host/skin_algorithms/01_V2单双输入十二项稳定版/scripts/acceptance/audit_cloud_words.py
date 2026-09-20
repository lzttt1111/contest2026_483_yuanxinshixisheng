from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import cv2
import numpy as np

NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
def normalized(text):
    return re.sub(r"AISIA-SINGLE-RGB-[A-Za-z0-9._-]+","AISIA-SINGLE-RGB-ID",text)
def doc(path):
    with ZipFile(path) as z:
        assert z.testzip() is None
        xml=ET.fromstring(z.read("word/document.xml"))
        tables=[normalized("|".join("".join(c.itertext()) for c in t.findall(".//w:t",NS))) for t in xml.findall(".//w:tbl",NS)]
        paragraphs=[normalized("".join(t.itertext())) for t in xml.findall(".//w:t",NS)]
        media=Counter()
        for name in z.namelist():
            if name.startswith("word/media/"):
                data=z.read(name)
                assert cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR) is not None
                media[hashlib.sha256(data).hexdigest()]+=1
        return tables, paragraphs, media, len(xml.findall(".//w:bookmarkStart",NS))
def compare_reports(local,cloud,alias):
    out=[]
    for edition in ("用户精简版","医生详细版"):
        left=list(local.glob("*"+edition+".docx"));right=list(cloud.glob("*"+edition+".docx"))
        assert len(left)==len(right)==1,(alias,edition,"report missing")
        a,b=doc(left[0]),doc(right[0])
        assert a[0]==b[0],(alias,edition,"table mismatch")
        assert a[2]==b[2],(alias,edition,"embedded media mismatch")
        assert a[3]==b[3],(alias,edition,"bookmarks")
        assert alias in "".join(a[1]) and alias in "".join(b[1]),(alias,"subject")
        out.append({"case":alias,"edition":edition,"tables":len(a[0]),"media":sum(a[2].values()),
                    "bookmarks":a[3],"tables_equal":True,"media_equal":True,
                    "local":str(left[0]),"cloud":str(right[0])})
    return out
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--institution-root",type=Path)
    p.add_argument("--consumer-root",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for profile,root in (("institution",a.institution_root),("consumer",a.consumer_root)):
        if root is None:continue
        for alias in ("clinic28-25","clinic28-09","clinic28-23"):
            if profile=="institution":
                local=root/"local_institution"/alias;cloud=root/"cloud_institution"/alias
                li=json.loads((local/"十二项检测结果索引.json").read_text())
                ci=json.loads((cloud/"十二项检测结果索引.json").read_text())
                assert li["成功项目数"]==ci["成功项目数"]==12
                for key,item in li["十二项结果"].items():
                    peer=ci["十二项结果"][key]
                    for role in ("主结果图",):
                        x=cv2.imread(str(local/item[role]));y=cv2.imread(str(cloud/peer[role]))
                        assert x is not None and y is not None and np.array_equal(x,y),(alias,key,role)
            else:
                candidates=[path.parent for path in (root/"local_consumer").rglob("十二项检测结果索引.json") if alias in str(path)]
                assert len(candidates)==1,(alias,"local result")
                local=candidates[0];cloud=root/"cloud_consumer"/alias
                bundle=json.loads((cloud/"cloud_response_bundle.json").read_text())
                assert bundle["task_count"]==11 and bundle["projected_result_count"]==12
                assert bundle["medical_report_generated"] is True
            row={"profile":profile,"case":alias,"word":compare_reports(local,cloud,alias)}
            rows.append(row)
            (a.output/"WORD_CROSSCHECK.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print("PASS",len(rows),"pairs",sum(len(r["word"]) for r in rows),"Word comparisons")

if __name__=="__main__":main()
