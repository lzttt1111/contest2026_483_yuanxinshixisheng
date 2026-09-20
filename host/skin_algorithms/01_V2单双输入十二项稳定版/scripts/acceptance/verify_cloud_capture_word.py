from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR","/tmp/dermavision-cloud-word-mpl")
os.environ.setdefault("YOLO_CONFIG_DIR","/tmp/dermavision-cloud-word-yolo")
from cloud.cloud_capture_word_report import CloudCaptureReportSession, export_clinic_reports
from src.detection_runtime.capture_manifest import load_clinic_capture_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--generate-medical-report",action="store_true",required=True)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.output.exists() and not a.resume: raise FileExistsError("fresh output required")
    a.output.mkdir(parents=True, exist_ok=a.resume)
    fixture=json.loads(a.fixture.read_text())
    rows=[]
    with (a.output/"runtime.log").open("a") as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            with CloudCaptureReportSession(a.runtime) as session:
                for capture in load_clinic_capture_manifest(a.fixture):
                    alias=capture.capture_alias
                    started=time.perf_counter()
                    local=a.output/"local_institution"/alias
                    if a.resume and local.is_dir():
                        index=json.loads((local/"十二项检测结果索引.json").read_text())
                        assert index["状态"]=="success" and index["成功项目数"]==12
                        assert len(list(local.glob("*.docx")))==2
                        local_reports={"retained":str(local)}
                    else:
                        local,local_reports=export_clinic_reports(session.orchestrator,capture,a.output/"local_institution",alias)
                    members=[r for r in fixture["images"] if r["capture_alias"]==alias]
                    payload=[{"filename":f"{alias}_{r['role']}_M.jpg","oss_key":r["relative_path"]} for r in members]
                    cloud,cloud_reports=session.generate(payload,a.output/"cloud_institution",subject_id=alias,
                        download_by_key=lambda key:(a.fixture.parent/key).read_bytes())
                    rows.append({"case":alias,"local":str(local),"cloud":str(cloud),
                        "local_reports":local_reports,"cloud_reports":cloud_reports,"seconds":time.perf_counter()-started})
                    (a.output/"REPORT_GENERATION.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2))
                    print(alias,"local/cloud reports generated",flush=True)
    print("PASS generated",len(rows)*4,"institution DOCX")

if __name__=="__main__":main()
