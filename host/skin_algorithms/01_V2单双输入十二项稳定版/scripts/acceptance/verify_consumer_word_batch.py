from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("fresh output required")
    a.output.mkdir(parents=True)
    input_root=a.output/"inputs";input_root.mkdir()
    rows=json.loads(a.fixture.read_text())["images"]
    sources=[]
    for alias in ("clinic28-25","clinic28-09","clinic28-23"):
        row=next(r for r in rows if r["capture_alias"]==alias and r["role"]=="RGB")
        source=a.fixture.parent/row["relative_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row["sha256"]
        target=input_root/(alias+"_RGB_M.jpg")
        shutil.copy2(source,target);sources.append((alias,target))
    env=os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"]="1"
    env["DERMAVISION_CAPTURE_PROFILE"]="consumer"
    def run(label,argv):
        with (a.output/(label+".log")).open("w") as log:
            subprocess.run([sys.executable,"-B",*argv],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(label,"PASS",flush=True)
    run("local_consumer",["run.py","--input-dir",str(input_root),"--glob","*_RGB_M.jpg","--no-recursive",
        "--output-dir",str(a.output/"local_consumer"),"--capture-profile","consumer","--output-profile","review",
        "--algorithms","all","--generate-medical-report","--no-resume","--stop-on-error",
        "--runtime-dir","/dev/shm/dermavision-consumer-word-20260914"])
    for alias,source in sources:
        dest=a.output/"cloud_consumer"/alias
        run("cloud_"+alias,["cloud/simulate_cloud_request.py","--input",str(source),"--output",str(dest),
            "--generate-medical-report","--report-subject-id",alias])
        run("pydantic_"+alias,["scripts/verify_cloud_pydantic_review.py","--root",str(dest)])
    (a.output/"GENERATION.json").write_text(json.dumps({"status":"passed","samples":3,"routes":["local_consumer","cloud_consumer"],"docx":12},indent=2))

if __name__=="__main__":main()
