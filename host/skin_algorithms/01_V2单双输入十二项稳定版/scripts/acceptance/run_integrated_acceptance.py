from __future__ import annotations
"""Single frozen run, stage receipts, no remote publication and no force retries."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cloud.cloud_consumer_word_batch import generate_cloud_consumer_report_batch

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def resources():
    info={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key,rest=line.split(":",1);info[key]=int(rest.split()[0])
    assert info["MemAvailable"]>=16*1024*1024,"less than 16 GiB available"
    assert info["SwapTotal"]-info["SwapFree"]<6*1024*1024,"swap limit"
    gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
    assert not gpu,"external GPU task present: "+gpu
def main():
    p=argparse.ArgumentParser();p.add_argument("--fixture",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",action="store_true");a=p.parse_args()
    a.output=a.output.resolve();a.fixture=a.fixture.resolve()
    if a.output.exists() and not a.resume:raise FileExistsError("fresh output required")
    a.output.mkdir(parents=True,exist_ok=a.resume)
    sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    receipt=a.output/"RUN_RECEIPT.json"
    history=json.loads(receipt.read_text()) if receipt.exists() else {"head":sha,"stages":[]}
    assert history["head"]==sha,"resume requires same frozen commit"
    inputs=a.output/"inputs";inputs.mkdir(exist_ok=True)
    fixture=json.loads(a.fixture.read_text())
    input_sha={}
    for number,alias in enumerate(("clinic28-25","clinic28-09","clinic28-23"),1):
        row=next(r for r in fixture["images"] if r["capture_alias"]==alias and r["role"]=="RGB")
        source=a.fixture.parent/row["relative_path"]
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row["sha256"]
        destination=inputs/(f"{number:02d}_"+alias+"_RGB_M.jpg")
        if destination.exists():assert hashlib.sha256(destination.read_bytes()).hexdigest()==row["sha256"]
        else:shutil.copy2(source,destination)
        input_sha[alias]=row["sha256"]
    history["input_sha256"]=input_sha
    def stage(name,args=None,call=None):
        prior=next((s for s in history["stages"] if s["name"]==name),None)
        if prior:
            assert prior["status"]=="passed","failed stage needs explicit bounded remediation"
            print(name,"REUSED",flush=True);return
        resources()
        event={"name":name,"status":"running","started":time.time()}
        history["stages"].append(event);write(receipt,history)
        try:
            if args is not None:
                cmd=[sys.executable,"-B",*args];event["command"]=cmd
                with (a.output/(name+".log")).open("w") as log:
                    subprocess.run(cmd,cwd=ROOT,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"},stdout=log,stderr=subprocess.STDOUT,check=True)
            else:call()
            event["status"]="passed"
            print(name,"PASS",flush=True)
        except Exception as error:
            event["status"]="failed";event["error"]=str(error);raise
        finally:
            event["seconds"]=round(time.time()-event["started"],3);write(receipt,history)
    stage("local_consumer",["run.py","--input-dir",str(inputs),"--glob","*_RGB_M.jpg","--no-recursive",
        "--output-dir",str(a.output/"local_consumer"),"--runtime-dir","/dev/shm/dermavision-integrated-consumer",
        "--capture-profile","consumer","--output-profile","review","--algorithms","all",
        "--generate-medical-report","--no-resume","--stop-on-error"])
    for profile in ("consumer","institution"):
        for service in ("dermavision","acne","wrinkle"):
            stage(profile+"_"+service,["scripts/acceptance/verify_cloud_capture_workers.py",
                "--fixture",str(a.fixture),"--output",str(a.output/("cloud_"+profile)/"services"/service),
                "--service",service,"--profile",profile])
    stage("cloud_consumer_word",call=lambda:generate_cloud_consumer_report_batch(
        inputs,a.output/"cloud_consumer"/"report_aggregate",Path("/dev/shm/dermavision-integrated-cloud-consumer")))
    stage("institution_reports",["scripts/acceptance/verify_cloud_capture_word.py","--fixture",str(a.fixture),
        "--output",str(a.output/"institution_reports"),"--runtime","/dev/shm/dermavision-integrated-clinic",
        "--generate-medical-report"])
    history["generation_complete"]=True;write(receipt,history)
    print("GENERATION COMPLETE; comparison and visual QA still required")
if __name__=="__main__":main()
