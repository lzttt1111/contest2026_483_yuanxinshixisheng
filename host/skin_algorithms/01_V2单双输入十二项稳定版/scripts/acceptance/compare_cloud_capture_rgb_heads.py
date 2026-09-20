from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ["DERMAVISION_CAPTURE_PROFILE"]="institution"
os.environ.setdefault("MPLCONFIGDIR","/tmp/dermavision-cloud-capture-mpl")
os.environ.setdefault("YOLO_CONFIG_DIR","/tmp/dermavision-cloud-capture-yolo")
from cloud.simulated_contract_versions import install_simulated_contract_versions
install_simulated_contract_versions()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cloud-root",type=Path,required=True)
    p.add_argument("--service",choices=["acne","wrinkle"],required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("fresh output required")
    a.output.mkdir(parents=True)
    fixture=json.loads(a.fixture.read_text())
    if a.service=="acne":
        from src.acne import worker
        from src.acne.artifact_policy import FORMAL_FAST_TOKEN
        algorithms=["acne",FORMAL_FAST_TOKEN]
        task=worker.analyze_image_v2
    else:
        from src.wrinkle import worker
        algorithms=["wrinkle"]
        task=worker.analyze_image
    pipeline=worker._get_pipeline()
    results=[]
    def upload(path,record,algo,name):
        key=f"{record}/{algo}/{name}"
        dest=a.output/"storage"/key
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
        return key
    worker.upload_via_internal_api=upload
    try:
        with (a.output/"runtime.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            for alias in ("clinic28-25","clinic28-09","clinic28-23"):
                row=next(r for r in fixture["images"] if r["capture_alias"]==alias and r["role"]=="RGB")
                source=a.fixture.parent/row["relative_path"]
                direct=pipeline.process_single(str(source),algorithms)
                assert direct["status"]=="success", direct
                worker.run_task_pipeline=lambda *args:direct
                worker.download_via_internal_api=lambda _:source.read_bytes()
                local=task.run(alias,source.name,None,[a.service])
                assert local["status"]=="success",local
                cloud=json.loads((a.cloud_root/a.service/alias/(a.service+".json")).read_text())
                if a.service=="acne":
                    assert local["raw_result"]["metrics"]==cloud["raw_result"]["metrics"]
                    local_images=[local["raw_result"]["overlay"]]
                    cloud_images=[cloud["raw_result"]["overlay"]]
                else:
                    for field in ("region_metrics","run_preset","successful_runs","failed_runs","region_analysis_status"):
                        assert local["raw_result"][field]==cloud["raw_result"][field],(alias,field)
                    keys=("stage2_overlay","region_overlay","texture_reference")
                    local_images=[local["raw_result"][key] for key in keys]
                    cloud_images=[cloud["raw_result"][key] for key in keys]
                for l,c in zip(local_images,cloud_images):
                    x=cv2.imread(str(a.output/"storage"/l));y=cv2.imread(str(a.cloud_root/a.service/"storage"/c))
                    assert x is not None and y is not None and np.array_equal(x,y),(alias,"pixels")
                results.append({"case":alias,"service":a.service,"metrics_equal":True,"pixels_equal":True})
                (a.output/"LOCAL_CLOUD_COMPARISON.json").write_text(json.dumps(results,indent=2))
        print("PASS local/cloud",a.service,len(results))
    finally:
        worker._close_worker_process()


if __name__=="__main__":main()
