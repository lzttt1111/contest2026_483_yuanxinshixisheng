from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ["DERMAVISION_CAPTURE_PROFILE"]="consumer"
os.environ.setdefault("MPLCONFIGDIR","/tmp/dermavision-cloud-capture-mpl")
from cloud.simulated_contract_versions import install_simulated_contract_versions
install_simulated_contract_versions()
from src import worker
from cloud_contracts import validate_worker_envelope


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError("fresh output required")
    args.output.mkdir(parents=True)
    fixture=json.loads(args.fixture.read_text())
    rows=[r for r in fixture["images"] if r["capture_alias"]=="clinic28-25"]
    payload=[{"filename":f"case_{r['role']}_M.jpg","oss_key":r["relative_path"]} for r in rows]
    rgb=next(r["relative_path"] for r in rows if r["role"]=="RGB")
    worker.download_via_internal_api=lambda key:(args.fixture.parent/key).read_bytes()
    def upload(path,record,algo,name):
        key=f"{record}/{algo}/{name}"
        target=args.output/key
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        return key
    worker.upload_via_internal_api=upload
    results=[]
    try:
        with (args.output/"runtime.log").open("w") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            for label,capture in (("before",None),("single_list",[{"filename":"phone.jpg","oss_key":rgb}]),
                                  ("four",payload),("after",None)):
                result=worker.analyze_image.run(label,rgb,None,["pores"],capture_images=capture)
                assert result["status"]=="success",result
                validate_worker_envelope("pores",result)
                results.append(result)
        for index in (1,3):
            assert results[index]["raw_result"]["metrics"]==results[0]["raw_result"]["metrics"]
            assert results[index]["debug_info"]["execution_mode"]==results[0]["debug_info"]["execution_mode"]=="single_algorithm_consumer"
            import cv2,numpy as np
            assert np.array_equal(cv2.imread(str(args.output/results[index]["raw_result"]["overlay"])),
                                  cv2.imread(str(args.output/results[0]["raw_result"]["overlay"])))
        assert results[2]["debug_info"]["execution_mode"]=="single_algorithm"
        (args.output/"RGB_REGRESSION.json").write_text(json.dumps({
            "status":"passed","order":["old_single","single_list","four","old_single"],
            "consumer_metrics_equal":True,"consumer_pixels_equal":True,"profile_restored":True,
        },indent=2))
        print("PASS RGB regression")
    finally:
        worker._close_worker_process()


if __name__=="__main__":main()
