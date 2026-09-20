from __future__ import annotations

"""Bounded real Worker validation using local bytes in place of HTTP storage."""

import argparse
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DERMAVISION_CAPTURE_PROFILE", "consumer")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dermavision-cloud-capture-mpl")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/dermavision-cloud-capture-yolo")

import aisia_contracts.scoring_input.v1  # Require real locked contracts, no mock models.
from cloud_contracts import validate_worker_envelope
from src.detection_runtime.cloud_task import close_capture_runtime


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_service(worker, service, fixture, destination, aliases, profile, selected=None):
    rows = fixture["images"]
    inputs = fixture["_root"]
    evidence = []
    record = {}
    worker.download_via_internal_api = lambda key: (inputs / key).read_bytes()
    def upload(path, record_id, algo, name):
        key = f"report/{record_id}/{algo}/{name}"
        target = destination / "storage" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return key
    worker.upload_via_internal_api = upload
    original_run = worker.run_task_pipeline
    def recording_run(*args):
        result = original_run(*args)
        record.clear()
        record.update(result)
        # Preserve algorithm evidence for offline/local comparison before cleanup.
        return result
    worker.run_task_pipeline = recording_run
    algorithms = (("redness","spots","brown","texture","pores","purple","surface_gloss","vascular","contour_firmness")
                  if service == "dermavision" else (service,))
    if selected:
        if not set(selected).issubset(algorithms):
            raise ValueError("unsupported targeted algorithms")
        algorithms = tuple(selected)
    task = worker.analyze_image_v2 if service == "acne" else worker.analyze_image
    try:
        for alias in aliases:
            members = [row for row in rows if row["capture_alias"] == alias]
            rgb = next(row for row in members if row["role"] == "RGB")
            if profile == "consumer":
                members = [rgb]
            payload = [{"filename":f"{alias}_{r['role']}_M.jpg","oss_key":r["relative_path"]} for r in members]
            for algorithm in algorithms:
                started = time.perf_counter()
                key = "acne_v2" if service == "acne" else algorithm
                job = alias
                record.clear()
                response = task.run(job,"",None,[algorithm],capture_images=payload)
                write_json(destination / alias / (algorithm+".json"),response)
                if response.get("status") != "success":
                    raise RuntimeError(f"{alias}/{algorithm}: {response}")
                validate_worker_envelope(key,response)
                scoring = response["raw_result"]["scoring_input"]
                assert scoring["capture_profile"] == profile
                assert scoring["report_id"] == alias
                assert scoring["source_image_sha256"] == rgb["sha256"]
                entry = {"case":alias,"algorithm":algorithm,"status":"success",
                         "seconds":round(time.perf_counter()-started,3)}
                evidence.append(entry)
                write_json(destination/"WORKER_RECEIPT.json",evidence)
                print(json.dumps(entry),flush=True)
    finally:
        worker._close_worker_process()
    return evidence


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--service",choices=["dermavision","acne","wrinkle"],required=True)
    parser.add_argument("--profile",choices=["consumer","institution"],default="institution")
    parser.add_argument("--algorithms",nargs="+")
    parser.add_argument("--cases",nargs="+",default=["clinic28-25","clinic28-09","clinic28-23"])
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a fresh output directory")
    args.output.mkdir(parents=True)
    fixture=json.loads(args.fixture.read_text(encoding="utf-8"))
    fixture["_root"]=args.fixture.parent
    for row in fixture["images"]:
        assert sha(args.fixture.parent/row["relative_path"])==row["sha256"]
    if args.service=="dermavision":
        from src import worker
    elif args.service=="acne":
        from src.acne import worker
    else:
        from src.wrinkle import worker
    with (args.output/"runtime.log").open("w",encoding="utf-8") as log:
        try:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                run_service(worker,args.service,fixture,args.output,args.cases,args.profile,args.algorithms)
            print("PASS", args.service, args.cases)
        except Exception as exc:
            write_json(args.output/"FAILURE.json", {"error":str(exc)})
            raise
        finally:
            close_capture_runtime()


if __name__ == "__main__":
    main()
