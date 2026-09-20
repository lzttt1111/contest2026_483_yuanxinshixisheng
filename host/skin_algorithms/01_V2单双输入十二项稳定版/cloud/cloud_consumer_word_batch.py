from __future__ import annotations
"""Batch the existing consumer CLI report chain without changing Worker envelopes."""
import os
from pathlib import Path
import subprocess
import sys

def generate_cloud_consumer_report_batch(input_dir: Path, output_root: Path, runtime_root: Path):
    if output_root.exists():
        raise FileExistsError("cloud report batch output already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    env=os.environ.copy()
    env["DERMAVISION_CAPTURE_PROFILE"]="consumer"
    command=[sys.executable,"-B",str(Path(__file__).resolve().parents[1]/"run.py"),
        "--input-dir",str(input_dir),"--glob","*_RGB_M.jpg","--no-recursive",
        "--output-dir",str(output_root),"--runtime-dir",str(runtime_root),
        "--capture-profile","consumer","--output-profile","review","--algorithms","all",
        "--generate-medical-report","--no-resume","--stop-on-error"]
    with output_root.with_suffix(".log").open("w",encoding="utf-8") as log:
        subprocess.run(command,cwd=Path(__file__).resolve().parents[1],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return tuple(sorted(output_root.rglob("*.docx")))
