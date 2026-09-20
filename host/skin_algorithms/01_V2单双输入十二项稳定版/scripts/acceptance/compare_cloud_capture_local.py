from __future__ import annotations

"""Compare full local clinic execution with independently executed Worker outputs."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("DERMAVISION_CAPTURE_PROFILE","institution")
os.environ.setdefault("MPLCONFIGDIR","/tmp/dermavision-cloud-capture-mpl")

from src.detection_runtime.capture_manifest import load_clinic_capture_manifest
from src.detection_runtime.clinic_provider import ClinicFourLightProvider
from src.detection_runtime.cloud_projection import project_clinic_result
from src.pipeline import DermaVisionPipeline
from src.capture_profile import CaptureProfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cloud-root",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fresh comparison directory required")
    args.output.mkdir(parents=True)
    captures=load_clinic_capture_manifest(args.fixture)
    provider=ClinicFourLightProvider()
    pipeline=DermaVisionPipeline(capture_profile=CaptureProfile.INSTITUTION)
    results=[]
    try:
        with (args.output/"runtime.log").open("w",encoding="utf-8") as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                for capture in captures:
                    alias=capture.capture_alias
                    cloud_dir=args.cloud_root/("dermavision25" if alias=="clinic28-25" else "dermavision09_23")
                    out=args.output/alias
                    out.mkdir()
                    prepared=provider.prepare(capture,out/"clinic")
                    items,receipt=provider.run_prepared(prepared,out/"clinic")
                    for algorithm in ("redness","brown","purple","surface_gloss","vascular"):
                        projected=project_clinic_result(algorithm,items,prepared,out/algorithm,receipt)
                        actual=json.loads((cloud_dir/alias/(algorithm+".json")).read_text())
                        expected=json.loads(Path(projected["results"][algorithm+"_metrics"]).read_text())
                        expected.pop("medical_metrics_v2",None)
                        actual_metrics = dict(actual["raw_result"]["metrics"])
                        # The cloud content-addressed capture alias differs from fixture aliases.
                        for metric in (expected, actual_metrics):
                            metric.pop("source_name", None)
                        assert actual_metrics==expected,(alias,algorithm,"metrics mismatch")
                        # Map like-for-like public media only, not local auxiliary basemaps.
                        fields=({"redness":["overlay","red_areas_overlay"],"brown":["overlay","brown_spots_overlay"],
                                 "purple":["uv_base","uv_spots_overlay","fluorescence_base","porphyrin_overlay"]}).get(algorithm,["overlay"])
                        result_keys=({"redness":["redness","red_areas_overlay"],"brown":["brown","brown_spots_overlay"],
                                      "purple":["purple_uv_base","purple_uv_spots_overlay","purple_fluorescence_base","purple_porphyrin_overlay"]}).get(algorithm,[algorithm])
                        import cv2,numpy as np
                        for field,key in zip(fields,result_keys):
                            cloud_image=cv2.imread(str(cloud_dir/"storage"/actual["raw_result"][field]))
                            local_image=cv2.imread(projected["results"][key])
                            assert cloud_image is not None and local_image is not None
                            assert np.array_equal(cloud_image,local_image),(alias,algorithm,field,"pixel mismatch")
                        results.append({"case":alias,"algorithm":algorithm,"metrics_equal":True,"pixels_equal":True})
                    base=pipeline.process_single(str(capture.channel("RGB_M").path),["spots","texture","pores","contour_firmness"])
                    assert base["status"]=="success"
                    for algorithm in ("spots","texture","pores","contour_firmness"):
                        raw=json.loads(Path(base["results"][algorithm+"_metrics"]).read_text())
                        raw.pop("medical_metrics_v2",None)
                        actual=json.loads((cloud_dir/alias/(algorithm+".json")).read_text())
                        assert actual["raw_result"]["metrics"]==raw,(alias,algorithm,"base metrics mismatch")
                        results.append({"case":alias,"algorithm":algorithm,"metrics_equal":True})
                    (args.output/"LOCAL_CLOUD_COMPARISON.json").write_text(
                        json.dumps(results,indent=2),encoding="utf-8")
                    print(alias,"PASS",flush=True)
    finally:
        provider.close()
        pipeline.close()
        (args.output/"LOCAL_CLOUD_COMPARISON.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("PASS local/cloud",len(results))


if __name__=="__main__":
    main()
