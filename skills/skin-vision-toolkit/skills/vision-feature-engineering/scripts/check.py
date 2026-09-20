TASK="feature"
"""Offline evidence checker. No model, network, training, deployment or input writes."""
import argparse,hashlib,json,math,sys
from pathlib import Path

def check(data,root):
    if data.get("task_kind")!=TASK:
        return {"status":"not_applicable","findings":[],"metrics":{}}
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):raise ValueError("non-finite number")
        if isinstance(value,dict):
            flags={'baseline_verified','calibration_required','calibrated','human_confirmed','valid','reset_all','cross_view_unique_claim','correspondence_verified','inference_requested','committed','artifacts_valid','point_overlay','registration_verified','clock_mapping_verified','mode_switch_clears_photos','document_claims_preserved'}
            for key,v in value.items():
                if key in flags and type(v) is not bool:raise ValueError('Boolean field required: '+key)
                finite(v)
        if isinstance(value,list):
            for v in value:finite(v)
    finite(data)
    if type(data['min_retained_ratio']) not in (int,float) or not math.isfinite(data['min_retained_ratio']):raise ValueError('finite numeric retained ratio required')
    d=data;findings=[];metrics={}
    def issue(code,detail):findings.append({"code":code,"detail":detail})

    masks=[d[k] for k in ("reference_skin","valid_mask","detection_mask")]
    height=len(masks[0]);width=len(masks[0][0]) if height else 0
    if not width or any(len(m)!=height or any(len(row)!=width for row in m) for m in masks):raise ValueError("mask shape mismatch")
    if any(x not in (0,1,False,True) for m in masks for row in m for x in row):raise ValueError("masks must be binary")
    ref,valid,det=masks
    reference=sum(map(sum,ref));area=sum(map(sum,valid));support=sum(det[y][x] and valid[y][x] for y in range(height) for x in range(width))
    retained=sum(ref[y][x] and valid[y][x] for y in range(height) for x in range(width))
    threshold=d["min_retained_ratio"]
    if not 0<=threshold<=1:raise ValueError("retained ratio out of range")
    ratio=retained/reference if reference else None
    if not reference:issue("empty_reference","Cannot judge skin removal without reference")
    elif ratio<threshold:issue("reference_skin_removed",{"retained":retained,"reference":reference})
    outside=sum(det[y][x] and not valid[y][x] for y in range(height) for x in range(width))
    if outside:issue("mask_outside_domain",outside)
    for r in d["instances"]:
        x,y=r["x"],r["y"]
        if type(x) is not int or type(y) is not int:raise ValueError("integer centroid required for this raster audit")
        if x<0 or y<0 or x>=width or y>=height or not valid[y][x]:issue("instance_outside_domain",r["id"])
    metrics.update(reference_pixels=reference,valid_pixels=area,support_pixels=support,retained_ratio=ratio,coverage=support/area if area else None)

    return {"status":"fail" if findings else "pass","findings":findings,"metrics":metrics}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--root",type=Path,help="Explicit allowed artifact root; default is input parent")
    args=parser.parse_args()
    try:
        if args.input.stat().st_size>32*1024*1024:raise ValueError("Input too large; provide bounded snapshot")
        data=json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(data,dict):raise ValueError("Input must be object")
        result=check(data,(args.root or args.input.parent).resolve())
        json.dumps(result,ensure_ascii=False,allow_nan=False)
    except (ValueError,TypeError,KeyError,IndexError,OSError,AttributeError,OverflowError) as exc:
        result={"status":"invalid_input","findings":[{"code":"invalid_input","detail":str(exc)}],"metrics":{}}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 2 if result["status"]=="invalid_input" else (1 if result["status"]=="fail" else 0)
if __name__=="__main__":sys.exit(main())
