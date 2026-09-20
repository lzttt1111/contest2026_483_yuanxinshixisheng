TASK="parity"
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
    if type(data['max_abs_tolerance']) not in (int,float) or not math.isfinite(data['max_abs_tolerance']):raise ValueError('finite numeric tolerance required')
    if 'coordinate_case' in data:
        cc=data['coordinate_case']
        for key in ('scale','tolerance_pixels'):
            if type(cc[key]) not in (int,float) or not math.isfinite(cc[key]):raise ValueError('finite number required: '+key)
        for value in cc['pad_xy']+[v for point in cc['model_points']+cc['original_points'] for v in point]:
            if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('finite coordinate number required')
    d=data;findings=[];metrics={}
    def issue(code,detail):findings.append({"code":code,"detail":detail})

    a=d["source"];b=d["target"];t=d["max_abs_tolerance"]
    if t<0:raise ValueError("negative tolerance")
    for obj in (a,b):
        if not obj["shape"] or any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in obj["shape"]):raise ValueError("invalid tensor shape")
        if math.prod(obj["shape"])!=len(obj["values"]):raise ValueError("tensor shape/length mismatch")
        if any(not isinstance(x,(int,float)) or isinstance(x,bool) for x in obj["values"]):raise ValueError("invalid tensor value")
    for obj in (a,b):
        if obj['layout'] not in ('NCHW','NHWC','CHW','HWC','NC','C','HW'):raise ValueError('unsupported tensor layout')
        if obj['color'] not in ('RGB','BGR','GRAY','RGBA','BGRA','none'):raise ValueError('unsupported tensor color')
    for key in ('runtime_provider','claimed_provider'):
        if type(d[key]) is not str or not d[key].strip():raise ValueError('nonempty provider required')
    contract=all(a[k]==b[k] for k in ("layout","color","shape"))
    if not contract:issue("tensor_contract_mismatch",{"source":{k:a[k] for k in ("layout","color","shape")},"target":{k:b[k] for k in ("layout","color","shape")}})
    diffs=[abs(x-y) for x,y in zip(a["values"],b["values"])] if len(a["values"])==len(b["values"]) else []
    if any(not math.isfinite(v) for v in diffs):raise ValueError('difference exceeds finite numeric range')
    maximum=max(diffs) if diffs else None
    if maximum is None or maximum>t:issue("numeric_parity_failed",maximum)
    if d["runtime_provider"]!=d["claimed_provider"]:issue("provider_mismatch",d["runtime_provider"])
    overlap=set(d["calibration_ids"])&set(d["test_ids"])
    if overlap:issue("calibration_leakage",sorted(overlap))
    rmse=(maximum*math.sqrt(sum((x/maximum)**2 for x in diffs)/len(diffs)) if maximum else 0.0) if diffs else None
    metrics.update(max_abs=maximum,rmse=rmse,provider=d["runtime_provider"],numeric_comparison_valid=contract)
    if 'coordinate_case' in d:
        c=d['coordinate_case'];scale=c['scale'];padding=c['pad_xy'];points=c['model_points'];reference=c['original_points'];tol=c['tolerance_pixels']
        if isinstance(scale,bool) or scale<=0 or tol<0 or len(padding)!=2 or len(points)!=len(reference) or not points:raise ValueError('invalid coordinate parity contract')
        restored=[]
        for point,expected in zip(points,reference):
            if len(point)!=2 or len(expected)!=2:raise ValueError('points require x/y')
            xy=[(point[j]-padding[j])/scale for j in (0,1)]
            if c.get('mirror_width') is not None:
                width=c['mirror_width']
                if type(width) is not int or width<1:raise ValueError('invalid mirror width')
                xy[0]=width-1-xy[0]
            restored.append(xy)
        errors=[math.hypot(a[0]-b[0],a[1]-b[1]) for a,b in zip(restored,reference)]
        metrics['coordinate_parity']={'restored_points':restored,'max_error_pixels':max(errors),'passed':max(errors)<=tol}
        if max(errors)>tol:issue('coordinate_parity_failed',metrics['coordinate_parity'])

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
