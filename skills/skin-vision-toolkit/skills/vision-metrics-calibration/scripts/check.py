TASK="metrics"
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
    for key in ('physical_scale','declared_p50','declared_p90'):
        value=data.get(key)
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value)):raise ValueError('finite number required: '+key)
    if data.get('physical_scale') is not None and data['physical_scale']<=0:raise ValueError('physical_scale must be positive')
    if type(data['declared_count']) is not int or data['declared_count']<0:raise ValueError('declared_count must be nonnegative integer')
    d=data;findings=[];metrics={}
    def issue(code,detail):findings.append({"code":code,"detail":detail})

    area=d["valid_pixels"];positive=d["positive_pixels"];values=d["values"]
    if not isinstance(area,int) or isinstance(area,bool) or not isinstance(positive,int) or isinstance(positive,bool):raise ValueError("pixel counts require integers")
    if any(not isinstance(x,(float,int)) or isinstance(x,bool) for x in values):raise ValueError("instance values must be finite numbers")
    if area<=0:issue("invalid_denominator",area)
    if 'observed_domain_pixels' in d:
        observed=d['observed_domain_pixels']
        if type(observed) is not int or observed<0:raise ValueError('invalid observed domain')
        if area>observed:issue('expanded_denominator',{'declared':area,'observed':observed})
    if positive<0 or positive>max(area,0):issue("support_outside_denominator",positive)
    if d["declared_count"]!=len(values):issue("count_mismatch",{"declared":d["declared_count"],"instances":len(values)})
    if d.get("cross_view_unique_claim") and not d.get("correspondence_verified"):issue("unverified_cross_view_count","No lesion correspondence")
    if d['unit'] not in ('pixel','pixels','px','px2','count','ratio','mm','mm2','um','um2','cm','cm2','m','m2'):raise ValueError('unsupported or missing unit')
    if d["unit"] in ("mm","mm2","um","um2","cm","cm2","m","m2") and not d.get("physical_scale"):issue("missing_physical_scale",d["unit"])
    def percentile(q):
        if not values:return None
        v=sorted(values);pos=(len(v)-1)*q;lo=int(pos);hi=min(lo+1,len(v)-1)
        weight=pos-lo
        return v[lo]*(1-weight)+v[hi]*weight
    metrics.update(count=len(values),coverage=positive/area if area>0 and 0<=positive<=area else None,density_per_100k_pixels=len(values)*100000/area if area>0 else None,p50=percentile(.5),p90=percentile(.9))
    for key in ('p50','p90'):
        declared=d.get('declared_'+key)
        if declared is not None and (metrics[key] is None or abs(declared-metrics[key])>1e-9):issue('quantile_mismatch',{'field':key,'declared':declared,'recomputed':metrics[key]})

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
