TASK="batch"
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
    d=data;findings=[];metrics={}
    def issue(code,detail):findings.append({"code":code,"detail":detail})

    if not d["config_hash"] or type(d['workers']) is not int or type(d['max_workers']) is not int or d["workers"]<1 or d["max_workers"]<1:raise ValueError("batch worker budgets must be positive integers")
    if d["workers"]>d["max_workers"]:issue("resource_budget_exceeded",{"workers":d["workers"],"budget":d["max_workers"]})
    plan=[];seen=set()
    for job in d["jobs"]:
        if type(job['committed']) is not bool or type(job['artifacts_valid']) is not bool:raise ValueError('completion evidence flags must be booleans')
        if job['status'] not in ('success','failed','running','pending','queued','cancelled'):raise ValueError('unknown job status')
        ident=job["id"]
        if type(ident) is not str or not ident.strip():raise ValueError('nonempty job identity required')
        if ident in seen:issue("duplicate_job",ident)
        seen.add(ident);same=job["config_hash"]==d["config_hash"]
        if not same:issue("cache_identity_mismatch",ident)
        complete=job["status"]=="success" and job["committed"] and job["artifacts_valid"] and same
        if job["status"]=="success" and not (job["committed"] and job["artifacts_valid"]):issue("incomplete_success",ident)
        plan.append({"id":ident,"action":"skip" if complete else "retry_or_recompute"})
    metrics.update(recovery_plan=plan,execution_started=False,preconditions={j['id']:['verify_previous_owner_stopped_or_lease_expired'] for j in d['jobs'] if j['status'] in ('running','pending','queued','cancelled')})

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
