"""CLI regression tests for the independent xhigh review; isolated synthetic inputs."""
import copy,json,subprocess,sys,tempfile,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CASES=json.loads((ROOT/'tests/cases.json').read_text(encoding='utf-8'))
BASE={c['input']['task_kind']:c for c in CASES if c['scenario']=='normal'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    tests=[]
    def add(kind,name,changes,expected):
        d=copy.deepcopy(BASE[kind]['input'])
        for path,value in changes:
            keys=path.split('.');obj=d
            for key in keys[:-1]:obj=obj[int(key)] if isinstance(obj,list) else obj[key]
            key=keys[-1]
            if isinstance(obj,list):obj[int(key)]=value
            else:obj[key]=value
        tests.append((kind,name,d,expected))
    for key in ('source_frame','target_frame','source_clock','target_clock'):
        for value in (None,'',True):
            add('integration',key+repr(value),[(key,value)],'invalid_input')
    add('integration','empty contract',[('consumer_required',[])],'invalid_input')
    for key in ('layout','color'):
        add('parity','null '+key,[('source.'+key,None),('target.'+key,None)],'invalid_input')
    add('parity','empty provider',[('runtime_provider',''),('claimed_provider','')],'invalid_input')
    for value in ('',None,'unknown'):
        add('metrics','unit '+repr(value),[('unit',value)],'invalid_input')
    add('metrics','cm2 without scale',[('unit','cm2')],'fail')
    add('metrics','cm2 with scale',[('unit','cm2'),('physical_scale',.1)],'pass')
    for value in ('runnning',None,''):
        add('batch','status '+repr(value),[('jobs.1.status',value)],'invalid_input')
    for flag in ('mask_edited','human_edited','protected'):
        add('data',flag,[('records.0.human_confirmed',False),('records.0.'+flag,True),('rerun_ids',['a'])],'fail')
    add('data','empty id',[('records.0.id','')],'invalid_input')
    add('spec','null requirement',[('requirements',[None])],'invalid_input')
    add('spec','string needs',[('requirements.0.needs','rgb')],'invalid_input')
    add('spec','string available',[('available','rgb')],'invalid_input')
    add('capture','list slots',[('config.slots',[])],'invalid_input')
    add('parity','overflow subtraction',[('source.shape',[1]),('target.shape',[1]),('source.values',[1e308]),('target.values',[-1e308])],'invalid_input')
    add('parity','stable rmse',[('source.shape',[1]),('target.shape',[1]),('source.values',[1e200]),('target.values',[0])],'fail')
    add('metrics','stable interpolation',[('values',[-1e308,1e308]),('declared_count',2)],'pass')
    add('publication','wrong binding',[('artifact_sha256',{'evidence.txt':'0'*64})],'fail')
    digest=hashlib.sha256((ROOT/'tests/assets/evidence.txt').read_bytes()).hexdigest()
    add('publication','correct binding',[('artifact_sha256',{'evidence.txt':digest})],'pass')
    results=[]
    with tempfile.TemporaryDirectory(prefix='xhigh_regression_') as td:
        path=Path(td)/'input.json'
        for kind,name,d,expected in tests:
            path.write_text(json.dumps(d),encoding='utf-8')
            run=subprocess.run([sys.executable,'-I','-B',str(ROOT/'skills'/BASE[kind]['skill']/'scripts/check.py'),str(path),'--root',str(ROOT/'tests/assets')],capture_output=True,text=True,encoding='utf-8')
            try:out=json.loads(run.stdout)
            except ValueError:out={'status':'no_json','stderr':run.stderr}
            ok=out.get('status')==expected and run.returncode==({'pass':0,'fail':1,'invalid_input':2}[expected])
            if name=='correct binding':ok=ok and out['metrics']['binding_verified'] is True
            results.append({'name':name,'passed':ok,'expected':expected,'output':out,'exit_code':run.returncode})
    report={'passed':all(r['passed'] for r in results),'count':len(results),'results':results}
    if args.output:args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
