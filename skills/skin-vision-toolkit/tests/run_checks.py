"""Execute frozen offline behavior cases; no imports from source projects."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(root):
 cases=json.loads((root/'tests/cases.json').read_text(encoding='utf-8'));results=[]
 with tempfile.TemporaryDirectory(prefix='vision_skill_cases_') as td:
  work=Path(td);assets=work/'assets';shutil.copytree(root/'tests/assets',assets)
  protected={p.name:digest(p) for p in assets.iterdir() if p.is_file()}
  for c in cases:
   path=work/(c['id']+'.json');path.write_text(json.dumps(c['input']),encoding='utf-8')
   started=time.perf_counter()
   proc=subprocess.run([sys.executable,'-I','-B',str(root/'skills'/c['skill']/'scripts/check.py'),str(path),'--root',str(assets)],capture_output=True,text=True,encoding='utf-8')
   try:r=json.loads(proc.stdout)
   except ValueError:r={'status':'invalid_output','findings':[],'stderr':proc.stderr[:500]}
   errors=[];expected=c['expected'];codes={x['code'] for x in r.get('findings',[])}
   if r.get('status')!=expected['status']:errors.append('status')
   if not set(expected['codes'])<=codes:errors.append('missing_findings')
   expected_exit=1 if expected['status']=='fail' else 0
   if proc.returncode!=expected_exit:errors.append('exit_code')
   m=r.get('metrics',{})
   if c['id']=='04-normal' and m.get('captured')!={'front':2}:errors.append('capture_frame')
   if c['id']=='04-failure' and m.get('captured'):errors.append('unsafe_capture')
   if c['id']=='05-normal' and (m.get('coverage')!=.25 or m.get('valid_pixels')!=4):errors.append('domain_arithmetic')
   if c['id']=='06-normal' and (m.get('count')!=4 or abs(m.get('p50',0)-2.5)>1e-10 or abs(m.get('p90',0)-3.7)>1e-10 or m.get('density_per_100k_pixels')!=4000):errors.append('metric_arithmetic')
   if c['id']=='06-failure' and m.get('coverage') is not None:errors.append('missing_denominator_not_null')
   if c['id']=='07-normal' and m.get('artifact_hashes',{}).get('evidence.txt')!=digest(assets/'evidence.txt'):errors.append('file_hash')
   if c['id']=='08-normal' and m.get('recovery_plan')!=[{'id':'a','action':'skip'},{'id':'b','action':'retry_or_recompute'}]:errors.append('resume_actions')
   if c['id']=='09-normal' and (m.get('max_abs',1)>.001 or not m.get('numeric_comparison_valid')):errors.append('parity')
   if c['id']=='10-failure' and m.get('point_transfer_allowed'):errors.append('unsafe_point_transfer')
   if digest(path)!=hashlib.sha256(json.dumps(c['input']).encode()).hexdigest():errors.append('input_modified')
   results.append({'id':c['id'],'passed':not errors,'errors':errors,'seconds':time.perf_counter()-started,'result':r})
  assert protected=={p.name:digest(p) for p in assets.iterdir() if p.is_file()},'Fixture files changed'
 return results
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--relocation',action='store_true');args=parser.parse_args()
 source=run(ROOT);relocated=[]
 if args.relocation:
  with tempfile.TemporaryDirectory(prefix='unrelated_project_') as td:
   target=Path(td)/'portable skill package'
   shutil.copytree(ROOT/'skills',target/'skills');shutil.copytree(ROOT/'tests',target/'tests')
   relocated=run(target)
 report={'kind':'deterministic_offline_checker_tests','not_llm_behavior_benchmark':True,'cases':source,'relocated_cases':relocated,'source_cases_sha256':digest(ROOT/'tests/cases.json'),'passed':all(x['passed'] for x in source+relocated),'source_count':len(source),'relocated_count':len(relocated),'no_model_or_project_dependency':True}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'passed':report['passed'],'source':len(source),'relocated':len(relocated),'failed':[x for x in source+relocated if not x['passed']]},ensure_ascii=False))
 return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
