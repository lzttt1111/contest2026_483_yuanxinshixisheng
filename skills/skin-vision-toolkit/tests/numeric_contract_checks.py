"""CLI regressions for boolean-as-number and valid numeric boundaries."""
import copy,json,subprocess,sys,tempfile,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 ap=__import__('argparse').ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 normal={c['skill']:c['input'] for c in json.loads((R/'tests/cases.json').read_text(encoding='utf-8')) if c['scenario']=='normal'}
 probes=[]
 def add(skill,path,value,expected='invalid_input',extra=None):
  d=copy.deepcopy(normal[skill]);node=d
  for key in path[:-1]:node=node[key]
  node[path[-1]]=value
  if extra:d.update(extra)
  probes.append((skill+'.'+'.'.join(map(str,path))+'='+repr(value),skill,d,expected))
 m='vision-metrics-calibration';p='vision-runtime-parity';c='pose-guided-capture'
 for v in (True,False,'1',float('nan'),float('inf'),-1,0):add(m,['physical_scale'],v,extra={'unit':'mm2'})
 for v in (True,False,'1',float('nan'),float('inf'),-1):add(p,['max_abs_tolerance'],v)
 for v in (True,False,1.0,'1',0,2):add(c,['config','yaw_sign'],v)
 for key in ('yaw_tolerance','pitch_limit','roll_limit','stable_ms','max_gap_ms','max_age_ms'):add(c,['config',key],True)
 add(c,['config','slots','front'],True)
 for key in ('t','age_ms','yaw','raw_yaw','pitch','roll'):add(c,['frames',0,key],True)
 for key in ('declared_p50','declared_p90','declared_count'):add(m,[key],True)
 add('vision-feature-engineering',['min_retained_ratio'],True)
 add('vision-feature-engineering',['instances',0,'x'],False)
 add('vision-data-review',['records',0,'revision'],True)
 for key in ('current_revision','result_revision','source_frame','target_frame'):add('vision-system-integration',[key],True)
 for field in ('scale','tolerance_pixels','pad_xy','model_points','original_points'):
  d=copy.deepcopy(normal[p]);d['coordinate_case']={'scale':1,'pad_xy':[0,0],'model_points':[[1,2]],'original_points':[[1,2]],'tolerance_pixels':0}
  d['coordinate_case'][field]=[True,0] if field=='pad_xy' else [[True,2]] if field.endswith('points') else True
  probes.append(('coordinate_bool_'+field,p,d,'invalid_input'))
 add(m,['physical_scale'],0.1,'pass',{'unit':'mm2'})
 add(p,['max_abs_tolerance'],0,'pass',{'target':copy.deepcopy(normal[p]['source'])})
 for v in (-1,1):add(c,['config','yaw_sign'],v,'pass')
 results=[]
 with tempfile.TemporaryDirectory(prefix='numeric_contract_') as td:
  tmp=Path(td)
  for n,(ident,skill,d,status) in enumerate(probes):
   inp=tmp/(str(n)+'.json');inp.write_text(json.dumps(d),encoding='utf-8')
   run=subprocess.run([sys.executable,'-I','-B',str(R/'skills'/skill/'scripts/check.py'),str(inp),'--root',str(R/'tests/assets')],capture_output=True,text=True,encoding='utf-8')
   try:out=json.loads(run.stdout)
   except ValueError:out={'status':'crash','stderr':run.stderr}
   results.append({'id':ident,'passed':out.get('status')==status and run.returncode==(0 if status=='pass' else 2),'exit_code':run.returncode,'result':out})
  (tmp/'src.json').write_text('{"count":1}',encoding='utf-8');(tmp/'dst.json').write_text('{"count":true}',encoding='utf-8')
  inp=tmp/'pub.json';inp.write_text(json.dumps({'task_kind':'publication','source_file':'src.json','report_file':'dst.json','artifact_files':[],'inference_requested':False}),encoding='utf-8')
  run=subprocess.run([sys.executable,'-I','-B',str(R/'skills/vision-result-publication/scripts/check.py'),str(inp),'--root',str(tmp)],capture_output=True,text=True,encoding='utf-8');out=json.loads(run.stdout)
  results.append({'id':'publication_bool_equals_one','passed':out['status']=='fail' and run.returncode==1,'result':out})
 report={'passed':all(x['passed'] for x in results),'count':len(results),'cases':results,'interpreter':sys.version.split()[0],'platform':sys.platform,'cli_tested':True,'production_inputs_used':False}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'passed':report['passed'],'count':len(results),'failed':[x['id'] for x in results if not x['passed']]}))
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
