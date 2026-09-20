"""Regression probes discovered by independent review, separate from original frozen 30."""
import copy,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 cases=json.loads((ROOT/'tests/cases.json').read_text(encoding='utf-8'))
 normal={c['skill']:c['input'] for c in cases if c['scenario']=='normal'}
 probes=[]
 d=copy.deepcopy(normal['vision-data-review']);d['rerun_ids']=['ghost']
 probes.append(('unknown-rerun','vision-data-review',d,'fail','unknown_rerun_id'))
 d=copy.deepcopy(normal['vision-data-review']);d['records'][0].pop('human_confirmed');d['rerun_ids']=['a']
 probes.append(('unknown-review-state','vision-data-review',d,'fail','unknown_review_state'))
 d=copy.deepcopy(normal['vision-batch-recovery']);d['jobs'][0]['committed']='false';d['jobs'][0]['artifacts_valid']='false'
 probes.append(('string-false','vision-batch-recovery',d,'invalid_input','invalid_input'))
 d=copy.deepcopy(normal['vision-system-integration']);d['producer_fields']=['capture_id']
 probes.append(('missing-contract-point','vision-system-integration',d,'fail','missing_contract_fields'))
 d=copy.deepcopy(normal['vision-spec-to-evidence']);d['requirements'][0].pop('unit')
 probes.append(('missing-unit','vision-spec-to-evidence',d,'fail','missing_unit'))
 for value,label in [(1.5,'fractional-workers'),(True,'boolean-workers')]:
  d=copy.deepcopy(normal['vision-batch-recovery']);d['workers']=value
  probes.append((label,'vision-batch-recovery',d,'invalid_input','invalid_input'))
 d=copy.deepcopy(normal['pose-guided-capture']);d['config']['min_frames']=True
 probes.append(('boolean-frame-count','pose-guided-capture',d,'invalid_input','invalid_input'))
 for skill,field in [('vision-system-integration','registration_verified'),('vision-metrics-calibration','cross_view_unique_claim')]:
  d=copy.deepcopy(normal[skill]);d[field]='false'
  probes.append(('false-string-'+field,skill,d,'invalid_input','invalid_input'))
 d=copy.deepcopy(normal['vision-data-review']);d['records'][0]['human_confirmed']='false'
 probes.append(('false-string-human','vision-data-review',d,'invalid_input','invalid_input'))
 d=copy.deepcopy(normal['vision-spec-to-evidence']);d['candidates'][0]['baseline_verified']='false'
 probes.append(('false-string-baseline','vision-spec-to-evidence',d,'invalid_input','invalid_input'))
 results=[]
 for name,skill,d,status,code in probes:
  spec=importlib.util.spec_from_file_location('probe_'+name,ROOT/'skills'/skill/'scripts/check.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  try:r=module.check(d,(ROOT/'tests/assets').resolve())
  except (ValueError,TypeError,KeyError,IndexError,OSError) as e:r={'status':'invalid_input','findings':[{'code':'invalid_input','detail':str(e)}],'metrics':{}}
  passed=r['status']==status and code in {x['code'] for x in r['findings']}
  if name.startswith('unknown-'):passed=passed and not r['metrics'].get('safe_rerun_ids')
  if name=='missing-contract-point':passed=passed and not r['metrics'].get('point_transfer_allowed')
  results.append({'id':name,'passed':passed,'result':r})
 out={'passed':all(x['passed'] for x in results),'cases':results}
 path=ROOT/'evaluation/review_regressions.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(out,ensure_ascii=False))
 return 0 if out['passed'] else 1
if __name__=='__main__':sys.exit(main())
