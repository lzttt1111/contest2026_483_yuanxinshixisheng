"""Additional tests derived from named plan gates, not from checker wording."""
import copy,importlib.util,json,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def check(name,d):
 s=importlib.util.spec_from_file_location(name,R/'skills'/name/'scripts/check.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 return m.check(d,(R/'tests/assets').resolve())
def main():
 base={c['skill']:c['input'] for c in json.loads((R/'tests/cases.json').read_text(encoding='utf-8')) if c['scenario']=='normal'}
 records=[]
 def test(name,passed,evidence):records.append({'name':name,'passed':bool(passed),'evidence':evidence})
 d=copy.deepcopy(base['pose-guided-capture']);d['config']['slots']={'left':-45};d['config']['yaw_sign']=-1
 for f in d['frames']:f['yaw']=45;f['raw_yaw']=45
 a=check('pose-guided-capture',d);test('mirror_correct',a['metrics']['captured']=={'left':2},a)
 d['config']['yaw_sign']=1;a=check('pose-guided-capture',d);test('mirror_wrong_does_not_capture_expected_side',not a['metrics']['captured'],a)
 d=copy.deepcopy(base['pose-guided-capture']);d['capture_revision']=4
 extension=[]
 for i,t in enumerate((300,400,500)):
  f=copy.deepcopy(d['frames'][0]);f['t']=t
  if i==0:f['retake']='front'
  extension.append(f)
 d['frames']+=extension;a=check('pose-guided-capture',d)
 test('retake_recaptures_new_frame',a['metrics']['captured']=={'front':5} and a['metrics']['capture_revision']==5,a)
 integ=copy.deepcopy(base['vision-system-integration']);integ['current_revision']=a['metrics']['capture_revision'];integ['result_revision']=4
 a=check('vision-system-integration',integ);test('retake_invalidates_previous_result',not a['metrics']['point_transfer_allowed'],a)
 d=copy.deepcopy(base['pose-guided-capture']);d['frames'][1]['face']='other'
 a=check('pose-guided-capture',d);test('face_switch_breaks_stability',not a['metrics']['captured'],a)
 d=copy.deepcopy(base['pose-guided-capture']);d['frames'][1]['t']=500;d['frames'][2]['t']=600
 a=check('pose-guided-capture',d);test('gap_breaks_stability',not a['metrics']['captured'],a)
 d=copy.deepcopy(base['vision-metrics-calibration']);d['observed_domain_pixels']=50
 a=check('vision-metrics-calibration',d);test('inflated_denominator_detected','expanded_denominator' in [x['code'] for x in a['findings']],a)
 d=copy.deepcopy(base['vision-metrics-calibration']);d['values']=[1,2,3,100];d['declared_p90']=46.1
 a=check('vision-metrics-calibration',d);test('do_not_average_percentiles',abs(a['metrics']['p90']-70.9)<1e-9 and any(x['code']=='quantile_mismatch' for x in a['findings']),a)
 d=copy.deepcopy(base['vision-runtime-parity']);d['coordinate_case']={'scale':0.5,'pad_xy':[0,10],'model_points':[[5,20]],'original_points':[[10,20]],'tolerance_pixels':0.001}
 a=check('vision-runtime-parity',d);test('letterbox_inverse',a['metrics']['coordinate_parity']['passed'],a)
 d['coordinate_case']['pad_xy']=[0,0]
 a=check('vision-runtime-parity',d);test('wrong_padding_detected',a['status']=='fail' and not a['metrics']['coordinate_parity']['passed'],a)
 d['coordinate_case'].update(pad_xy=[0,10],mirror_width=100,original_points=[[89,20]])
 a=check('vision-runtime-parity',d);test('inverse_then_mirror',a['metrics']['coordinate_parity']['passed'],a)
 d['coordinate_case']['scale']=0
 try:check('vision-runtime-parity',d);rejected=False
 except ValueError:rejected=True
 test('zero_scale_rejected',rejected,{'ValueError':rejected})
 report={'passed':all(x['passed'] for x in records),'count':len(records),'cases':records,'source_projects_loaded':False}
 (R/'evaluation/completion_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'passed':report['passed'],'count':len(records),'failed':[x['name'] for x in records if not x['passed']]}))
 if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
