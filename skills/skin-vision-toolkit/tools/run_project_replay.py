"""Replay a cached project result without model imports or writes to the source project."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_check(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'skills'/name/'scripts/check.py')
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.check
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result-root',type=Path,required=True);args=ap.parse_args();source=args.result_root.resolve()
 p=source/'水光检测完整结果.json';q=source/'02_front/pores/metrics.json'
 d=json.loads(p.read_text(encoding='utf-8'));item=d['views']['front']['modules']['pores'];saved=json.loads(q.read_text(encoding='utf-8'))
 evidence=(source/item['evidence']).resolve();assert evidence.is_relative_to(source)
 before={p.name:sha(p),q.relative_to(source).as_posix():sha(q),evidence.relative_to(source).as_posix():sha(evidence)}
 assert sha(evidence)==item['evidence_sha256']
 with np.load(evidence,allow_pickle=False) as f:
  valid=f['valid_mask']>0;mask=f['mask']>0
  area=int(valid.sum());positive=int((valid&mask).sum())
 values=[x['measurement_area'] for x in item['instances']]
 inp={'task_kind':'metrics','valid_pixels':area,'positive_pixels':positive,'values':values,'declared_count':item['metrics']['count'],'cross_view_unique_claim':False,'correspondence_verified':False,'unit':'pixel','physical_scale':None}
 result=load_check('vision-metrics-calibration')(inp,source)
 assert result['status']=='pass'
 assert area==item['metrics']['analysis_pixels']
 assert abs(result['metrics']['coverage']-item['metrics']['coverage'])<1e-9
 assert abs(result['metrics']['density_per_100k_pixels']-item['metrics']['density_per_100k_pixels'])<1e-9
 # Two separately saved source projections, copied without image/identity information.
 out=ROOT/'evaluation/project_replay';out.mkdir(parents=True,exist_ok=True)
 for name,v in [('source.json',item['metrics']),('report.json',saved['metrics'])]:
  (out/name).write_text(json.dumps(v,ensure_ascii=False)+'\n',encoding='utf-8')
 (out/'source_hashes.json').write_text(json.dumps(before,indent=2)+'\n',encoding='utf-8')
 publication=load_check('vision-result-publication')({'task_kind':'publication','source_file':'source.json','report_file':'report.json','artifact_files':['source_hashes.json'],'inference_requested':False},out)
 assert publication['status']=='pass'
 after={p.name:sha(p),q.relative_to(source).as_posix():sha(q),evidence.relative_to(source).as_posix():sha(evidence)}
 assert before==after
 receipt={'passed':True,'source_alias':'existing_rgb_triplet/front/pores','source_hashes':before,'metrics':result,'publication':publication,'source_files_unchanged':True,'inference_executed':False,'numpy_used_for_saved_npz_only':True,'limitation':'Checks cached quantities and projections; does not certify detector accuracy or physical skin measurements.'}
 (ROOT/'evaluation/project_replay.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'passed':True,'count':len(values),'valid_pixels':area,'support_pixels':positive,'source_unchanged':True,'inference':False}))
if __name__=='__main__':main()
