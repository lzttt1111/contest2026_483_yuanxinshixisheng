"""Offline fit from frozen saved evidence; never imports inference modules."""
from pathlib import Path
import argparse,collections,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloud_three.independent_spots import VERSION,GROUPS,WEIGHTS,measures,score

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def winpath(value):return Path(value.replace('D:\\','/mnt/d/').replace('\\','/'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();rows=[json.loads(x) for x in a.cache.read_text().splitlines()]
 frozen=json.loads((a.freeze/'freeze_manifest.json').read_text())['inputs']
 sources={x['relative_path']:x['source'] for x in json.loads((a.freeze/'private/provenance.json').read_text())['inputs']}
 identities={x['input_sha256']:x for x in frozen}
 assert len(rows)==len(identities)==1000 and len({x['input_sha256'] for x in rows})==1000
 assert collections.Counter(x['split'] for x in rows)=={'development':800,'independent_confirmation':200}
 assert len({json.dumps(x['parameters'],sort_keys=True) for x in rows})==1
 # Freeze engineering definition before independent confirmation. Not clinical calibration.
 ref={'version':VERSION,'score_direction':'higher_is_better','weights':WEIGHTS,'minimum_valid_area':2000,'minimum_reference_count':100,'detector_parameters':rows[0]['parameters'],'region_groups':GROUPS,'regions':{},'calibration_kind':'engineering_empirical_rank_not_clinical_validation','source_manifest_sha256':sha(a.freeze/'freeze_manifest.json')}
 accepted=[];excluded=[];proof=[]
 for i,r in enumerate(rows):
  f=identities[r['input_sha256']];s=sources[f['relative_path']]
  assert r['split']==f['split']
  index=winpath(s['v2_index_path']);raw=Path(r['source_result'])
  assert sha(index)==s['v2_index_sha256'] and sha(raw)==r['source_sha256']
  assert sha(winpath(s['v2_input_copy_path']))==r['input_sha256']==s['v2_input_copy_sha256']
  idx=json.loads(index.read_text());spots=json.loads((index.parent/idx['items']['spots']['量化JSON']).read_text())
  assert spots['medical_metrics_v2']==r['medical_metrics_v2'] and spots['parameters']==r['parameters']
  proof.append({'input_sha256':r['input_sha256'],'result_sha256':r['source_sha256'],'index_sha256':s['v2_index_sha256'],'split':r['split']})
  try:accepted.append((r,measures(r)))
  except ValueError as e:excluded.append({'input_sha256':r['input_sha256'],'split':r['split'],'reason':str(e)})
  if (i+1)%100==0:print('verified',i+1,flush=True)
 for region in ('full_face',*GROUPS):
  eligible=[m[region] for r,m in accepted if r['split']=='development' and m[region]['quality']=='可评估' and m[region]['valid_area']>=ref['minimum_valid_area']]
  ref['regions'][region]={key:sorted(m[key] for m in eligible) for key in WEIGHTS}
  assert len(eligible)>=ref['minimum_reference_count']
 ref['development_source_count']=sum(r['split']=='development' for r,m in accepted)
 ref['identity_proof_sha256']=hashlib.sha256(json.dumps(proof,sort_keys=True).encode()).hexdigest()
 confirmation=[]
 for r,m in accepted:
  if r['split']!='independent_confirmation':continue
  scored,trace=score(r,ref)
  for region,node in zip(('full_face',*GROUPS),(scored,*scored['regional_scores']['items'])):
   assert node['score'] is None or 0<=node['score']<=100
   if m[region]['count']>0:assert node['score']!=100
  confirmation.append({'input_sha256':r['input_sha256'],'result':scored})
 a.output.mkdir(parents=True,exist_ok=True)
 for name,data in [('independent_spots_reference.json',ref),('identity_proof.json',proof),('confirmation.json',confirmation),('fit_receipt.json',{'definition':ref['calibration_kind'],'excluded':excluded,'accepted':dict(collections.Counter(r['split'] for r,m in accepted)),'confirmation_count':len(confirmation),'inference_runs':0})]:
  (a.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 print('completed',len(accepted),'excluded',len(excluded),flush=True)
if __name__=='__main__':main()
