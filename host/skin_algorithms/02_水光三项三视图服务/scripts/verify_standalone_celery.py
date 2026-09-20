"""Run the delivered package without development-tree assets or business code."""
from pathlib import Path
import os,sys,json,tempfile,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'runtime').mkdir(exist_ok=True)
RUNTIME=Path(tempfile.mkdtemp(prefix='standalone_celery_',dir=ROOT/'runtime'))
for key in ('SHUIGUANG_PREPROCESS_MODE','SHUIGUANG_FACE_PARSER_MODEL','PYTHONPATH','DERMAVISION_V3_WORKSPACE','DERMAVISION_V3_SCORE_REFERENCE'):
 os.environ.pop(key,None)
os.environ.update(SHUIGUANG_INPUT_ROOT=str(ROOT/'examples'),SHUIGUANG_RUNTIME_ROOT=str(RUNTIME),SHUIGUANG_CLOUD_DEVICE='cuda:0',SHUIGUANG_BROKER_URL='filesystem://',SHUIGUANG_RESULT_BACKEND='file://'+str(RUNTIME/'backend'),PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(ROOT))
from cloud_three.tasks import app
from cloud_three.public_scores import ScoreResponse

def main():
 log=(RUNTIME/'worker.log').open('w')
 worker=subprocess.Popen([sys.executable,'-m','celery','-A','cloud_three.tasks:app','worker','--pool=solo','--concurrency=1','-Q','shuiguang_diagnostic,shuiguang_scores','--loglevel=INFO','--hostname=package-self-contained@%h'],cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT)
 checks=[]
 latest_results={}
 try:
  for sample in ('one','two'):
   request={'task_id':'standalone_'+sample,'images':[{'image_id':v,'path':f'{sample}/inputs/{v}.jpg'} for v in ('right','front','left')],'mirrored':False}
   start=time.monotonic();r=app.send_task('shuiguang.analyze_three_views_diagnostic',args=[request],queue='shuiguang_diagnostic').get(timeout=300,interval=1)
   assert r.get('schema_version')=='shuiguang_diagnostic_v1',r
   contract=ScoreResponse.model_validate(r['scores'])
   assert r['schema_version']=='shuiguang_diagnostic_v1'
   assert r['purpose']=='diagnostic_old_reference_not_calibrated'
   assert r['preprocessing']=='bisenet_rgb_diagnostic_v1'
   assert r['reference_matches_new_preprocessing'] is False
   assert len(contract.pores.regions)==9 and len(contract.spots.regions)==6 and len(contract.surface_gloss.regions)==9
   assert all(0<=item.score<=100 for item in (contract.pores,contract.spots,contract.surface_gloss))
   assert all(item.score is not None and item.severity is not None for item in contract.spots.regions)
   latest_results[sample]=r
   job=RUNTIME/'diagnostic_jobs'/request['task_id'];identity=json.loads((job/'request.json').read_text());timing=json.loads((job/'resident_timing.json').read_text())
   assert identity['preprocessing']=='bisenet_rgb_diagnostic_v1'
   assert identity['model_sha256']=='0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'
   assert timing['algorithms']==['pores','spots','surface_gloss'] and timing['services']==['dermavision']
   trace=json.loads((job/'scoring_evidence_trace.json').read_text());assert set(trace['doctor_pipeline_projects'])=={'rgb','pores','surface_gloss'}
   again=app.send_task('shuiguang.analyze_three_views_diagnostic',args=[request],queue='shuiguang_diagnostic').get(timeout=30,interval=1);assert again==r
   formal=app.send_task('shuiguang.analyze_three_views_scores',args=[request],queue='shuiguang_scores').get(timeout=300,interval=1)
   formal_contract=ScoreResponse.model_validate(formal)
   assert all(item.score is not None and item.severity is not None for item in formal_contract.surface_gloss.regions)
   rec={'sample':sample,'elapsed_including_queue':time.monotonic()-start,'timing':timing,'structure_contract_verified':True,'latest_bisenet_scores_used':True,'legacy_fallback_non_null':True,'default_mode_is_bisenet':True,'package_model_sha_verified':True,'idempotent':True,'formal_legacy_fallback_enabled':True};checks.append(rec);print(json.dumps(rec),flush=True)
  # The formal fallback call is intentionally also a real three-item score;
  # therefore the resident counter is at least two after two diagnostic calls.
  assert checks[1]['timing']['resident_reuse_index']>=2
  artifact={'stage':'three_item_legacy_score_fallback','runtime_relative_path':RUNTIME.relative_to(ROOT).as_posix(),'business_code_root':'vendor/v3','model_relative_path':'vendor/v3/models/face_parsing_resnet18.onnx','python_environment_external_permitted':True,'transport':'filesystem_test_only','checks':checks}
  (RUNTIME/'verification.json').write_text(json.dumps(artifact,indent=2)+'\n')
  (ROOT/'STANDALONE_DIAGNOSTIC_ACCEPTANCE.json').write_text(json.dumps(artifact,indent=2)+'\n')
  (ROOT/'LATEST_CLOUD_RUN_ONE.json').write_text(json.dumps(latest_results['one'],ensure_ascii=False,indent=2)+'\n')
  (ROOT/'LATEST_CLOUD_RUN_TWO.json').write_text(json.dumps(latest_results['two'],ensure_ascii=False,indent=2)+'\n')
 finally:
  worker.terminate()
  try:worker.wait(timeout=30)
  except subprocess.TimeoutExpired:worker.kill();worker.wait()
  log.close()
if __name__=='__main__':main()
