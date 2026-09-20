from pathlib import Path
import json,time,requests
ROOT=Path(__file__).resolve().parents[1]
request=json.loads((ROOT/'request_scores_example.json').read_text());request['task_id']='http_diagnostic_one_v1'
guards={}
for route,code in [('score-jobs','SCORING_REFERENCE_NOT_READY'),('jobs','LEGACY_TASK_DISABLED')]:
 blocked=requests.post('http://127.0.0.1:8897/api/'+route,json=request,timeout=10)
 assert blocked.status_code==409 and blocked.json()['detail']['code']==code
 guards[route]=code
response=requests.post('http://127.0.0.1:8897/api/diagnostic-jobs',json=request,timeout=10);response.raise_for_status();queued=response.json()
assert response.status_code==202
deadline=time.monotonic()+180
while time.monotonic()<deadline:
 response=requests.get('http://127.0.0.1:8897/api/diagnostic-jobs/'+queued['queue_id'],timeout=10);response.raise_for_status();result=response.json()
 if result.get('schema_version')=='shuiguang_diagnostic_v1':break
 if result.get('status')=='failed':raise RuntimeError(result)
 time.sleep(1)
else:raise TimeoutError('HTTP diagnostic task did not complete')
assert result==json.loads((ROOT/'examples/diagnostic_one/result.json').read_text())
receipt={'http_submit':202,'http_query':200,'queue_id':queued['queue_id'],'result_matches_real_example':True,'scope':'local_diagnostic_only','http_409_guards':guards}
(ROOT/'LOCAL_HTTP_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
