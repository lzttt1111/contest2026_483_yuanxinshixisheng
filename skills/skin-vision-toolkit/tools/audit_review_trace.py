"""Measure public execution metadata, without exporting dialogue or reasoning."""
import argparse,hashlib,json,re
from pathlib import Path
from datetime import datetime
R=Path(__file__).resolve().parents[1]
def dt(x):return datetime.fromisoformat(x.replace('Z','+00:00'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--transcript',type=Path,required=True);args=ap.parse_args()
 phases=[];current=None
 for raw in args.transcript.open(encoding='utf-8'):
  row=json.loads(raw);d=row.get('payload',{});kind=row.get('type');typ=d.get('type');ts=row.get('timestamp')
  if kind=='event_msg' and typ=='task_started':current={'turn_id':d.get('turn_id'),'started_at':ts,'tool_calls':0,'user_input_request_calls':0,'model':None,'effort':None}
  if current is None:continue
  if kind=='turn_context':current['model']=d.get('model');current['effort']=d.get('effort')
  if kind=='response_item' and typ in ('function_call','custom_tool_call'):
   current['tool_calls']+=1
   text=d.get('name','')+' '+str(d.get('input',d.get('arguments','')))
   if re.search(r'request_user_input|request_onboarding_input',text):current['user_input_request_calls']+=1
  if kind=='response_item' and typ=='message' and d.get('role')=='assistant':
   text=''.join(x.get('text','') for x in d.get('content',[]))
   try:answer=json.loads(text)
   except ValueError:answer={}
   if answer.get('phase') in ('baseline','with_skill','recheck','completion_audit'):current['phase']=answer['phase']
  if kind=='event_msg' and typ=='task_complete':
   if current.get('phase'):
    current['completed_at']=ts;current['wall_seconds']=(dt(ts)-dt(current['started_at'])).total_seconds();phases.append(current)
   current=None
 assert {x['phase'] for x in phases}=={'baseline','with_skill','recheck','completion_audit'}
 result={'phases':phases,'source_sha256':hashlib.sha256(args.transcript.read_bytes()).hexdigest(),'definition':'Wall time from task_started to task_complete, includes model generation and tools. Input request count counts recorded request-user-input calls, not all hypothetical helpful clarifications.','per_case_wall_time':'not_instrumented; do not divide aggregate into claimed measurements','causal_speedup':'not_supported','raw_content_exported':False}
 (R/'evaluation/comparison_measurements.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(phases,ensure_ascii=False))
if __name__=='__main__':main()
