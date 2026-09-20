"""Save only public final review messages from the explicitly named evaluator."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--transcript',type=Path,required=True);ap.add_argument('--phase',choices=['baseline','with_skill','recheck','completion_audit','numeric_recheck'],required=True);args=ap.parse_args()
 found=None
 for raw in args.transcript.open(encoding='utf-8'):
  r=json.loads(raw);p=r.get('payload',{})
  if p.get('type')!='agent_message' or p.get('author')!='/root/independent_skill_validation':continue
  text='\n'.join(x.get('text','') for x in p.get('content',[]) if x.get('type') in ('text','input_text','output_text'))
  if text.startswith('Message Type: FINAL_ANSWER') and '\nPayload:\n' in text:
   text=text.split('\nPayload:\n',1)[1]
  try:d=json.loads(text)
  except ValueError:continue
  if d.get('phase')==args.phase:
   found={'source_event_timestamp':r.get('timestamp'),'source_event_sha256':hashlib.sha256(raw.encode()).hexdigest(),'author':p['author'],'review':d}
 if found is None:raise SystemExit('Matching final JSON not yet found')
 out=ROOT/'evaluation'/('independent_'+args.phase+'.json')
 if out.exists():raise SystemExit('Review already saved')
 out.write_text(json.dumps(found,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('Saved authentic public review:',args.phase)
if __name__=='__main__':main()
