"""Snapshot bounded evidence metadata, never copy full chats or private pictures."""
import argparse,hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--logs-root',type=Path,required=True);ap.add_argument('--water-root',type=Path,required=True);ap.add_argument('--pc-root',type=Path,required=True);args=ap.parse_args()
 requested=json.loads((ROOT/'evidence/selection.json').read_text(encoding='utf-8'));mapping=requested['sources'];wanted=requested['selections'];records=[]
 for key,relative in mapping.items():
  path=(args.logs_root/relative).resolve()
  assert path.is_relative_to(args.logs_root.resolve())
  file_hash=digest(path);matches={x['seq']:x['skill'] for x in wanted if x['source']==key}
  for line in path.open(encoding='utf-8'):
   e=json.loads(line)
   if e['seq'] not in matches:continue
   assert e['role']=='user'
   records.append({'skill':matches[e['seq']],'source_alias':key,'source_relative_path':relative,'source_file_sha256':file_hash,'session_id':e['session_id'],'ts':e['ts'],'seq':e['seq'],'event_sha256':hashlib.sha256(line.encode()).hexdigest(),'evidence_type':'user_accepted_plan' if 'PLEASE IMPLEMENT THIS PLAN' in e.get('text','') else 'user_direct_request','summary':'See skill-specific project-case.md; full private source is not distributed.'})
 assert len(records)==10
 pairs=[('water_capture',args.water_root/'shuiguang/capture.py'),('water_pipeline',args.water_root/'shuiguang/pipeline.py'),('water_regional_tests',args.water_root/'tests/test_regional_v2.py'),('water_audit_side_effects',args.water_root/'tools/audit_regional_v2.py'),('pc_web_modes',args.pc_root/'web/app.js'),('pc_capture_gate',args.pc_root/'lab/state.py')]
 code=[]
 for name,p in pairs:
  code.append({'id':name,'relative_path':str(p.relative_to(args.water_root if name.startswith('water') else args.pc_root)).replace('\\','/'),'sha256':digest(p),'bytes':p.stat().st_size})
 now=datetime.now(timezone.utc).isoformat()
 (ROOT/'evidence/method_sources.json').write_text(json.dumps({'captured_at':now,'records':records,'not_proof_new_skill_used_historically':True},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (ROOT/'evidence/source_code.json').write_text(json.dumps({'captured_at':now,'files':code,'source_roots_not_distributed':True,'observations':['PC mode switch calls reset(all).','Regional audit invokes enrich_result and rebuild; not a read-only source command.','Water capture requires three distinct RGB views; PC two-slot contract must not be assumed equivalent.']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('Evidence: 10 user records; 6 source snapshots; no raw dialogue exported.')
if __name__=='__main__':main()
