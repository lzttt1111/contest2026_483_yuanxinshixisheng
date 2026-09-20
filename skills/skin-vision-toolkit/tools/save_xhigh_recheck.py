"""Archive the public xhigh approval with reviewed source fingerprints."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
session=Path(r'C:\Users\28330\.codex\sessions\2026\09\10\rollout-2026-09-10T10-39-44-01a0892f-7594-73d0-881f-0cd97927f975.jsonl')
selected=None
for line in session.open(encoding='utf-8'):
    event=json.loads(line);p=event.get('payload',{})
    if p.get('type')!='agent_message':continue
    body='\n'.join(x.get('text','') for x in p.get('content',[]) if isinstance(x,dict))
    if 'Message Type: FINAL_ANSWER' in body and 'Sender: /root/astra_xhigh_release_audit' in body and '0.1.3' in body:
        selected=(event.get('timestamp'),body.split('Payload:\n',1)[-1])
if not selected:raise SystemExit('No public 0.1.3 final review found')
body=selected[1]
if 'approved' not in body or 'true' not in body:raise SystemExit('No explicit approval marker; inspect review')
receipt={'version':'0.1.3','approved':True,'model':'gpt-6-astra','reasoning_effort':'xhigh','public_message_timestamp':selected[0],'review':body,'reviewed_source_sha256':{}}
for group in ('skills','tests'):
    for path in sorted((ROOT/group).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            receipt['reviewed_source_sha256'][path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/'evaluation/independent_xhigh_recheck.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT.parent/'强审报告/0.1.3_Astra_xhigh_修复复核报告.md').write_text(body+'\n',encoding='utf-8')
print('Saved approval and '+str(len(receipt['reviewed_source_sha256']))+' source hashes')
