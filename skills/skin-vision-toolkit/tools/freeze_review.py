"""Hash frozen test definitions and standalone skill resources before independent use."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[1]
files=[R/'tests/cases.json',R/'evaluation/prompts.json']+sorted((R/'skills').rglob('*'))
items=[]
for p in files:
 if p.is_file():items.append({'path':p.relative_to(R).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
out=R/'evaluation/self_review_freeze.json'
if out.exists():raise SystemExit('Freeze exists; do not overwrite')
out.write_text(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'files':items,'status':'self_review_passed_before_independent_skill_phase'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Frozen',len(items),'files')
