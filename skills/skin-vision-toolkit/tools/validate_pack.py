"""Package metadata and side-effect surface audit; use alongside behavior tests."""
import argparse,ast,hashlib,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--quick-validator',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 errors=[];records=[]
 for skill in sorted((ROOT/'skills').iterdir()):
  if not skill.is_dir():continue
  p=skill/'SKILL.md';text=p.read_text(encoding='utf-8')
  q=subprocess.run([sys.executable,'-X','utf8','-B',str(args.quick_validator),str(skill)],capture_output=True,text=True,encoding='utf-8')
  if q.returncode:errors.append(skill.name+': quick_validate '+q.stdout+q.stderr)
  for doc in skill.rglob('*.md'):
   body=doc.read_text(encoding='utf-8')
   for link in re.findall(r'\]\(([^)]+)\)',body):
    if not link.startswith(('http:','https:','#')) and not (doc.parent/link).exists():errors.append(skill.name+': broken '+link)
   if re.search(r'[A-Za-z]:\\|\\\\wsl|/home/lztt',body):errors.append(skill.name+': private absolute path')
  script=skill/'scripts/check.py';tree=ast.parse(script.read_text(encoding='utf-8'))
  forbidden={'torch','onnxruntime','mediapipe','requests','urllib','subprocess','socket'}
  imports=set()
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):imports.update(a.name.split('.')[0] for a in node.names)
   if isinstance(node,ast.ImportFrom):imports.add((node.module or '').split('.')[0])
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('write_text','write_bytes','unlink','rename','replace','mkdir','rmdir'):errors.append(skill.name+': write call '+node.func.attr)
  if imports&forbidden:errors.append(skill.name+': heavy/network/process import')
  ui=(skill/'agents/openai.yaml').read_text(encoding='utf-8')
  if '$'+skill.name not in ui:errors.append(skill.name+': UI invocation')
  records.append({'skill':skill.name,'quick_validate_exit':q.returncode,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'imports':sorted(imports)})
 if len(records)!=10:errors.append('Expected 10 skills')
 report={'passed':not errors,'errors':errors,'skills':records,'note':'Static checks and quick_validate do not prove model decision quality.'}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'passed':report['passed'],'skills':len(records),'errors':errors},ensure_ascii=False))
 return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
