"""Build a portable, checksummed local archive; no upload or installation."""
import hashlib,json,sys,tempfile,zipfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 required=['README.md','VALIDATION.md','LICENSE','NOTICE','VERSION','evaluation/independent_baseline.json','evaluation/independent_with_skill.json','evaluation/independent_recheck.json','evaluation/self_checks.json','evaluation/linux_checks.json','evaluation/review_regressions.json','evaluation/structure_checks.json','evaluation/project_replay.json']
 for name in required:
  if not (ROOT/name).is_file():raise RuntimeError('Missing release evidence: '+name)
 for name in ('completion_checks','requirements_audit'):
  if not json.loads((ROOT/'evaluation'/(name+'.json')).read_text(encoding='utf-8'))['passed']:raise RuntimeError('Completion gate failed: '+name)
 audit=json.loads((ROOT/'evaluation/independent_completion_audit.json').read_text(encoding='utf-8'))['review']
 if audit['blocking_issues'] or audit['release_recommendation']['approved'] is not True:raise RuntimeError('Independent completion gate not approved')
 for name in ['self_checks','linux_checks','review_regressions','structure_checks','project_replay']:
  if not json.loads((ROOT/'evaluation'/(name+'.json')).read_text(encoding='utf-8'))['passed']:raise RuntimeError('Unpassed validation: '+name)
 review=json.loads((ROOT/'evaluation/independent_recheck.json').read_text(encoding='utf-8'))['review']
 if review['release_recommendation']['blocking_issues_remaining']:raise RuntimeError('Independent blockers remain')
 version=(ROOT/'VERSION').read_text().strip();release=ROOT.parent/'发布包';release.mkdir(exist_ok=True)
 for name in ('numeric_contract_windows','numeric_contract_linux'):
  result=json.loads((ROOT/'evaluation'/(name+'.json')).read_text(encoding='utf-8'))
  if not result['passed'] or result['count']<52:raise RuntimeError('Numeric type gate failed: '+name)
 numeric_review=json.loads((ROOT/'evaluation/independent_numeric_recheck.json').read_text(encoding='utf-8'))['review']
 if numeric_review['blocking_issues'] or numeric_review['release_recommendation']['approved'] is not True:raise RuntimeError('Numeric independent gate failed')
 if version=='0.1.3':
  for name in ('xhigh_regressions_013_windows','xhigh_regressions_013_linux','run_checks_013_windows','run_checks_013_linux','numeric_contract_checks_013_windows','numeric_contract_checks_013_linux','review_regressions_013_windows','completion_checks_013_windows','structure_checks_013'):
   if not json.loads((ROOT/'evaluation'/(name+'.json')).read_text(encoding='utf-8'))['passed']:raise RuntimeError('0.1.3 gate failed: '+name)
  approval=json.loads((ROOT/'evaluation/independent_xhigh_recheck.json').read_text(encoding='utf-8'))
  if approval.get('approved') is not True or approval.get('version')!=version:raise RuntimeError('Current xhigh approval required')
  for name,expected in approval['reviewed_source_sha256'].items():
   if sha((ROOT/name).read_bytes())!=expected:raise RuntimeError('Source changed after xhigh review: '+name)
 archive=release/('visual-ai-skills-'+version+'.zip')
 if archive.exists():raise RuntimeError('Refuse to overwrite existing release')
 files=[]
 for p in sorted(ROOT.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.tmp') or p.name in ('FILE_MANIFEST.json','SHA256SUMS.txt'):continue
  if p.is_symlink():raise RuntimeError('Symlink in portable release')
  files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size})
 manifest={'version':version,'files':files,'count':len(files),'models_or_private_images_included':False,'installed_globally':False,'uploaded':False}
 (ROOT/'FILE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (ROOT/'SHA256SUMS.txt').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in files),encoding='utf-8')
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for x in files:z.write(ROOT/x['path'],'visual-ai-skills/'+x['path'])
  for name in ('FILE_MANIFEST.json','SHA256SUMS.txt'):z.write(ROOT/name,'visual-ai-skills/'+name)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for x in files:assert sha(z.read('visual-ai-skills/'+x['path']))==x['sha256']
  with tempfile.TemporaryDirectory(prefix='released_skill_unpack_') as td:
   z.extractall(td);package=Path(td)/'visual-ai-skills'
   result=subprocess.run([sys.executable,'-I','-B',str(package/'tests/run_checks.py'),'--output',str(Path(td)/'test-results.json')],capture_output=True,text=True,encoding='utf-8')
   assert result.returncode==0,result.stdout+result.stderr
   for script in ('review_regressions.py','completion_checks.py','xhigh_regressions.py'):
    result=subprocess.run([sys.executable,'-I','-B',str(package/'tests'/script)],capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
   result=subprocess.run([sys.executable,'-I','-B',str(package/'tests/numeric_contract_checks.py'),'--output',str(Path(td)/'numeric-results.json')],capture_output=True,text=True,encoding='utf-8')
   assert result.returncode==0,result.stdout+result.stderr
 receipt={'archive':archive.name,'sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'manifest_file_count':len(files),'archive_crc_verified':True,'every_entry_hash_verified':True,'unpacked_30_cases_passed':True,'uploaded':False}
 (release/'发布校验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
