"""Inventory static delivery assets; never mutate user runtime/results."""
from pathlib import Path
import os,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','runtime','__pycache__','.pytest_cache','hair_repair_research'}
def row(p):return {'path':p.relative_to(ROOT).as_posix(),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 rows=[]
 for folder,dirs,files in os.walk(ROOT):
  dirs[:]=[d for d in dirs if d not in SKIP]
  for name in files:
   p=Path(folder)/name
   if p==ROOT/'PACKAGE_MANIFEST.json' or p.suffix in {'.pyc','.pyo'}:continue
   if not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('External asset link: '+str(p))
   rows.append(row(p))
 (ROOT/'PACKAGE_MANIFEST.json').write_text(json.dumps({'package':ROOT.name,'stage':'bisenet_diagnostic_not_formal_calibration','files':sorted(rows,key=lambda r:r['path'])},ensure_ascii=False,indent=2)+'\n')
 if '--sync-parent-water-only' in sys.argv:
  parent=ROOT.parent/'PACKAGE_MANIFEST.json'
  if not parent.is_file():raise ValueError('Parent submission manifest not found')
  previous=json.loads(parent.read_text());prefix=ROOT.name+'/'
  kept=[r for r in previous['files'] if not r['path'].startswith(prefix)]
  updated=[{**r,'path':prefix+r['path']} for r in rows]
  own=row(ROOT/'PACKAGE_MANIFEST.json');updated.append({**own,'path':prefix+own['path']})
  previous['files']=sorted(kept+updated,key=lambda r:r['path'])
  parent.write_text(json.dumps(previous,ensure_ascii=False,indent=2)+'\n')
 print('Static manifest refreshed:',len(rows),'files; runtime untouched')
if __name__=='__main__':main()
