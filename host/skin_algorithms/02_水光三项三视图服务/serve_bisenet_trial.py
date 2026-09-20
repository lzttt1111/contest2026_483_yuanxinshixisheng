"""Local read-only BiSeNet deployment trial; never publishes uncalibrated scores."""
from pathlib import Path
import json
import cv2,numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,Response
ROOT=Path(__file__).resolve().parent;TRIAL=ROOT/'runtime/bisenet_deployment'
app=FastAPI(title='BiSeNet水光三项本机试运行')
def record(sample,view):
 if sample not in ('one','two') or view not in ('left','front','right'):raise HTTPException(404)
 data=json.loads((TRIAL/'detection_results.json').read_text())
 return next(r for r in data['results'] if r['sample']==sample and r['view']==view)
def checked(value):
 path=Path(value).resolve()
 if not path.is_relative_to(TRIAL.resolve()) or not path.is_file():raise HTTPException(404)
 return path
def display_count(module,metrics):
 raw=metrics.get({'pores':'总计','spots':'spot_count','surface_gloss':'油光区域数量'}[module])
 if isinstance(raw,(int,float)):return raw
 if isinstance(raw,dict):
  # 油光的量化 JSON 按区域保存数量；网页只显示总量，不能把对象
  # 直接交给模板，否则会出现 [object Object]。
  for key in ('总计','total','count','数量'):
   if isinstance(raw.get(key),(int,float)):return raw[key]
  vals=[v for v in raw.values() if isinstance(v,(int,float))]
  return sum(vals) if vals else 0
 return raw
@app.get('/')
def index():return FileResponse(ROOT/'trial_web/index.html')
@app.get('/comparison.js')
def comparison_script():return FileResponse(ROOT/'trial_web/comparison.js',media_type='text/javascript')
@app.get('/api/state')
def state(sample:str='one',module:str='pores'):
 if module not in ('pores','spots','surface_gloss'):raise HTTPException(404)
 rows=[]
 for view in ('left','front','right'):
  r=record(sample,view);item=r['result']['十二项结果'][module];metrics=json.loads(checked(item['量化JSON']).read_text())
  count=display_count(module,metrics)
  rows.append({'view':view,'count':count,'seconds':r['seconds'],'input_sha256':r['input_sha256'],'status':item['状态']})
 return {'sample':sample,'module':module,'preprocessing':'BiSeNet ResNet18 + MediaPipe眉毛30px','algorithms':['pores','spots','surface_gloss'],'scoring_status':'comparison_pending_not_calibrated','views':rows}
@app.get('/api/comparison')
def comparison(sample:str='one',module:str='pores'):
 record(sample,'front')
 if module not in ('pores','spots','surface_gloss'):raise HTTPException(404)
 path=checked(TRIAL/'score_comparison_v2'/(sample+'.json'));data=json.loads(path.read_text())
 return {'purpose':data['purpose'],'reference_matches_new_preprocessing':False,
  'before':data['before']['scores'][module],'after':data['after']['scores'][module],
  'measurements':{side:([r for r in data[side]['measurements'] if r['region']=='full_face' and r['module']==('01' if module=='pores' else '02')] if module!='spots' else data[side]['spots_measurements'].get('full_face',{})) for side in ('before','after')}}
@app.get('/media/{sample}/{view}/{kind}/{module}')
def media(sample:str,view:str,kind:str,module:str):
 if module not in ('pores','spots','surface_gloss'):raise HTTPException(404)
 r=record(sample,view);item=r['result']['十二项结果'][module]
 if kind=='detection':return FileResponse(checked(item['主结果图']))
 if kind=='old':return FileResponse(checked(TRIAL/'preprocess'/sample/view/'legacy_Final_skin.png'))
 if kind=='new':
  metrics=checked(item['量化JSON']);npz=checked(metrics.parents[2]/'preprocessed/doctor_v3_rgb.npz')
  with np.load(npz) as a:
   image=a['image'].copy();valid=a['valid']>0;image[valid]=(image[valid]*.5+np.array((0,255,0))*.5).astype(np.uint8)
  ok,encoded=cv2.imencode('.png',image)
  if not ok:raise HTTPException(500)
  return Response(encoded.tobytes(),media_type='image/png')
 raise HTTPException(404)
if __name__=='__main__':
 import uvicorn
 uvicorn.run(app,host='127.0.0.1',port=8896)
