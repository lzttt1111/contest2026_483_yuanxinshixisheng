"""Independent RGB spot burden. No other detector is accepted as input.

Six disjoint displayed regions aggregate the detector's 13 native partitions.
Area and counts are additive; percentiles of different regions are never averaged.
"""
import bisect
import hashlib
import json
import math
from pathlib import Path

VERSION='independent-visible-spots-area-density-v1'
GROUPS={
 'forehead':('额头','眉间'),
 'nose':('鼻部',),
 'left_cheek':('画面左鼻旁','画面左颧部','画面左面颊'),
 'right_cheek':('画面右鼻旁','画面右颧部','画面右面颊'),
 'perioral':('口周',),
 'chin':('画面左下颌','画面右下颌','下巴'),
}
LABELS={'full_face':'全面部','forehead':'额部（含眉间）','nose':'鼻部','left_cheek':'画面左面颊（含颧部、鼻旁）','right_cheek':'画面右面颊（含颧部、鼻旁）','perioral':'口周','chin':'下巴及下颌'}
WEIGHTS={'area_ratio':0.55,'density':0.45}

def number(value):
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
  raise ValueError('missing or invalid independent spot evidence')
 return float(value)

def raw_row(row):
 aux=row['auxiliary_metrics'];scope=aux['analysis_scope'];counts=aux['count_and_density']
 count=sum(number(counts[k]) for k in ('punctate_spot_count','patch_spot_count','confluent_candidate_count'))
 if not count.is_integer():raise ValueError('noninteger count')
 area=number(scope['valid_skin_area_px']);feature=number(aux['scope_and_burden']['total_feature_area_px'])
 if (count==0)!=(feature==0):raise ValueError('count/area evidence contradiction')
 return {'valid_area':area,'feature_area':feature,'count':int(count),'quality':scope['evaluation_status']}

def measures(metrics):
 med=metrics['medical_metrics_v2']
 if med['metrics_version']!='medical_metrics_v2_20260728':raise ValueError('measurement definition mismatch')
 native={row['analysis_region']:raw_row(row) for row in med['region_metrics']}
 result={'full_face':raw_row(med['overall_metrics'])}
 expected=set(n for names in GROUPS.values() for n in names)
 if set(native)-expected:raise ValueError('region schema mismatch')
 incomplete=[region for region,names in GROUPS.items() if set(names)-set(native)]
 if len(incomplete)>1:raise ValueError('multiple incomplete aggregate regions')
 for region,names in GROUPS.items():
  if region in incomplete:continue
  rows=[native[n] for n in names]
  result[region]={key:sum(r[key] for r in rows) for key in ('valid_area','feature_area','count')}
  # Regions with zero valid pixels do not contribute; all observed parts must be assessable.
  result[region]['quality']='可评估' if all(r['quality']=='可评估' for r in rows if r['valid_area']>0) else '不可评估'
 if incomplete:
  # The source omits native areas below 100 px. With exactly one affected
  # aggregate its additive totals are uniquely determined by conservation.
  # Never infer percentile/intensity or distribute a remainder over two areas.
  region=incomplete[0]
  residual={key:result['full_face'][key]-sum(result[r][key] for r in GROUPS if r!=region) for key in ('valid_area','feature_area','count')}
  if any(value<0 for value in residual.values()):raise ValueError('invalid regional conservation remainder')
  observed=[native[n] for n in GROUPS[region] if n in native]
  if any(residual[key]+0.1<sum(row[key] for row in observed) for key in residual):raise ValueError('remainder contradicts observed subregions')
  residual['quality']=result['full_face']['quality']
  residual['reconstruction']='unique_additive_remainder_from_full_face'
  result[region]=residual
 for key in ('valid_area','feature_area','count'):
  total=sum(result[r][key] for r in GROUPS)
  if abs(total-result['full_face'][key])>max(0.1,abs(total)*1e-5):raise ValueError('region/full-face reconciliation failed: '+key)
 if result['full_face']['count']!=metrics['spot_count']:raise ValueError('public count mismatch')
 for row in result.values():
  row['area_ratio']=row['feature_area']/row['valid_area'] if row['valid_area'] else None
  row['density']=row['count']*100000/row['valid_area'] if row['valid_area'] else None
 return result

def grade(score):
 return '未见明显' if score>=81 else '轻度' if score>=61 else '中度' if score>=41 else '较明显' if score>=21 else '显著'

def score(metrics,reference):
 if reference['version']!=VERSION:raise ValueError('independent reference version mismatch')
 if metrics['parameters']!=reference['detector_parameters']:raise ValueError('detector parameters mismatch')
 try:rows=measures(metrics)
 except (ValueError,KeyError,TypeError) as error:
  reason='independent_spots_evidence_incomplete'
  base={'score':None,'severity':None,'score_status':'unavailable','score_reason':reason}
  return {**base,'name':'可见色斑','score_view':'front','score_direction':'higher_is_better','scoring_version':VERSION,
   'regional_scores':{'score_view':'front','score_direction':'higher_is_better','region_schema':'independent_spots_six_regions_v1','region_side_convention':'image_left_right','scoring_version':VERSION,
   'items':[{**base,'region':r,'name':LABELS[r]} for r in GROUPS]}},{'error':str(error)}
 output=[];trace={}
 for region in ('full_face',*GROUPS):
  row=rows[region]
  node={'region':region,'name':LABELS[region],'score':None,'severity':None,'score_status':'unavailable','score_reason':None}
  ref=reference['regions'].get(region)
  if row['quality']!='可评估' or row['valid_area']<reference['minimum_valid_area']:
   node['score_reason']='insufficient_valid_area_or_quality'
  elif not ref or min(len(ref[k]) for k in WEIGHTS)<reference['minimum_reference_count']:
   node['score_reason']='insufficient_independent_reference'
  elif row['count']==0:
   node.update(score=100,severity='未见明显',score_status='available',score_reason=None)
  else:
   components={}
   for key in WEIGHTS:
    distribution=ref[key];value=row[key]
    # Empirical rank including the observed value; positive burden cannot score 100.
    rank=(bisect.bisect_left(distribution,value)+bisect.bisect_right(distribution,value)+1)/2
    components[key]=100*(1-rank/(len(distribution)+1))
   raw=sum(WEIGHTS[k]*components[k] for k in WEIGHTS)
   displayed=min(99,int(math.floor(raw+0.5)))
   node.update(score=displayed,severity=grade(displayed),score_status='available',score_reason=None)
   row={**row,'component_scores':components,'raw_score':raw}
  trace[region]=row;output.append(node)
 full=output[0]
 return {**{k:full[k] for k in ('score','severity','score_status','score_reason')},'name':'可见色斑','score_view':'front','score_direction':'higher_is_better','scoring_version':VERSION,'score_basis':VERSION,'regional_scores':{'score_view':'front','score_direction':'higher_is_better','region_schema':'independent_spots_six_regions_v1','region_side_convention':'image_left_right','scoring_version':VERSION,'items':output[1:]}},trace
