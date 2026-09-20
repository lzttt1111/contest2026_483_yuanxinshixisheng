import copy
import pytest
from cloud_three.independent_spots import GROUPS,VERSION,measures,score

def metric():
 def row(name,area,count):return {'analysis_region':name,'auxiliary_metrics':{'analysis_scope':{'valid_skin_area_px':area,'evaluation_status':'可评估'},'count_and_density':{'punctate_spot_count':count,'patch_spot_count':0,'confluent_candidate_count':0},'scope_and_burden':{'total_feature_area_px':count*10}}}
 names=[name for names in GROUPS.values() for name in names]
 return {'parameters':{},'spot_count':len(names),'medical_metrics_v2':{'metrics_version':'medical_metrics_v2_20260728','overall_metrics':row('全面部',len(names)*3000,len(names)),'region_metrics':[row(name,3000,1) for name in names]}}

def reference():return {'version':VERSION,'detector_parameters':{},'minimum_valid_area':2000,'minimum_reference_count':100,'regions':{r:{'density':[10.0]*100,'area_ratio':[0.1]*100} for r in ('full_face',*GROUPS)}}

def test_conservation_and_positive_scores():
 m=metric();rows=measures(m);assert sum(rows[k]['count'] for k in GROUPS)==13
 out,_=score(m,reference());assert 0<=out['score']<100
 assert all(0<=x['score']<100 for x in out['regional_scores']['items'])

def test_single_missing_aggregate_exact_remainder():
 m=metric();expected=measures(m);m['medical_metrics_v2']['region_metrics'].pop(1)
 actual=measures(m)
 for key in ('valid_area','feature_area','count'):assert actual['forehead'][key]==expected['forehead'][key]
 assert actual['forehead']['reconstruction']=='unique_additive_remainder_from_full_face'
 result,_=score(m,reference());assert [r['region'] for r in result['regional_scores']['items']]==list(GROUPS)

def test_multiple_missing_regions_not_fabricated():
 m=metric();m['medical_metrics_v2']['region_metrics']=[r for r in m['medical_metrics_v2']['region_metrics'] if r['analysis_region'] not in ('眉间','鼻部')]
 out,trace=score(m,reference());assert out['score'] is None
 assert all(r['score'] is None for r in out['regional_scores']['items'])

def test_zero_is_not_missing():
 m=metric();m['spot_count']=0
 for row in [m['medical_metrics_v2']['overall_metrics'],*m['medical_metrics_v2']['region_metrics']]:
  row['auxiliary_metrics']['count_and_density']['punctate_spot_count']=0
  row['auxiliary_metrics']['scope_and_burden']['total_feature_area_px']=0
 out,_=score(m,reference());assert out['score']==100
 m['medical_metrics_v2']['overall_metrics']['auxiliary_metrics']['analysis_scope']['valid_skin_area_px']=None
 out,_=score(m,reference());assert out['score'] is None

def test_foreign_detector_parameters_rejected():
 m=metric();m['parameters']={'changed':True}
 with pytest.raises(ValueError,match='parameters'):score(m,reference())

def test_other_detector_cannot_change_score():
 m=metric();expected=score(m,reference());m['brown']={'mask':'anything'};m['uv_spots']={'count':999}
 assert score(m,reference())==expected
