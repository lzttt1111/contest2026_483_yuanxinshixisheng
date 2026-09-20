import sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/v3'))
from cloud_three.observed_zero import apply_observed_zero

def fixture():
    scores={k:{'score':50,'regional_scores':{'items':[{'region':'left_cheek','score':None,'severity':None}]}} for k in ('pores','surface_gloss')}
    rows=[];basis={}
    for name,value in [('gloss_area',0),('gloss_high_area',0),('gloss_mean',None)]:
        key='02.'+name
        reason='missing_gloss_mask' if value is not None else 'no_valid_targets'
        state='measured' if value is not None else 'no_targets'
        rows.append({'metric_id':key,'region':'left_cheek','value':value,'status':'measured' if value is not None else 'unavailable','measurement_status':state,'reason':reason})
        basis[key+':left_cheek']={'effective_area_px':2000,'quality_status':'PASS','quality_reason':None,'reason':None if value is not None else reason,'measurement_status':state,'pixel_score_summary':{'count':0}}
    return scores,{'measurements':rows,'basis':basis}

def test_proven_zero_scores_100_without_imputing_mean():
    scores,payload=fixture();original=deepcopy(payload)
    result=apply_observed_zero(scores,payload)
    r=scores['surface_gloss']['regional_scores']['items'][0]
    assert (r['score'],r['severity'])==(100,'未见明显')
    assert len(result['applied'])==1 and payload==original
    assert scores['pores']['regional_scores']['items'][0]['score'] is None

def test_nonzero_or_missing_or_bad_quality_never_gets_100():
    for change in ('nonzero','quality','small','missing','unknown_pixels'):
        scores,payload=fixture()
        if change=='nonzero':payload['measurements'][0]['value']=.1
        elif change=='quality':payload['basis']['02.gloss_area:left_cheek']['quality_status']='FAIL'
        elif change=='small':payload['basis']['02.gloss_area:left_cheek']['effective_area_px']=0
        elif change=='missing':payload['measurements'].pop()
        else:payload['basis']['02.gloss_mean:left_cheek']['pixel_score_summary']={}
        assert apply_observed_zero(scores,payload)['applied']==[]
        assert scores['surface_gloss']['regional_scores']['items'][0]['score'] is None

def test_nonblocking_warning_is_preserved_in_zero_proof():
    scores,payload=fixture()
    for basis in payload['basis'].values():basis['quality_status']='WARNING'
    original=deepcopy(payload)
    result=apply_observed_zero(scores,payload)
    assert len(result['applied'])==1
    assert result['applied'][0]['proof']['quality_statuses']==['WARNING']
    assert payload==original
    scores,payload=fixture()
    for basis in payload['basis'].values():
        basis['quality_status']='WARNING';basis['quality_reason']='blurred'
    assert apply_observed_zero(scores,payload)['applied']==[]

def test_pores_zero_count_requires_complete_geometry_and_domain():
    scores,payload=fixture();payload={'measurements':[],'basis':{}}
    for name,value in [('density',0),('area_p50',None),('large_density',0)]:
        key='01.'+name;state='measured' if value is not None else 'no_targets'
        reason=None if value is not None else 'no_valid_targets'
        payload['measurements'].append({'metric_id':key,'region':'left_cheek','value':value,'status':'measured' if value is not None else 'unavailable','measurement_status':state,'reason':reason})
        payload['basis'][key+':left_cheek']={'quality_status':'WARNING','quality_reason':None,'effective_area_px':2000,'reason':reason,'count':0,'unlocated_instance_count':0,'instance_areas_px2':[]}
    original=deepcopy(payload)
    assert apply_observed_zero(scores,payload)['applied'][0]['project']=='pores'
    assert scores['pores']['regional_scores']['items'][0]['score']==100
    scores,_=fixture();original['basis']['01.density:left_cheek']['unlocated_instance_count']=1
    assert apply_observed_zero(scores,original)['applied']==[]
