from cloud_three.independent_gloss import score_gloss,VERSION

def fixture():
    scores={'surface_gloss':{'score':None,'severity':None,'regional_scores':{'items':[]}}}
    payload={'measurements':[],'basis':{},'scores':{}}
    for metric,value,raw in [('gloss_area',.1,60),('gloss_high_area',.02,30)]:
        key='02.'+metric+':full_face'
        payload['measurements'].append({'metric_id':'02.'+metric,'region':'full_face','status':'measured','value':value})
        payload['basis'][key]={'quality_status':'PASS','quality_reason':None,'effective_area_px':5000}
        payload['scores'][key]={'score_status':'candidate','reference_status':'compatible','score_raw':raw,'trace':{'reference_identity_validation':{'status':'matched'},'formal_report_eligible':True,'reference_version':'fixture','reference_source':'fixture'}}
    return scores,payload

def test_weights_same_region_no_mean_required():
    s,p=fixture();t=score_gloss(s,p)
    assert s['surface_gloss']['score']==47
    assert abs(t['regions'][0]['score_raw']-(60*25+30*20)/45)<1e-9
    assert s['surface_gloss']['scoring_version']==VERSION

def test_missing_or_mismatched_reference_not_imputed():
    for kind in ('missing','mismatch','quality'):
        s,p=fixture();key='02.gloss_high_area:full_face'
        if kind=='missing':p['scores'].pop(key)
        if kind=='mismatch':p['scores'][key]['trace']['reference_identity_validation']['status']='mismatched'
        if kind=='quality':p['basis'][key]['quality_status']='FAIL'
        score_gloss(s,p);assert s['surface_gloss']['score'] is None
