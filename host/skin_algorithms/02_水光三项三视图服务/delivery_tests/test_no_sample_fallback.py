from copy import deepcopy
from unittest.mock import patch
from cloud_three.scoring_evidence import apply_legacy_fallback
from cloud_three.service_scores import VERSION

def test_no_sample_file_read_or_score_imputation(monkeypatch):
    monkeypatch.setenv('SHUIGUANG_LEGACY_SCORE_FALLBACK','1')
    values={'surface_gloss':{'score':54,'severity':'中度','regional_scores':{'items':[
      {'region':'left_cheek','score':None,'severity':None,'score_reason':'reference_missing'},
      {'region':'right_cheek','score':100,'severity':'未见明显','score_reason':None}]}}}
    before=deepcopy(values)
    with patch('pathlib.Path.read_text',side_effect=AssertionError('fixed sample must not be read')):
        assert apply_legacy_fallback(values)==[]
    assert values==before
    assert VERSION!='bisenet-independent-three-diagnostic-20260915-6'

def test_health_disables_fallback_even_with_legacy_flag(monkeypatch):
    monkeypatch.setenv('SHUIGUANG_LEGACY_SCORE_FALLBACK','1')
    from cloud_three.api import health
    assert health()['legacy_score_fallback'] is False
