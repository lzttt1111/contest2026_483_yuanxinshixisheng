from types import SimpleNamespace
import pytest
from cloud_three.service_scores import execute_scores

def request():
 return {'task_id':'gate_test','images':[{'image_id':str(i),'path':f'{i}.jpg'} for i in range(3)],'mirrored':False}

def test_formal_request_validates_inputs_with_fallback_disabled(monkeypatch):
 monkeypatch.setenv('SHUIGUANG_PREPROCESS_MODE','bisenet_rgb_diagnostic_v1')
 monkeypatch.setenv('SHUIGUANG_LEGACY_SCORE_FALLBACK','0')
 result=execute_scores(request())
 assert result['error']['code']=='INVALID_INPUT'

def test_unconfigured_worker_defaults_to_bisenet_not_legacy(monkeypatch):
 monkeypatch.delenv('SHUIGUANG_PREPROCESS_MODE',raising=False)
 monkeypatch.delenv('SHUIGUANG_LEGACY_SCORE_FALLBACK',raising=False)
 from cloud_three.settings import preprocessing_mode
 assert preprocessing_mode()=='bisenet_rgb_diagnostic_v1'
 assert execute_scores(request())['error']['code']=='INVALID_INPUT'

def test_diagnostic_requires_selected_mode(monkeypatch):
 monkeypatch.setenv('SHUIGUANG_PREPROCESS_MODE','legacy')
 assert execute_scores(request(),diagnostic=True)['error']['code']=='DIAGNOSTIC_MODE_REQUIRED'

def test_legacy_task_cannot_run_in_new_worker(monkeypatch):
 monkeypatch.setenv('SHUIGUANG_PREPROCESS_MODE','bisenet_rgb_diagnostic_v1')
 from cloud_three.tasks import analyze_three_views
 assert analyze_three_views.run(request())['error']['code']=='LEGACY_TASK_DISABLED'

def test_resident_does_not_silently_switch_preprocessing(monkeypatch):
 from cloud_three import resident_scoring
 monkeypatch.setattr(resident_scoring,'_SESSION',SimpleNamespace(device='cuda:0',preprocessing='legacy'))
 monkeypatch.setenv('SHUIGUANG_PREPROCESS_MODE','bisenet_rgb_diagnostic_v1')
 with pytest.raises(RuntimeError,match='切换预处理'):resident_scoring._get('cuda:0')
