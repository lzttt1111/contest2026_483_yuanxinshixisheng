"""Optional headless HTTP gateway. Input paths are backend-managed mounted files."""
import os
import uuid
from fastapi import FastAPI,HTTPException
from .contracts import AnalyzeRequest
from .tasks import app as celery_app,analyze_three_views
from .tasks import compose_three_views_v2,analyze_three_views_scores
from .tasks import analyze_three_views_diagnostic
from .contracts_v2 import ComposeRequest
from .settings import preprocessing_mode

app=FastAPI(title="水光三项JSON服务",version="1.1")
def legacy_score_fallback_enabled():
    return False
@app.post('/api/diagnostic-jobs',status_code=202)
def diagnostic_submit(request:AnalyzeRequest):
    queued=analyze_three_views_diagnostic.delay(request.model_dump())
    return {'task_id':request.task_id,'queue_id':queued.id,'status':'queued','purpose':'diagnostic_not_formal_scoring'}

@app.get('/api/diagnostic-jobs/{queue_id}')
def diagnostic_status(queue_id:str):return status(queue_id)
@app.get("/health")
def health():
    return {"status":"ok","service":"shuiguang_three",
            "algorithms":["pores","spots","surface_gloss"],
            "legacy_score_fallback":legacy_score_fallback_enabled()}

@app.post("/api/score-jobs",status_code=202)
def score_submit(request:AnalyzeRequest):
    queued=analyze_three_views_scores.delay(request.model_dump())
    return {"task_id":request.task_id,"queue_id":queued.id,"status":"queued",
            "score_source":"current_image_evidence_old_reference"}

@app.get("/api/score-jobs/{queue_id}")
def score_status(queue_id:str):
    return status(queue_id)

@app.post("/api/v2/jobs",status_code=202)
def compose_submit(request:ComposeRequest):
    queued=compose_three_views_v2.delay(request.model_dump())
    return {"task_id":request.task_id,"queue_id":queued.id,"status":"queued"}

@app.get("/api/v2/jobs/{queue_id}")
def compose_status(queue_id:str):
    result=status(queue_id)
    if result.get("schema_version") not in (None,"shuiguang_cloud_v2"):
        raise HTTPException(409,"此任务不属于JSON v2")
    if result.get("schema_version")=="shuiguang_cloud_v2" and result.get("status")=="success":
        from .public_scores import public_scores
        return public_scores(result)
    return result
@app.post("/api/jobs",status_code=202)
def submit(request:AnalyzeRequest):
    if preprocessing_mode()!='legacy':
        raise HTTPException(409,{'code':'LEGACY_TASK_DISABLED','message':'新预处理不允许提交旧推理任务'})
    queued=analyze_three_views.delay(request.model_dump())
    return {"task_id":request.task_id,"queue_id":queued.id,"status":"queued"}
@app.get("/api/jobs/{queue_id}")
def status(queue_id:str):
    try:uuid.UUID(queue_id)
    except ValueError:raise HTTPException(400,"无效queue_id")
    result=celery_app.AsyncResult(queue_id)
    if result.successful():return result.result
    if result.failed():return {"status":"failed","error":{"code":"WORKER_FAILED","message":"任务执行异常"}}
    return {"queue_id":queue_id,"status":result.state.lower(),
        "progress":result.info if isinstance(result.info,dict) else None}
