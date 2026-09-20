import os
from celery import Celery
from celery.signals import worker_ready
from .settings import RUNTIME,preprocessing_mode

broker=os.environ.get("SHUIGUANG_BROKER_URL","redis://127.0.0.1:6379/0")
backend=os.environ.get("SHUIGUANG_RESULT_BACKEND","redis://127.0.0.1:6379/1")
if backend.startswith("file://"):
    from pathlib import Path
    Path(backend.removeprefix("file://")).mkdir(parents=True,exist_ok=True)
app=Celery("shuiguang_three",broker=broker,backend=backend)
app.conf.update(task_serializer="json",result_serializer="json",accept_content=["json"],
    task_default_queue="shuiguang_three",worker_prefetch_multiplier=1,
    task_track_started=True,result_expires=86400,broker_connection_retry_on_startup=True)
if broker=="filesystem://":
    queue=RUNTIME/"queue";queue.mkdir(parents=True,exist_ok=True)
    folders={name:queue/name for name in ("data","processed","control")}
    for p in folders.values():p.mkdir(exist_ok=True)
    app.conf.broker_transport_options=dict(data_folder_in=str(folders["data"]),data_folder_out=str(folders["data"]),
        processed_folder=str(folders["processed"]),control_folder=str(folders["control"]),store_processed=True)
@app.task(name="shuiguang.analyze_three_views",bind=True)
def analyze_three_views(self,request):
    if preprocessing_mode()!='legacy':
        from .service import failure
        return failure(request.get('task_id','unknown'),'LEGACY_TASK_DISABLED','新预处理worker不执行旧版推理任务，请使用显式诊断任务')
    from .service import execute
    def progress(stage,**details):
        self.update_state(state="PROGRESS",meta={"stage":stage,**details})
    return execute(request,progress)

@app.task(name="shuiguang.compose_three_views_v2",queue="shuiguang_three_v2")
def compose_three_views_v2(request):
    from .compose_v2 import compose
    from .public_scores import public_scores
    return public_scores(compose(request))

@app.task(name="shuiguang.analyze_three_views_scores",bind=True,queue="shuiguang_scores")
def analyze_three_views_scores(self,request):
    from .service_scores import execute_scores
    def progress(stage,**details):self.update_state(state="PROGRESS",meta={"stage":stage,**details})
    return execute_scores(request,progress)

@worker_ready.connect
def preload_three_item_models(sender=None,**kwargs):
    if os.environ.get("SHUIGUANG_PRELOAD","1").strip().lower() not in {"0","false","no"}:
        from .resident_scoring import preload
        preload()

@app.task(name='shuiguang.analyze_three_views_diagnostic',bind=True,queue='shuiguang_diagnostic')
def analyze_three_views_diagnostic(self,request):
    from .service_scores import execute_scores
    def progress(stage,**details):self.update_state(state='PROGRESS',meta={'stage':stage,**details})
    return execute_scores(request,progress,diagnostic=True)
