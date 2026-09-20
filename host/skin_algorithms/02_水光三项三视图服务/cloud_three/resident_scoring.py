"""Process-resident three-item V3 scoring engine for a solo Celery worker."""
from __future__ import annotations
import atexit
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from .settings import ROOT,RUNTIME,VENDOR,preprocessing_mode

CODE=VENDOR/"v3"
REFERENCE=VENDOR/"v3_reference.json"
ALGORITHMS=("pores","spots","surface_gloss")
_LOCK=threading.Lock()
_SESSION=None

class ThreeItemResident:
    def __init__(self,device:str):
        if not device.startswith("cuda:"):
            raise RuntimeError("三项常驻评分当前要求CUDA设备，例如cuda:0")
        if not (CODE/"run.py").is_file() or not REFERENCE.is_file():
            raise RuntimeError("V3执行器或评分参考缺失")
        self.device=device
        self.preprocessing=preprocessing_mode()
        self.counter=0
        RUNTIME.mkdir(parents=True,exist_ok=True)
        reference=RUNTIME/"resident_assets"/"v3_reference.json"
        reference.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REFERENCE,reference)
        if hashlib.sha256(reference.read_bytes()).digest()!=hashlib.sha256(REFERENCE.read_bytes()).digest():
            raise RuntimeError("评分参考复制校验失败")
        os.environ["DERMAVISION_V3_WORKSPACE"]=str(RUNTIME)
        os.environ["DERMAVISION_V3_SCORE_REFERENCE"]=str(reference)
        os.environ["DERMAVISION_CLINICAL_SPEC"]="v3"
        os.environ["PYTHONDONTWRITEBYTECODE"]="1"
        if str(CODE) not in sys.path:sys.path.insert(0,str(CODE))
        import src
        if not Path(src.__file__).resolve().is_relative_to(CODE.resolve()):
            raise RuntimeError("V3源码导入路径不在包内")
        from src.doctor_v3.stage1_runtime import configure_runtime
        configure_runtime(RUNTIME/"resident_runtime"/str(os.getpid()))
        from src.capture_profile import CaptureProfile
        from src.nine_analysis import NineAnalysisOrchestrator
        self.orchestrator=NineAnalysisOrchestrator(
            RUNTIME/"resident_engine",cuda_device=device.split(":",1)[1],
            stage_input=True,algorithms=ALGORITHMS,capture_profile=CaptureProfile.CONSUMER)
        began=time.monotonic();self.cold_start=self.orchestrator.start()
        self.cold_start_seconds=time.monotonic()-began

    def close(self):
        self.orchestrator.close()

    def score(self,front:Path,job:Path):
        self.counter+=1
        sample=("score_"+hashlib.sha256((job.name+str(self.counter)).encode()).hexdigest()[:20])
        began=time.monotonic()
        result=self.orchestrator.analyze(front,repetition=self.counter,sample_id=sample,timeout_seconds=600)
        if result.get("状态")!="success" or int(result.get("成功项目数",0))!=len(ALGORITHMS):
            raise RuntimeError("三项检测未全部成功")
        source=self.orchestrator.run_root/f"repeat_{self.counter}"/sample
        from .scoring_evidence import score_saved
        scores,trace=score_saved(source,sample)
        (job/"scoring_evidence_trace.json").write_text(json.dumps(trace,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (job/"scores_internal.json").write_text(json.dumps(scores,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        timing={"mode":"resident_three_item","cold_start_seconds":self.cold_start_seconds,
          "preprocessing":self.preprocessing,
          "task_seconds":time.monotonic()-began,"resident_reuse_index":self.counter,
          "algorithms":list(ALGORITHMS),"services":sorted(self.orchestrator.clients)}
        (job/"resident_timing.json").write_text(json.dumps(timing,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        return scores

def _get(device):
    global _SESSION
    if _SESSION is None:_SESSION=ThreeItemResident(device)
    elif _SESSION.device!=device:raise RuntimeError("常驻评分进程不能在任务间切换GPU")
    elif _SESSION.preprocessing!=preprocessing_mode():raise RuntimeError('常驻进程不能在任务间切换预处理版本')
    return _SESSION

def preload(device_name=None):
    from .settings import device as configured_device
    with _LOCK:
        session=_get(device_name or configured_device())
        return {"cold_start_seconds":session.cold_start_seconds,"algorithms":list(ALGORITHMS),"services":sorted(session.orchestrator.clients)}

def score_front(front,job,device):
    with _LOCK:return _get(device).score(Path(front),Path(job))

def close():
    global _SESSION
    if _SESSION is not None:_SESSION.close();_SESSION=None
atexit.register(close)
