"""End-to-end: three unordered images -> front selection -> V3 score-only JSON."""
import hashlib
import json
import shutil
import time
import os
from pathlib import Path
from .contracts import AnalyzeRequest
from .public_scores import public_from_v3_extract
from .service import RequestError,safe_inputs,locked,atomic_json,failure
from .settings import ROOT,RUNTIME,bootstrap,device,preprocessing_mode

VERSION="current-image-zero-gloss-coverage-20260920-3"
def execute_scores(payload,progress=None,*,diagnostic=False):
    progress=progress or (lambda *a,**k:None)
    request=AnalyzeRequest.model_validate(payload)
    mode=preprocessing_mode()
    # Old reference parameters may score current measurements; fixed sample
    # result values never participate. Missing evidence remains unavailable.
    fallback_enabled=False
    if diagnostic and mode!='bisenet_rgb_diagnostic_v1':
        return failure(request.task_id,'DIAGNOSTIC_MODE_REQUIRED','诊断worker必须明确使用BiSeNet预处理')
    model_sha=None
    if mode!='legacy':
        model=Path(os.environ.get('SHUIGUANG_FACE_PARSER_MODEL',str(ROOT/'vendor/v3/models/face_parsing_resnet18.onnx'))).resolve()
        if not model.is_relative_to(ROOT.resolve()) or not model.is_file():
            return failure(request.task_id,'MODEL_ASSET_INVALID','模型必须存在于当前独立沙箱内')
        model_sha=hashlib.sha256(model.read_bytes()).hexdigest()
    purpose='diagnostic' if diagnostic else 'formal'
    job=RUNTIME/('diagnostic_jobs' if diagnostic else 'score_jobs')/request.task_id
    try:
        records=safe_inputs(request)
        identity={"request":request.model_dump(),"images":[x[2] for x in records],
          "version":VERSION,"purpose":purpose,"preprocessing":mode,"model_sha256":model_sha,"legacy_score_fallback":fallback_enabled,
          "v3_reference_sha256":hashlib.sha256((ROOT/"vendor/v3_reference.json").read_bytes()).hexdigest(),
          "independent_spots_reference_sha256":hashlib.sha256((ROOT/"vendor/independent_spots_reference.json").read_bytes()).hexdigest()}
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        with locked(RUNTIME/"locks"/(purpose+"_score_"+request.task_id+".lock")):
            if (job/"request.json").is_file():
                prior=json.loads((job/"request.json").read_text())
                if prior["fingerprint"]!=fingerprint:
                    return failure(request.task_id,"TASK_ID_CONFLICT","任务ID已绑定其他输入或评分版本")
                if (job/"result.json").is_file():return json.loads((job/"result.json").read_text())
                return failure(request.task_id,"INTERRUPTED","上次任务未正常结束，请使用新任务ID")
            with locked(RUNTIME/"locks"/"gpu.lock"):
                job.mkdir(parents=True)
                atomic_json(job/"request.json",{"fingerprint":fingerprint,**identity})
                began=time.monotonic()
                try:
                    bootstrap()
                    from shuiguang.runtime import configure
                    from shuiguang.io import read_image,write_image
                    from shuiguang.orientation import inspect_photos
                    configure(device())
                    images=job/"inputs";images.mkdir()
                    paths=[]
                    for i,(_,source,sha) in enumerate(records):
                        raw=images/(str(i)+source.suffix.lower());shutil.copyfile(source,raw)
                        if hashlib.sha256(raw.read_bytes()).hexdigest()!=sha:raise RequestError("INPUT_CHANGED","输入复制期间发生变化")
                        normalized=images/(str(i)+"_rgb.png");write_image(normalized,read_image(raw,request.mirrored));paths.append(normalized)
                    progress("orientation")
                    orientation=inspect_photos(paths)
                    atomic_json(job/"orientation.json",orientation)
                    if not orientation["confident"]:raise RequestError("ORIENTATION_UNCERTAIN","未能明确识别一张正面及左右侧脸")
                    front=paths[orientation["assignment"]["front"]]
                    progress("v3_scoring",view="front")
                    from .live_scoring import score_front
                    result=public_from_v3_extract(score_front(front,job,device()))
                    if diagnostic:
                        result={'schema_version':'shuiguang_diagnostic_v1','purpose':'diagnostic_old_reference_not_calibrated','preprocessing':mode,'reference_matches_new_preprocessing':False,'scores':result}
                    atomic_json(job/"result.json",result)
                    atomic_json(job/"receipt.json",{"status":"success","version":VERSION,
                      "seconds":time.monotonic()-began,"front_image_id":request.images[orientation["assignment"]["front"]].image_id,
                      "orientation_version":orientation["orientation_version"],"output":"score_only_json",
                      "word_generated":False,"detector_models_trained":False})
                    progress("complete")
                    return result
                except Exception as exc:
                    import traceback
                    (job/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
                    code=exc.code if isinstance(exc,RequestError) else "ANALYSIS_FAILED"
                    message=exc.message if isinstance(exc,RequestError) else "视角识别或V3评分未完成，请查看任务日志"
                    result=failure(request.task_id,code,message);atomic_json(job/"result.json",result);return result
    except RequestError as exc:return failure(request.task_id,exc.code,exc.message)
