import fcntl
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from PIL import Image
from .contracts import AnalyzeRequest,Success,Failure
from .settings import RUNTIME,ROOT,bootstrap,input_root,device

class RequestError(ValueError):
    def __init__(self,code,message):
        self.code,self.message=code,message
        super().__init__(message)

def atomic_json(path,data):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    os.replace(temp,path)

@contextmanager
def locked(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RequestError("BUSY","任务正在执行，请稍后查询")
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

def failure(task_id,code,message):
    return Failure(task_id=task_id,error={"code":code,"message":message}).model_dump()

def safe_inputs(request):
    root=input_root()
    records=[]
    for image in request.images:
        relative=Path(image.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RequestError("INVALID_INPUT","图片必须使用输入目录内的相对路径")
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RequestError("INVALID_INPUT","图片不存在或超出输入目录")
        if path.stat().st_size>25*1024*1024:
            raise RequestError("INVALID_INPUT","单张图片超过25MB")
        with Image.open(path) as photo:
            if photo.format not in ("JPEG","PNG") or photo.mode not in ("RGB","RGBA") or photo.width*photo.height>32000000:
                raise RequestError("INVALID_INPUT","需要有效RGB JPEG或PNG原始照片")
            photo.verify()
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        records.append((image,path,sha))
    if len({r[2] for r in records})!=3:
        raise RequestError("DUPLICATE_IMAGES","三张照片不能重复")
    return records

def format_result(request,assignment,data,stats,scores):
    from shuiguang.config import REGIONS
    views={v:request.images[i].image_id for v,i in assignment.items()}
    selected={}
    meta={"pores":("毛孔","v2_visible_pores"),"spots":("可见色斑","v2_visible_spots_component"),
          "surface_gloss":("表面油光","v2_oiliness_tendency")}
    for module,(name,basis) in meta.items():
        account=stats["display_reconciliation"]["modules"][module]
        names=[r for r in account["visible_regions"] if not r.endswith("_eye")]
        rows=[]
        for region in names:
            primary=stats["regions"][region]["modules"][module]
            counts={v:stats["views"][v]["modules"][module]["regions"][region]["count"] for v in views}
            rows.append(dict(region=region,name=REGIONS[region],**counts,
                primary_view=primary["primary_view"],
                primary_count=primary["metrics"]["count"] if primary["metrics"] else None,
                supplementary_views=[a["view"] for a in primary["alternatives"] if a["has_findings"]]))
        totals={v:account["views"][v]["total_count"] for v in views}
        remaining={v:totals[v]-sum(row[v] or 0 for row in rows) for v in views}
        selected[module]=dict(name=name,**scores[module],score_view="front",score_basis=basis,
            total_count=totals,regions=rows,unassigned_count=remaining)
    return Success(task_id=request.task_id,main_image_id=views["front"],views=views,results=selected).model_dump()

def execute(payload,progress=None):
    progress=progress or (lambda *a,**k:None)
    request=AnalyzeRequest.model_validate(payload)
    job=RUNTIME/"jobs"/request.task_id
    try:
        records=safe_inputs(request)
        signature={"request":request.model_dump(),"images":[r[2] for r in records],
                   "vendor_manifest_sha256":hashlib.sha256((ROOT/"vendor_manifest.json").read_bytes()).hexdigest()}
        fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        with locked(RUNTIME/"locks"/(request.task_id+".lock")):
            receipt=job/"request.json"
            if receipt.exists():
                prior=json.loads(receipt.read_text())
                if prior["fingerprint"]!=fingerprint:
                    return failure(request.task_id,"TASK_ID_CONFLICT","任务ID已绑定其他输入或版本，请使用新ID")
                if (job/"result.json").exists():
                    return json.loads((job/"result.json").read_text())
                return failure(request.task_id,"INTERRUPTED","上次任务未正常结束，请使用新任务ID重试")
            with locked(RUNTIME/"locks"/"gpu.lock"):
                job.mkdir(parents=True)
                atomic_json(receipt,{"fingerprint":fingerprint,**signature})
                started=time.monotonic()
                try:
                    bootstrap()
                    from shuiguang.runtime import configure
                    configure(device())
                    from shuiguang.io import read_image,write_image
                    from shuiguang.orientation import inspect_photos
                    snapshot=job/"inputs";snapshot.mkdir()
                    paths=[]
                    for ordinal,(_,source,expected) in enumerate(records):
                        copy=snapshot/(str(ordinal)+source.suffix.lower())
                        shutil.copyfile(source,copy)
                        if hashlib.sha256(copy.read_bytes()).hexdigest()!=expected:
                            raise RequestError("INPUT_CHANGED","复制期间输入照片发生变化")
                        normalized=snapshot/(str(ordinal)+"_rgb.png")
                        write_image(normalized,read_image(copy,request.mirrored))
                        paths.append(normalized)
                    progress("orientation")
                    orientation=inspect_photos(paths)
                    atomic_json(job/"orientation.json",orientation)
                    if not orientation["confident"]:
                        raise RequestError("ORIENTATION_UNCERTAIN","未能明确识别一张正面及左右侧脸，请重新采集")
                    assignment=orientation["assignment"]
                    arranged={v:paths[assignment[v]] for v in ("left","front","right")}
                    from .analysis import analyze_views
                    data,stats=analyze_views(arranged,request.task_id,progress)
                    # Save only compact accounting evidence, no front-end pages or rendered overlays.
                    atomic_json(job/"counts.json",stats["display_reconciliation"])
                    progress("scoring",view="front")
                    from .scoring import score_front
                    scores=score_front(arranged["front"],job,device())
                    result=format_result(request,assignment,data,stats,scores)
                    atomic_json(job/"result.json",result)
                    atomic_json(job/"receipt.json",{"status":"success","seconds":time.monotonic()-started,
                        "orientation":orientation["orientation_version"],"counting":stats["version"],
                        "public_detectors":["pores","spots","surface_gloss"],"front_scoring":"stable_v2_full_pipeline"})
                    progress("complete")
                    return result
                except Exception as exc:
                    import traceback
                    (job/"error.log").write_text(traceback.format_exc(),encoding="utf-8")
                    code=exc.code if isinstance(exc,RequestError) else "ANALYSIS_FAILED"
                    message=exc.message if isinstance(exc,RequestError) else "检测或评分未完成，请核对输入并查看任务日志"
                    result=failure(request.task_id,code,message)
                    atomic_json(job/"result.json",result)
                    return result
    except RequestError as exc:return failure(request.task_id,exc.code,exc.message)
    except (OSError,ValueError) as exc:
        return failure(request.task_id,"INVALID_INPUT","输入文件无法读取或不是有效图片")
