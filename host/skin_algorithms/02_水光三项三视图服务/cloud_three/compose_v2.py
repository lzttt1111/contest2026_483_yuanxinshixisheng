"""Compose from persisted results; no image inference, ECDF fitting, or Word parsing."""
import os
from pathlib import Path
from PIL import Image,ImageOps
from .contracts import Success
from .contracts_v2 import ComposeRequest,ComposedResult,FailureV2
from .settings import ROOT,RUNTIME
from .v3_saved import read,sha,digest,within,extract,load_bundle,EvidenceError,VERSION
from .service import locked,atomic_json,RequestError

def root():
    return Path(os.environ.get("SHUIGUANG_RESULT_ROOT",str(ROOT))).resolve()

def failure(task_id,code,message):
    return FailureV2(task_id=task_id,error={"code":code,"message":message}).model_dump()

def counts_and_identity(folder):
    data=Success.model_validate(read(folder/"result.json")).model_dump()
    req=read(folder/"request.json")
    orientation=read(folder/"orientation.json")
    assignment=orientation["assignment"]
    if set(assignment)!=set(("left","front","right")) or set(assignment.values())!={0,1,2}:
        raise EvidenceError("INVALID_COUNTS_IDENTITY","数量结果缺少有效三视图来源")
    images=req["request"]["images"]
    if data["task_id"]!=req["request"]["task_id"]:
        raise EvidenceError("INVALID_COUNTS_IDENTITY","数量结果任务与输入回执不一致")
    if any(data["views"][v]!=images[i]["image_id"] for v,i in assignment.items()):
        raise EvidenceError("INVALID_COUNTS_IDENTITY","图片ID与自动视图映射不一致")
    index=assignment["front"]
    normalized=within(folder,"inputs/"+str(index)+"_rgb.png")
    normalized_sha=sha(normalized)
    score_evidence=read(folder/"score_evidence.json")
    if score_evidence.get("front_sha256")!=normalized_sha:
        raise EvidenceError("COUNTS_INPUT_CHANGED","正面快照与已完成任务证据哈希不一致")
    allowed={normalized_sha}
    proofs={"normalized_sha256":normalized_sha,"source_conversion_verified":False}
    originals=[p for p in (folder/"inputs").glob(str(index)+".*") if p.suffix.lower() in (".png",".jpg",".jpeg")]
    if len(originals)==1 and not req["request"].get("mirrored",False):
        original=originals[0].resolve()
        if not original.is_relative_to(folder.resolve()):raise EvidenceError("INVALID_REFERENCE","输入快照路径越界")
        source_sha=sha(original)
        if source_sha!=req["images"][index]:raise EvidenceError("COUNTS_INPUT_CHANGED","原始输入快照哈希不一致")
        with Image.open(original) as a,Image.open(normalized) as b:
            first=ImageOps.exif_transpose(a).convert("RGB")
            second=b.convert("RGB")
            if first.size==second.size and first.tobytes()==second.tobytes():
                allowed.add(source_sha)
                proofs.update(source_conversion_verified=True,original_sha256=source_sha)
    return data,allowed,proofs

def compose(request):
    task_id=request.get("task_id","invalid") if isinstance(request,dict) else "invalid"
    try:
        req=ComposeRequest.model_validate(request)
        folder=within(root(),req.counts_job_ref)
        counts,allowed,identity=counts_and_identity(folder)
        report=None;source=None
        if req.v3_result_ref:
            report,source=load_bundle(within(root(),req.v3_result_ref))
            if source["front_sha256"] not in allowed:
                raise EvidenceError("V3_INPUT_MISMATCH","V3评分不属于数量结果的正面照片，拒绝绑定")
        source_hashes={name:sha(folder/name) for name in ("result.json","request.json","orientation.json","score_evidence.json")}
        code_hashes={name:sha(Path(__file__).parent/name) for name in ("compose_v2.py","v3_saved.py","contracts_v2.py","v3_rules/scoring.py")}
        fingerprint=digest({"request":req.model_dump(),"counts":source_hashes,"identity":identity,
                            "v3":source,"adapter":VERSION,"code":code_hashes})
        destination=RUNTIME/"compose_v2"/req.task_id
        with locked(RUNTIME/"locks"/("compose_v2_"+req.task_id+".lock")):
            if (destination/"receipt.json").is_file():
                prior=read(destination/"receipt.json")
                if prior["fingerprint"]!=fingerprint:
                    return failure(req.task_id,"TASK_ID_CONFLICT","任务ID已绑定其他结果或评分版本")
                return ComposedResult.model_validate(read(destination/"result.json")).model_dump()
            scores=extract(report, None if report else "missing_same_image_v3_result")
            for key,value in scores.items():
                value.update({field:counts["results"][key][field] for field in ("total_count","regions","unassigned_count")})
            result=ComposedResult(task_id=req.task_id,main_image_id=counts["main_image_id"],
                                  views=counts["views"],results=scores).model_dump()
            destination.mkdir(parents=True,exist_ok=True)
            atomic_json(destination/"result.json",result)
            atomic_json(destination/"receipt.json",{"fingerprint":fingerprint,"adapter":VERSION,
                "counts_hashes":source_hashes,"image_binding":identity,"v3_source":source,
                "detector_inference_executed":False,"scoring_recalculated":False})
            return result
    except EvidenceError as exc:return failure(task_id,exc.code,exc.message)
    except RequestError as exc:return failure(task_id,exc.code,exc.message)
    except (ValueError,OSError,KeyError,TypeError):
        return failure(task_id,"INVALID_SAVED_RESULT","已有结果格式、身份或文件引用无效")
