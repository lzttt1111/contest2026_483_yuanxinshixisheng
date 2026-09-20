"""Patient-facing missing-score reasons backed by recorded evidence contracts."""
def explain(mid,key,region,item,trace):
 why=trace.get("reference_contract_error") or trace.get("reason") or item.get("reason","")
 if why in ("insufficient_region_for_scoring","insufficient_valid_region_or_rejected_quality"):return "有效区域不足"
 if why=="rejected_input_quality":return "本次图像质量不足"
 if mid in ("09","10","11") and region!="full_face" and item.get("value") is None:return "缺少该分区的2.5D指标"
 if mid=="08" and key=="depth":return "缺少该分区的凹陷估计及参考"
 if mid=="08" and key in ("main_count","length_burden"):return "历史仅有纹段统计，无此项同定义参考"
 if "spectral" in why:return "缺少同光源参考"
 if "domain_p90" in why or "pixel_p90" in why:return "缺少同统计范围的强度参考"
 if "roi" in why or "region_transfer" in why:return "缺少该分区的同定义参考"
 if "diffuse_background" in why:return "缺少弥漫背景的历史参考"
 if "multi_supported" in why:return "缺少多图支持色斑的历史参考"
 if "grid_cluster" in why:return "缺少受累网格的历史参考"
 if "value" not in item:return "缺少完整的评分参考"
 if item.get("value") is None:return "缺少该项量化依据"
 return "缺少该指标的同定义参考"
def annotate(model):
 for mid,module in model["modules"].items():
  for region,metrics in module["metrics"].items():
   for key,item in metrics.items():
    if item.get("score") is None:
     trace=model["score_trace"].get(mid+"."+key+":"+region,{})
     item["no_score_reason"]=explain(mid,key,region,item,trace)
   for key,item in metrics.items():
    if item.get("score") is not None or "value" in item:continue
    names=item.get("missing",[])
    child_keys=[key+"."+n if mid=="03" and key in ("spots","brown","uv") else n for n in names]
    reasons=[metrics[n].get("no_score_reason") for n in child_keys if n in metrics]
    reasons=[r for r in dict.fromkeys(reasons) if r]
    if reasons:item["no_score_reason"]="；".join(reasons[:2])
   target=module if region=="full_face" else module["regions"].get(region,{})
   if target.get("score") is None:
    reasons=[metrics[k].get("no_score_reason") for k in target.get("missing",[]) if k in metrics]
    reasons=[r for r in dict.fromkeys(reasons) if r]
    target["no_score_reason"]="；".join(reasons[:2]) if reasons else explain(mid,"_module",region,target,{})
 return model
