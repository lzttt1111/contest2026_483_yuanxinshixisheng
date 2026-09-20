"""Bind one completed detection's internal evidence to a portable V3 report."""
from pathlib import Path
import json
from .bundle import publish, sha256
from .measurements import read_evidence, measure_source, complete_measurements
from .report_model import build_report

MEDIA_MODULES = {
    "pores": ["01"], "surface_gloss":["02"], "porphyrin":["02"],
    "spots":["03"],"brown":["03"],"uv_spots":["03"],"redness":["04"],
    "vascular":["05"],"acne":["06"],"wrinkle":["07","08","09"],
    "texture":["10"],"contour_firmness":["11"]
}
CAPTIONS = {"pores":"可见毛孔检测图","surface_gloss":"表面油光检测图","porphyrin":"卟啉相关图像",
"spots":"普通色斑检测图","brown":"棕色色素图像","uv_spots":"UV相关色素图像",
"redness":"红区检测参考图（包含局灶红色表现）","vascular":"线状血管样结构检测图",
"acne":"痤疮样候选检测参考图","wrinkle":"二维纹路检测参考图（非三维深度测量）",
"texture":"二维表面纹理参考图（非三维平整度测量）","contour_firmness":"二维轮廓参考图（非三维组织位移测量）"}

def export_v3(result_root, source_root, runtime_items, *, subject_id, profile, captured_at=None, generate_pdf=False, registration=None, index_items=None, reference_path=None, source_media_root=None, input_hashes=None, stage1=False):
    result_root, source_root = Path(result_root), Path(source_root)
    if stage1:
        from .stage1_runtime import require_output
        require_output(result_root)
    media_root=Path(source_media_root) if source_media_root is not None else result_root
    from .calibration import load_reference
    calibration = load_reference(reference_path,profile)
    evidence = {}
    # Resolve only the current sample's known output folders, never the input corpus.
    for project, item in runtime_items.items():
        parent = Path(item["主结果图"]).parent
        candidates = [parent / ("doctor_v3_"+project+".npz"), parent.parent / ("doctor_v3_"+project+".npz")]
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates)>1:
            raise ValueError("ambiguous V3 evidence: "+project)
        if candidates:
            evidence[project] = candidates[0]
    for path in source_root.rglob("doctor_v3_*.npz"):
        project = path.stem.removeprefix("doctor_v3_")
        if project not in evidence:
            evidence[project] = path
    measured, assets, data, metadata = [], {}, {}, {}
    contour_item=runtime_items.get("contour_firmness",{})
    if contour_item.get("主结果图"):
        complete_geometry=Path(contour_item["主结果图"]).parent/"轮廓紧致度完整指标.json"
        if complete_geometry.is_file() and not complete_geometry.is_symlink():
            assets["evidence/contour_complete.json"]=complete_geometry
    if "evidence/contour_complete.json" not in assets:
        saved_geometry = source_root / "evidence/contour_complete.json"
        if saved_geometry.is_file() and not saved_geometry.is_symlink():
            assets["evidence/contour_complete.json"] = saved_geometry
    for project, path in sorted(evidence.items()):
        arrays, meta = read_evidence(path)
        data[project], metadata[project] = arrays, meta
        assets["evidence/"+path.name] = path
        assets["evidence/"+path.with_suffix(".json").name] = path.with_suffix(".json")
    from .phenotypes import classify_acne, measure_acne, measure_lines, diffuse, measure_diffuse, supported_spots
    original_data = dict(data)
    labels=[]
    if "acne" in data:
        data["acne"], labels = classify_acne(data["acne"],metadata["acne"],data.get("pores"))
        measured.extend(measure_acne(data["acne"],labels,profile))
    if "wrinkle" in data:
        measured.extend(measure_lines(data["wrinkle"],profile,
                         high_precision=calibration.get("imaging",{}).get("high_precision_texture") is True))
        from .wrinkle_metrics import measure as measure_stable
        measured.extend(measure_stable(data["wrinkle"],profile))
    registration = registration or {}
    cp_allowed = profile=="consumer" or registration.get("CP_M") is True
    uv_allowed = profile=="consumer" or registration.get("365_M") is True
    if "spots" in data:
        supported = supported_spots(data["spots"],data.get("brown") if cp_allowed else None,
                                    data.get("uv_spots") if uv_allowed else None)
        if supported is not None:
            data["spots"]=supported
            metadata["spots"]["multiview_verified"]=True
            if profile=="consumer":
                metadata["spots"]["source_kind"]="rgb_proxy"
    diffuse_arrays = diffuse(data["redness"],data.get("vascular"),data.get("acne"),
                              cross_channel_authorized=cp_allowed) if "redness" in data else None
    if diffuse_arrays is not None:
        measured.extend(measure_diffuse(diffuse_arrays,profile))
    for project,arrays in data.items():
        if project=="redness" and diffuse_arrays is not None:
            continue
        measured.extend(measure_source(project,arrays,metadata[project],profile,thresholds=calibration.get("thresholds")))
    measured = complete_measurements(measured,profile)
    stage_payload = None
    if stage1:
        from .stage1_pipeline import analyze,configured_reference
        from .stage1_config import measurement_config
        inputs = list(result_root.glob("00_输入图片.*")) + list((result_root/"00_输入图像").glob("*.jpg"))
        hashes = input_hashes if input_hashes is not None else {p.name:sha256(p) for p in inputs}
        stage_payload, derived, visual = analyze(original_data, metadata, subject_id=subject_id,
             profile=profile, input_sha256=hashes, registration=registration, thresholds=measurement_config(calibration),
             reference=configured_reference())
        measured = stage_payload["measurements"]
        receipt=result_root/"运行回执.json"
        if receipt.is_file():
            batch=json.loads(receipt.read_text(encoding="utf8")).get("batch",{})
            if batch.get("signature"):stage_payload["batch_signature"]=batch["signature"]
        diffuse_arrays = visual.get("diffuse_arrays", diffuse_arrays)
        labels = visual.get("acne_labels", labels)
        data["acne"] = visual.get("classified_acne", data.get("acne"))
        if data.get("acne") is None:data.pop("acne",None)
        if "wrinkle" in data:
            data["wrinkle"] = dict(data["wrinkle"], stage1_figures=derived)
        import io
        if derived:
            buffer=io.BytesIO();__import__("numpy").savez_compressed(buffer,**derived)
            derived_path=result_root/"_stage1_derived.npz";derived_path.write_bytes(buffer.getvalue())
            assets["evidence/stage1_derived.npz"]=derived_path
    if index_items is None:
        index = json.loads((result_root/"十二项检测结果索引.json").read_text(encoding="utf-8"))
        indexed = index.get("items") or index.get("十二项结果")
    else:
        indexed = index_items
    if not indexed or len(indexed)!=12:
        raise ValueError("V3 requires a complete twelve-item source index")
    media = []
    for project,item in indexed.items():
        relatives = [item["主结果图"],*item.get("附加结果图",[])]
        for number,relative in enumerate(relatives):
            source = (media_root/relative)
            if source.is_symlink() or not source.resolve().is_relative_to(media_root.resolve()):
                raise ValueError("escaped source media")
            target = "media/"+project+"_"+str(number)+source.suffix.lower()
            assets[target] = source
            media.append({"path":target,"modules":MEDIA_MODULES[project],"caption":CAPTIONS[project],"sha256":sha256(source)})
    from .figures import generate as generate_figures
    for path,entry in generate_figures(data,diffuse_arrays,labels,result_root/"_v3_figures"):
        assets[entry["path"]]=path
        media.append(entry)
    report = build_report(subject_id,profile,measured,media,captured_at=captured_at,
                          references=None if stage1 else calibration.get("references"))
    report["calibration_version"] = calibration.get("sha256","uncalibrated")
    report["reference_notice"] = calibration.get("report_notice")
    if reference_path is not None:
        assets["scoring/reference.json"] = Path(reference_path)
    report["evidence_projects"] = sorted(evidence)
    report["registration"] = registration
    if stage_payload is not None:report["stage1"] = stage_payload
    inputs = list(result_root.glob("00_输入图片.*")) + list((result_root/"00_输入图像").glob("*.jpg"))
    report["input_sha256"] = input_hashes if input_hashes is not None else {p.name:sha256(p) for p in inputs}
    from .phenotypes import PARAMETERS
    report["algorithm_parameters"] = PARAMETERS
    report["implementation_gaps"] = sorted({m["reason"] for m in measured if m.get("reason") in
        ("missing_v3_measurement_evidence","multi_image_support_not_yet_verified","requires_focal_and_vascular_exclusion")})
    destination = publish(result_root/"doctor_v3",report,assets)
    if stage_payload is not None:
        from .stage1_pipeline import save
        save(result_root/"v3_scoring",stage_payload)
        if generate_pdf:
            from .stage1_delivery import prepare_word
            from .word_report import generate_word
            target=prepare_word(result_root,destination,stage_payload,cloud=index_items is not None)
            generate_word(__import__("json").loads((target/"报告数据/report.json").read_text(encoding="utf8")),
                          target,target/"报告数据/templates")
    elif generate_pdf:
        from .pdf_report import generate
        generate(destination,result_root/"PDF报告")
    return destination
