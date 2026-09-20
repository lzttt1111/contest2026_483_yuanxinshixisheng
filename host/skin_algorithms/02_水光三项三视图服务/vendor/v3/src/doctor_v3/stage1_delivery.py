"""Shared V3 output seam; no models, fixed sample identities or frozen input fallbacks."""
import json
from pathlib import Path
from .bundle import sha256, write_json
from .stage1_pipeline import from_result, save
from .stage1_runtime import require_output


def export_scoring(source_root, destination, *, subject_id, profile, image,
                   runtime_items=None, thresholds=None, reference=None, registration=None,batch_signature=None):
    destination = require_output(destination)
    payload, _, _, _ = from_result(source_root, subject_id=subject_id, profile=profile,
                                  input_sha256={Path(image).name: sha256(image)},
                                  runtime_items=runtime_items, thresholds=thresholds,
                                  reference=reference, registration=registration)
    if batch_signature is not None:payload["batch_signature"]=batch_signature
    from .evaluation_scope import annotate_payload_scope
    annotate_payload_scope(payload)
    save(destination / "v3_scoring", payload)
    return payload


def default_catalog():
    root = Path(__import__("os").environ.get("DERMAVISION_V3_WORKSPACE", Path(__file__).resolve().parents[3])).resolve()
    path = root / "06_批跑与评分研究/merged_reference_20260909/catalog_r2.json.gz"
    if not path.is_file():
        raise ValueError("V3 historical reference catalog is missing")
    return path


def prepare_word(result_root, bundle_root, payload, *, cloud=False, catalog=None):
    from .evaluation_scope import annotate_payload_scope
    annotate_payload_scope(payload)
    from .word_delivery import prepare
    from .word_images import prepare_images
    from .stage1_word import attach_stage1
    from .word_package import finalize
    root = require_output(result_root)
    target = root / "检测结果" if cloud else root
    model = prepare(root, bundle_root, catalog or default_catalog(), target,
                    cloud=cloud, in_place=not cloud)
    prepare_images(target, bundle_root, root if cloud else None)
    report_path = target / "报告数据/report.json"
    model = attach_stage1(json.loads(report_path.read_text(encoding="utf8")), payload)
    write_json(report_path, model)
    write_json(target / "报告数据/stage1_quantitative.json", payload)
    complete_path = target / "十二项完整量化指标.json"
    complete = json.loads(complete_path.read_text(encoding="utf8"))
    complete["doctor_v3"].update(scores=model["modules"], score_trace=model["score_trace"],
                                 report_sha256=sha256(report_path), stage1=payload)
    write_json(complete_path, complete)
    finalize(target)
    return target
