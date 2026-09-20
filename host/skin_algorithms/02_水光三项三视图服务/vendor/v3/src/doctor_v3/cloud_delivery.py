"""V3 reports from actual simulated Worker outputs, with no second inference."""
from pathlib import Path
from .bundle import contained,write_json
from .delivery import export_v3

def export_cloud(root, bundle, subject_id, *, generate_pdf=False, captured_at=None, reference_path=None):
    from .stage1_runtime import require_output
    root=require_output(root)
    # The optional Word projector reads the same current response bundle. During
    # live simulation the public sanitized bundle is finalized after this call.
    if generate_pdf and not (root/"cloud_response_bundle.json").exists():
        write_json(root/"cloud_response_bundle.json",bundle)
    items={}
    for name,task in bundle["tasks"].items():
        response=task["response"]
        if response.get("status")!="success":
            raise ValueError("failed worker cannot produce a complete V3 report")
        raw=response["raw_result"]
        if name=="purple":
            roles={"uv_spots":("uv_spots_overlay","uv_base"),"porphyrin":("porphyrin_overlay","fluorescence_base")}
        elif name=="wrinkle":
            roles={"wrinkle":("stage2_overlay",)}
        else:
            roles={"acne" if name in ("acne","acne_v2") else name:("overlay",)}
        for project,keys in roles.items():
            paths=[]
            for key in keys:
                relative="simulated_oss/"+raw[key]
                contained(root,relative)
                paths.append(relative)
            items[project]={"主结果图":paths[0],"附加结果图":paths[1:]}
    return export_v3(root,root/"_doctor_v3_evidence",{},subject_id=subject_id,
                     profile="consumer",captured_at=captured_at,generate_pdf=generate_pdf,
                     index_items=items,reference_path=reference_path,stage1=True)
