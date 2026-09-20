"""Bind this result's own formal media and saved V3 evidence without inference."""
import json,shutil
from pathlib import Path
from .bundle import contained,sha256,write_json,load

def colour_pair_positions(item):
 paths=[item["主结果图"],*item.get("附加结果图",[])]
 if len(paths)!=2:raise ValueError("exact base and marker pair required")
 if "底图" in Path(paths[1]).name:return 1,0
 roles=[i.get("role") for i in item.get("images",[])]
 if roles and roles[0]=="overlay" and roles[1] in ("red_areas_overlay","brown_spots_overlay"):return 0,1
 if "实例图" in Path(paths[1]).name:return 0,1
 raise ValueError("unknown colour media contract")

def refresh_colour_pairs(root):
 root=Path(root);report_path=root/"报告数据/report.json"
 model=json.loads(report_path.read_text(encoding="utf-8"))
 index=json.loads((root/"十二项检测结果索引.json").read_text(encoding="utf-8"))
 items=index.get("items",index.get("十二项结果"))
 for mid,project in (("03","brown"),("04","redness")):
  item=items[project];paths=[item["主结果图"],*item.get("附加结果图",[])]
  positions=colour_pair_positions(item)
  for group in model["module_figures"][mid]:
   if len(group)==2 and group[0]["role"]==project+"_base":
    for image,position in zip(group,positions):
     image["path"]=paths[position];image["sha256"]=sha256(contained(root,paths[position]))
 write_json(report_path,model)
 full_path=root/"十二项完整量化指标.json";full=json.loads(full_path.read_text(encoding="utf-8"))
 full["doctor_v3"].update(report_sha256=sha256(report_path),report_media=model["module_figures"])
 write_json(full_path,full)
 from .word_package import finalize
 finalize(root)

def prepare_images(result_root,bundle_root,cloud_source=None):
 import cv2,numpy as np
 from .measurements import read_evidence
 root=Path(result_root).resolve();bundle_root=Path(bundle_root).resolve();payload=load(bundle_root)
 index=json.loads((root/"十二项检测结果索引.json").read_text(encoding="utf-8"))
 items=index.get("items",index.get("十二项结果"));groups={f"{i:02d}":[] for i in range(1,12)}
 image_root=root/"十二项检测/13_报告衍生图";image_root.mkdir(exist_ok=True)
 def copied(path,role,caption):
  path=Path(path)
  if not path.is_file() or path.is_symlink():raise ValueError("missing formal media "+role)
  if path.resolve().is_relative_to(root.resolve()):
   return {"path":path.relative_to(root).as_posix(),"sha256":sha256(path),"role":role,"caption":caption}
  target=image_root/(role+path.suffix.lower());shutil.copyfile(path,target)
  return {"path":target.relative_to(root).as_posix(),"sha256":sha256(target),"role":role,"caption":caption}
 def add(module,rows):
  groups[module].append(rows)
 def image(project,number=0):
  item=items[project];paths=[item["主结果图"],*item.get("附加结果图",[])]
  return contained(root,paths[number])
 def pair(project,base,marker,caption):
  return [copied(image(project,base),project+"_base",caption+"底图"),
          copied(image(project,marker),project+"_markers",caption+"特征点结果")]
 add("01",[copied(image("pores"),"pores","可见毛孔检测结果")])
 add("02",[copied(image("surface_gloss"),"gloss","表面油光检测结果")])
 cloud=items["uv_spots"].get("images",[{}])[0].get("role")=="uv_base"
 add("02",pair("porphyrin",0 if cloud else 1,1 if cloud else 0,"卟啉"))
 add("03",[copied(image("spots"),"spots","可见色斑检测结果")])
 from .phenotypes import supported_spots
 from .measurements import measure_source
 spot_arrays,_=read_evidence(bundle_root/"evidence/doctor_v3_spots.npz")
 brown_arrays,_=read_evidence(bundle_root/"evidence/doctor_v3_brown.npz")
 uv_arrays,_=read_evidence(bundle_root/"evidence/doctor_v3_uv_spots.npz")
 registration=payload.get("registration",{})
 cp_allowed=payload["capture_profile"]=="consumer" or registration.get("CP_M") is True
 uv_allowed=payload["capture_profile"]=="consumer" or registration.get("365_M") is True
 supported=supported_spots(spot_arrays,brown_arrays if cp_allowed else None,uv_arrays if uv_allowed else None)
 if payload.get("stage1"):
  with np.load(bundle_root/"evidence/stage1_derived.npz",allow_pickle=False) as saved:
   mask=saved.get("spots_supported")
  supported={**spot_arrays,"instances":mask,"continuous":mask} if mask is not None else None
 if supported is not None:
  measured=measure_source("spots",supported,{"multiview_verified":True},payload["capture_profile"])
  expected={(m["metric_id"],m["region"]):m for m in payload["measurements"] if m["module"]=="03"}
  for m in measured:
   previous=expected.get((m["metric_id"],m["region"]),{})
   if not payload.get("stage1") and m["value"] is not None and previous.get("value") is not None and not np.isclose(m["value"],previous["value"],rtol=1e-5,atol=1e-6):
    raise ValueError("supported-spots figure differs from saved V3 measurements")
  canvas=cv2.imread(str(image("spots")),cv2.IMREAD_COLOR)
  if canvas is None or canvas.shape[:2]!=supported["valid"].shape:raise ValueError("spots figure canvas mismatch")
  contours,_=cv2.findContours((supported["instances"]>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(canvas,contours,-1,(255,180,40),2,cv2.LINE_AA)
  selected=image_root/"spots_supported.jpg";cv2.imencode(".jpg",canvas)[1].tofile(str(selected))
  add("03",[copied(selected,"spots_supported","综合色斑标注结果（圈选区域为多图支持目标）")])
 add("03",pair("brown",*colour_pair_positions(items["brown"]),"棕区"))
 add("03",pair("uv_spots",0 if cloud else 1,1 if cloud else 0,"UV色斑"))
 add("04",pair("redness",*colour_pair_positions(items["redness"]),"红区"))
 # These are the already-saved new-definition diagrams, not rerun classifiers.
 from .wrinkle_frames import definitions as frame_definitions,draw as draw_frames
 wrinkle_arrays,_=read_evidence(bundle_root/"evidence/doctor_v3_wrinkle.npz")
 frame_regions=frame_definitions(wrinkle_arrays)
 for module,caption in (("04","弥漫性泛红分布"),("06","红斑样、丘疹样及丘脓疱样目标分布"),
                        ("07","干燥性细纹检测结果"),("08","稳定性线性皱纹检测结果")):
  entry=next((m for m in payload["media"] if m["path"].endswith("v3_"+module+".jpg")),None)
  if not entry:raise ValueError("missing same-definition V3 figure "+module)
  source=contained(bundle_root,entry["path"])
  if sha256(source)!=entry["sha256"]:raise ValueError("saved V3 figure hash mismatch")
  if module in ("07","08"):
   canvas=cv2.imread(str(source),cv2.IMREAD_COLOR)
   if canvas is None:raise ValueError("wrinkle figure unreadable")
   framed=draw_frames(canvas,frame_regions,module)
   target=image_root/("v3_"+module+"_framed.jpg")
   cv2.imencode(".jpg",framed,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(target))
   source=target
  add(module,[copied(source,"v3_"+module,caption)])
 add("05",[copied(image("vascular"),"vascular","线状血管样结构检测结果")])
 # V2 named groove diagram: use the exact role if retained, otherwise draw the
 # same stored named region skeleton, never a different local/cloud sample.
 if len(items["wrinkle"].get("附加结果图",[]))>=2:
  groove=image("wrinkle",2)
 else:
  arrays,_=read_evidence(bundle_root/"evidence/doctor_v3_wrinkle.npz")
  if cloud_source is None:raise ValueError("actual cloud source required for named groove proof")
  cloud_bundle=json.loads((Path(cloud_source)/"cloud_response_bundle.json").read_text(encoding="utf-8"))
  raw=cloud_bundle["tasks"]["wrinkle"]["response"]["raw_result"]
  summary_path=contained(cloud_source,"simulated_oss/"+raw["medical_report_csv_v2"]).parent/"summary_json.json"
  summary=json.loads(summary_path.read_text(encoding="utf-8"))
  from src.wrinkle.algorithms.wrinkle_detection_algorithm import filtered_region_indices,build_face_filter_masks,build_aesthetic_region_masks
  from src.wrinkle.report_region_overlays import render_legacy_group_overlays
  points=arrays["landmarks"];regions=filtered_region_indices(len(points))
  filters=build_face_filter_masks(arrays["image"],points,regions,arrays["valid"])
  defs=build_aesthetic_region_masks(points,regions,filters,arrays["skeleton"])
  import tempfile
  with tempfile.TemporaryDirectory(prefix=".named-wrinkle-",dir=root/"报告数据") as stage:
   replay=render_legacy_group_overlays(arrays["image"],defs,stage)
   proof=summary["report_group_overlays"]["09"]
   if replay["09"]["legacy_centerline_pixels"]!=proof["legacy_centerline_pixels"]:
    raise ValueError("replayed groove is not the saved named groove evidence")
   groove=image_root/"grooves.jpg";shutil.copyfile(replay["09"]["path"],groove)
 add("09",[copied(groove,"grooves","结构性沟纹检测结果")])
 add("10",[copied(image("texture"),"texture","皮肤纹理检测结果")])
 derived=root/"十二项检测/13_报告衍生图"
 options=[p for p in derived.glob("*") if p.is_file() and "表面" in p.name]
 if len(options)==1:surface=options[0]
 else:
  arrays,meta=read_evidence(bundle_root/"evidence/doctor_v3_texture.npz")
  rgb,_=read_evidence(bundle_root/"evidence/doctor_v3_rgb.npz")
  canvas=rgb["image"].copy()
  if canvas.shape[:2]!=arrays["valid"].shape:raise ValueError("texture/RGB canvas mismatch")
  for instance in meta.get("instances",[]):
   x,y=(int(round(v)) for v in instance["centroid"])
   if 0<=y<canvas.shape[0] and 0<=x<canvas.shape[1] and arrays["valid"][y,x]:
    cv2.circle(canvas,(x,y),2,(20,120,240),-1,cv2.LINE_AA)
  surface=image_root/"surface_generated.jpg";cv2.imencode(".jpg",canvas)[1].tofile(str(surface))
 add("10",[copied(surface,"surface_irregularity","表面不规则分布")])
 add("11",[copied(image("contour_firmness"),"contour","面部轮廓检测结果")])
 # Persist only relative references. Renderer verifies every image before binding.
 report_path=root/"报告数据/report.json";model=json.loads(report_path.read_text(encoding="utf-8"))
 model["module_figures"]=groups;model["report_format"]="doctor_v3_illustrated_word_v1"
 write_json(report_path,model)
 complete_path=root/"十二项完整量化指标.json";complete=json.loads(complete_path.read_text(encoding="utf-8"))
 complete["doctor_v3"]["report_sha256"]=sha256(report_path)
 complete["doctor_v3"]["report_media"]=groups
 write_json(complete_path,complete)
 from .word_package import finalize
 finalize(root)
 return groups

def insert_figures(table,model,module,root):
 from docx.shared import Cm
 from docx.enum.text import WD_ALIGN_PARAGRAPH
 from docx.oxml import OxmlElement
 from docx.oxml.ns import qn
 # The approved change is confined to the chapter's former image placeholder.
 table.rows[0].cells[0].merge(table.rows[0].cells[1])
 for p in table.rows[0].cells[0].paragraphs:p.paragraph_format.keep_with_next=True
 for group in model["module_figures"][module]:
  row=table.add_row();cell=row.cells[0].merge(row.cells[1])
  trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement("w:cantSplit"))
  p=cell.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
  for index,item in enumerate(group):
   path=contained(root,item["path"])
   if sha256(path)!=item["sha256"]:raise ValueError("report image SHA mismatch")
   if index:p.add_run("  ")
   shape=p.add_run().add_picture(str(path),width=Cm(8.6 if len(group)==2 else 13.72))
   shape._inline.docPr.set("descr",item["caption"])
   shape._inline.docPr.set("name",item["role"])
  caption=cell.add_paragraph("    ".join(item["caption"] for item in group))
  caption.alignment=WD_ALIGN_PARAGRAPH.CENTER
  caption.paragraph_format.keep_with_next=False
  caption.paragraph_format.keep_together=True
