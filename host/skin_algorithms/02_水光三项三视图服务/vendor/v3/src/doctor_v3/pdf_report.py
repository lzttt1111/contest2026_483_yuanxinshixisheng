"""Direct dual-PDF rendering: no torch, detector, original input, Word or office suite."""
from pathlib import Path
import os
import shutil
import sys
import tempfile
from xml.sax.saxutils import escape
from .bundle import load, contained
from .formal_text import acquisition_note, assert_formal_text, explanation
from .history import compare
from .registry import REGION_LABELS

def dependencies():
    root = Path(__file__).resolve().parents[2]
    deps = root.parent / "07_运行依赖"
    sys.path.insert(0, str(deps / "pdf"))
    font = deps / "fonts/extracted/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    if not font.is_file():
        raise FileNotFoundError("V3 PDF font is not installed")
    return font

def format_value(value, unit=""):
    if value is None:
        return "—"
    if unit == "比例":
        return f"{value*100:.2f}%"
    return str(round(value,4)) if isinstance(value,float) else str(value)

def status_label(module, payload):
    if module["score"]["score"] is not None:
        return module["score"]["grade"]
    if module["id"] in ("09","10","11"):
        return "本次条件暂不支持"
    if any(k.startswith(module["id"]+".") and s.get("score") is not None for k,s in payload.get("score_trace",{}).items()):
        return "部分指标可评分，综合暂不评分"
    if any(m["module"]==module["id"] and m["value"] is not None for m in payload["measurements"]):
        return "量化结果可查看，暂不评分"
    return "本次暂不可评估"

def generate(result_dir, output_dir, previous_result=None):
    payload = load(result_dir)
    previous = load(previous_result) if previous_result is not None else None
    history = compare(payload, previous)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("PDF output must be an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    font = dependencies()
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont("V3Chinese", str(font), subfontIndex=0))
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Image, KeepTogether
    from pypdf import PdfReader
    normal = ParagraphStyle("body", fontName="V3Chinese",fontSize=10.5,leading=17,spaceAfter=7,wordWrap="CJK")
    small = ParagraphStyle("small", parent=normal,fontSize=9,leading=13)
    title = ParagraphStyle("title", parent=normal,fontSize=20,leading=29,spaceAfter=16)
    heading = ParagraphStyle("heading",parent=normal,fontSize=14,leading=21,spaceBefore=12,spaceAfter=9,keepWithNext=True)
    def para(text, style=normal):
        text = str(text)
        assert_formal_text(text)
        return Paragraph(escape(text).replace("\n","<br/>"),style)
    def table(rows, widths):
        obj = Table([[para(cell,small) for cell in row] for row in rows],colWidths=widths,repeatRows=1,hAlign="LEFT")
        obj.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf1f4")),
             ("GRID",(0,0),(-1,-1),.35,colors.HexColor("#c9cfd5")),("VALIGN",(0,0),(-1,-1),"TOP"),
             ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
             ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        return obj
    stage = Path(tempfile.mkdtemp(prefix=".pdf-",dir=output))
    paths = []
    try:
        for edition,label in (("home","居家版"),("institution","机构版")):
            path = stage / ("AISIA_皮肤检测报告_"+label+".pdf")
            class ReportDocument(SimpleDocTemplate):
                def afterFlowable(self, flowable):
                    key = getattr(flowable, "bookmark_key", None)
                    if key is not None:
                        self.canv.bookmarkPage(key)
                        self.canv.addOutlineEntry(flowable.getPlainText(),key,level=0)
            doc = ReportDocument(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=46,bottomMargin=43,
                                    title="AISIA皮肤检测报告（"+label+"）",author="AISIA")
            width = A4[0]-84
            story = [para("AISIA 皮肤检测报告",title),para(label,heading),
                     para("受检者编号："+payload["subject_id"]),
                     para("采集时间："+(payload.get("captured_at") or "未提供")),
                     para("报告生成时间："+payload["generated_at"][:10]),para(acquisition_note(payload["capture_profile"])),
                     para("本报告用于图像表型评估与复查参考，不作为疾病确诊依据。"),
                     para("结果总览",heading)]
            if payload.get("reference_notice"):
                story.insert(-1,para(payload["reference_notice"],small))
            story.append(table([["检测项目","状态评分","结果状态"]]+[[m["title"],format_value(m["score"]["score"]),status_label(m,payload)] for m in payload["modules"]],[width*.48,width*.18,width*.34]))
            story.append(para("状态评分越高，表示相应图像表现的负担越低。未提供分数的项目，请阅读该章节的评估说明。",small))
            for module in payload["modules"]:
                chapter = para(module["id"]+"  "+module["title"],heading)
                chapter.bookmark_key = "module-"+module["id"]
                story += [PageBreak(),chapter,
                          para("本次结果",heading)]
                score = module["score"]
                if score["score"] is None:
                    story.append(para(explanation("requires_calibrated_3d" if module["id"] in ("09","10","11") else "missing_compatible_reference")))
                else:
                    story.append(para(f"初步状态参考分：{score['score']}分　参考等级：{score['grade']}"))
                sub_scores=module.get("sub_scores",{}).get("full_face",{})
                if sub_scores:
                    labels={"surface_gloss":"表面油光","porphyrin":"毛囊卟啉","spots":"色斑","brown":"棕色色素","uv":"UV色素"}
                    story.append(table([["分项结果","状态评分","程度"]]+[
                        [labels[k],format_value(s["score"]),s["grade"] if s["score"] is not None else "暂不评分"]
                        for k,s in sub_scores.items()],[width*.5,width*.25,width*.25]))
                entries = [m for m in payload["measurements"] if m["module"] == module["id"] and m["region"]=="full_face"]
                if entries:
                    rows = [["核心指标","本次结果","单位","初步参考分"]]
                    for m in entries:
                        s=payload.get("score_trace",{}).get(m["metric_id"]+":"+m["region"],{}).get("score")
                        rows.append([m["name"],format_value(m["value"],m["unit"]),m["unit"] if m["value"] is not None else "—",
                                     str(s)+"分" if s is not None else "暂不评分"])
                    story.append(table(rows,[width*.40,width*.18,width*.22,width*.20]))
                    reasons = list(dict.fromkeys(explanation(m.get("reason")) for m in entries if m["value"] is None))
                    story += [para(r,small) for r in reasons]
                if edition=="home" and module["id"]=="08":
                    rows=[["皱纹类型","本次量化结果","状态评分"]]
                    for region,label in (("forehead","额纹"),("glabella","眉间纹"),("crow_feet","鱼尾纹"),("perioral","口周纹")):
                        metric_id="08."+("main_count" if region in ("forehead","glabella") else "area")
                        entry=next((m for m in payload["measurements"] if m["metric_id"]==metric_id and m["region"]==region),None)
                        value="—" if entry is None else format_value(entry["value"],entry["unit"])+("条" if entry["value"] is not None and entry["unit"]=="条" else "")
                        rows.append([label,value,format_value(module["regions"][region]["score"])])
                    story.append(table(rows,[width*.3,width*.4,width*.3]))
                if edition == "institution":
                    story.append(para("分区结果",heading))
                    rows = [["区域","状态评分／程度","本次量化结果"]]
                    for r,s in module["regions"].items():
                        if r=="full_face":
                            continue
                        local=[m for m in payload["measurements"] if m["module"]==module["id"] and m["region"]==r and m["value"] is not None]
                        details=[]
                        for m in local:
                            ms=payload.get("score_trace",{}).get(m["metric_id"]+":"+m["region"],{}).get("score")
                            details.append(m["name"]+"："+format_value(m["value"],m["unit"])+(f"（参考分{ms}）" if ms is not None else ""))
                        detail="；".join(details) or "本次暂不提供该区域量化评估"
                        rows.append([REGION_LABELS.get(r,"相关区域"),format_value(s["score"])+"／"+(s["grade"] if s["score"] is not None else "暂不评分"),detail])
                    story.append(table(rows,[width*.22,width*.23,width*.55]))
                    primary={"03":"spots.area","08":"area"}.get(module["id"])
                    if primary is None:
                        from .registry import MODULES
                        primary=next(iter(MODULES[int(module["id"])-1].weights))
                    keyed={(m["metric_id"],m["region"]):m for m in payload["measurements"]}
                    comparisons=[]
                    for m in payload["measurements"]:
                        if m["metric_id"]!=module["id"]+"."+primary or not m["region"].startswith("left_"):
                            continue
                        other=keyed.get((m["metric_id"],m["region"].replace("left_","right_",1)))
                        if other and m["value"] is not None and other["value"] is not None:
                            comparisons.append([REGION_LABELS[m["region"]]+"／"+REGION_LABELS[other["region"]],
                                                format_value(m["value"],m["unit"]),format_value(other["value"],other["unit"])])
                    story.append(para("左右对应区域",heading))
                    if comparisons:
                        story.append(table([["对应区域（主要指标）","左侧","右侧"],*comparisons],[width*.5,width*.25,width*.25]))
                        story.append(para("此处仅列相同指标的左右数值，不将原始数值差直接解释为疾病或严重程度差异。",small))
                    else:
                        story.append(para("本次条件暂不支持对应区域的量化比较。",small))
                    if score["score"] is None:
                        story.append(para("本次暂不依据未分级指标判断主要严重度来源。",small))
                selected = [m for m in payload["media"] if module["id"] in m["modules"]]
                derived = [m for m in selected if m.get("derived")]
                if derived:
                    selected=derived
                if module["id"]=="09" and selected:
                    selected=selected[-1:]
                for number, media in enumerate(selected):
                    image = Image(str(contained(result_dir,media["path"])))
                    image_height = (180 if edition=="home" else 190) if module["id"]=="08" else (220 if edition=="home" else 260)
                    scale = min(width/image.imageWidth,image_height/image.imageHeight,1.0)
                    image.drawWidth = image.imageWidth*scale
                    image.drawHeight = image.imageHeight*scale
                    block = [image,para(media["caption"],small),Spacer(1,8)]
                    if number==0:
                        block.insert(0,para("检测结果图",heading))
                    story.append(KeepTogether(block))
                history_start=len(story)
                story.append(para("与上次相比",heading))
                history_rows = [r for r in history["items"] if r["metric_id"].startswith(module["id"]+".") and r["region"]=="full_face"]
                if history_rows:
                    names = {m["metric_id"]:m["name"] for m in entries}
                    def delta_text(row):
                        if row["delta"] is None:
                            return "—"
                        if row.get("unit")=="比例":
                            return f"{row['delta']*100:+.2f}个百分点"
                        return format_value(row["delta"])
                    rows = [["项目","上次","本次","变化"]]+[[names.get(r["metric_id"],"相关指标"),format_value(r["previous"],r.get("unit","")),format_value(r["current"],r.get("unit","")),delta_text(r)] for r in history_rows]
                    story.append(table(rows,[width*.46,width*.18,width*.18,width*.18]))
                    story.append(para("仅对具备可比条件的项目展示数值变化；变化的实际意义需结合复查条件及专业评估。",small))
                else:
                    story.append(table([["对比项目","上次","本次","变化"],["状态评分","—",format_value(score["score"]),"—"]],[width*.46,width*.18,width*.18,width*.18]))
                    story.append(para(explanation("no_history"),small))
                story.append(para(("医学说明：" if edition=="institution" else "温馨提示：")+module["note"],small))
                story[history_start:]=[KeepTogether(story[history_start:])]
            def footer(canvas, document):
                canvas.saveState()
                canvas.setFont("V3Chinese",8)
                canvas.drawString(42,25,"AISIA 皮肤检测报告 · "+label)
                canvas.drawRightString(A4[0]-42,25,str(document.page))
                canvas.restoreState()
            doc.build(story,onFirstPage=footer,onLaterPages=footer)
            reader = PdfReader(str(path))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            assert_formal_text(text)
            if len(reader.pages)<12 or any(m["title"] not in text for m in payload["modules"]):
                raise ValueError("incomplete PDF chapters")
            paths.append(path)
        # Both editions must finish and pass text validation before publication.
        published = []
        try:
            for path in paths:
                target = output/path.name
                os.replace(path,target)
                published.append(target)
        except Exception:
            for path in published:
                path.unlink()
            raise
        return published
    finally:
        shutil.rmtree(stage)
