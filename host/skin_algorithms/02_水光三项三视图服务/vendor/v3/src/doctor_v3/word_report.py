"""Generate the two text-only physician DOCX editions from saved report data."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import tempfile
from .word_tables import bind_table
from .word_text import bind_text, heading
from .formal_text import assert_formal_text

def generate_word(model,output_dir,template_root=None):
    if not model.get("subject_id") or any(c in model["subject_id"] for c in '/\\:*?"<>|'):
        raise ValueError("invalid report subject filename")
    active_root=Path(template_root) if template_root else Path(__file__).resolve().parents[2]/"templates/doctor_v3_text"
    if (active_root/"ACTIVE_TEMPLATE.json").is_file():
        from .word_source_report import generate_from_source
        return generate_from_source(model,output_dir,active_root)
    raise ValueError("latest physician Word template authority is missing; PDF-layout fallback is disabled")

def _legacy_pdf_layout_not_for_current_reports(model,output_dir,template_root=None):
    from docx import Document
    from docx.shared import Pt,Cm,RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
    template_root=Path(template_root) if template_root else Path(__file__).resolve().parents[2]/"templates/doctor_v3_text"
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    names=[f"AISIA_{model['subject_id']}_{name}版.docx" for name in ("居家","机构")]
    if any((output/n).exists() for n in names):raise FileExistsError("report already exists")
    stage=Path(tempfile.mkdtemp(prefix=".word-v3-",dir=output))
    audits=[]
    try:
        for edition,name,filename in zip(("home","institution"),("居家","机构"),names):
            layout=json.loads((template_root/(edition+"_source_layout.json")).read_text(encoding="utf-8"))
            doc=Document()
            section=doc.sections[0]
            section.page_width=Pt(layout["page_sizes"][0][0]);section.page_height=Pt(layout["page_sizes"][0][1])
            section.left_margin=section.right_margin=Pt(72)
            section.top_margin=section.bottom_margin=Pt(60)
            for style_name in ("Normal","Title","Heading 1","Heading 2"):
                st=doc.styles[style_name];st.font.name="宋体";st.font.color.rgb=RGBColor(0,0,0)
                st.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"),"宋体")
            body_size=12 if edition=="home" else 10.5
            doc.styles["Title"].font.size=Pt(30 if edition=="home" else 25.1)
            st=doc.styles["Normal"];st.font.size=Pt(body_size)
            st.paragraph_format.space_after=Pt(5);st.paragraph_format.line_spacing=1.15
            doc.add_paragraph("皮肤检测"+name+"版报告","Title")
            doc.add_paragraph("受检者编号："+model["subject_id"])
            doc.add_paragraph("采集时间："+(model.get("captured_at") or "未提供")+"    报告日期："+model["generated_at"][:10])
            audit={"edition":edition,"template_sha256":layout["source_sha256"],"blocks":[],"exceptions":[]}
            def paragraph(text,style=None,size=None):
                assert_formal_text(text)
                p=doc.add_paragraph(text,style)
                if size:
                    for run in p.runs:run.font.size=Pt(size)
                return p
            for module,blocks in layout["modules"].items():
                state={}
                for b in blocks:
                    if b["kind"]=="table":
                        rows=bind_table(model,module,b["rows"])
                        table=doc.add_table(rows=0,cols=len(rows[0]))
                        table.alignment=WD_TABLE_ALIGNMENT.LEFT;table.autofit=False
                        total=sum(b["widths"]); available=min(451,total)
                        for column,width in zip(table.columns,b["widths"]):column.width=Pt(available*width/total)
                        for n,row in enumerate(rows):
                            cells=table.add_row().cells
                            for c,value in zip(cells,row):
                                assert_formal_text(str(value));c.text=str(value);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                                for p in c.paragraphs:
                                    p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
                                    for run in p.runs:
                                        run.font.size=Pt(body_size);run.font.bold=n==0
                                props=c._tc.get_or_add_tcPr()
                                borders=OxmlElement("w:tcBorders")
                                for edge in ("top","left","bottom","right"):
                                    e=OxmlElement("w:"+edge);e.set(qn("w:val"),"single");e.set(qn("w:sz"),"4");e.set(qn("w:color"),"D9D9D9");borders.append(e)
                                props.append(borders)
                                if n==0:
                                    shade=OxmlElement("w:shd");shade.set(qn("w:fill"),"F2F2F2");props.append(shade)
                            if n==0:
                                repeat=OxmlElement("w:tblHeader");table.rows[n]._tr.get_or_add_trPr().append(repeat)
                        audit["blocks"].append({"source":b["id"],"kind":"table","header":rows[0],"row_labels":[r[:2] if b["rows"][0][0]=="皱纹类型" and len(r)>4 else r[:1] for r in rows[1:]]})
                    else:
                        text=bind_text(model,module,b,state,edition)
                        if text is None:
                            audit["exceptions"].append({"source":b["id"],"reason":"nonapplicable_history_or_authoring_note"});continue
                        is_heading=heading(b,edition)
                        style="Heading 1" if b["kind"]=="chapter" else "Heading 2" if is_heading else None
                        p=paragraph(text,style,size=b.get("font_size") if is_heading else body_size)
                        if is_heading:p.paragraph_format.keep_with_next=True
                        audit["blocks"].append({"source":b["id"],"kind":"heading" if is_heading else "paragraph","text":text})
                audit["exceptions"].extend(state.get("exceptions",[]))
            footer=section.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
            field=OxmlElement("w:fldSimple");field.set(qn("w:instr"),"PAGE");footer._p.append(field)
            path=stage/filename;doc.save(path)
            import zipfile
            with zipfile.ZipFile(path) as z:
                if z.testzip() or any(n.startswith("word/media/") for n in z.namelist()):
                    raise ValueError("text-only DOCX CRC/media gate failed")
            audit["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            audits.append(audit)
        published=[]
        try:
            for name in names:
                os.replace(stage/name,output/name);published.append(output/name)
        except Exception:
            for p in published:p.unlink()
            raise
        return published,audits
    finally:
        shutil.rmtree(stage)
