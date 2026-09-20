"""Fill native physician DOCX templates in place; preserve their styles and layout."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import zipfile
from .formal_text import assert_formal_text
from .word_slots import score_text,grade_text
from .word_tables import bind_table,distribution,description
from .word_text import bind_text
from .word_public_labels import formal_label

TITLES=("毛孔","油脂分泌","面部色素","弥漫性泛红","毛细血管样增生","毛囊炎症及痤疮",
        "干燥性细纹","稳定线性皱纹","结构性沟纹","皮肤平整度","轮廓紧致度")
def paragraph_text(p,text):
    text=text.replace("暂不评分","暂无评分")
    assert_formal_text(text)
    # Keep pPr, bookmarks and the first run's character formatting.
    if p.runs:
        p.runs[0].text=text
        for run in p.runs[1:]:run.text=""
    else:p.add_run(text)

def cell_text(cell,text):
    if cell.text==text:return
    paragraphs=cell.paragraphs
    paragraph_text(paragraphs[0],text)
    for p in paragraphs[1:]:paragraph_text(p,"")

def overview(model,table):
    for row in table.rows[1:]:
        title=row.cells[0].text.strip()
        module=str(TITLES.index(title)+1).zfill(2)
        m=model["modules"][module]
        values=[title,score_text(m)+" / "+grade_text(m),distribution(model,module,"主要受累区域"),
                distribution(model,module,"主要严重度来源")]
        for c,text in zip(row.cells,values):cell_text(c,formal_label(text,module))

def hero(model,module,table):
    m=model["modules"][module]
    left,right=table.rows[0].cells
    first=left.paragraphs[0]
    if len(first.runs)>=2:
        first.runs[0].text=score_text(m)+"\n"
        first.runs[1].text=grade_text(m)
        for run in first.runs[2:]:run.text=""
    else:paragraph_text(first,score_text(m)+"\n"+grade_text(m))
    if len(left.paragraphs)>1:
        summary=formal_label(distribution(model,module,"主要严重度来源"),module)
        if m.get("assessment_title"):summary=m["assessment_title"]+"状态评分"+("；"+summary if summary!="—" else "")
        paragraph_text(left.paragraphs[1],summary)
    for p in right.paragraphs:
        paragraph_text(p,"")
    from docx.oxml.ns import qn
    for node in list(right._tc.iter(qn("w:drawing"))):
        node.getparent().remove(node)
    for node in list(right._tc.iter(qn("w:pict"))):
        node.getparent().remove(node)

def front_text(model,text):
    if text.startswith("报告性质："):
        return "受检者编号："+model["subject_id"]
    if text.startswith("生成日期："):
        return "报告日期："+model["generated_at"][:10]
    if text.startswith("重要说明：") or text.startswith("内容来源："):
        return None
    if text.startswith("本报告整理"):
        return "本报告汇总本次11类面部检测结果，包括各模块的状态评分、主要区域及纵向变化。"
    if text.startswith(("中央面部：","双侧面颊：","纹理与结构：","轮廓：")):
        label=text.split("：",1)[0]
        groups={"中央面部":["01","02","03","04","05"],"双侧面颊":["03","04","06","10"],
                "纹理与结构":["07","08","09"],"轮廓":["11"]}[label]
        if label=="轮廓":
            value=model["modules"]["11"]
            return "轮廓：整体状态评分为"+score_text(value)+"，结果等级为"+grade_text(value)+"。"
        selected=[]
        allowed={"中央面部":{"forehead","nose","left_nasal","right_nasal","chin"},
                 "双侧面颊":{"left_cheek","right_cheek","left_zygoma","right_zygoma","left_inner_cheek","right_inner_cheek"},
                 "纹理与结构":None}[label]
        for m in groups:
            if allowed is None:location=distribution(model,m,"主要受累区域")
            else:
                from .registry import REGION_LABELS
                values=[(r,s) for r,s in model["modules"][m]["regions"].items() if r in allowed and s.get("score") is not None and s["score"]<81]
                values.sort(key=lambda x:x[1]["score"])
                location="、".join(REGION_LABELS[r] for r,s in values[:2]) or "—"
            if location not in ("—","未见明显异常区域"):
                selected.append(TITLES[int(m)-1]+"："+location)
        return label+"："+"；".join(selected[:3])+"。" if selected else label+"：相关可评估项目未见明显异常表现。"
    return text.replace("示例报告","报告").replace("纵向变化示例","纵向变化").replace("标准化照片和三维模型","标准化面部图像")

def populate(template,model,destination,edition,media_root=None):
    from docx import Document
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    from docx.oxml.ns import qn
    doc=Document(template)
    before_sections=[s._sectPr.xml for s in doc.sections]
    before_styles=doc.styles.element.xml
    source_sha=hashlib.sha256(Path(template).read_bytes()).hexdigest()
    module=None;state={};audit={"source_sha256":source_sha,"template_sha256":source_sha,"blocks":[],"exceptions":[]}
    for index,node in enumerate(list(doc.element.body)):
        kind=node.tag.rsplit("}",1)[-1]
        locator=f"word/document.xml/body/{index}"
        if kind=="p":
            p=Paragraph(node,doc._body);text=p.text.strip()
            if not text:continue
            style=p.style.name
            if text in TITLES and style=="Heading 1":
                module=str(TITLES.index(text)+1).zfill(2);state={}
                audit["blocks"].append({"source":locator,"kind":"chapter","module":module,"text":text})
                continue
            if text=="报告使用说明":module=None
            if module is None:
                replacement=front_text(model,text)
            elif text.startswith("温馨提示："):
                replacement=text
                if "当前综合色素问题以" in text:
                    tail=text.split("  ",1)
                    replacement="温馨提示："+("  "+tail[1] if len(tail)>1 else "")
                replacement=replacement.replace("及3D面部曲面","").replace("及3D表面形貌","")
                replacement=replacement.replace("标准化拍摄和3D检测条件下","标准化拍摄条件下").replace("外观和三维表现","外观表现").replace("和3D重建条件","及检测条件").replace("三维形态状态","图像形态状态")
                state["medical"]=False
            elif style.startswith("Heading") and not ("重点关注：" in text or text.endswith("。")):
                state["section"]=text;state["medical"]="医学说明" in text
                state["history"]=any(x in text for x in ("纵向","上次","追踪","变化"))
                state["history_written"]=False
                replacement=re.sub(r"（完善疾病库后.*?）","",text)
            else:
                b={"id":locator,"kind":"text","text":text.lstrip("• ").strip(),"font_size":0}
                replacement=bind_text(model,module,b,state,"institution")
                if text.startswith("• ") and replacement and not state.get("history"):
                    replacement="• "+replacement
            if replacement is None:
                node.getparent().remove(node)
                audit["exceptions"].append({"source":locator,"reason":"example_or_nonapplicable_history_text"})
            else:
                changed=formal_label(replacement,module)
                if changed!=replacement:
                    audit["exceptions"].append({"source":locator,"reason":"image_only_groove_label","before":replacement,"after":changed})
                replacement=changed
                if replacement!=p.text:paragraph_text(p,replacement)
                audit["blocks"].append({"source":locator,"kind":"paragraph","module":module,"text":replacement})
        elif kind=="tbl":
            table=Table(node,doc._body)
            rows=[[c.text.strip() for c in row.cells] for row in table.rows]
            if rows[0][0]=="检测模块":
                overview(model,table)
                audit["blocks"].append({"source":locator,"kind":"overview","header":rows[0]})
            elif len(rows)==1 and len(rows[0])==2 and "图片预留" in rows[0][1]:
                if module is None:raise ValueError("score card without module")
                hero(model,module,table)
                if model.get("module_figures"):
                    from .word_images import insert_figures
                    insert_figures(table,model,module,media_root)
                audit["blocks"].append({"source":locator,"kind":"score_card","module":module})
                audit["exceptions"].append({"source":locator,"reason":"text_only_image_placeholder_cleared"})
            else:
                if module is None:raise ValueError("unmapped front-matter table")
                from .evaluation_scope import retained_table_rows
                retained=retained_table_rows(model,module,rows)
                if len(retained)!=len(rows):
                    removed=[i for i in range(len(rows)) if i not in retained]
                    audit["exceptions"].append({"source":locator,"reason":"approved_assessment_scope",
                        "removed_rows":[rows[i][0] for i in removed],
                        "scope_version":model["assessment_scope"]["version"]})
                    for i in reversed(removed):
                        tr=table.rows[i]._tr
                        tr.getparent().remove(tr)
                    rows=[rows[i] for i in retained]
                bound=bind_table(model,module,rows)
                if len(bound)!=len(rows):raise ValueError("table row drift")
                original_bound=bound
                bound=[[formal_label(str(v),module,bool(model.get("stage1_versions"))) for v in row] for row in bound]
                if bound!=original_bound:
                    audit["exceptions"].append({"source":locator,"reason":"image_only_groove_label",
                        "before_labels":[r[0] for r in original_bound],"after_labels":[r[0] for r in bound]})
                for row,values in zip(table.rows,bound):
                    for cell,value in zip(row.cells,values):cell_text(cell,str(value))
                for cell in table.rows[0].cells:
                    for p in cell.paragraphs:p.paragraph_format.keep_with_next=True
                audit["blocks"].append({"source":locator,"kind":"table","module":module,"header":bound[0],
                                        "row_labels":[r[0] for r in bound[1:]]})
    # Remove only orphaned placeholder media after clearing all placeholder drawings.
    for rel in list(doc.part.rels.values()):
        if rel.reltype.endswith(("/header","/footer")):
            for node in rel.target_part.element.iter(qn("w:t")):
                if node.text:node.text=node.text.replace("示例","")
    doc.core_properties.title="AISIA皮肤检测报告"
    doc.core_properties.subject=model["subject_id"]
    doc.core_properties.author="AISIA"
    doc.core_properties.comments=""
    used={n.get(qn("r:embed")) for n in doc.element.iter(qn("a:blip"))}
    for rid,rel in list(doc.part.rels.items()):
        if rel.reltype.endswith("/image") and rid not in used:doc.part.drop_rel(rid)
    if before_sections!=[s._sectPr.xml for s in doc.sections] or before_styles!=doc.styles.element.xml:
        raise ValueError("native section/style preservation failed")
    doc.save(destination)
    with zipfile.ZipFile(destination) as package:
        if package.testzip():raise ValueError("DOCX CRC failed")
        from xml.etree import ElementTree as ET
        root=ET.fromstring(package.read("word/document.xml"))
        texts="\n".join(n.text or "" for n in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
        assert_formal_text(texts)
        for token in ("请替换","图片预留区","非真实检测","示例报告","内容来源："):
            if token in texts:raise ValueError("template placeholder leaked: "+token)
        if not model.get("module_figures") and any(n.startswith("word/media/") for n in package.namelist()):
            raise ValueError("text-only report contains media")
    audit["sha256"]=hashlib.sha256(Path(destination).read_bytes()).hexdigest()
    return audit

def generate_from_source(model,output_dir,template_root):
    import os,shutil,tempfile
    root=Path(template_root);output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    authority=json.loads((root/"ACTIVE_TEMPLATE.json").read_text(encoding="utf-8"))
    if authority["format"]!="docx":raise ValueError("latest native Word template required")
    stage=Path(tempfile.mkdtemp(prefix=".word-native-",dir=output))
    paths=[];audits=[];published=[]
    try:
        for edition,label in (("home","居家"),("institution","机构")):
            spec=authority["editions"][edition];template=root/spec["file"]
            if hashlib.sha256(template.read_bytes()).hexdigest()!=spec["sha256"]:raise ValueError("physician template changed")
            name=f"AISIA_{model['subject_id']}_{label}版.docx"
            if (output/name).exists():raise FileExistsError("report already exists")
            path=stage/name
            audits.append(populate(template,model,path,edition,root.parent.parent));paths.append(path)
        for p in paths:
            target=output/p.name;os.replace(p,target);published.append(target)
        return published,audits
    except Exception:
        for p in published:p.unlink()
        raise
    finally:
        shutil.rmtree(stage)
