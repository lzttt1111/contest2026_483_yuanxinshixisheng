import csv
import pytest
from src.medical_v2_delivery import document_rows, validate_csv_matches_document

def document():
    return {"总体指标":{"辅助指标":{"分析范围":{"检测范围":"全面部","评估状态":"ASSESSABLE","有效皮肤面积（像素）":100}},
             "核心指标":{"范围":{"P50单体面积（像素）":2.5,"P90单体面积（像素）":4.0}},
             },"分区指标":[]}
def write_csv(path, squared=False, wrong=False, duplicate=False):
    rows=document_rows(document())
    names=list(rows[0])
    if squared:names=[n.replace("单体面积（像素）","单体面积（像素²）") for n in names]
    values=list(rows[0].values())
    if wrong:values[-1]=99
    if duplicate:names.append("核心-P50单体面积（像素）");values.append(2.5)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(names);writer.writerow(values)
@pytest.mark.parametrize("squared",[False,True])
def test_pore_area_unit_alias_preserves_numeric_validation(tmp_path,squared):
    p=tmp_path/"metrics.csv";write_csv(p,squared=squared)
    validate_csv_matches_document(p,document())
    write_csv(p,squared=squared,wrong=True)
    with pytest.raises(ValueError):validate_csv_matches_document(p,document())
def test_ambiguous_area_alias_is_rejected(tmp_path):
    p=tmp_path/"metrics.csv";write_csv(p,squared=True,duplicate=True)
    with pytest.raises(ValueError,match="重复"):validate_csv_matches_document(p,document())
