"""Public success payload: exactly three score-only items. Internal evidence stays intact."""
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator
Severity=Literal["未见明显","轻度","中度","较明显","显著"]
class Strict(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
class Score(Strict):
    score:float|None=Field(ge=0,le=100,description="0至100状态分，高分状态更好；缺测为null。色斑为独立面积密度口径，毛孔/油光沿用V3")
    severity:Severity|None=Field(description="程度等级；缺测为null")
    @model_validator(mode="after")
    def paired(self):
        if (self.score is None)!=(self.severity is None):raise ValueError("评分与程度必须同时存在或为空")
        return self
class RegionScore(Score):
    region:str=Field(description="本项评分区域英文ID，左右按画面；色斑为六个合并区域，其他项保持原生区域")
    name:str=Field(description="区域中文名称")
class Item(Score):
    name:str=Field(description="检测项目中文名称")
    regions:list[RegionScore]=Field(description="本项正面分区评分与程度")
class ScoreResponse(Strict):
    pores:Item=Field(description="毛孔")
    spots:Item=Field(description="可见色斑")
    surface_gloss:Item=Field(description="纯表面油光")
def public_scores(document):
    if document.get("status")=="failed":return document
    if "results" not in document:
        return ScoreResponse.model_validate(document).model_dump()
    output={}
    for key in ("pores","spots","surface_gloss"):
        item=document["results"][key]
        output[key]={"name":item["name"],"score":item["score"],"severity":item["severity"],
          "regions":[{k:r[k] for k in ("region","name","score","severity")}
                     for r in item["regional_scores"]["items"]]}
    return ScoreResponse.model_validate(output).model_dump()

def public_from_v3_extract(scores):
    return ScoreResponse.model_validate({key:{"name":value["name"],"score":value["score"],
      "severity":value["severity"],"regions":[{field:r[field] for field in ("region","name","score","severity")}
      for r in value["regional_scores"]["items"]]} for key,value in scores.items()}).model_dump()
