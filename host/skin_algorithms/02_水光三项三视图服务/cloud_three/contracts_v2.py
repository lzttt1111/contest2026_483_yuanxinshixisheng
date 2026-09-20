"""Saved-result composition contract. Counts and native V3 score regions are separate."""
from typing import Literal
from pydantic import Field,model_validator
from .contracts import Strict,Identifier,Counts,Views,RegionResult,ErrorInfo

Severity=Literal["未见明显","轻度","中度","较明显","显著"]

class ComposeRequest(Strict):
    task_id:Identifier=Field(description="JSON v2组装任务唯一ID")
    counts_job_ref:str=Field(min_length=1,description="已有三视图任务目录，相对SHUIGUANG_RESULT_ROOT")
    v3_result_ref:str|None=Field(default=None,description="可选的已导入正面V3证据目录，相对同一允许根")

class Score(Strict):
    score:float|None=Field(ge=0,le=100,description="V3状态评分；高分更好，缺测为null")
    severity:Severity|None=Field(description="V3程度等级，缺测为null")
    score_status:Literal["available","unavailable"]=Field(description="评分可用状态")
    score_reason:str|None=Field(description="不可评分原因；可用时为null")
    @model_validator(mode="after")
    def validate_score(self):
        if self.score_status=="available":
            if self.score is None or self.severity is None or self.score_reason is not None:
                raise ValueError("可用评分必须具有数值与等级，不带缺项原因")
        elif self.score is not None or self.severity is not None or not self.score_reason:
            raise ValueError("不可用评分必须为null并注明原因")
        return self

class RegionalScore(Score):
    region:str=Field(description="V3原生区域ID，不等同于数量表区域")
    name:str=Field(description="V3区域中文名；左右以画面为准")

class RegionalScores(Strict):
    score_view:Literal["front"]="front"
    score_direction:Literal["higher_is_better"]="higher_is_better"
    region_schema:Literal["v3_native"]="v3_native"
    region_side_convention:Literal["image_left_right"]="image_left_right"
    scoring_version:str|None=Field(description="实际V3系统、ROI和评分身份；来源不可验证时null")
    items:list[RegionalScore]=Field(description="每项固定的V3原生区域列表，缺项仍保留")
    @model_validator(mode="after")
    def check_regions(self):
        if len({r.region for r in self.items})!=len(self.items):raise ValueError("重复评分区域")
        if any(r.score_status=="available" for r in self.items) and not self.scoring_version:
            raise ValueError("有分数时必须有评分版本")
        return self

class ItemV2(Score):
    name:str
    score_view:Literal["front"]="front"
    score_direction:Literal["higher_is_better"]="higher_is_better"
    scoring_version:str|None
    score_basis:Literal["v3_visible_pores","v3_visible_spots","v3_surface_gloss"]
    total_count:Counts
    regions:list[RegionResult]
    unassigned_count:Counts
    regional_scores:RegionalScores
    @model_validator(mode="after")
    def reconcile(self):
        if self.scoring_version!=self.regional_scores.scoring_version:raise ValueError("整体和分区版本不一致")
        if self.score_status=="available" and not self.scoring_version:raise ValueError("有评分时必须有版本")
        if len({r.region for r in self.regions})!=len(self.regions):raise ValueError("重复数量区域")
        for view in ("left","front","right"):
            total=getattr(self.total_count,view);remaining=getattr(self.unassigned_count,view)
            vals=[getattr(r,view) for r in self.regions]
            if total is None:
                if remaining is not None or any(v is not None for v in vals):raise ValueError("未测视图有数量")
            elif remaining is None or sum(v or 0 for v in vals)+remaining!=total:raise ValueError("数量无法对账")
        return self

class ResultsV2(Strict):
    pores:ItemV2
    spots:ItemV2
    surface_gloss:ItemV2

class ComposedResult(Strict):
    schema_version:Literal["shuiguang_cloud_v2"]="shuiguang_cloud_v2"
    task_id:Identifier
    status:Literal["success"]="success"
    main_image_id:str
    views:Views
    results:ResultsV2
    @model_validator(mode="after")
    def check_binding(self):
        if self.main_image_id!=self.views.front or len(set(self.views.model_dump().values()))!=3:
            raise ValueError("三视图映射无效")
        bases={"pores":"v3_visible_pores","spots":"v3_visible_spots","surface_gloss":"v3_surface_gloss"}
        for key,value in bases.items():
            if getattr(self.results,key).score_basis!=value:raise ValueError("评分项目错绑")
        return self

class FailureV2(Strict):
    schema_version:Literal["shuiguang_cloud_v2"]="shuiguang_cloud_v2"
    task_id:str
    status:Literal["failed"]="failed"
    error:ErrorInfo
