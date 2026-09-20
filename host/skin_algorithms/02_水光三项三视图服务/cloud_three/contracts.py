from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

View=Literal["left","front","right"]
Count=Annotated[int,Field(strict=True,ge=0)]
Identifier=Annotated[str,Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
class Strict(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
class InputImage(Strict):
    image_id:Identifier=Field(description="原始上传图片唯一ID")
    path:str=Field(min_length=1,description="相对服务输入根目录的RGB文件路径")
class AnalyzeRequest(Strict):
    task_id:Identifier=Field(description="业务检测任务ID；同ID同输入幂等返回，不同输入拒绝")
    images:list[InputImage]=Field(min_length=3,max_length=3,description="乱序三张RGB照片")
    mirrored:bool=Field(default=False,strict=True,description="输入是否镜像自拍，true时统一去镜像")
    @model_validator(mode="after")
    def unique(self):
        if len({i.image_id for i in self.images})!=3:raise ValueError("图片ID不能重复")
        return self
class Counts(Strict):
    left:Count|None
    front:Count|None
    right:Count|None
class Views(Strict):
    left:str
    front:str
    right:str
class RegionResult(Counts):
    region:Literal["forehead","nose","left_eye","right_eye","left_cheek","right_cheek","perioral","chin"]
    name:str
    primary_view:View|None
    primary_count:Count|None
    supplementary_views:list[View]
    @model_validator(mode="after")
    def check_primary(self):
        if self.primary_view is None:
            if self.primary_count is not None:raise ValueError("无主观察时数量必须null")
        elif self.primary_count is None or self.primary_count!=getattr(self,self.primary_view):
            raise ValueError("主观察数量与对应视图不一致")
        if len(set(self.supplementary_views))!=len(self.supplementary_views) or self.primary_view in self.supplementary_views:
            raise ValueError("补充视图重复或包含主观察")
        return self
class Item(Strict):
    name:str
    score:float|None=Field(ge=0,le=100)
    severity:Literal["未见明显","轻度","中度","较明显","显著"]|None
    score_view:Literal["front"]="front"
    score_basis:Literal["v2_visible_pores","v2_visible_spots_component","v2_oiliness_tendency"]
    total_count:Counts
    regions:list[RegionResult]
    unassigned_count:Counts
    @model_validator(mode="after")
    def reconcile(self):
        if (self.score is None)!=(self.severity is None):raise ValueError("评分和程度必须同时存在或为空")
        if len({r.region for r in self.regions})!=len(self.regions):raise ValueError("重复区域")
        for view in ("left","front","right"):
            total=getattr(self.total_count,view)
            remaining=getattr(self.unassigned_count,view)
            if total is None:
                if remaining is not None or any(getattr(r,view) is not None for r in self.regions):
                    raise ValueError("未评估视图不能带数值")
            elif remaining is None or sum(getattr(r,view) or 0 for r in self.regions)+remaining!=total:
                raise ValueError("分区数量与全图总数不一致")
        return self
class Results(Strict):
    pores:Item
    spots:Item
    surface_gloss:Item
class Success(Strict):
    schema_version:Literal["shuiguang_cloud_v1"]="shuiguang_cloud_v1"
    task_id:Identifier
    status:Literal["success"]="success"
    main_image_id:str
    views:Views
    results:Results
    @model_validator(mode="after")
    def check_views(self):
        if self.main_image_id!=self.views.front:raise ValueError("主图必须等于正面图")
        if len(set(self.views.model_dump().values()))!=3:raise ValueError("三视图图片ID不能重复")
        expected={"pores":"v2_visible_pores","spots":"v2_visible_spots_component","surface_gloss":"v2_oiliness_tendency"}
        for key,basis in expected.items():
            if getattr(self.results,key).score_basis!=basis:raise ValueError("评分来源与项目不匹配")
        return self
class ErrorInfo(Strict):
    code:str
    message:str
class Failure(Strict):
    schema_version:Literal["shuiguang_cloud_v1"]="shuiguang_cloud_v1"
    task_id:str
    status:Literal["failed"]="failed"
    error:ErrorInfo
