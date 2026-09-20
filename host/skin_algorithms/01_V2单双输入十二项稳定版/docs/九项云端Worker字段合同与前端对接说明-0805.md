# 九项云端 Worker 字段合同与前端对接说明-0805

## 1. 对接结论

本版不删除、不改名、不改变旧 Worker 字段类型，仅完成两项云端升级：

1. 紫区 `raw_result.metrics` 恢复为前端可直接使用的英文扁平整数字段；完整脸返回 12 个字段，局部脸返回 2 个总计字段。
2. 八个 Worker 、九项结果均增加 Pydantic 严格合同校验和中文字段说明。校验通过后返回原始 `dict`，不会重新序列化或改写旧字段。

必须按以下顺序发布：

```text
ai-skin-backend 先允许新合同
→ 前端按本文档增加可选字段读取
→ Worker 发布
```

Pydantic 在 Worker 侧只负责“声明和校验返回 JSON”，不会自动修改后端 `AlgorithmEnvelope` 或 `to_response`。后端仍需依据本文档同步其严格模型和字段映射。

## 2. 任务、队列和调用参数

八个任务对应九项结果；`purple` 一个任务同时产生 UV 色斑和紫质两项。

| 结果 | Celery Task | Queue | `algorithms` |
|---|---|---|---|
| 红区 | `dermavision.analyze_image` | `redness` | `["redness"]` |
| 斑点 | `dermavision.analyze_image` | `spots` | `["spots"]` |
| 棕区 | `dermavision.analyze_image` | `brown` | `["brown"]` |
| 纹理 | `dermavision.analyze_image` | `texture` | `["texture"]` |
| 毛孔 | `dermavision.analyze_image` | `pores` | `["pores"]` |
| UV色斑+紫质 | `dermavision.analyze_image` | `purple` | `["purple"]` |
| 痤疮 | `acne-detection-worker.analyze_image` | `acne-detection-worker` | `["acne"]` |
| 皱纹 | `wrinkle.analyze_image` | `wrinkle` | `["wrinkle"]` |

八个任务的四个参数顺序固定为：

```text
task_id, oss_key, oss_result_prefix, algorithms
```

- `task_id`：业务记录 ID，也是成功信封的 `record_id`。
- `oss_key`：输入图片 OSS 对象键。
- `oss_result_prefix`：可选输出 OSS 前缀。
- `algorithms`：单元素列表，必须与当前 Queue 匹配。

### 2.1 代码位置与并行启动

```text
src/worker.py          八任务统一注册表 + DermaVision六队列Worker
src/acne/worker.py     痤疮Worker
src/wrinkle/worker.py  皱纹Worker
run_cloud_worker.py    八个目标的统一启动器
```

生产并行运行不是调用一个新的聚合任务，而是同时启动八个
`concurrency=1` 的独立 Worker，由后端并发提交八个旧任务：

```bash
sudo systemctl enable --now dermavision-workers.target
```

完整运维命令见 `deploy/README.md`。

## 3. 信封合同

### 3.1 成功信封

| JSON 路径 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `record_id` | string | 是 | 任务追踪 ID |
| `status` | string | 是 | 固定为 `success` |
| `meta_data.name` | string | 是 | 当前算法名 |
| `meta_data.version` | string | 是 | 固定为字符串 `"1"` |
| `raw_result` | object | 是 | OSS 文件键、量化结果和质量信息 |
| `debug_info` | object | 是 | CSV、展示索引、耗时和执行模式 |

```json
{
  "record_id": "6a61770e3423989c53a05d63",
  "status": "success",
  "meta_data": {"name": "purple", "version": "1"},
  "raw_result": {},
  "debug_info": {}
}
```

### 3.2 失败信封

DermaVision 和皱纹的基本失败格式：

```json
{
  "record_id": "6a61770e3423989c53a05d63",
  "status": "failed",
  "error_message": "错误说明"
}
```

痤疮可额外返回 `error_code`，皱纹的部分失败路径可额外返回 `error_details`。失败信封不会伪造成功的 `raw_result`。

## 4. 六项 DermaVision `raw_result`

| 算法 | 固定字段 |
|---|---|
| redness | `overlay`, `metrics`, `quality_score`, `quality_status`, `quality_flags`, `red_areas_overlay` |
| spots | `overlay`, `metrics`, `quality_score`, `quality_status`, `quality_flags` |
| brown | `overlay`, `metrics`, `quality_score`, `quality_status`, `quality_flags`, `brown_spots_overlay` |
| texture | `overlay`, `metrics`, `quality_score`, `quality_status`, `quality_flags` |
| pores | `overlay`, `metrics`, `quality_score`, `quality_status`, `quality_flags` |
| purple | `uv_base`, `uv_spots_overlay`, `fluorescence_base`, `porphyrin_overlay`, `metrics`, `quality_score`, `quality_status`, `quality_flags` |

`overlay` 及紫区四张图字段均为 OSS key，由后端 `to_response` 转换为预签名 URL。

### 4.1 红区量化字段

红区 `raw_result.metrics` 保留旧版英文工程结构。前端核心展示字段如下：

| JSON 路径 | 中文名称 | 类型 | 单位/范围 |
|---|---|---|---|
| `raw_result.metrics.red_area_ratio` | 红区面积占比 | number | 0～1 |
| `raw_result.metrics.high_red_area_ratio` | 高红区面积占比 | number | 0～1 |
| `raw_result.metrics.mean_redness` | 平均红度 | number | 0～1工程值 |
| `raw_result.metrics.p50_redness` | P50红度 | number | 0～1工程值 |
| `raw_result.metrics.p90_redness` | P90红度 | number | 0～1工程值 |
| `raw_result.metrics.p95_redness` | P95红度 | number | 0～1工程值 |
| `raw_result.metrics.redness_burden` | 红度综合负担 | number | 工程指标 |
| `raw_result.metrics.red_feature_count` | 局灶红色实例数 | integer | 个 |
| `raw_result.metrics.red_feature_area` | 局灶红色实例总面积 | integer | 像素 |
| `raw_result.metrics.red_feature_area_ratio` | 局灶红色面积占比 | number | 0～1 |

`region_statistics` 保留分区的 `area/red_area_ratio/high_red_area_ratio/mean_redness/p50_redness/p90_redness/p95_redness/max_redness/redness_burden`。`red_feature_region_distribution` 保留分区的 `count/area/area_ratio`。`red_feature_locations` 为兼容旧字段，正式公开返回为空数组。

### 4.2 斑点量化字段

| JSON 路径 | 中文名称 | 类型 | 单位/范围 |
|---|---|---|---|
| `raw_result.metrics.spot_count` | 可见斑点总数 | integer | 个 |
| `raw_result.metrics.spot_area_ratio` | 斑点面积占比 | number | 0～1 |
| `raw_result.metrics.spot_confidence` | 平均工程置信度 | number | 0～1 |
| `raw_result.metrics.mean_deltaE` | 平均综合色差 | number | Delta E工程值 |
| `raw_result.metrics.small_spot_count` | 小型斑点数 | integer | 个 |
| `raw_result.metrics.large_spot_count` | 大片状斑点数 | integer | 个 |
| `raw_result.metrics.merged_spot_count` | 融合去重后斑点数 | integer | 个 |
| `raw_result.metrics.salient_spot_count` | 显著颜色异常候选数 | integer | 个 |
| `raw_result.metrics.analysis_zone_area` | 斑点有效分析面积 | integer | 像素 |
| `raw_result.metrics.mean_spot_area` | 平均斑点面积 | number | 像素 |
| `raw_result.metrics.median_spot_area` | P50斑点面积 | number | 像素 |

`region_distribution` 中每个分区保留 `count/area_ratio`。`spot_locations` 和 `large_spot_locations` 为兼容字段，正式公开返回空数组。其余算法参数和过滤统计仍保持旧英文字段不变。

### 4.3 棕区、纹理和毛孔用户简表

这三项保留旧合同的中文 key，不做英文改名。完整脸返回：

| 算法 | `metrics` 字段 | 含义 | 类型/单位 |
|---|---|---|---|
| 棕区 | `总计/额头/左脸颊/右脸颊/鼻部/下巴` | 棕色斑全脸总数及各区数量 | integer，个 |
| 纹理 | `总计/凸起样（黄）/凹陷样（蓝）/额头/左脸颊/右脸颊/鼻部/下巴` | 二维纹理总数、双类型数量和分区数量 | integer，个 |
| 毛孔 | `总计/额头/左脸颊/右脸颊/鼻部/下巴` | 可见毛孔全脸总数及各区数量 | integer，个 |

局部脸仅返回 `{"\u603b\u8ba1": <integer>}`。本次云端合并不修改 `src/engines/pores_engine.py`，毛孔仍使用当前 `dev` 的正式版。

### 4.4 紫区英文精简指标

完整脸必须且只能返回以下 12 个整数字段：

| JSON 路径 | 中文名称 | 详细含义 | 类型 | 单位 |
|---|---|---|---|---|
| `raw_result.metrics.uv_spots_total` | UV色斑总数 | 有效分析区域的UV色斑实例总数 | integer | 个 |
| `raw_result.metrics.uv_spots_forehead` | 额头UV色斑数 | 额头区域UV色斑实例数 | integer | 个 |
| `raw_result.metrics.uv_spots_left_cheek` | 画面左脸颊UV色斑数 | 画面左侧脸颊UV色斑实例数 | integer | 个 |
| `raw_result.metrics.uv_spots_right_cheek` | 画面右脸颊UV色斑数 | 画面右侧脸颊UV色斑实例数 | integer | 个 |
| `raw_result.metrics.uv_spots_nose` | 鼻部UV色斑数 | 鼻部区域UV色斑实例数 | integer | 个 |
| `raw_result.metrics.uv_spots_chin` | 下巴UV色斑数 | 下巴区域UV色斑实例数 | integer | 个 |
| `raw_result.metrics.porphyrin_total` | 紫质总数 | 有效分析区域的紫质实例总数 | integer | 个 |
| `raw_result.metrics.porphyrin_forehead` | 额头紫质数 | 额头区域紫质实例数 | integer | 个 |
| `raw_result.metrics.porphyrin_left_cheek` | 画面左脸颊紫质数 | 画面左侧脸颊紫质实例数 | integer | 个 |
| `raw_result.metrics.porphyrin_right_cheek` | 画面右脸颊紫质数 | 画面右侧脸颊紫质实例数 | integer | 个 |
| `raw_result.metrics.porphyrin_nose` | 鼻部紫质数 | 鼻部区域紫质实例数 | integer | 个 |
| `raw_result.metrics.porphyrin_chin` | 下巴紫质数 | 下巴区域紫质实例数 | integer | 个 |

局部脸只返回 `uv_spots_total` 和 `porphyrin_total`。紫区用户 CSV 固定两行，分别为“紫外线色斑”和“紫质”，其数值与上述 JSON 字段逐项一致。紫区不返回 `medical_metrics_v2` 或 `medical_report_csv_v2`。

完整脸 `metrics` 示例：

```json
{
  "uv_spots_total": 506,
  "uv_spots_forehead": 87,
  "uv_spots_left_cheek": 134,
  "uv_spots_right_cheek": 184,
  "uv_spots_nose": 30,
  "uv_spots_chin": 71,
  "porphyrin_total": 447,
  "porphyrin_forehead": 26,
  "porphyrin_left_cheek": 121,
  "porphyrin_right_cheek": 216,
  "porphyrin_nose": 51,
  "porphyrin_chin": 33
}
```

局部脸 `metrics` 示例：

```json
{
  "uv_spots_total": 128,
  "porphyrin_total": 94
}
```

上述字段在各自对应的完整脸/局部脸合同中均为必填字段；任意中文 key、额外 key、嵌套宽表或非整数值都会被 Pydantic 拒绝。

## 5. 痤疮 `raw_result`

### 5.1 17 个 OSS key

```text
acne_summary
acne_circles
acne_raw_boxes
acne_detections
acne_skin_mask
acne_forbidden_mask
acne_standardized
max_recall_heatmap
diffuse_erythema_heatmap
focal_candidate_heatmap
max_recall_circles
combined_candidate_circles
max_recall_debug
max_recall_json
original_yolo_circles
original_unsupervised_circles
original_combined_circles
```

### 5.2 10 个结构化字段

| JSON 路径 | 类型 | 含义 |
|---|---|---|
| `raw_result.acne_presence` | object/null | 痤疮检出状态和中文说明 |
| `raw_result.acne_count` | number/null | 疑似痤疮候选数 |
| `raw_result.region_counts` | array | 面部分区名称及候选数 |
| `raw_result.grading_status` | string/null | Acne-LDS分级状态 |
| `raw_result.grading_reason` | string/null | 未分级或异常原因 |
| `raw_result.detector_status` | string/null | 检测器执行状态 |
| `raw_result.detector_reason` | string/null | 检测器未执行或异常原因 |
| `raw_result.input_mode` | string/null | 输入图模式 |
| `raw_result.detection_scope` | string/null | 痤疮检测范围 |
| `raw_result.量化结果` | object | 旧版前端中文量化结果 |

`raw_result.量化结果.疑似痤疮圈选数量` 的单位为“个”。`raw_result.量化结果.痤疮严重程度等级.等级` 在评级成功时为 1～4 级，`.注释` 是对等级或不可评级原因的中文说明。

## 6. 皱纹 `raw_result`

### 6.1 13 个 OSS key

```text
analysis_face
preprocessed_face
stage1_candidates
vote_heatmap
stage2_overlay
stage2_centerline
face_filter_debug
texture_reference
comparison
region_overlay
region_tiles
region_metrics_csv
summary_json
```

### 6.2 6 个结构化字段

| JSON 路径 | 类型 | 含义 |
|---|---|---|
| `raw_result.region_metrics` | array | 各区皱纹量化指标 |
| `raw_result.run_preset` | string/null | 运行预设，正式保持 `balanced` |
| `raw_result.device` | string/null | 实际推理设备 |
| `raw_result.successful_runs` | integer/null | 多视图成功推理次数 |
| `raw_result.failed_runs` | integer/null | 多视图失败推理次数 |
| `raw_result.region_analysis_status` | string/null | 分区分析状态 |

`region_metrics` 的每一项包含：

| 字段 | 中文名称 | 单位 |
|---|---|---|
| `region_key` | 稳定英文分区标识 | — |
| `region_name` | 分区中文名称 | — |
| `short_name` | 分区英文缩写 | — |
| `relative_score` | 分区相对得分 | 0～100 |
| `segment_count` | 分区纹路线段数 | 条 |
| `wrinkle_pixels` | 分区皱纹中心线像素数 | 像素 |
| `mean_segment_length` | 平均线段长度 | 像素 |
| `max_segment_length` | 最大线段长度 | 像素 |
| `density_per_10k` | 分区皱纹密度 | 像素/1万分区像素 |
| `share_pct` | 分区皱纹占比 | % |
| `area_px` | 分区有效面积 | 像素 |

## 7. 医学 V2 可选字段

支持的算法可在不改写旧 `metrics` 的前提下增加：

| JSON 路径 | 类型 | 说明 |
|---|---|---|
| `raw_result.medical_metrics_v2` | object | 英文结构的医学量化增量指标 |
| `raw_result.medical_report_csv_v2` | string | 医学 V2 CSV 的 OSS key |

这两个字段是可选加法字段：生成或上传失败时可以省略，不影响旧任务成功返回。紫区明确不返回这两个字段。`meta_data.version` 仍为 `"1"`，医学指标版本由 `medical_metrics_v2.metrics_version` 单独管理。

## 8. Worker、后端和前端职责边界

| 组件 | 责任 |
|---|---|
| Worker | 运行算法，上传 OSS，构造五字段信封，在返回前通过 Pydantic 校验 |
| ai-skin-backend | 先用 `AlgorithmEnvelope` 验证信封，再通过对应版本的 `to_response` 将 OSS key 转为 URL并映射前端结构 |
| 前端 | 按稳定 JSON 路径读取展示字段，对可选字段做缺省兼容 |

禁止前端直接依赖 Pydantic 内部类名，如 `FullMedicalMetricsV2`。类名只是开发者 Schema 名称，不是生产 JSON 字段。对接应以本文档的完整 JSON 路径为准。

## 9. 本地云端合同验收

生成八个真实 Worker 的本地 OSS 模拟结果：

```bash
uv run --frozen python cloud/simulate_cloud_request.py \
  --input "<测试图片>" \
  --output output/cloud-pydantic-review-visia2-20260805
```

校验图片、CSV、JSON、五字段信封和 Pydantic 合同：

```bash
uv run --frozen python scripts/verify_cloud_pydantic_review.py \
  --root output/cloud-pydantic-review-visia2-20260805
```

启动验收页和 Swagger：

```bash
CLOUD_REVIEW_ROOT="$PWD/output/cloud-pydantic-review-visia2-20260805" \
uv run --frozen uvicorn cloud.review_api:app --host 0.0.0.0 --port 8765
```

- 验收页：`http://127.0.0.1:8765/review`
- Swagger：`http://127.0.0.1:8765/docs`
- 合同目录：`http://127.0.0.1:8765/api/contracts`
- 单任务结果：`http://127.0.0.1:8765/api/results/<algorithm>`

验收页字段表固定展示：`JSON路径 / 中文名称 / 当前值 / 数据类型 / 单位 / 详细说明 / 是否必填`。

## 10. 发布门禁和回滚

发布前必须确认：

1. 根 `WORKER_CONTRACT.md`、`src/acne/WORKER_CONTRACT.md`和 `src/wrinkle/WORKER_CONTRACT.md` 全部通过合同测试。
2. 五字段信封、Task、Queue、四参数顺序和 `version="1"` 未变。
3. 紫区完整脸只有 12 个英文整数指标，用户 CSV 只有两行数据。
4. 痤疮 17+10 和皱纹 13+6 字段不得减少。
5. `run.py`、`src/nine_analysis/*`、毛孔引擎和三份 `WORKER_CONTRACT.md` 必须与合并时的 `dev` 一致。

若后端尚未接受医学 V2 两个可选字段，只需在 Worker 环境中设置：

```text
ENABLE_MEDICAL_METRICS_V2=false
```

该回滚只关闭新增医学字段，不影响旧 `metrics`、结果图或五字段信封。
