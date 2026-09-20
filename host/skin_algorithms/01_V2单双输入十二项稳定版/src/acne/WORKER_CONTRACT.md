# 痤疮Worker接口合同

## 信封格式（worker 返回）

acne worker 的正式成功任务必须返回统一六字段信封。v1与v2共用同一
`src.acne.worker:celery_app`，但Queue、Task、schema和公开raw_result不同。

| 字段 | 类型 | 说明 |
|------|------|------|
| `record_id` | str | 任务追踪 ID |
| `status` | str | 任务状态：`"success"` / `"failed"` |
| `schema_version` | str | 数据合同版本：v1任务=`"1"`，acne_v2=`"2"` |
| `meta_data` | dict | v1=`{"name":"acne","version":"1"}`；v2 version=`"2"` |
| `raw_result` | dict | v1保留历史完整结构；v2严格为5个公开键 |
| `debug_info` | dict | 调试信息（展示结果、版本、耗时） |

### v1信封结构示例

```json
{
  "record_id": "6a61770e...",
  "status": "success",
  "schema_version": "1",
  "meta_data": {"name": "acne", "version": "1"},
  "raw_result": {
    "acne_summary": "acne/v1/.../acne_summary.jpg",
    "acne_circles": "acne/v1/.../acne_circles.json",
    "acne_count": 15,
    "region_counts": [...],
    "量化结果": {...}
  },
  "debug_info": {
    "display_result": "...",
    "algorithm_version": "1",
    "elapsed_seconds": 8.3
  }
}
```

### acne v2精简公开合同

- Queue：`acne_v2`
- Task：`dermavision.analyze_image`
- 四参数：`task_id, oss_key, oss_result_prefix, algorithms`
- `schema_version="2"`，`meta_data={"name":"acne","version":"2"}`
- `raw_result`精确键：`overlay/metrics/quality_score/quality_status/quality_flags`
- `metrics`精确包含`疑似痤疮数量`六分区和`痤疮严重程度等级`。
- formal-fast跳过MaxRecall；不得返回热力图、模型路径或内部调试对象。
- 普通用户加法Queue为`consumer_acne_v2`，Task和四参数不变；Profile来自Queue/进程环境，公开raw_result仍精确为上述5键。
- consumer标准化人脸使用公共毛发Mask上限；institution`acne_v2`保持历史Mask行为。

### raw_result 字段

**OSS key 字段**（17 个，后端 `to_response` 转预签名 URL）：
`acne_summary` / `acne_circles` / `acne_raw_boxes` / `acne_detections` / `acne_skin_mask` / `acne_forbidden_mask` / `acne_standardized` / `max_recall_heatmap` / `diffuse_erythema_heatmap` / `focal_candidate_heatmap` / `max_recall_circles` / `combined_candidate_circles` / `max_recall_debug` / `max_recall_json` / `original_yolo_circles` / `original_unsupervised_circles` / `original_combined_circles`

**结构化数据**（10 个，前端直接使用）：
`acne_presence` / `acne_count` / `region_counts` / `grading_status` / `grading_reason` / `detector_status` / `detector_reason` / `input_mode` / `detection_scope` / `量化结果`

### v1/v2版本边界

acne v1的`schema_version`、`meta_data.version`和`debug_info.algorithm_version`
固定为`"1"`；acne_v2的`schema_version`和`meta_data.version`固定为`"2"`。
两者都不从`settings.algorithm_version`动态读取。

### 失败返回

失败时返回 `{"record_id": ..., "status": "failed", "error_code": "...", "error_message": "..."}`，后端 `_check_group` 检查 status 后直接报错。

---

## 部署注意

acne worker 与 ai-skin-backend 需要同时部署。版本不匹配会导致：
- worker 返回旧格式 -> ai-skin-backend `validate_worker_result` 报错
- ai-skin-backend 未更新 -> 读取返回空数据
