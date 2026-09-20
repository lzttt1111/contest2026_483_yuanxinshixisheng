# 皱纹Worker接口合同

## 信封格式（worker 返回）

wrinkle worker 的 `analyze_image` 正式成功任务必须返回统一六字段信封：

| 字段 | 类型 | 说明 |
|------|------|------|
| `record_id` | str | 任务追踪 ID |
| `status` | str | 任务状态：`"success"` / `"failed"` |
| `schema_version` | str | 数据合同版本，当前为`"1"` |
| `meta_data` | dict | 算法元数据：`{"name": "wrinkle", "version": "1"}`。version 写死 `"1"` |
| `raw_result` | dict | 13个历史OSS键+6个结构化字段+医学V2 JSON/CSV |
| `debug_info` | dict | 调试信息（展示结果、版本、耗时、输出目录） |

### 信封结构示例

```json
{
  "record_id": "6a61770e...",
  "status": "success",
  "schema_version": "1",
  "meta_data": {"name": "wrinkle", "version": "1"},
  "raw_result": {
    "analysis_face": "wrinkle/v1/.../analysis_face.jpg",
    "preprocessed_face": "wrinkle/v1/.../preprocessed_face.jpg",
    "region_overlay": "wrinkle/v1/.../region_overlay.jpg",
    "region_metrics": [...],
    "run_preset": "...",
    "device": "...",
    "successful_runs": 3,
    "failed_runs": 0,
    "region_analysis_status": "completed"
  },
  "debug_info": {
    "display_result": "...",
    "algorithm_version": "1",
    "elapsed_seconds": 12.5,
    "output_dir": "/tmp/wrinkle_xxx"
  }
}
```

### raw_result 字段

**OSS key 字段**（13 个，后端 `to_response` 转预签名 URL）：
`analysis_face` / `preprocessed_face` / `stage1_candidates` / `vote_heatmap` / `stage2_overlay` / `stage2_centerline` / `face_filter_debug` / `texture_reference` / `comparison` / `region_overlay` / `region_tiles` / `region_metrics_csv` / `summary_json`

**结构化数据**（6 个，前端直接使用）：
`region_metrics` / `run_preset` / `device` / `successful_runs` / `failed_runs` / `region_analysis_status`

**医学V2加法字段**：`medical_metrics_v2` / `medical_report_csv_v2`。已有字段不得删除、改名或换类型。

### version 写死 "1"

`meta_data.version` 和 `debug_info.algorithm_version` 都写死 `"1"`，不再从环境变量 `settings.algorithm_version` 读取。

### consumer加法Queue

- 普通用户Queue为`consumer_wrinkle`，Task、四参数、六字段信封和全部raw_result键不变。
- consumer标准化人脸使用公共毛发Mask作为检测上限；institution`wrinkle`保持历史Mask行为。
- Profile只由Queue/进程环境指定，禁止图片像素自动分流。

### 失败返回

失败时返回 `{"record_id": ..., "status": "failed", "error_message": "..."}`，后端 `_check_group` 检查 status 后直接报错。

---

## 部署注意

wrinkle worker 与 ai-skin-backend 需要同时部署。版本不匹配会导致：
- worker 返回旧格式 -> ai-skin-backend `validate_worker_result` 报错
- ai-skin-backend 未更新 -> 读取返回空数据
