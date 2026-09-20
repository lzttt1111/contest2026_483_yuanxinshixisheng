# DermaVision Worker接口合同

## 信封格式（worker 返回）

dermavision worker 的 `analyze_image` 任务必须返回统一信封格式。信封包含 6 个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `record_id` | str | 任务追踪 ID，同时用作 OSS 结果路径组件。后端 AlgorithmEnvelope 接受此字段但不参与核心校验 |
| `status` | str | 任务状态：`"success"` / `"failed"`。后端 `_check_group` 检查此字段，failed 会直接报错 |
| `schema_version` | str | **数据格式版本**，从 aisia-contracts 的 `SCHEMA_VERSION` 常量取（SSOT）。backend 用它找对应 pydantic model 校验。详见 包内 Pydantic 合同 |
| `meta_data` | dict | 算法元数据：`{"name": <算法名>, "version": "1"}`。version 是算法 git 版本（信息性），与 schema_version 不同 |
| `raw_result` | dict | 算法原始结果（OSS key 形式的图片路径 + 量化指标 + 质量信息）。后端读取时调 `to_response` 转换为前端格式 |
| `debug_info` | dict | 调试信息（CSV 报告路径、计时、执行模式）。不直接给前端，由 `to_response` 提取需要的字段 |

### 信封结构示例

```json
{
  "record_id": "6a61770e3423989c53a05d63",
  "status": "success",
  "schema_version": "2",
  "meta_data": {
    "name": "redness",
    "version": "1"
  },
  "raw_result": {
    "overlay": "dermavision/v1/6a61770e.../03_RBX红区结果图.jpg",
    "metrics": { ... },
    "quality_score": 74.89,
    "quality_status": "WARNING",
    "quality_flags": ["low_contrast"]
  },
  "debug_info": {
    "report_csv": "dermavision/v1/6a61770e.../redness_report.csv",
    "timing_seconds": { "download": 0.5, "pipeline": 3.2, "upload": 0.3 },
    "execution_mode": "single_algorithm"
  }
}
```

### raw_result 字段（各算法差异）

| 算法 | 共有字段 | 特有字段 |
|------|----------|----------|
| redness | overlay, metrics, quality_score, quality_status, quality_flags | red_areas_overlay |
| spots | overlay, metrics, quality_score, quality_status, quality_flags | - |
| brown | overlay, metrics, quality_score, quality_status, quality_flags | brown_spots_overlay |
| texture | overlay, metrics, quality_score, quality_status, quality_flags | - |
| pores | overlay, metrics, quality_score, quality_status, quality_flags | - |
| purple | metrics, quality_score, quality_status, quality_flags | uv_base, uv_spots_overlay, fluorescence_base, porphyrin_overlay |
| surface_gloss | overlay, metrics, quality_score, quality_status, quality_flags | medical_report_csv_v2 |
| vascular | overlay, metrics, quality_score, quality_status, quality_flags | medical_report_csv_v2 |
| contour_firmness | overlay, metrics, quality_score, quality_status, quality_flags | medical_report_csv_v2 |

- `overlay`: OSS key（结果图），后端 `to_response` 转预签名 URL
- `metrics`: 量化指标 dict，redness/spots 为完整英文 dict 透传，
  brown/texture/pores 为历史精简中文 dict，新增三项为专用Pydantic定义的精简中文 dict。
- `quality_score`/`quality_status`/`quality_flags`: 图片质量信息
- `purple` 为独立新增算法，正式返回四张图：UV底图、UV色斑实例图、
  荧光UV底图和紫质实例图；现有五项字段保持不变。

### schema_version 与 meta_data.version

两个 version 含义不同，不要混：

- **`schema_version`**（信封顶层）：**数据格式版本**。redness/spots/brown/texture/pores = `"2"`；purple/acne/wrinkle 及新增三项 = `"1"`。新增三项必须在 `aisia-contracts@v0.2.0` 先发布后才可部署 Worker。
- **`meta_data.version`**（信封内）：**算法 git 版本**，信息性，当前统一写 `"1"`。backend 落库记录，不做结构判断。

worker 从 contract 导入 `SCHEMA_VERSION` 常量填顶层 `schema_version`，`meta_data.version` 保持 `"1"`。详见 包内 Pydantic 合同。

⚠️ **铁律**：contract 的 `v{N}.py` 发布（打 tag）后不可修改。加字段/改字段 = 新建 `v{N+1}.py` + bump `SCHEMA_VERSION` + contract 打新 tag + 两消费方更新 `@tag`。

### 多算法兼容模式

`_build_envelope_response` 仅用于单算法任务（`len(algorithms) == 1 and algorithms[0] in _ENVELOPE_ALGOS`）。多算法兼容模式（理论上不会走到）返回旧格式，后端 `validate_worker_result` 会报错。

### 失败返回

失败时返回 `{"record_id": ..., "status": "failed", "error_message": "..."}`，后端 `_check_group` 检查 status 后直接报错，不走 `validate_worker_result`。

---

## 部署注意

dermavision 与 ai-skin-backend 需要同时部署。版本不匹配会导致：
- worker 返回旧格式 -> ai-skin-backend `validate_worker_result` 报错（pydantic `extra="forbid"`）
- ai-skin-backend 未更新 -> 读取返回空数据（`to_response` 从 `raw_result` 取字段，旧数据没有 `raw_result`）

## 2026-08-24 双Profile加法合同

- 现有`redness/spots/brown/texture/pores/purple/surface_gloss/vascular/contour_firmness` Queue永久代表`institution`，不得改名或改变厂家机构效果。
- `consumer_*`为普通用户加法Queue；Task、四参数、`schema_version`、`meta_data`和`raw_result`字段集合与对应机构算法相同。
- Profile由Queue显式选择，禁止根据分辨率、EXIF或像素内容自动推断。
- consumer DermaVision的既有`debug_info.execution_mode`值为`single_algorithm_consumer`；不得增加前端字段或公开内部Mask。
- institution继续要求厂家RED/BROWN sidecar；consumer红棕和血管禁止调用sidecar。
- 本地两套Profile统一使用`production_proxy_v1`和同一137项显式公式；只有完整证据的模块可`score_valid=true`，缺证据必须`score=null/grade=不可评估/score_valid=false`。当前仅3个冻结锚点、非人群常模；普通用户本地review默认生成Word，云端Worker默认不生成。
