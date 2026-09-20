> 最新修复已生效，当前行为与验收请先看 [修复说明_20260920.md](../修复说明_20260920.md)。固定样例回退已停用，正面采集槽位已新增；下文中对应旧限制和上线声明为历史说明。

# 水光三项服务：BiSeNet三项兼容上线版

当前只运行毛孔、独立可见色斑、纯表面油光。默认预处理为BiSeNet；不设置环境变量也不会回退旧毛发规则。旧规则仅保留为显式历史对照，不作为新服务默认值。

## 从这里开始

- 完整启动与逐字段说明：[水光三项检测技术与部署说明_20260915.md](水光三项检测技术与部署说明_20260915.md)
- 模型来源、SHA与用途：[模型资产与用途说明.md](模型资产与用途说明.md)
- 真实前后评分对照：[BiSeNet评分诊断对照_20260915.md](BiSeNet评分诊断对照_20260915.md)
- 选型结论：[预处理五十图方案选择结论_20260915.md](预处理五十图方案选择结论_20260915.md)

## 当前可运行入口

诊断Celery任务：`shuiguang.analyze_three_views_diagnostic`，队列：`shuiguang_diagnostic`。
正式三项任务：`shuiguang.analyze_three_views_scores`，队列：`shuiguang_scores`。
HTTP：`POST /api/diagnostic-jobs`，用返回的queue_id查询。
CLI单次冒烟测试（使用包内第一组样例，需已安装锁定依赖与CUDA）：

```bash
SHUIGUANG_INPUT_ROOT="$PWD/examples" SHUIGUANG_CLOUD_DEVICE=cuda:0 uv run --no-sync python run_diagnostic_json.py --request request_scores_example.json --output runtime/example_diagnostic.json
```

CLI每次新进程都可能冷启动；连续任务应使用常驻Celery worker，不用CLI反复启动来评估稳态性能。
模型已放在`vendor/v3/models/face_parsing_resnet18.onnx`；不要从沙箱外导入业务代码或模型。

诊断响应包含用途标识和三项scores。高分状态更好；区域评分只使用正面，色斑不进入Brown/UV混合评分链。

## 当前评分兼容策略

本次不重跑1000例。新BiSeNet预处理仍实际运行三项检测；凡新预处理没有分区参考而产生的`null`，由包内 `vendor/legacy_score_fallback.json` 读取已经保存的旧评分补齐。回退只填缺失值，不覆盖新结果，并在 `scoring_evidence_trace.json` 留下使用清单。这样可立即返回完整的整体分、分区分和严重程度。

因此当前结果的 `reference_matches_new_preprocessing` 仍为 `false`：它是“新预处理量化 + 旧参考缺项回退”的可上线兼容结果，不应描述为已完成新参考标定。待以后获得时间，可只重建参考配置再关闭回退。默认启用回退；设置 `SHUIGUANG_LEGACY_SCORE_FALLBACK=0` 会重新启用正式参考门禁并返回 `SCORING_REFERENCE_NOT_READY`。

`examples/diagnostic_one/result.json`和`diagnostic_two/result.json`是两组真实Celery诊断返回。
`examples/current_demo`保留较早历史结果（包括当时撤回色斑评分的null），不能当成当前诊断接口的唯一示例。

本轮真实Celery使用本地文件系统测试传输，两组三视图、正式入口、常驻复用、幂等和回退补齐均通过；生产部署仍需把broker/backend换成Redis。1000例没有重跑，也没有推送远端。
