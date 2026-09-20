---
name: vision-metrics-calibration
description: "Define and audit visual metrics, denominators, per-view aggregation and versioned calibration. Use for counts, coverage, density, percentiles and score provenance; not detector tuning or automatic clinical grading."
---

# 量化统计与评分标定

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 明确检测实例、有效域、单位、缺失状态和统计对象；分母必须是本项真正检测过的区域，不能扩大到未检测皮肤。
2. 单视角观察和分区正式汇总分层；无可靠病灶对应时不跨视图相加成全脸唯一数量。主视角按检测项和局部质量选择，辅助发现单独显示。
3. 分位数从选定原始实例/像素重算，不平均分位数。物理单位依赖真实标尺，空域/缺证据返回不可评估。
4. 仅在明确有评分任务时流式建立质量/设备分层参考分布，隔离开发确认集及正式/候选配置，记录权重来源和版本。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

指标字典、分母与单位检查、统计对账、参考分布、候选配置和差异解释。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

水光当前无总分需求，不因套用旧方案新建评分。工程默认权重不冒充医生标定，视觉响应不等同于含水量或病理严重度。

## 离线辅助检查

工具计算单个观察域的样本分位数和数量/面积统计，并拒绝无对应的全脸唯一声明；不自动选择主视角或拟合评分权重。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
