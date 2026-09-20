---
name: vision-generalization
description: "Review cross-subject or cross-scene visual-model experiments, leakage, checkpoint selection, synthetic data and calibration provenance. Use before training or interpreting benchmark results; never starts training itself."
---

# 模型训练与跨场景泛化

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 冻结现有划分与基线，按人物、Session、原图哈希、近重复簇检查泄漏；目录叫train不代表已经确认最终划分。
2. 分别测量检测框→ROI→关键点/分类→最终区域输出，判断上游裁剪域差还是模型误差。
3. 训练增强、合成素材和量化校准只取协议允许的数据；验证集选checkpoint，测试集按预先冻结协议评估。
4. 限定候选与资源预算，保留失败版本和困难样本。数据扩充后重新检查人物/序列泄漏，不把单人物高分宣称泛化。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

划分审计、基线实验表、分层指标、困难样本队列和模型选择依据。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

分析请求不能启动GPU、训练或改标注。不同测试集的AP/PCK不能直接排名，不能因版本号更新就设为默认。

## 离线辅助检查

脚本检查清单分组泄漏、测试选型和校准/合成来源；不评估模型精度。near-duplicate cluster必须由上游提供，未知时如实标记。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
