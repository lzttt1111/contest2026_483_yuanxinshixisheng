---
name: vision-result-publication
description: "Publish consistent visual-AI JSON/CSV, overlays, web views and reports from one result source. Use for artifact mapping and report-only changes; not inference, score recomputation or generic copywriting."
---

# 结果图与报告同源交付

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 指定本次唯一结果索引、指标与图片版本；分清只改显示和需要改变检测的任务。
2. 报告和网页引用同次已有证据，保证数字、等级、缺项状态和图片哈希一致；不在报告阶段重新运行模型或评分。
3. 医生模板版本由当前任务指定，建立检测项到章节映射；不可用项不变成正常结论。
4. 只在隔离复制件上做重建/渲染验收，确认过程不加载模型；检查链接、单位、图例、长文本、表格和最终版式。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

结果索引、字段/章节映射、文件哈希、同源检查、展示产物及重建回执。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

不能复用历史图片冒充新推理；标题含audit不表示只读，先审查脚本副作用。脚本一致不替代DOCX/PDF实际渲染检查。

## 离线辅助检查

检查器读取显式root下两个JSON投影及引用文件，比较实际内容并计算SHA；默认只打印JSON，不写原文件。--root必须是本次允许的副本目录。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
