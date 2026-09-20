---
name: vision-runtime-parity
description: "Verify visual-model portability across runtimes with tensor, preprocessing, decoding, coordinate and provider evidence. Use for ONNX/CPU/mobile/RKNN/C++ conversion checks; not training or claiming hardware results from simulation."
---

# 模型跨运行时迁移

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 冻结源模型、输入、模型SHA及预后处理合同：颜色、布局、归一化、letterbox、ROI、输出头与坐标空间。
2. 从相同输入逐层对拍输入张量、原始输出、解码和最终坐标；先找差异层，不靠最终截图相似判断。
3. 先验证浮点路线，再在授权下量化；校准只用规定训练数据，阈值由当前任务确定。
4. 记录实际provider、冷/热启动、资源与时效。PC、模拟器、实板分开；发布清单记录已验证范围和缺项。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

模型合同、转换/对拍记录、误差表、运行provider与资源实测、可交付模型清单。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

能加载不等于结果等价；不能把CPU回退写成NPU或把模拟器写成实板。新下载/量化/部署不因调用Skill自动获授权。

## 离线辅助检查

比较小型已保存张量数组及声明合同，不加载ONNX/PyTorch；合同不匹配时数值差只作诊断，numeric_comparison_valid=false。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
