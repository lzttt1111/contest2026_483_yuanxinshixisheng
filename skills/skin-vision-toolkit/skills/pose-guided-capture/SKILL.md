---
name: pose-guided-capture
description: "Design and replay pose-guided capture state machines, including mirror signs, raw/stable pose, freshness, identity continuity, retake and slot state. Not identity authentication or skin analysis."
---

# 姿态引导与自动采集

## 使用入口

先读取任务目标、操作性质、允许输入/基线、输出目录及资源限制。只做分析时不转为实施；任务不涉及本Skill能力时说明不适用，不强行调用工具。

## 工作流程

1. 先核对找脸、裁剪、颜色通道、归一化与角度顺序，区分相机镜像和受检者左右；方向变换写在配置中。
2. 原始角度、稳定角度和显示刷新分别保留；采集同时检查raw与stable，不能让滤波滞后触发错拍。
3. 按配置的任意槽位、稳定时长、最小帧数、间隔/时效与人脸连续性判断触发；失锁/过期/换人中断连续窗口。
4. 单侧重拍/整轮重置明确清哪些状态；保存触发帧的原始字节和frame_id，更新capture revision，不能保存下一帧截图。

## 输出与解释

需要完整产物的组织示例时，阅读[工作示例](references/worked-example.md)。示例是教学/回放材料，不代替当前任务的真实证据。

状态机、参数、回放轨迹、拍照槽位及元数据合同、重拍测试。

结论先行，并列出实际检查、证据文件/版本、未解决项。区分用户需求、计划、静态检查、模拟回放与实测，不把检查器pass解释为整个产品已通过。

## 关键边界

人脸空间锁定不是身份认证；不以单个ONNX大小推断真实FPS。没有原始角度或时间域说明时不得直接接自动拍照。

## 离线辅助检查

回放器使用显式配置，支持reset_all/retake事件标记；不会访问摄像头。需将真实状态机日志适配为frames快照，不能当作生产相机替换实现。

先读[输入合同与解读](references/contract.md)，把已有材料适配为有限JSON快照；不要求业务API迁移到该测试格式。运行：

~~~text
python -B scripts/check.py INPUT.json --root ALLOWED_ARTIFACT_ROOT
~~~

路径相对当前Skill目录；也可直接使用脚本绝对路径。Python 3.10+、仅标准库。输出stdout JSON；退出码0为pass或not_applicable，1为发现证据问题，2为输入不完整/无效。必须同时读取status，不能只看退出码。没有网络、模型、摄像头或文件写操作。

需要本项目案例时阅读[参考案例](references/project-case.md)，历史参数仅为案例，不覆盖当前用户要求。不要为小任务加载其他无关Skill。
