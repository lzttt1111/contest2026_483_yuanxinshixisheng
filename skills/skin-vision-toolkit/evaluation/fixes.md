# 独立评审修复记录

仅修改Skill沙箱。

- R1：未知rerun ID报错，不进入safe列表。
- R2：缺失人工状态报缺证据，safe为空；已明确布尔False才是未确认。
- R3：完成标记严格布尔，workers/max_workers正整数；running恢复增加执行者停止/租约失效前提。
- R4：任一接口阻断时point_transfer_allowed=false，并保留requires_live_verification。
- R5：缺unit报告missing_unit、要求补证据。
- R6：clock字段明确为域身份而非类型，跨设备同类时钟仍需映射。
- R7：三元素张量例明确为摘要，完整NCHW实际需要四维及真实图证据。
- 同类问题：所有检查器声明的布尔字段统一严格验证，capture min_frames不能接受布尔。
- 文档收口：CLI的fail/invalid_input与函数异常处理语义统一。

修复前5项实际失败回执：review_regressions_before.json；修复后12项：review_regressions.json。独立复核报告保留全部判定与局限。
