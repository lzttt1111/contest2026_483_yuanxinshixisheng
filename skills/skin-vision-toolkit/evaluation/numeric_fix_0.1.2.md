# 0.1.2 数值合同修复

来源：水光主对话复测0.1.1指出physical_scale=true、max_abs_tolerance=true、yaw_sign=true被接受；本分支在修改前已独立复现。

修复范围：
- physical_scale非空时必须有限且大于0；布尔和数字字符串拒绝。
- max_abs_tolerance必须有限非负；yaw_sign必须严格整数-1或1。
- 同类角度、帧数值、槽位、坐标、分位数声明、保留比例、版本号排除布尔；二值Mask和合法状态布尔仍允许。
- 报告投影不再利用Python中true==1的宽松比较，保留JSON类型。

52项CLI测试覆盖原3问题、True/False/字符串/NaN/Inf、无效范围及合法正数/零容差/±1方向。Windows和Linux均通过；原30+迁移30及两组12项测试继续通过。独立复核回执在independent_numeric_recheck.json。

没有修改冻结原30题或源项目。0.1.0/0.1.1归档保留，最终提交使用0.1.2解压后的源文件。未推送、未全局安装、未改Hook。
