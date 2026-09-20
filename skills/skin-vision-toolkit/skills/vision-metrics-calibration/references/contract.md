# 输入合同与结果解读

数值参数不接受JSON布尔值或数字字符串；浮点量必须有限。整数型计数/版本/方向使用严格整数，yaw_sign只接受-1或1；物理标尺为正数、数值容差为非负数。二值Mask和明确的状态布尔字段仍按各自合同接受布尔值。

可选observed_domain_pixels是已实际检测的域上限，declared分母大于它时报告expanded_denominator。可选declared_p50/declared_p90与values重新计算的分位数比较；不能通过平均已有分位数得到合并分位数。

本文件是只读检查器的适配合同，不是强制产品接口。

valid_pixels和positive_pixels为整数；values是一组真实实例数值，declared_count与其长度同口径。密度输出图像域每10万像素；单位换算需要physical_scale。P50/P90采用排序后线性插值，空values返回null。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "metrics",
  "valid_pixels": 100,
  "positive_pixels": 20,
  "values": [
    1,
    2,
    3,
    4
  ],
  "declared_count": 4,
  "cross_view_unique_claim": false,
  "correspondence_verified": false,
  "unit": "pixel",
  "physical_scale": null
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为metrics时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
# 0.1.3 输入与证据边界补充

unit 支持 pixel/pixels/px/px2/count/ratio，以及 mm/mm2/um/um2/cm/cm2/m/m2；其他单位先由适配器显式规范化，不猜测。所有物理单位必须提供正数 physical_scale（单位对应的每像素尺度），仅审声明，不转换实例量；values 应已处于声明单位。分位数用加权插值避免差值溢出。
