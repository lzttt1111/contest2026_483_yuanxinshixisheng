# 输入合同与结果解读

数值参数不接受JSON布尔值或数字字符串；浮点量必须有限。整数型计数/版本/方向使用严格整数，yaw_sign只接受-1或1；物理标尺为正数、数值容差为非负数。二值Mask和明确的状态布尔字段仍按各自合同接受布尔值。

本文件是只读检查器的适配合同，不是强制产品接口。

三个Mask同形、二值，instances为栅格整数中心x/y。输出valid_pixels、support_pixels、retained_ratio及coverage。实际大图应保存证据并调用真实算法适配，不能把这个小栅格检查器当检测模型。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "feature",
  "reference_skin": [
    [
      1,
      1
    ],
    [
      1,
      1
    ]
  ],
  "valid_mask": [
    [
      1,
      1
    ],
    [
      1,
      1
    ]
  ],
  "detection_mask": [
    [
      1,
      0
    ],
    [
      0,
      0
    ]
  ],
  "instances": [
    {
      "id": "a",
      "x": 0,
      "y": 0
    }
  ],
  "min_retained_ratio": 0.75
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为feature时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
