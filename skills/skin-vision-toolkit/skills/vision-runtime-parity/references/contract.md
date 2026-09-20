# 输入合同与结果解读

数值参数不接受JSON布尔值或数字字符串；浮点量必须有限。整数型计数/版本/方向使用严格整数，yaw_sign只接受-1或1；物理标尺为正数、数值容差为非负数。二值Mask和明确的状态布尔字段仍按各自合同接受布尔值。

可选coordinate_case用于检查二维letterbox还原：scale为正缩放比，pad_xy=[左padding,上padding]，model_points与original_points为同长非空xy列表，tolerance_pixels为非负误差门槛。先执行(x-pad_x)/scale、(y-pad_y)/scale；若提供正整数mirror_width，再按width-1-x反镜像。输出还原点和欧氏误差，不估算真实世界坐标；不支持的仿射/透视映射需另行提供已验证变换，不自动近似。

本检查器比较的是已保存的数值快照，不解析模型图或证明布局标签真实。下方三元素shape=[1,3]是抽象通道摘要，layout表示来源声明，不是完整NCHW张量轴定义；真实NCHW完整张量应提供N/C/H/W四维shape及对应全部值。简化样例只验证数值与声明一致，不可用于证明完整模型布局迁移。

本文件是只读检查器的适配合同，不是强制产品接口。

source/target含shape、layout、color、values(展平)；max_abs_tolerance显式给定。RMSE和最大绝对误差由真实数值计算。provider声明必须与实际回执对照；calibration_ids/test_ids用于泄漏检查。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "parity",
  "source": {
    "layout": "NCHW",
    "color": "RGB",
    "shape": [
      1,
      3
    ],
    "values": [
      0.1,
      0.2,
      0.3
    ]
  },
  "target": {
    "layout": "NCHW",
    "color": "RGB",
    "shape": [
      1,
      3
    ],
    "values": [
      0.1001,
      0.2001,
      0.3001
    ]
  },
  "max_abs_tolerance": 0.001,
  "runtime_provider": "CPU",
  "claimed_provider": "CPU",
  "calibration_ids": [
    "train1"
  ],
  "test_ids": [
    "test1"
  ]
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为parity时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
# 0.1.3 输入与证据边界补充

layout 限 NCHW/NHWC/CHW/HWC/NC/C/HW；color 限 RGB/BGR/GRAY/RGBA/BGRA/none。provider 为非空后端声明（不代表实测）。摘要张量不执行布局维度推断。数值差溢出返回 invalid_input；RMSE 使用缩放计算。坐标恢复或其他派生值超出有限范围时 CLI 返回结构化 invalid_input。
