# 输入合同与结果解读

数值参数不接受JSON布尔值或数字字符串；浮点量必须有限。整数型计数/版本/方向使用严格整数，yaw_sign只接受-1或1；物理标尺为正数、数值容差为非负数。二值Mask和明确的状态布尔字段仍按各自合同接受布尔值。

capture_revision可选，默认0且必须为非负整数；任一retake或reset_all请求使其增加1。输出新版本用于上层判旧结果失效，不会自行修改任何产品结果文件。

本文件是只读检查器的适配合同，不是强制产品接口。

config定义slots名称到角度、yaw_sign(±1)、各容差与ms预算。frames含递增t、age_ms、valid、face、yaw、raw_yaw、pitch、roll；所选yaw_sign同时作用raw/stable。换人必须先显式reset_all，不能自动完成同次检查。complete=false不代表算法失败，可能只是尚未完成稳定采集。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "capture",
  "config": {
    "slots": {
      "front": 0
    },
    "yaw_tolerance": 5,
    "pitch_limit": 10,
    "roll_limit": 10,
    "stable_ms": 200,
    "min_frames": 3,
    "max_gap_ms": 150,
    "max_age_ms": 100,
    "yaw_sign": 1
  },
  "frames": [
    {
      "t": 0,
      "age_ms": 0,
      "valid": true,
      "face": "p1",
      "yaw": 0,
      "raw_yaw": 0,
      "pitch": 0,
      "roll": 0
    },
    {
      "t": 100,
      "age_ms": 0,
      "valid": true,
      "face": "p1",
      "yaw": 0,
      "raw_yaw": 0,
      "pitch": 0,
      "roll": 0
    },
    {
      "t": 200,
      "age_ms": 0,
      "valid": true,
      "face": "p1",
      "yaw": 0,
      "raw_yaw": 0,
      "pitch": 0,
      "roll": 0
    }
  ]
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为capture时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
# 0.1.3 输入与证据边界补充

config、slots 必须为对象，slots 非空且槽位名非空；frames 为对象列表。此回放器不验证原始帧唯一性，适配器必须去重 frame_id；complete 才表示槽位完成，status=pass 不等于已经拍照。
