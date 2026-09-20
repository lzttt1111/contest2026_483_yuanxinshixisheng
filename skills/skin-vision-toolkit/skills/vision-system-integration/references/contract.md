# 输入合同与结果解读

数值参数不接受JSON布尔值或数字字符串；浮点量必须有限。整数型计数/版本/方向使用严格整数，yaw_sign只接受-1或1；物理标尺为正数、数值容差为非负数。二值Mask和明确的状态布尔字段仍按各自合同接受布尔值。

source_clock/target_clock必须是时钟域身份（设备/进程或共享时钟源及启动epoch），不是单纯的monotonic/unix类型名称。合成样例用monotonic作为同一已约定域的短别名；两台设备各自的monotonic应写成不同域ID并验证映射。仅字符串一致不证明实物时钟同步，point_transfer_allowed只是给定快照内未发现阻断，仍需实际坐标/时钟证据。

本文件是只读检查器的适配合同，不是强制产品接口。

producer_fields/consumer_required是实际适配合同字段清单；current_revision/result_revision同一采集组内比较。不同frame点叠加需要已验证配准；不同时钟需要映射。脚本不自动造新API或修代码。

输入样例（合成数据，不是临床或硬件实测）：

~~~json
{
  "task_kind": "integration",
  "producer_fields": [
    "capture_id",
    "revision",
    "frame_id"
  ],
  "consumer_required": [
    "capture_id",
    "revision"
  ],
  "current_revision": 2,
  "result_revision": 2,
  "source_frame": 10,
  "target_frame": 10,
  "point_overlay": true,
  "registration_verified": false,
  "source_clock": "monotonic",
  "target_clock": "monotonic",
  "clock_mapping_verified": false,
  "mode_switch_clears_photos": true,
  "document_claims_preserved": false
}
~~~

CLI输出包含status、findings(code/detail)和metrics。结构/类型无效或无法继续计算时invalid_input；能明确定位的业务缺证据可返回fail并给出finding（如缺单位、未知人工状态），两者均不是通过。task_kind不为integration时not_applicable。直接调用check函数需处理ValueError/TypeError等异常，CLI main已处理。实际任务需另行核对数据来源真实性。

所有工具只读取显式输入，默认不更改任何文件。大规模清单先形成限定快照；输入上限32MiB防止误读原始长日志。需要批量检查时在获授权后按明确分片调用，不由工具自行扫描项目。
# 0.1.3 输入与证据边界补充

帧身份必须为非空字符串或非负整数；时钟域身份必须非空。producer_fields、consumer_required 必须为非空字符串列表。身份来自上游实际实例，不能把通用时钟类型名当成已验证的同一时钟。检查器不验证身份真实性。
