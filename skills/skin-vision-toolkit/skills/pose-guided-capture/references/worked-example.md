# 完整产物示例：稳定拍照与重拍

合成配置：front目标0度，yaw容差5度，pitch/roll上限10度，稳定200ms、至少3帧，间隔不超过150ms，年龄不超过100ms。

回放：t=0/100/200三帧同一face，raw/stable角度均为0。第三帧触发front。保存规范是第三帧原始字节、frame_id、时间域、raw/stable角度和capture revision，不取后续截图。

状态机：等待有效脸→累积稳定窗口→完成槽位；旧帧/失锁/换人中断窗口。重拍front清除此槽并增加采集revision；t=300/400/500重新稳定后保存新第三帧。旧分析revision与新revision不符，应标失效而非覆盖。

镜像回放：同样raw yaw=+45，显式yaw_sign=-1映射到left=-45；设错为+1时不能触发该left槽。参数是本例输入，不是所有设备固定要求。
