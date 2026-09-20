"""本地合同测试的离线版本常量。

生产 Worker 仍必须安装锁定的 ``aisia-contracts``；此处只复用云端
模拟器已有的版本注入，使无 Codeup 权限的离线验收仍可校验本地严格合同。
"""

from cloud.simulated_contract_versions import install_simulated_contract_versions


install_simulated_contract_versions()
