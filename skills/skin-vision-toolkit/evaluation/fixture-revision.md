## 题面修订1

2026-09-10，主代理在编写完整Skill后、首次运行验收前发现02-normal原题面将human_confirmed=true样本列入rerun_ids，却预期pass，与人工标注保护原则矛盾。只将该normal题面的rerun_ids改为空，失败题面和预期规则不变；原题面保存在cases-v1.json。不能通过放松代码保护来迎合错误fixture。后续独立验证使用修订后题面，其余输入不变。
