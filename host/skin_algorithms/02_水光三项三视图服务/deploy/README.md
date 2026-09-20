# 水光三项云端部署

## 安装目录

```text
/opt/dermavision/shuiguang-three
```

```bash
cd /opt/dermavision/shuiguang-three
uv sync --frozen
sudo install -d -m 0750 /etc/dermavision
sudo cp deploy/shuiguang.env.example /etc/dermavision/shuiguang.env
sudo cp deploy/shuiguang-score-worker.service /etc/systemd/system/
sudo cp deploy/shuiguang-score-api.service /etc/systemd/system/
sudo systemctl daemon-reload
```

按服务器实际路径、用户、Redis和GPU修改环境文件及service中的User/Group。环境文件不得写入源码仓库真实密码。

## 启动

```bash
sudo systemctl enable --now shuiguang-score-worker.service
sudo systemctl enable --now shuiguang-score-api.service
systemctl status shuiguang-score-worker.service
systemctl status shuiguang-score-api.service
```

Worker必须使用solo池和concurrency=1。启动时默认预热常驻resident；日志出现worker ready后再接流量。

## 停止与重启

```bash
sudo systemctl restart shuiguang-score-worker.service
sudo systemctl restart shuiguang-score-api.service
sudo systemctl stop shuiguang-score-api.service
sudo systemctl stop shuiguang-score-worker.service
```

## 日志

```bash
journalctl -u shuiguang-score-worker.service -f
journalctl -u shuiguang-score-api.service -f
```

健康检查：

```bash
curl http://127.0.0.1:8893/health
```

生产应通过反向代理暴露API，不直接公开Redis或8893端口。
