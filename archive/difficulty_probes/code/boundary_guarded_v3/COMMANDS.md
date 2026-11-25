# v3 上传、串行执行与查看

在 Mac 终端上传独立包；不需要重新上传模型、环境和旧输出。当前 SSH 端口为 42172。

```bash
rsync -avh --progress --partial \
  --exclude='outputs/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.DS_Store' --exclude='*.log' --exclude='*.pid' \
  -e "ssh -p 42172 -i $HOME/.ssh/dsai4207_4090" \
  --rsync-path="mkdir -p /root/code/boundary_guarded_v3 && rsync" \
  "/Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/code/boundary_guarded_v3/" \
  root@xj-member.bitahub.com:/root/code/boundary_guarded_v3/
```

登录：

```bash
ssh -p 42172 -i "$HOME/.ssh/dsai4207_4090" root@xj-member.bitahub.com
```

在服务器终端，结束旧 v2 测试，再启动本版。停止命令同时匹配脚本名与旧包的工作目录，不使用可能过期的 PID 文件；不删除旧输出。若旧任务已停止，循环不会终止其他目录的任务。等待旧目录运行锁释放，避免与旧进程重叠。

```bash
for guarded_old_pid in $(pgrep -f 'scripts/(run_boundary_parallel|evaluate_boundary)\.py' || true); do
  if [ "$(readlink -f "/proc/$guarded_old_pid/cwd" 2>/dev/null)" = /root/code/boundary_confirm_v2 ]; then
    kill -TERM "$guarded_old_pid" 2>/dev/null || true
  fi
done
if [ -f /root/code/boundary_confirm_v2/outputs/.boundary-run.lock ]; then
  flock -w 30 /root/code/boundary_confirm_v2/outputs/.boundary-run.lock true || exit 1
fi
cd /root/code/boundary_guarded_v3 || exit 1
bash run_guarded.sh --plan || exit 1
nohup bash run_guarded.sh >> guarded.log 2>&1 < /dev/null &
echo $! > guarded.pid
```

实时监控（每 5 秒显示题号、token 数、速率、当前文本、五分钟剩余时间、各类终止计数）：

```bash
cd /root/code/boundary_guarded_v3 || exit 1
python3 -S scripts/monitor_guarded.py --follow
```

总日志：

```bash
tail -n 50 -f /root/code/boundary_guarded_v3/guarded.log
```

继续本版已停止的运行，只需执行上述 `nohup` 与 PID 记录两行。运行锁会拒绝重复启动。关闭本地电脑不会关闭已在远端启动的 nohup 任务，远端开发机需要保持开机。

仅做本地 CPU 回归检查：

```bash
cd "/Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/code/boundary_guarded_v3" || exit 1
python3 -S tests/run_all.py
```

完成后在 Mac 下载整个新输出目录：

```bash
rsync -avh --progress --partial \
  -e "ssh -p 42172 -i $HOME/.ssh/dsai4207_4090" \
  root@xj-member.bitahub.com:/root/code/boundary_guarded_v3/outputs/ \
  "/Users/xu./Downloads/boundary_guarded_v3_outputs/"
```
