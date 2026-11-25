# 从旧 Base 服务器直接部署两台 RL 机器

旧机为 `ssh -p 42172 root@xj-member.bitahub.com`。旧机继续当前 Base；两台新机分别执行同一套 `run_shard.py` 的第 0、1 批。每批 100 题、每题 16 次。新机使用 Linux x86_64、CPython 3.11、单张 RTX 4090。

机器 A：`ssh -p 42369 root@xj-member.bitahub.com`，固定第 0 批。机器 B：`ssh -p 42186 root@xj-member.bitahub.com`，固定第 1 批。新机之间不共享输出目录，旧机源端口始终是 42172。

## 1. 本地电脑：将小型分批代码上传旧机一次

在搬好的 `code` 目录执行。无需先把模型下载到本地，也无需已有 `payload/ready.json`。

```bash
bash math_l4_rl_split_v1/upload_direct_tools.sh 42172 root@xj-member.bitahub.com
```

此命令只同步独立分批代码和四个安装配置文件，排除模型、回答、payload 和运行日志；不会重启当前 Base 或控制器。上传会在目标机逐文件校验 SHA256；出现 `[DIRECT TOOLS READY]` 才表示代码已在正确目录且校验通过。如果此前已上传但新机报缺少 `pull_from_source.sh`，请使用 1.2.1 的修复脚本重新执行本节命令，再从新机重新取代码。旧机可以作为两台新机的文件来源，但不要在旧机启动 RL。

## 2. 两台新机：分别从旧机取代码、模型和数据

在每台新机终端执行同一组命令。新机需要 bash、rsync、SSH、flock 和 Python 3.11；如果 Python 不在默认位置，先设置 `RLVR_SYSTEM_PYTHON`。先在前台完成 SSH 认证和传输，按提示输入旧机登录密码；`pull_from_source.sh` 也支持通过 `RLVR_SOURCE_SSH_KEY` 指定新机上已有且获授权的私钥路径。

```bash
mkdir -p /root/code/math_l4_rl_split_v1 &&
rsync -avh --progress --partial \
  --exclude='/payload/' --exclude='/outputs/' --exclude='/validation/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='*.log' --exclude='.DS_Store' \
  -e 'ssh -p 42172' \
  root@xj-member.bitahub.com:/root/code/math_l4_rl_split_v1/ \
  /root/code/math_l4_rl_split_v1/ &&
cd /root/code/math_l4_rl_split_v1 &&
bash pull_from_source.sh 42172 root@xj-member.bitahub.com
```

大文件走旧机到新机的 SSH 连接，不经过本地电脑。传输包含约 4.55 GB Base 权重、选中的约 43.7 MB LoRA adapter、固定代码和数据，以及候选 128/256/384 的选点校验证据。实际推理加载 checkpoint-384。脚本按固定清单取文件，不复制活动的 test_base/test_rl 回答、run state 或优化器状态。

出现 `[DIRECT READY]` 后继续。传输中断可再次执行；完整文件复用，完成后检查哈希与选点记录。SSH 连通性和真实传输速度尚未实测；如果新机不能登录旧机，先处理 SSH 网络或认证。至少等 Base 结束、两台新机取完文件并通过校验，再关闭旧机。

## 3. 安装环境并后台运行各自分片

机器 A（42369）：

```bash
cd /root/code/math_l4_rl_split_v1 &&
nohup bash -c 'bash setup_worker.sh --shard-index 0 && exec ../.venv/bin/python -u scripts/run_shard.py --experiment-root ../math_l4_v1 --shard-index 0' \
  >> rl-batch-0.log 2>&1 < /dev/null &
```

机器 B（42186）：

```bash
cd /root/code/math_l4_rl_split_v1 &&
nohup bash -c 'bash setup_worker.sh --shard-index 1 && exec ../.venv/bin/python -u scripts/run_shard.py --experiment-root ../math_l4_v1 --shard-index 1' \
  >> rl-batch-1.log 2>&1 < /dev/null &
```

分别查看 `tail -n 60 -f /root/code/math_l4_rl_split_v1/rl-batch-0.log` 或 `rl-batch-1.log`。

中断后重跑同一启动命令，会校验并跳过完整已保存回答；当前未保存的回答重新生成。保持输出目录不变，不要在进程仍运行时重复启动或重新拉文件。安装阶段约 4 GB 固定依赖由各新机下载。后续下载两批结果及合并按 `COMMANDS.md` 第 7、8 节执行。

旧机已有的停止控制器继续工作，实际日志为 `/root/code/math_l4_rl_split_v1/base-stop-fix.log`。不要 kill Base，也不要执行旧项目 `run.sh all`。
