# 三台机器：旧机完成 Base，两台新机分别运行 RL

推荐使用 `DIRECT_TRANSFER.md` 的服务器直传方式：本地只上传少量代码到旧机，两台新机直接从旧机拉模型和固定实验文件。本页保留本地电脑中转方案，以及两种方式共用的结果下载和合并命令。

旧机仍是 `ssh -p 42172 root@xj-member.bitahub.com`，仅完成当前 Base。新机 A 为 42369（第 0 批），新机 B 为 42186（第 1 批），主机均为 root@xj-member.bitahub.com。第一、第二批 RL 分别运行在另外两台单卡 RTX 4090 上。新机不必等待 Base 完成。

## 1. Mac：给旧机上传停止控制器

以下 Mac 命令在同一个本地终端执行。

```bash
rl_bundle="/Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/code/math_l4_rl_split_v1"
bash "$rl_bundle/upload_base_stop.sh" 42172 root@xj-member.bitahub.com
```

这里只上传外部控制器和参考配置，不上传模型，不改原运行代码或回答文件。

## 2. 旧机：继续当前 Base，完成后停止

在旧机终端执行一次。它保留现有 Base 进程，接着当前进度等待，不从头开始；如果此前启动了旧版自动切换监控器，会精确停用该监控器。

```bash
cd /root/code/math_l4_rl_split_v1 &&
nohup ../.venv/bin/python -u scripts/watch_base_only.py \
  --experiment-root ../math_l4_v1 \
  --run-dir outputs/math_l4_v1_fix1 \
  >> base-stop.log 2>&1 < /dev/null &
```

启动后应看到 `[PIDFD READY]`、`[BASE ONLY WATCH]` 和 `[WAIT BASE]`（若 Base 已结束则进入完整性校验）。旧版本若报 `Safe watcher replacement requires Linux pidfd support.`，原 Base 没有被停止，但控制器已退出；先重新上传更新，再执行上面的启动命令，不要 kill 原 Base，也不要重启 `run.sh all`。日志采用追加方式，判断时看最后的新记录。

同时查看实际生成进度和停止控制器：

```bash
tail -n 30 -f \
  /root/code/math_l4_v1/experiment-fix1.log \
  /root/code/math_l4_rl_split_v1/base-stop.log
```

完成标志是 `[STOPPED AFTER BASE]`。凭据位于 `/root/code/math_l4_rl_split_v1/outputs/base_stop_receipt.json`，包含 `complete=true`、`rl_started=false` 和完整 Base 样本哈希。原流水线的 `run_state.json` 此时可能显示 `interrupted`，表示它按计划结束后续阶段；Base 的完整结果不受影响。

控制器在 Base 完整保存且验证通过后停止原流水线。原流程若抢先进入了 RL，属于该流程的 RL 子进程也会停止，已有文件保留。若另一独立 `run_shard.py` 已经在旧机运行，控制器会明确报错并保留它，避免误杀另一项任务；不要把失败日志当成已停止。

## 3. Mac：增量下载当前模型、数据和结果，准备新机上传包

```bash
bash "$rl_bundle/download_current.sh" 42172 root@xj-member.bitahub.com
```

默认更新 `~/Downloads/math_l4_selected_20261104`。已存在的模型文件会复用；脚本会补齐三个候选的选点证据并完整校验。看到 `[PACKAGE 4/4] READY` 后上传。

## 4. Mac：分别上传到两台新机器

以下已填入两台新机地址：42369 对应第 0 批，42186 对应第 1 批；旧机 42172 仍仅运行 Base。

第一台新机运行第 0 批：

```bash
bash "$rl_bundle/upload_batch.sh" 0 '42369' 'root@xj-member.bitahub.com'
```

第二台新机运行第 1 批：

```bash
bash "$rl_bundle/upload_batch.sh" 1 '42186' 'root@xj-member.bitahub.com'
```

每台需要完整 Base 和选中 adapter。上传不会包含活动的 Base/RL 回答、run state 或虚拟环境；再次上传不会删除服务器已生成的分片回答。

## 5. 第一台新机：安装并运行第 0 批，可断点续跑

新机使用 Linux x86_64、Python 3.11、单张 RTX 4090。依赖约 4 GB，下载有进度且可续传。若 Python 3.11 不在 PATH，先设置 `RLVR_SYSTEM_PYTHON` 为它的可执行路径。

```bash
cd /root/code/math_l4_rl_split_v1 &&
nohup bash -c 'bash setup_worker.sh --shard-index 0 && exec ../.venv/bin/python -u scripts/run_shard.py --experiment-root ../math_l4_v1 --shard-index 0' \
  >> rl-batch-0.log 2>&1 < /dev/null &
```

```bash
tail -n 60 -f /root/code/math_l4_rl_split_v1/rl-batch-0.log
```

## 6. 第二台新机：安装并运行第 1 批，可断点续跑

```bash
cd /root/code/math_l4_rl_split_v1 &&
nohup bash -c 'bash setup_worker.sh --shard-index 1 && exec ../.venv/bin/python -u scripts/run_shard.py --experiment-root ../math_l4_v1 --shard-index 1' \
  >> rl-batch-1.log 2>&1 < /dev/null &
```

```bash
tail -n 60 -f /root/code/math_l4_rl_split_v1/rl-batch-1.log
```

两批各 100 题、每题 16 次，共各 1,600 次回答。再次启动同一批会校验并跳过已完整保存的回答；保持输出目录不变。日志使用 `>>` 保留历史。正在运行时不要重复启动；全局锁会阻止同一 GPU 再开任务。

续跑粒度是完整回答，尚未保存的当前回答会从头生成，不恢复中间 token。半行 JSONL 会被拒绝而非静默覆盖。guard 以 75 退出且已有持久化进展时，最多自动恢复两次；其他故障修复后再运行相同命令。`nohup` 可跨本地关机和 SSH 断线，但远程实例必须持续开机。

## 7. Mac：分别回收 Base 和两批 RL

旧机完成后，再增量下载完整 Base 和选点证据：

```bash
bash "$rl_bundle/download_current.sh" 42172 root@xj-member.bitahub.com
```

第一台新机：

```bash
rl_result="$HOME/Downloads/math_l4_selected_20261104/math_l4_v1/outputs/math_l4_v1_fix1/rl_shards"
mkdir -p "$rl_result"
rsync -avh --progress --partial \
  --include='/shard_0/***' --include='/worker_0_*.json' --exclude='*' \
  -e "ssh -p 42369 -i $HOME/.ssh/dsai4207_4090" \
  root@xj-member.bitahub.com:/root/code/math_l4_v1/outputs/math_l4_v1_fix1/rl_shards/ \
  "$rl_result/"
```

第二台新机：

```bash
rsync -avh --progress --partial \
  --include='/shard_1/***' --include='/worker_1_*.json' --exclude='*' \
  -e "ssh -p 42186 -i $HOME/.ssh/dsai4207_4090" \
  root@xj-member.bitahub.com:/root/code/math_l4_v1/outputs/math_l4_v1_fix1/rl_shards/ \
  "$rl_result/"
```

## 8. Mac：合并与 Base/RL 对比

```bash
rl_snapshot="$HOME/Downloads/math_l4_selected_20261104/math_l4_v1"
python3 -B "$rl_bundle/scripts/merge_shards.py" --experiment-root "$rl_snapshot" &&
python3 -B "$rl_snapshot/scripts/analyze.py" \
  --base outputs/math_l4_v1_fix1/test_base \
  --rl outputs/math_l4_v1_fix1/test_rl_split_merged \
  --output outputs/math_l4_v1_fix1/comparison_split_v1
```

要求完整 Base 和两批完整 RL，校验通过才输出比较。合并和评分不加载模型、不需要本地 GPU。旧机 `test_rl/` 若有少量抢跑回答，会保留原证据，但不自动导入新分片；当前迁移针对已完成训练与选点后的独立 RL 测试。
