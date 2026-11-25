# DSAI4207：两台新机 RL 测评交接

更新：2026-11-04（北京时间）。请接着部署和测评；模型训练及 checkpoint 选择已完成，无需重新训练。

**实验和当前进度**

- 模型：Qwen3.5-2B-Base；RL 方法：LoRA + GRPO。数据：MATH Level 4。正式测试共 200 题，每题 16 次采样，最终比较 Base/RL 的 pass@1、pass@4、pass@16。
- 已选 `formal/checkpoint-384`，验证集答对 309/512（60.35%）。这是验证结果，正式测试尚未汇总。训练不再修改参数；推理需要完整 Base 权重和这个 LoRA adapter。
- 旧机：`ssh -p 42172 root@xj-member.bitahub.com`。原项目 `/root/code/math_l4_v1`，运行目录 `outputs/math_l4_v1_fix1`。最后用户日志确认仍在 `TEST BASE`，停止控制器已通过 `[PIDFD READY]` 并输出 `[WAIT BASE]`；没有更晚的完整进度证据。
- **不要 kill 当前 Base，不要重跑 `run.sh all`，不要重复启动控制器。** 它等待 Base 完整保存并验证，再停止原流水线。完成标志为 `[STOPPED AFTER BASE]`，凭据为 `/root/code/math_l4_rl_split_v1/outputs/base_stop_receipt.json`。原 run state 随后显示 `interrupted` 可以是正常的计划停止。
- 两台额外机器分别跑 RL 第 0、1 批，各 100 题 × 16 = 1,600 次回答。它们可立即开始，无需等待 Base。机器 A 为 `ssh -p 42369 root@xj-member.bitahub.com`（第 0 批），机器 B 为 `ssh -p 42186 root@xj-member.bitahub.com`（第 1 批）；尚无新机成功启动的证据。旧机 42172 不属于这两台。

**换本地电脑要带什么**

推荐保留 `project/code/` 的目录结构；最少需要以下文件。新本地环境需 Python 3.11+、bash、rsync、SSH，无需 GPU。

```text
code/math_l4_rl_split_v1/                         # 完整目录，当前版本 1.2.1
code/scripts/prepare_packages.py
code/requirements-gpu.txt
code/packages/linux-cp311-cu128/manifest.json
code/packages/linux-cp311-cu128/requirements-lock.txt
```

本地 `.venv` 和已下载的 Linux wheels 不必搬。可选携带旧电脑 `~/Downloads/math_l4_selected_20261104/`，放到新电脑同一相对位置以复用模型下载。SSH 脚本默认使用 `$HOME/.ssh/dsai4207_4090`；新电脑须具备可用登录权限，密钥路径不同时设置 `RLVR_SSH_KEY`。不要把私钥内容写进交接文档或发给 AI。

**推荐传输：旧机直接到两台新机**

完整命令见 `code/math_l4_rl_split_v1/DIRECT_TRANSFER.md`。本地在 `code` 目录执行一次小型代码同步：

```bash
bash math_l4_rl_split_v1/upload_direct_tools.sh 42172 root@xj-member.bitahub.com
```

看到 `[DIRECT TOOLS READY]` 后，旧机才具备已同步的完整分批代码。两台新机分别按该文档第 2 节用 rsync 从旧机取 `math_l4_rl_split_v1`，然后运行：

```bash
cd /root/code/math_l4_rl_split_v1 &&
bash pull_from_source.sh 42172 root@xj-member.bitahub.com
```

模型和数据走旧机→新机，不经过本地电脑。每台接收完整 Base（约 4.55 GB）、checkpoint-384 LoRA 参数（约 43.7 MB），以及 128/256/384 的选点证据；实际推理使用 384。出现 `[DIRECT READY]` 才继续安装运行。新机需能 SSH 登录旧机；此时前台完成认证，随后才用 nohup 安装和测评。两台新机的实际传输尚未确认。

旧机至少保留到 Base 完成、两台新机取完文件并通过校验。原本经本地电脑中转的方案仍在 `COMMANDS.md`；该方案的 `payload/ready.json` 含本地绝对路径，换电脑后必须重新构建。

**两台新服务器：安装并运行**

每台使用 Linux x86_64、Python 3.11、单张 RTX 4090，与 Base 的测评条件保持一致；新机配置尚需核实。安装脚本下载固定依赖并检查环境，必要时设置 `RLVR_SYSTEM_PYTHON` 指向 Python 3.11。

机器 A（42369，第 0 批）：

```bash
cd /root/code/math_l4_rl_split_v1 &&
nohup bash -c 'bash setup_worker.sh --shard-index 0 && exec ../.venv/bin/python -u scripts/run_shard.py --experiment-root ../math_l4_v1 --shard-index 0' \
  >> rl-batch-0.log 2>&1 < /dev/null &
```

机器 B（42186，第 1 批）：

```bash
cd /root/code/math_l4_rl_split_v1 &&
nohup bash -c 'bash setup_worker.sh --shard-index 1 && exec ../.venv/bin/python -u scripts/run_shard.py --experiment-root ../math_l4_v1 --shard-index 1' \
  >> rl-batch-1.log 2>&1 < /dev/null &
```

各自用 `tail -n 60 -f /root/code/math_l4_rl_split_v1/rl-batch-0.log` 或 `rl-batch-1.log` 看日志。中断后运行相同命令、保持输出目录不变：完整保存的回答会校验并跳过，未保存的当前回答重新生成。进程仍在运行时不要重复启动。

**后续与注意事项**

- 分片结果：`/root/code/math_l4_v1/outputs/math_l4_v1_fix1/rl_shards/shard_0/`、`shard_1/`。完整下载及合并命令见 `code/math_l4_rl_split_v1/COMMANDS.md` 第 7、8 节。最终合并 3,200 条 RL 回答，再与 3,200 条 Base 回答比较。
- 旧机实际日志：`/root/code/math_l4_v1/experiment-fix1.log`；当前有效控制器日志：`/root/code/math_l4_rl_split_v1/base-stop-fix.log`。COMMANDS.md 的通用示例使用 `base-stop.log`，本次排查请看 `base-stop-fix.log`。
- 上传修复：1.2 的绝对路径 `/./` 配合 macOS openrsync 可能造成文件落错目录。1.2.1 已改为先进入 code 再用相对路径，并在目标机逐文件校验 SHA256。此前看到 rsync 成功但找不到 `pull_from_source.sh`，需重新上传修复版小型代码，再让新机拉取；不要删除模型或回答。
- 已解决：旧机 conda Python 3.11.15 缺少两个 pidfd 接口；1.1.1 用 Linux 系统调用兼容，用户已确认服务器自检成功。原 Base 进程未被打断。无需更换 Python 或重新安装旧环境。
- 保留冻结数据、判分器、采样种子、原生上下文上限、300 秒超时与重复检测策略；不要在迁移中改评分规则或吞掉损坏缓存。少量历史判分/标注问题应在保存原始回答后另行统一审计。
- 原有 185 项 CPU 检查通过，新增 45 项直传检查也通过，包括分片合并、完整回答续跑、上传保护和控制器兼容；新机安装及真实 GPU 推理尚未验证。先阅读 `README.md`、`DIRECT_TRANSFER.md`、`COMMANDS.md`、`memory.md`，确认实际 SSH 地址和现场状态后继续。
