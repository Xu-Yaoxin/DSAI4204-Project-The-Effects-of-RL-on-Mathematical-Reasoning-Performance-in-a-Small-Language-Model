# MATH Level 4 / 2B commands

当前服务器：`ssh -p 42172 root@xj-member.bitahub.com`。新目录 `/root/code/math_l4_v1`，复用 `/root/code/.venv` 和 `/root/code/models/Qwen3.5-2B-Base`。无需再下载模型或安装新依赖。

## 1.0.1 自检修复 / CPU fixture patch

如果旧运行报 `SimpleNamespace has no attribute set_num_threads`，先用下方 rsync 命令更新整包。错误发生在训练前的测试替身中；修复不改变模型、数据、判分器或生产代码。新包哈希与旧失败记录不同，使用新目录启动，保留原记录：

```bash
cd /root/code/math_l4_v1 || exit 1
nohup bash run.sh all --run-dir outputs/math_l4_v1_fix1 >> experiment-fix1.log 2>&1 < /dev/null &
echo $! > experiment-fix1.pid
```

```bash
tail -n 80 -f /root/code/math_l4_v1/experiment-fix1.log
```

```bash
cd /root/code/math_l4_v1 || exit 1
python3 -S scripts/monitor.py --run-dir outputs/math_l4_v1_fix1 --follow
```

此运行以后的续跑、分阶段执行均保留 `--run-dir outputs/math_l4_v1_fix1`。最终结果位于 `outputs/math_l4_v1_fix1/comparison/`。下方不带 `--run-dir` 的通用命令适用于没有旧运行记录的首次启动。

## 通用命令 / General commands

在 Mac 上传独立包；旧结果和新运行的 outputs 都不会被覆盖：

```bash
rsync -avh --progress --partial \
  --exclude='outputs/' --exclude='.venv/' --exclude='models/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' \
  --exclude='*.log' --exclude='*.pid' \
  -e "ssh -p 42172 -i $HOME/.ssh/dsai4207_4090" \
  --rsync-path="mkdir -p /root/code/math_l4_v1 && rsync" \
  "/Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/code/math_l4_v1/" \
  root@xj-member.bitahub.com:/root/code/math_l4_v1/
```

登录服务器：

```bash
ssh -p 42172 -i "$HOME/.ssh/dsai4207_4090" root@xj-member.bitahub.com
```

后台运行完整流程；自动检查文件、数据、环境和 GPU，再做工程验收、pilot、正式训练、验证集选点及 Base/RL 测试：

```bash
cd /root/code/math_l4_v1 || exit 1
nohup bash run.sh all >> experiment.log 2>&1 < /dev/null &
echo $! > experiment.pid
```

实时查看阶段与当前评测回答，每五秒刷新：

```bash
cd /root/code/math_l4_v1 || exit 1
python3 -S scripts/monitor.py --run-dir outputs/math_l4_v1 --follow
```

查看训练指标、checkpoint 保存和报错日志：

```bash
tail -n 80 -f /root/code/math_l4_v1/experiment.log
```

`Ctrl+C` 只停止日志/监控查看，不会终止 nohup 后台任务。远端开发机需要保持运行；本地电脑可关闭。

中断后，重新执行同一条 `nohup bash run.sh all` 命令。程序核验并复用已完成阶段，训练从最新完整且有哈希封存的 checkpoint 恢复；评测复用已保存的完整回答。未完成的那一次回答会重新生成。代码、配置、数据或模型发生变化时，程序拒绝混用旧运行。

只做准备检查，不训练：

```bash
cd /root/code/math_l4_v1 || exit 1
bash prepare.sh
```

只查看完整计划（不加载模型）：

```bash
cd /root/code/math_l4_v1 || exit 1
bash run.sh plan
```

其他阶段入口：`bash run.sh train` 完成工程/pilot/正式训练；`bash run.sh evaluate` 在已接受的正式训练上完成开发选点、Base/RL 测试和分析；`bash run.sh analyze` 只重放最终分析。这些入口仍使用相同的固定配置与目录，不改变实验。

完成后在 Mac 下载整轮结果。正式分析还需开发选点及 checkpoint 身份证据，因此建议保留整个 outputs，不能只下载最终 summary：

```bash
rsync -avh --progress --partial \
  -e "ssh -p 42172 -i $HOME/.ssh/dsai4207_4090" \
  root@xj-member.bitahub.com:/root/code/math_l4_v1/outputs/ \
  "$HOME/Downloads/math_l4_v1_outputs/"
```

本地 CPU 回归与数据复核：

```bash
cd "/Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/code/math_l4_v1" || exit 1
python3 -S tests/run_all.py
python3 -S scripts/prepare_data.py --verify
bash run.sh plan
```
