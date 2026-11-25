# DSAI4207 RLVR v1.4.0

**当前 MATH Level 4 最终结果入口（2026-11-06）：[FINAL_RESULTS.md](outputs/FINAL_RESULTS.md)。** 原始 Base/RL 回答和最终 v4 评分的完整地址均在该索引中，评分目录为 `outputs/rescore_20261106/`。本文件后文的 `outputs/rescore_v1_4/` 与 `rescore_local.sh` 属于之前的 GSM8K 实验。

## 100 题 Baseline 难度摸底 / Boundary probe

新增独立评测包 `boundary_probe_v1/`：5 个题组各 20 题，只测原始 Base；支持同一张 4090 上两个进程分片生成、断点恢复、完整合并与分组成功率统计。服务器现有环境与模型可直接复用。上传、nohup、日志和结果下载命令见 [README_BOUNDARY.md](README_BOUNDARY.md)。本地 GPU 推理与实际加速尚未验证。

完整说明见 [READM.md](READM.md)，版本记录见 [memory.md](memory.md)。Qwen3.5-2B-Base 的云端 v1.2.2 实验已完成；本版本修复判分器并提供本地离线重评分（offline rescoring）。

在本地 `project/code` 目录一键运行：

```bash
bash rescore_local.sh
```

需要本机 Python 3.10+，已用 Python 3.11 验证。使用 `python3 -S` 和标准库，不安装包、不加载模型、不联网、不使用 GPU。先运行相关回归测试，再读取相邻的 `../server_results/run_20261031/code/`，对测试集 6,400 条及三个开发候选 2,400 条已保存回答重新判分，并计算配对 bootstrap 区间。

结果位于 `outputs/rescore_v1_4/`，先看 `main_analysis/report.md` 和 `diagnostics.json`。逐条变化在 `changed_scores.jsonl`，完成与完整性凭据为 `ready.json`。相同输入与代码的完整结果会校验后复用；遇到已有目录冲突，用 `--output outputs/rescore_review_2` 指定新目录。其他归档位置可传 `--archive /absolute/path/to/downloaded/code`。

原归档及每条原始评分均保留。新规则同时应用于 Base、RL 和全部开发候选；原 checkpoint-384 不重新选择。这是事后评分敏感性分析，不会修正旧训练中的奖励，也不能证明推理过程正确。

只运行所有无需模型的测试：

```bash
python3 -S tests/run_all.py
```

测试报告：`outputs/validation/v1_4/test_report.json`。未来使用新判分器训练必须使用新运行目录并重新通过工程和 pilot 验收；本次本地命令不会启动训练。云端安装、单卡/多卡及同版本恢复说明见完整文档；v1.3 真实 GPU 中断恢复仍待实测。


## 2026-11-01：200 题原生长度复核

独立上传包在 `boundary_confirm_v2/`。两组各 100 道新题（GSM-Symbolic main、原始 MATH test Level 4 标量无图子集），只测 Base，每题 4 次，共 800 个回答。输出额度按用户确认设置为 262,144 减实际输入 token 数。复用服务器环境和模型，默认同卡两进程，按完整回答续跑。

上传和 nohup 命令见 [COMMANDS.md](boundary_confirm_v2/COMMANDS.md)，协议与运行说明见 [README.md](boundary_confirm_v2/README.md)。新版判分器仅在该独立包使用，旧训练与结果不变。原生长度 GPU 占用与速度尚未实测，长时间重复输出仍可能达到上限；不可用上一轮短输出耗时推算本轮完成时间。


## 2026-11-01：当前边界复核使用 guarded v3

新的入口为 `boundary_guarded_v3/run_guarded.sh`：同一时间仅一个模型进程；每次回答限时 300 秒，超时计错、0 分；固定阈值检测精确重复并提前停止，每 5 秒显示 token 进度与实际输出。复用 v2 相同的 200 题、每题 4 次与原生长度额度，使用新的结果目录，不混入旧停止规则下的回答。

无需新增依赖。上传、结束旧任务、nohup 串行启动与实时查看命令见 [guarded v3 COMMANDS](boundary_guarded_v3/COMMANDS.md)，协议及验证范围见 [guarded v3 README](boundary_guarded_v3/README.md)。下面/上面的 v2 并行说明仅保留为历史版本记录；当前建议以本节为准。v3 已做 CPU 线程、子进程与缓存回归，真实 4090 推理仍需部署验证。


## 2026-11-01：MATH Level 4 下一轮正式实验

新的独立包为 `math_l4_v1/`：Qwen3.5-2B-Base、LoRA-GRPO，512训练/128验证/200测试，判分器3.0.0；用户确认Base/RL各16次采样，比较pass@1/4/16。训练2048-token上限，评测保留原生长度与300秒超时，严格单卡串行。数据排除历史校准题，验证集冻结checkpoint后才测试。

完整上传与nohup命令见 [MATH COMMANDS](math_l4_v1/COMMANDS.md)，协议见 [MATH READM](math_l4_v1/READM.md)。包内已含实际数据，复用服务器现有环境和模型。11套件658项CPU回归及上传异地目录计划通过；实际新GPU训练未在本机执行，`bash run.sh all`会先工程/pilot验收。旧GSM和边界复核包继续保留为历史版本。
