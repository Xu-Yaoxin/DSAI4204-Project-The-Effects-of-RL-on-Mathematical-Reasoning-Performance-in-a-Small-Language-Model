# Boundary guarded v3 — 串行、实时输出、五分钟限时

本包复用服务器 `/root/code/.venv` 和 `models/Qwen3.5-2B-Base`，无需重新安装或下载模型。启动入口是 `bash run_guarded.sh`，所有运行路径相对本包；脚本会自动切换到自身目录。

## 本次规则 / Protocol

- 数据与 v2 完全相同：100 题 GSM-Symbolic main、100 题原始 MATH test Level 4 的标量、无图子集；仅测 Base。每题独立采样 4 次，共 800 次回答。
- **串行 / Sequential：同一时间只有一个模型推理进程。** 两个内部数据分片依次执行，禁止同时启动；不是两模型并行。每次生成 1 个回答。
- **单次回答限时 / Answer deadline：300 秒。** 从该次 `generate` 调用前开始计时，包含首 token 等待；模型加载不计入。到时仍未结束，记录 `answer_timeout`，`correct=false`、`reward=0`。即使部分输出包含正确数字，也不能获得分数。
- 停止请求需要当前解码步骤返回。最多再给 15 秒清理时间；如果线程仍无响应，先持久保存本次超时记录，再退出该 worker，由监督进程校验后续跑下一次回答。300 秒是评分截止线，进程清理和重新加载可能额外耗时。
- **重复保护 / Repetition guard：** 每增加 32 个 tokens 检查一次；相同 token 片段连续重复至少 4 次，且覆盖至少 1,024 个 tokens 才打断。检测周期范围为 1–256 tokens。比如同一个 token 连续 1,024 次，或同一个 256-token 片段连续 4 次会触发。它检测精确重复，不能识别所有语义无意义的内容；其他长输出仍受 300 秒限制。
- 保持原生长度额度：`262144 - 实际输入 token 数`。不改温度、top-p、提示词或重复惩罚，不因看到 `####` 就提前结束回答。
- 额外记录连续 120 秒无新 token 的 `no_token_timeout`，用于区分推理失去响应和一直正常输出。该类不是已验证的数学错误。

## 如何计分 / Scoring

| 终止原因 | 记录 | 主指标中的处理 |
|---|---|---|
| 自然 EOS | 数值判分器核对最终答案 | 正确为 1，否则为 0 |
| 五分钟未结束 | `answer_timeout`，明确计错、0 分 | 计入全部 800 次的分母 |
| 精确重复、原生长度耗尽、120 秒无 token | 原因单独记录，`correct/reward=null` | 没有获得成功分；数学正确性未知 |
| OOM、模型异常、文件错误 | 基础设施失败，停止运行并留日志 | 不伪造答错记录 |

主要报告为全部请求次数中的成功率，即 `已确认正确次数 / 请求次数`。另报自然完成后的正确率、超时率、重复率和未知比例，并为未知项给出上下界。不要只用完成回答的正确率选数据，这会忽略失败与超时。超时零分属于本轮明确规定的运行政策，不能据此声称模型的数学推理一定错误。

每条样本保留完整 token IDs、可评分文本 `text`、包含特殊 tokens 的 `raw_text`、时间及停止证据。逐条 token 轨迹位于 `worker_N/live/*.tokens.jsonl`。实时预览保留特殊 tokens，便于发现空白或占位符循环。已写完的终止记录经过校验、flush 和 fsync，重启后跳过；被手动中断而尚未保存完整记录的那一次会重新生成。

## 运行 / Run

上传与停止旧任务的完整命令见 [COMMANDS.md](COMMANDS.md)。先在本地或服务器查看计划：

```bash
bash run_guarded.sh --plan
```

服务器后台执行：

```bash
cd /root/code/boundary_guarded_v3 || exit 1
nohup bash run_guarded.sh >> guarded.log 2>&1 < /dev/null &
echo $! > guarded.pid
```

同一命令用于中断后续跑。程序校验数据、模型、评分和生成协议身份；不混用旧 v2 的 49 条结果。旧文件与旧训练代码不变。

每 5 秒查看当前回答与剩余时间：

```bash
python3 -S scripts/monitor_guarded.py --follow
```

`Ctrl+C` 只关闭查看器，后台继续。总日志与具体模型日志：

```bash
tail -n 50 -f guarded.log
```

```bash
tail -n 30 -f outputs/guarded_v3/worker_0.log outputs/guarded_v3/worker_1.log
```

第二个分片尚未启动时 `worker_1.log` 不存在属于正常状态；用实时查看器即可。当前完整文本在 `outputs/guarded_v3/worker_N/live/current.txt`，合并进度在 `outputs/guarded_v3/live/current.json`。监控只是读取文件，不运行模型。

完成后看 `outputs/guarded_v3/analysis/report.md` 与 `summary.json`；原始合并记录为 `outputs/guarded_v3/base/samples.jsonl`。两片均完整才会合并并统计。出现需要退出 worker 的保护情况，每次启动允许至多 2 次自动重启；如果本片已经完成则无需重启、不占额度。普通报错不自动重试。

## 验证范围 / Validation

```bash
python3 -S tests/run_all.py
```

测试无需模型、GPU 或网络，覆盖真实 CPU 线程队列、子进程超时与退出、缓存恢复、严格串行、监控和评分统计。时间相关测试使用加速测试时钟，不需要真的等待五分钟。计划检查使用真实 200 题与冻结数据哈希；模型包尚未导入。

**本机没有执行 RTX 4090 推理，因此尚不能保证实际吞吐、显存峰值或 CUDA 故障恢复时间。** 启动时要求至少 11 GiB 空闲显存、16 GiB 可用主机内存；该门槛是保守预检，不是显存足够的证明。真实运行仍可能因硬件错误或 OOM 停止；已有记录保留。

实现沿用已安装 Transformers 5.3.0 的 streamer 输出接口及逐解码步骤检查的 MaxTimeCriteria；相关接口依据 [固定版本 generation 源码](https://raw.githubusercontent.com/huggingface/transformers/v5.3.0/src/transformers/generation/utils.py) 与 [stopping criteria 源码](https://raw.githubusercontent.com/huggingface/transformers/v5.3.0/src/transformers/generation/stopping_criteria.py)。超时另由主线程监视，避免只依赖生成线程自行返回。

数据来源、许可和抽样细节见 `data/boundary_confirm_v2/` 及 `provenance/`；保留 v2 来源目录名便于逐字节核对。内部监督脚本沿用 `run_boundary_parallel.py` 文件名，但本版仅接受 `--workers 1`，用户只需使用 `run_guarded.sh`。
