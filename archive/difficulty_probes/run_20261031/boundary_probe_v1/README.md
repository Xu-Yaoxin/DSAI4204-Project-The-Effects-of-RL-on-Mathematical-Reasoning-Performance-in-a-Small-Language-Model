# Baseline 难度摸底：100 题 / Boundary probe v1

只评测原始 Qwen3.5-2B-Base，不加载 LoRA adapter、不训练模型。目标是寻找单次成功率（pass@1）约 40%–50% 的题组，为下一轮独立 Base/RL 比较确定合适难度。本数据集没有使用任何模型输出选题；这轮属于探索性难度校准，不是正式确认实验。

## 题目组成

| 题组 | 题数 | 来源及作用 |
|---|---:|---|
| GSM-Symbolic main | 20 | 基础题型的名称和数值变化，作为参照 |
| GSM-Symbolic P1 | 20 | 原题型增加一个条件，观察更长推理链 |
| GSM-Symbolic P2 | 20 | 原题型增加两个条件，观察进一步增难 |
| MATH-500 Level 1–3 数值子集 | 20 | Level 1/2/3 分别 6/7/7 题 |
| MATH-500 Level 4–5 数值子集 | 20 | Level 4/5 各 10 题 |

GSM 三组使用同一批 20 个原题家族，匹配 instance 序号；具体名称和数值可能随来源版本变化，不宣称每个数字都严格受控。已排除此前 200 道主测试题对应的 GSM 家族，并排除与已用训练/开发/测试题完全规范化重合的问题。题目文字保留原文，选择种子为 20261031。

MATH 只保留无需图片且最终答案为整数、有限小数或分数的题。500 条源记录中，42 条因图形需求排除、109 条因非标量答案排除，剩余 349 条用于固定随机抽样。这是特定数值子集，不能冒称完整 MATH-500 成绩。跨数据集的真实难度顺序要看本次结果。

题目清单：`data/boundary_v1/questions.md`。完整数值答案、来源、原答案：`questions.jsonl`。生成提示只使用 `question`，标准答案、原解答和 canary 均不进入提示。清单和原始来源哈希见 `manifest.json`。官方公开题目可能存在预训练接触，当前检查不能证明不存在污染。

## 文件与上传

本地完整源码和原始池位于主 `project/code`；供服务器运行的是独立子目录 `boundary_probe_v1/`。上传后位于 `/root/code/boundary_probe_v1/`，通过相对路径读取服务器已有的 `../models/Qwen3.5-2B-Base` 和 `../.venv/bin/python`。这次不覆盖旧训练代码、判分器、检查点或历史输出，也不需要重新安装依赖或下载模型。

在 Mac 执行，SSH 使用你此前提供的端口 42260：

```bash
cd "/Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/code"
rsync -avh --progress \
  --exclude='outputs/' --exclude='*.log' --exclude='__pycache__/' --exclude='*.pyc' \
  -e "ssh -p 42260 -i $HOME/.ssh/dsai4207_4090" \
  --rsync-path="mkdir -p /root/code/boundary_probe_v1 && rsync" \
  boundary_probe_v1/ root@xj-member.bitahub.com:/root/code/boundary_probe_v1/
```

## 云端双进程后台运行

在服务器终端执行：

```bash
cd /root/code/boundary_probe_v1
nohup bash run_boundary_parallel.sh >> boundary-parallel.log 2>&1 < /dev/null &
```

默认每题独立采样 4 次，共 400 条回答。两个独立进程共享同一张 GPU，各负责 50 题、200 条回答；每进程同时生成一个答案。两个进程都使用原始 Base，不加载 RL adapter。温度 0.6、top-p 0.95、top-k 0，BF16，固定种子。输出上限 1,024 tokens，输入上限 2,048 tokens。相较旧主实验的 512-token 输出预算，本轮增加了长度空间；两个实验的数字不能直接混称同协议结果。

入口先检查模型可用的单张 BF16 GPU、CPU 配额、全局空闲显存和容器可用内存。双进程启动要求至少 14 GiB 空闲显存、28 GiB 可用内存（名义 30 GB 主机通常达不到此余量，应顺序运行）；这是保守的工程门槛，不能保证运行时峰值。每进程默认最多 4 个 CPU intra-op 线程、1 个 inter-op 线程，并按 CPU 配额降低线程数。主机图中 56 GiB 内存和约 4.63 GiB 历史显存峰值支持尝试，1024-token 新预算的实际峰值仍需本轮确认。

每个进程保存独立日志和断点。任一进程失败，主进程停止另一个，保留已有结果，不生成不完整的总成绩。两个进程完成后，严格检查分片归属、400 条完整网格、原始模型、生成协议和评分，再按原题序合并。并行与串行使用相同的题目种子和生成设置，但不承诺不同执行方式下 GPU 浮点运算逐比特一致。

脚本按条保存答案，显示进度、耗时和 ETA。重复执行会检查配置、题目、源码、模型、依赖版本及已有答案，只补生成缺失部分；完成结果会先经过严格缓存校验再复用，不要求空闲 GPU 或初始化 CUDA。若最后一行 JSONL 被强制中断导致不完整，脚本保留文件并报错，不会悄悄删改记录。并发锁会阻止同一包内重复运行。

查看日志：

```bash
tail -n 50 -f /root/code/boundary_probe_v1/boundary-parallel.log
```

完成标志是 `✅ [PARALLEL COMPLETE] 100%`。两个详细日志为 `outputs/boundary_v1_parallel/worker_0.log` 和 `worker_1.log`。关闭本地电脑不会停止已经由 nohup 启动的远程进程；服务器实例本身须保持运行。

可选的快速模式每题只回答一次，输出单独保存，不能把其 20 题/组的比例当成精确边界：

```bash
nohup bash run_boundary_parallel.sh --quick >> boundary-parallel-quick.log 2>&1 < /dev/null &
```

默认与快速模式不要同时运行。串行与并行入口也共享一个运行锁；旧训练位于包外，必须先确认旧训练已结束。单次模式使用 Wilson 区间；默认重复模式按整个题目家族 bootstrap。4 次回答不会把一组 20 道题变成 80 道独立题。

若两进程发生 OOM 或资源不足，在原任务已经退出后，用下面的命令依次恢复同样两个分片，已完成答案会保留：

```bash
nohup bash run_boundary_parallel.sh --sequential >> boundary-parallel.log 2>&1 < /dev/null &
```

顺序恢复仍使用 `outputs/boundary_v1_parallel/`；单 worker 门槛为 7 GiB 空闲显存、14 GiB 可用内存。全新串行对照可运行 `bash run_boundary.sh`，其独立输出为 `outputs/boundary_v1/`，不要与分片恢复混淆。

两个进程不保证加速两倍。普通 CUDA 多进程共享 GPU 涉及上下文时间片，是否提速要看整体耗时和每秒输出 tokens；代码不启用 MPS、不更改服务器 GPU 调度。[NVIDIA 的说明](https://docs.nvidia.com/deploy/mps/architecture.html)。CPU 线程数的限制依据 [PyTorch 多进程建议](https://docs.pytorch.org/docs/2.10/notes/multiprocessing.html)。

## 看哪些结果

默认并行结果位于 `outputs/boundary_v1_parallel/`：

- `worker_0/`、`worker_1/`：独立可恢复分片；`invocation-*.json` 记录本次资源、退出码、总耗时及聚合 tokens/s。
- `base/samples.jsonl`：所有原始回答、正确性、截断和提取诊断。
- `base/manifest.json`：模型和数据身份、生成预算、软件版本、完成状态。
- `analysis/report.md`、`summary.json`：五组 pass@1、95% 区间、pass@4、格式/提取失败率、截断率、长度统计。
- `analysis/calibration_candidates.json`：按距离 45% 的远近排列题组，不自动挑题或选模型。
- `analysis/per_question.jsonl`：每题四次的正确次数，保留所有题目。

优先看 pass@1；pass@4 是四次至少一次答对的比例，两者不能混淆。如果某组 pass@1 落在 40%–50%，先标记为候选；每组仅 20 题，区间通常较宽。若截断或提取失败较多，应先审查回答和预算，不能直接说模型不会做。

确定候选来源后，用同类的未使用新题再确认难度，随后比较 Base/RL。GSM 后续确认须排除整个摸底模板家族，而不只是换一组数字。当前只运行 Base，因此还不能判断哪个题组的 RL 增益最大。

下载结果，在 Mac 执行：

```bash
cd "/Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/code"
mkdir -p outputs/boundary_download
rsync -avh --progress \
  -e "ssh -p 42260 -i $HOME/.ssh/dsai4207_4090" \
  root@xj-member.bitahub.com:/root/code/boundary_probe_v1/outputs/boundary_v1_parallel/ \
  outputs/boundary_download/
```

## 本地验证与复现

主 code 目录内：

```bash
python3 -S scripts/prepare_boundary_data.py --verify
bash run_boundary_parallel.sh --plan
python3 -S tests/run_all.py --output outputs/validation/boundary_v1/test_report.json
```

发布包内的预览命令为 `bash run_boundary_parallel.sh --plan`，只检查包哈希、数据和预算，不需要模型或 GPU。包的源文件清单在 `bundle_manifest.json`，完整原始池保留本地，不重复上传。

真实 GPU 生成及双进程加速尚未实测；本地验证覆盖数据抽样、哈希与隔离、Base-only 调用、提示不含 gold、模拟生成、分片断点复用、完整合并、篡改拒绝、分组统计、资源门槛及上传目录迁移。调度测试确实启动两个 CPU fixture 子进程，验证并发启动、失败清理及顺序恢复；这不能代替真实 GPU 验收。完整测试报告随包保留在 `validation/test_report.json`。

## 来源与许可

Mirzadeh, Iman, et al. 2024. “GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models.” arXiv:2410.05229. [官方数据](https://huggingface.co/datasets/apple/GSM-Symbolic)。固定 revision `93b5b3758d9d9841ffe81d6cd2ae2b030685b078`；数据标注 CC-BY-NC-ND-4.0，许可文本随包保留。

Hendrycks, Dan, et al. 2021. “Measuring Mathematical Problem Solving With the MATH Dataset.” NeurIPS. [MATH 原仓库](https://github.com/hendrycks/math)。本次使用 [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)，其卡片说明来源为 OpenAI PRM800K 的 500 题子集，固定 revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`。保留原 MATH 的 MIT 许可；H4 数据卡未另行声明许可，来源区别在清单中记录。
