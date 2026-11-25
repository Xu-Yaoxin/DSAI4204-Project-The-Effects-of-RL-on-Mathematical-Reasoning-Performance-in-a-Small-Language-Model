# DSAI4207：RLVR 数学推理实验

我们使用 Qwen3.5-2B-Base，通过可验证奖励强化学习（RLVR）、组相对策略优化（GRPO）和低秩适配（LoRA），研究强化学习是否提高单次答对概率，以及有限次数采样下的题目覆盖率。这是围绕[原论文](docs/papers/2504.13837v5_RLVR_reasoning_capacity.pdf)研究问题的小规模改编复现，模型、训练规模和采样预算与原论文不同。

先完成 GSM8K 实验及难度摸底，再选定 MATH Level 4：512 题训练、128 题开发、200 题测试。按开发集选择 checkpoint-384，Base/RL 各完成 3,200 次测试回答。修复答案提取规则后，使用评分器 v4 统一重评分已有回答，没有重新生成答案或按测试成绩选择模型。

## 直接找到需要的文件

| 内容 | 路径与用途 |
|---|---|
| MATH 训练代码 | [code/math_l4_v1](code/math_l4_v1/)，保留配置、数据、测试和原训练评分器 v3 |
| 训练环境 | [requirement.txt](requirement.txt)，84 个包的固定版本和 SHA-256；安装说明见 [ENVIRONMENT.md](docs/ENVIRONMENT.md) |
| 选中 RL 模型 | [models/rl_checkpoint384](models/rl_checkpoint384/)，包含 LoRA 权重、配置、tokenizer 和最终训练恢复状态 |
| Baseline 原始回答 | [results/raw/baseline.jsonl](results/raw/baseline.jsonl)，200 题 × 16 次，未改变生成内容 |
| RL 原始回答 | [results/raw/rl.jsonl](results/raw/rl.jsonl)，两批合并，共 3,200 条 |
| Baseline 最终评分 | [results/final/base/samples.jsonl](results/final/base/samples.jsonl) |
| RL 最终评分 | [results/final/rl/samples.jsonl](results/final/rl/samples.jsonl) |
| 最终统计／逐题结果 | [summary.json](results/final/summary.json)、[per_question.jsonl](results/final/per_question.jsonl) |
| 结果分析 | [ANALYSIS.md](ANALYSIS.md)，解释指标、置信区间、评分变化和结论限制 |
| 17 页小组展示 | [Canva 演示稿](https://www.canva.com/design/DAHbzTUbxt8/view)（访问沿用 Canva 权限） · [12 分钟英文讲稿](presentation/output/Canva_17_Slides_Script_EN.docx) · [中文讲稿](presentation/output/Canva_17_Slides_Script_ZH.docx) |
| pass@k 结果图 | [Base 与 LoRA-GRPO 对比图](figures/figure2_pass_at_k_endpoint_fit_log2.png) · [根据逐题评分重绘](figures/plot_figure2_endpoint_fits.py) |
| 原论文与 proposal | [docs/papers](docs/papers/)、[proposal PDF](docs/DSAI4207_LLM_Proposal_Overleaf%284%29.pdf) |
| 首轮 GSM8K 实验 | [archive/gsm8k](archive/gsm8k/)，保留训练代码、参数、回答、训练报告及原／重评结果 |
| 难度摸底与来源证据 | [archive/README.md](archive/README.md) |

原始回答文件中的评分是生成时的旧评分。当前结论使用 `results/final/`：其中 `text` / `raw_text` 保留回答，`correct` / `status` / `reward` 是最终评分，`old_score` 是旧评分审计记录。原始和最终文件逐条对应，每组均有 3,200 条。

## 当前结果

| 指标 | Base | RL | 提升（百分点） |
|---|---:|---:|---:|
| pass@1 | 43.53% | 61.50% | +17.97 |
| pass@4 | 69.63% | 81.93% | +12.29 |
| pass@16 | 85.50% | 94.00% | +8.50 |

最终正确回答数为 1,393 → 1,968；16 次至少答对一次的题数为 171 → 188。结果支持本实验协议下成功率提高，不能直接证明完整推理能力边界扩大。重复停止、300 秒时限、剩余提取失败和单次训练种子均限制结论；具体证据与敏感性分析见 [ANALYSIS.md](ANALYSIS.md)。

## 使用与复现

以下命令均在 `project` 目录执行。仅检查文件、权重和统计，不调用模型或判分器：

```bash
python3 -B -S maintenance/verify_project.py
```

在 Linux x86_64、Python 3.11 和 NVIDIA GPU 服务器上安装环境、下载固定版本 Base 并进行预检：

```bash
bash setup_environment.sh
```

确认准备完成后，重新运行同一 MATH 训练及评测流程：

```bash
nohup bash train.sh > reproduction.log 2>&1 < /dev/null &
tail -n 60 -f reproduction.log
```

新结果写入 `code/math_l4_v1/outputs/math_l4_reproduction/`，不会覆盖本项目已归档的结果。相同运行目录下重启该命令会按原代码验证缓存并恢复完整检查点。查看计划而不运行模型可执行 `bash train.sh plan --run-dir outputs/math_l4_reproduction`。

若以后需要重新评分，运行 `bash rescore_math.sh`。它使用迁移后的冻结来源并写入全新的 `results/recomputed_v4/`；当前正式结果 `results/final/` 保留不变。本次整理未执行训练或重评分。

## 参数与存档说明

RL 使用 LoRA，`models/rl_checkpoint384/adapter_model.safetensors` 约 43.7 MB，须与固定版本 Base 配合使用。Base 是公开模型，约 4.57 GB；项目保留版本和逐文件哈希，可通过安装脚本重新下载，未复制其完整权重。

最终 MATH checkpoint 已从 Downloads 完整复制，共 13 个文件，并逐项核对封存哈希。额外保留工程测试、pilot 训练及环境记录。当前模型参数、数据、原始回答、评分和主要复核流程不依赖 Downloads；其旧绝对地址只保留在历史来源记录中。

`archive/math_l4/experiment/` 是冻结来源包，包含三个开发候选及选点证据。部分内容与方便直接使用的 `models/`、`results/raw/` 导出相同，这是有记录的保留。不要单独删改冻结包内部文件：原来源清单和最终评分身份依赖其哈希。

首轮 GSM8K 的优化器、调度器和随机数恢复状态已按确认删除；其模型权重、训练记录和答案保留。安装包缓存、重复部署 payload 和临时文件也已清理。路径迁移见 [relocation.json](maintenance/relocation.json)，删除清单见 [cleanup.json](maintenance/cleanup.json)，版本记录见 [memory.md](memory.md)。

## Group presentation

[English 10-minute presentation, Canva link and speaker script](presentation/README.md). The final deck uses the verified MATH Level 4 results and contains 12 main slides plus a reference appendix.
