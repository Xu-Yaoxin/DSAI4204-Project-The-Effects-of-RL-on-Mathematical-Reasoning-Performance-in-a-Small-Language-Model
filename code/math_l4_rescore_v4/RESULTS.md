当前项目入口见 [README.md](../../README.md)，详细分析见 [ANALYSIS.md](../../ANALYSIS.md)。以下路径已按迁移记录更新，统计值未变。

**判分器 v4 本地重评分结果（2026-11-06）**

全部 6,400 条回答已统一处理，模型、问题、标准答案和生成内容未改变。只有自然结束的回答应用 v4；超时和重复停止保留原评分。新数据中 old_score 保存旧判分，rescore 保存来源、版本和哈希。

| 模型 | 原正确数 / 3,200 | 新正确数 / 3,200 | 错误转正确 | 正确转错误 | 全部评分变化记录 |
|---|---:|---:|---:|---:|---:|
| base | 1393 | 1393 | 0 | 0 | 0 |
| rl | 1950 | 1968 | 18 | 0 | 25 |

| 新评分指标 | Base | RL | RL−Base（百分点） | 配对 95% 区间 |
|---|---:|---:|---:|---:|
| pass@1 | 43.53% | 61.50% | +17.97 | [+15.09, +20.91] |
| pass@4 | 69.63% | 81.93% | +12.29 | [+8.94, +15.70] |
| pass@16 | 85.50% | 94.00% | +8.50 | [+4.00, +13.00] |

按题目看，16 次中至少答对一次的题数从 171/200 增至 188/200：两者均成功 167 题，仅 RL 成功 21 题，仅 Base 成功 4 题，两者均未成功 8 题。25 条评分变化中，18 条恢复为正确，另外 7 条提取出了错误答案，正确数没有增加；没有原先正确的回答被改判错误。

本次结果支持：LoRA-GRPO 提高了这组 MATH Level 4 题目上的单次成功率，也提高了有限采样下的观察覆盖率。单次成功率提升比 pass@16 提升多 9.47 个百分点，配对 95% 区间为 [4.37, 14.75] 个百分点。这支持“收益主要体现在更稳定地答对”，不能据此断言完整推理能力边界不变或已经扩大。

重复停止产生的 unknown 为 Base 211 条、RL 138 条；主指标把这些回答视为没有确认成功，但不将其称为已证明的数学错误。如果分别把 unknown 的未知结局取最好或最坏情况，pass@1 提升仍在 [+11.38, +22.28] 个百分点内，而 pass@16 提升范围为 [-2.50, +14.00] 个百分点。这是敏感性界限，不是置信区间；因此 pass@16 的提升对未知结局处理不完全稳健。

300 秒超时从 219/3,200（6.84%）增至 320/3,200（10.00%），重复停止则从 6.59% 降至 4.31%。三台机器记录的 GPU 均为 RTX 4090，但实际生成速度不同；相同的墙钟时限仍会影响可生成的 token 数。应把指标解释为既定生成与停止协议下的表现，不据此声称 RL 推理更快。

这是根据已观察到的解析问题进行的事后重评分。checkpoint-384 的选择沿用原开发集结果；没有根据测试结果重选模型。bootstrap 仍按题目配对，2,000 次、种子 42。一次训练种子、有限采样、固定 300 秒规则和不同机器的吞吐差异，仍限制结论的推广。

修复限于有明确定义的标题和最终答案句式，不把所有提取失败都改判正确。空 #### 后换行给裸数字、冲突答案、未知或否定表述等仍按既有保守规则处理。该判分器核对最终标量答案，并不证明推理过程全部正确。

**完成后复核与旧结果清理**

用户完成全量重评分后，本次仅检查已有产物，未再次运行判分器。已核对 7 个输出文件、124 个冻结来源文件、2 个来源审计文件的哈希，以及全部 6,400 条回答的生成字段、完整配对采样和 888 条非 EOS 评分。逐题 pass@k 与全部配对 bootstrap 区间另经独立统计复算，结果一致。详见 [final_integrity_review.json](validation/final_integrity_review.json)。这些核对确认产物完整、统计一致，不等于对所有数学推理或提取失败完成了人工裁定。

已删除 9 个被 v4 替代的独立旧报告、图表及中间统计文件，详见 [superseded_scores_cleanup.json](validation/superseded_scores_cleanup.json)。原始下载、模型备份、冻结实验及诊断证据保留；其中旧评分是来源证据，当前 `ready.json` 依赖其哈希，不应删除或改写。后续报告使用本页与 v4 `summary.json`。

**原始回答地址**

- baseline: [samples.jsonl](</Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/results/raw/baseline.jsonl>)
- rl_shard_0: [samples.jsonl](</Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/archive/math_l4/experiment/outputs/math_l4_v1_fix1/rl_shards/shard_0/samples.jsonl>)
- rl_shard_1: [samples.jsonl](</Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/archive/math_l4/experiment/outputs/math_l4_v1_fix1/rl_shards/shard_1/samples.jsonl>)
- rl_merged: [samples.jsonl](</Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/archive/math_l4/experiment/outputs/math_l4_v1_fix1/test_rl_split_merged/samples.jsonl>)

**重评分结果地址**

- base: [samples.jsonl](</Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/results/final/base/samples.jsonl>)
- rl: [samples.jsonl](</Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/results/final/rl/samples.jsonl>)
- 完整 Base/RL 对比: [summary.json](</Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/results/final/summary.json>)
- 全部评分变化: [changes.jsonl](</Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/results/final/changes.jsonl>)
- 完成状态和哈希清单: [ready.json](</Users/xu./Library/Mobile Documents/com~apple~CloudDocs/PolyU/Y4/DSAI 4207/project/results/final/ready.json>)
